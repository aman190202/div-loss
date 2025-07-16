from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import os
import cv2
import warnings
from collections import defaultdict
from datasets.utils import RandomGamma

# Helper: parse COLMAP intrinsics
def _K_from_params(model, params):
    if model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        raise NotImplementedError(f"Model {model} not supported")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

def _read_cameras(path):
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cid, model, w, h, *params = line.strip().split()
            cid, w, h = int(cid), int(w), int(h)
            params = list(map(float, params))
            cams[cid] = {
                'K': _K_from_params(model, params),
                'width': w,
                'height': h,
                'model': model
            }
    return cams

def _q2R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x*y - z*w),     2 * (x*z + y*w)],
        [    2 * (x*y + z*w), 1 - 2 * (x**2 + z**2),     2 * (y*z - x*w)],
        [    2 * (x*z - y*w),     2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
    ], dtype=np.float32)

def _read_images(path):
    images, name2id = {}, {}
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith('#'):
            i += 1
            continue
        toks = lines[i].split()
        img_id = int(toks[0])
        q = np.array(list(map(float, toks[1:5])), dtype=np.float32)
        t = np.array(list(map(float, toks[5:8])), dtype=np.float32)
        cam_id = int(toks[8])
        name = toks[9]
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = _q2R(q)
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c).astype(np.float32)
        images[img_id] = dict(w2c=w2c, c2w=c2w, cam_id=cam_id, name=name)
        name2id[name] = img_id
        i += 2
    return images, name2id

def _read_pairs(path, topk):
    pairs = defaultdict(lambda: ([], []))
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            ref, src, *score = line.strip().split()
            pairs[int(ref)][0].append(int(src))
            pairs[int(ref)][1].append(float(score[0]) if score else 1.0)
    for ref, (srcs, scores) in pairs.items():
        while len(srcs) < topk:
            srcs.extend(srcs[:topk - len(srcs)])
            scores.extend(scores[:topk - len(scores)])
    return pairs

def _read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        w, h = map(int, f.readline().decode().split())
        scale = float(f.readline().decode().strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        return np.flipud(data.reshape(h, w))

class MVSDataset(Dataset):
    def __init__(self, root, scene_list, mode, nviews=5, ndepths=192, interval_scale=1.06, **kwargs):
        super().__init__()
        self.root = root
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.tgt_w, self.tgt_h = 640, 512
        self.random_view = kwargs.get('random_view', False)

        self.scene_list = (
            [l.strip() for l in open(scene_list)] if isinstance(scene_list, str) else list(scene_list)
        )
        self.metas = self._build_metas()

        self.tf_rgb = transforms.Compose([
            transforms.Resize((self.tgt_h, self.tgt_w)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _build_metas(self):
        metas = []
        for scene in self.scene_list:
            sparse_path = os.path.join(self.root, scene, "sparse")
            cams = _read_cameras(os.path.join(sparse_path, "cameras.txt"))
            imgs, _ = _read_images(os.path.join(sparse_path, "images.txt"))
            pairs = _read_pairs(os.path.join(sparse_path, "pairs.txt"), self.nviews - 1)

            for ref, (srcs, _) in pairs.items():
                metas.append((scene, ref, srcs[:self.nviews - 1], cams, imgs))
        return metas

    def __len__(self):
        return len(self.metas)

    def _get_proj_mat(self, K, w2c):
        P = np.zeros((2, 4, 4), dtype=np.float32)
        P[0] = w2c
        P[1, :3, :3] = K
        return P

    def _load_img(self, path):
        return self.tf_rgb(Image.open(path))

    def __getitem__(self, idx):
        scene, ref_id, src_ids, cams, imgs_meta = self.metas[idx]
        view_ids = [ref_id] + src_ids

        imgs = []
        proj_mats = []
        depth_ms, mask_ms, depth_vals = None, None, None

        for i, vid in enumerate(view_ids):
            meta = imgs_meta[vid]
            cam = cams[meta["cam_id"]]
            img_path = os.path.join(self.root, scene, "images", meta["name"])

            img = self._load_img(img_path)
            h0, w0 = cam["height"], cam["width"]
            sx, sy = self.tgt_w / w0, self.tgt_h / h0

            K = cam["K"].copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy

            proj_mats.append(self._get_proj_mat(K, meta["w2c"]))
            imgs.append(img)

            if i == 0 and self.mode != 'test':
                depth_path = os.path.join(self.root, scene, "depth", f"depth_{vid:08d}.pfm")
                if os.path.isfile(depth_path):
                    depth = _read_pfm(depth_path).astype(np.float32)
                    depth = cv2.resize(depth, (self.tgt_w, self.tgt_h), interpolation=cv2.INTER_NEAREST)
                else:
                    # fallback: random uniform depth
                    near, far = 0.1, 100.0
                    depth = np.random.uniform(near, far, size=(self.tgt_h, self.tgt_w)).astype(np.float32)

                mask = (depth > 0).astype(np.float32)
                depth_ms = {
                    "stage1": cv2.resize(depth, (self.tgt_w // 4, self.tgt_h // 4), interpolation=cv2.INTER_NEAREST),
                    "stage2": cv2.resize(depth, (self.tgt_w // 2, self.tgt_h // 2), interpolation=cv2.INTER_NEAREST),
                    "stage3": depth
                }
                mask_ms = {
                    "stage1": cv2.resize(mask, (self.tgt_w // 4, self.tgt_h // 4), interpolation=cv2.INTER_NEAREST),
                    "stage2": cv2.resize(mask, (self.tgt_w // 2, self.tgt_h // 2), interpolation=cv2.INTER_NEAREST),
                    "stage3": mask
                }
                dmin = np.min(depth[depth > 0])
                dmax = np.max(depth)
                depth_vals = np.linspace(dmin, dmin + self.ndepths * self.interval_scale, self.ndepths, dtype=np.float32)

        # --- ensure depth is always present (random fallback as in custom_train) ---
        if depth_ms is None:
            near, far = 0.1, 100.0
            rand_depth = np.random.uniform(near, far, size=(self.tgt_h, self.tgt_w)).astype(np.float32)
            rand_mask = np.ones_like(rand_depth, dtype=np.float32)
            depth_ms = {
                "stage1": cv2.resize(rand_depth, (self.tgt_w // 4, self.tgt_h // 4), interpolation=cv2.INTER_NEAREST),
                "stage2": cv2.resize(rand_depth, (self.tgt_w // 2, self.tgt_h // 2), interpolation=cv2.INTER_NEAREST),
                "stage3": rand_depth
            }
            mask_ms = {
                "stage1": cv2.resize(rand_mask, (self.tgt_w // 4, self.tgt_h // 4), interpolation=cv2.INTER_NEAREST),
                "stage2": cv2.resize(rand_mask, (self.tgt_w // 2, self.tgt_h // 2), interpolation=cv2.INTER_NEAREST),
                "stage3": rand_mask
            }
            depth_vals = np.linspace(near, far, self.ndepths, dtype=np.float32)

        imgs = torch.stack(imgs, dim=0)
        proj = np.stack(proj_mats, axis=0)
        proj_ms = {
            "stage1": proj,
            "stage2": proj.copy(),
            "stage3": proj.copy()
        }
        proj_ms["stage2"][:, 1, :2] *= 2
        proj_ms["stage3"][:, 1, :2] *= 4

        return {
            "imgs": imgs,
            "proj_matrices": proj_ms,
            "depth": depth_ms,
            "depth_values": depth_vals,
            "mask": mask_ms
        }
