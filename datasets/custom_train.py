# ---------------------------------------------------------------------------
#  colmap_mvs_dataloader.py   ·  v3 – adds SIMPLE_RADIAL camera support
# ---------------------------------------------------------------------------
"""
Drop‑in PyTorch `Dataset` for COLMAP‑style scene folders **with supervisory
views and optional random depth**. Now supports the following intrinsic
models directly from `cameras.txt` (all others still raise):

* PINHOLE                – fx, fy, cx, cy
* SIMPLE_PINHOLE         – f,  cx, cy           (fx = fy = f)
* SIMPLE_RADIAL          – f,  cx, cy, k        (radial k ignored)
* SIMPLE_RADIAL_FISHEYE  – f,  cx, cy, k        (radial k ignored)

If you need more models (OPENCV, RADIAL, …) just extend the `_K_from_params`
helper at the top of the file.

Folder layout expected for each *scene*:

```
root/
  scene_A/
    images/       # *.jpg / *.png – names must match images.txt
    sparse/
       cameras.txt images.txt pairs.txt           # standard COLMAP TXT export
    depth/        # depth_<image_id>.pfm          # optional
  scene_B/ …
```

Author: OpenAI ChatGPT (2025‑07‑15)
License: MIT
"""
from __future__ import annotations

import os
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple
from torchvision import transforms
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets.utils import RandomGamma

# ════════════════════════════  CAMERA HELPERS  ════════════════════════════




def _K_from_params(model: str, params: List[float]) -> np.ndarray:
    """Build a 3×3 intrinsic matrix *ignoring distortion*.

    Supported models and expected param order come from COLMAP docs.
    """
    if model == "PINHOLE":                 # fx, fy, cx, cy
        fx, fy, cx, cy = params
    elif model in ("SIMPLE_PINHOLE",       # f, cx, cy
                   "SIMPLE_RADIAL",       # f, cx, cy, k
                   "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        raise NotImplementedError(
            f"Camera model '{model}' not implemented. Add logic in _K_from_params().")

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K


def _read_cameras_txt(path: str) -> Dict[int, Dict]:
    """Parse COLMAP **cameras.txt** into {camera_id: {K,w,h,model}}."""
    cams = {}
    with open(path, "r") as f:
        for ln in f:
            if ln.lstrip().startswith("#") or ln.strip() == "":
                continue
            cid, model, w, h, *params = ln.strip().split()
            cid, w, h = int(cid), int(w), int(h)
            params = list(map(float, params))
            K = _K_from_params(model, params)
            cams[cid] = dict(K=K, width=w, height=h, model=model)
    return cams


def _q2R(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) ➜ 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w)],
        [2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)]
    ], dtype=np.float32)


def _read_images_txt(path: str):
    """Parse COLMAP **images.txt**. Returns (images dict, name→id map)."""
    images, name2id = {}, {}
    with open(path, "r") as f:
        lines = [ln.rstrip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("#"):
            i += 1
            continue
        tok = lines[i].split()
        (img_id, qw,qx,qy,qz, tx,ty,tz, cam_id, name) = tok[:10]
        img_id, cam_id = int(img_id), int(cam_id)
        q = np.array([float(qw),float(qx),float(qy),float(qz)], np.float32)
        t = np.array([float(tx),float(ty),float(tz)], np.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3,:3] = _q2R(q)
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c).astype(np.float32)
        # skip POINTS2D line
        i += 2
        images[img_id] = dict(w2c=w2c, c2w=c2w, cam_id=cam_id, name=name)
        name2id[name] = img_id
    return images, name2id


def _read_pairs_txt(path: str, topk: int):
    """Return dict[ref] = ([src_ids], [scores]) from pairs.txt."""
    out = defaultdict(lambda: ([], []))
    with open(path) as f:
        for ln in f:
            if ln.strip()=="" or ln.startswith("#"):
                continue
            parts = ln.strip().split()
            ref, src = map(int, parts[:2])
            score = float(parts[2]) if len(parts) > 2 else 1.0
            out[ref][0].append(src)
            out[ref][1].append(score)
    # pad to length ≥ topk
    for ref,(srcs,scores) in out.items():
        if len(srcs) < topk:
            need = topk - len(srcs)
            srcs.extend(srcs[:need])
            scores.extend(scores[:need])
    return out

# ═══════════════════════  DATASET IMPLEMENTATION  ════════════════════════
class MVSDataset(Dataset):
    """PyTorch Dataset compatible with MVSNet‑style pipelines."""

    def __init__(
        self,
        root: str,
        scene_list,
        mode: str = "train",
        nviews: int = 5,
        ndepths: int = 192,
        depth_interval: float = 1.06,
        random_view: bool = False,
        sup_view_selection: str = "topk",  # "topk" | "score"
        # random_depth: bool = True,
        # depth_range: Tuple[float, float] = (0.1, 100.0),
        # img_transform=None,
        **kwargs
    ):
        super().__init__()
        assert mode in ("train", "val", "test")
        assert sup_view_selection in ("topk", "score")
        self.root = root
        self.scenes = (
            [l.strip() for l in open(scene_list) if l.strip()]
            if isinstance(scene_list, str) else list(scene_list)
        )
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.d_int = depth_interval
        self.random_view = random_view
        self.sup_sel = sup_view_selection
        self.random_depth = True
        self.depth_range = (0.1, 100.0)
        self.img_tf = None
        # target spatial resolution (H,W) – match DTU settings (512×640)
        self.tgt_h, self.tgt_w = 512, 640

        self.metas = self._build_meta_list()
        self.define_transforms()

    # ─────────────────────────── meta list ──────────────────────────────
    def _build_meta_list(self):
        metas = []
        for scene in self.scenes:
            sp = os.path.join(self.root, scene, "sparse")
            cams = _read_cameras_txt(os.path.join(sp, "cameras.txt"))
            imgs, _ = _read_images_txt(os.path.join(sp, "images.txt"))
            pairs_path = os.path.join(sp, "pairs.txt")
            if os.path.isfile(pairs_path):
                pairs = _read_pairs_txt(pairs_path, self.nviews-1)
            else:
                ids = sorted(imgs.keys())
                pairs = {i: ( [j for j in ids if j!=i][:self.nviews-1], [1.0]*(self.nviews-1)) for i in ids}
            for ref,(src_ids,scores) in pairs.items():
                metas.append((scene, ref, src_ids, scores, cams, imgs))
        print(f"[ColmapMVSDataset] {len(metas)} samples over {len(self.scenes)} scenes")
        return metas

    # ─────────────────────── helper functions ───────────────────────────
    @staticmethod
    def _proj(K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
        P = np.zeros((2,4,4), dtype=np.float32)
        P[0] = w2c
        P[1,:3,:3] = K
        return P
    
    def define_transforms(self):
        resize_tf = transforms.Resize((self.tgt_h, self.tgt_w))
        self.transform_aug = transforms.Compose([
            transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
            resize_tf,
            transforms.ToTensor(),
            RandomGamma(min_gamma=0.5, max_gamma=2.0, clip_image=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_seg = transforms.Compose([
            resize_tf,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def _ms(arr: np.ndarray):
        """Quarter / half / full‑res dict."""
        h,w = arr.shape[-2:]
        return {
            "stage1": cv2.resize(arr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(arr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": arr,
        }

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        scene, ref_id, src_ids, scores, cams, imgs_meta = self.metas[idx]

        # choose working views
        if self.random_view:
            perm = torch.randperm(len(src_ids))[: self.nviews - 1]
            view_ids = [ref_id] + [src_ids[i] for i in perm]
        else:
            view_ids = [ref_id] + src_ids[: self.nviews - 1]

        # supervisory set
        if self.sup_sel == "topk":
            sup_ids = [ref_id] + src_ids[: self.nviews - 1]
        else:  # score‑weighted random sample
            probs = np.array(scores); probs /= probs.sum()
            choice = np.random.choice(len(src_ids), self.nviews - 1, False, probs)
            sup_ids = [ref_id] + [src_ids[i] for i in sorted(choice)]

        cache_img, cache_proj = {}, {}
        H = W = None

        def _load(iid):
            if iid in cache_img:
                return
            meta = imgs_meta[iid]
            cam = cams[meta["cam_id"]]
            img_path = os.path.join(self.root, scene, "images", meta["name"])

            # --- read & resize image to target resolution ---
            pil_img = Image.open(img_path)
            pil_img_resized = pil_img.resize((self.tgt_w, self.tgt_h), Image.BILINEAR)
            np_img = np.asarray(pil_img_resized, np.float32) / 255.0

            # --- adjust intrinsics to new resolution ---
            K_orig = cam["K"]
            scale_x = self.tgt_w / cam["width"]
            scale_y = self.tgt_h / cam["height"]
            K = K_orig.copy()
            K[0, 0] *= scale_x      # fx
            K[0, 2] *= scale_x      # cx
            K[1, 1] *= scale_y      # fy
            K[1, 2] *= scale_y      # cy

            cache_img[iid] = np_img
            cache_proj[iid] = self._proj(K, meta["w2c"])
            nonlocal H, W
            H, W = np_img.shape[:2]

        for vid in set(view_ids + sup_ids):
            _load(vid)

        # original images (normalised to 0-1 already in cache)
        imgs = torch.stack([
            torch.from_numpy(cache_img[i].transpose(2, 0, 1)) for i in view_ids
        ])

        # data-augmented RGB images (jitter + gamma) for consistency with
        # other loaders such as `dtu_train.py`
        imgs_aug = torch.stack([
            self.transform_aug(Image.open(os.path.join(self.root, scene, "images", imgs_meta[i]["name"])))
            for i in view_ids
        ])
        proj = np.stack([cache_proj[i] for i in view_ids])
        proj_ms = {"stage1": proj,
                   "stage2": proj.copy(),
                   "stage3": proj.copy()}
        proj_ms["stage2"][:, 1, :2] *= 2
        proj_ms["stage3"][:, 1, :2] *= 4

        sup_imgs = torch.from_numpy(
            np.stack([cache_img[i] for i in sup_ids]).transpose(0, 3, 1, 2)
        ).float()
        # variance‑normalise per‑image
        sup_imgs = (sup_imgs - sup_imgs.mean(dim=(2,3), keepdim=True)) / (
            sup_imgs.var(dim=(2,3), keepdim=True, unbiased=False).sqrt() + 1e-8)
        sup_proj = np.stack([cache_proj[i] for i in sup_ids])
        sup_proj_ms = {"stage1": sup_proj,
                       "stage2": sup_proj.copy(),
                       "stage3": sup_proj.copy()}
        sup_proj_ms["stage2"][:, 1, :2] *= 2
        sup_proj_ms["stage3"][:, 1, :2] *= 4

        # depth / mask
        depth_ms = mask_ms = depth_vals = None
        if self.mode != "test":
            pfm = os.path.join(self.root, scene, "depth", f"depth_{ref_id:08d}.pfm")
            if os.path.isfile(pfm):
                full = self._read_pfm(pfm).astype(np.float32)
                depth_ms = self._ms(full)
                mask_ms = self._ms((full > 0).astype(np.float32))
                depth_vals = np.linspace(full.min(), full.min() + self.d_int * self.ndepths,
                                         self.ndepths, dtype=np.float32)
        if depth_ms is None and self.random_depth:
            near, far = self.depth_range
            full = np.random.uniform(near, far, (H, W)).astype(np.float32)
            depth_ms = self._ms(full)
            mask_ms = self._ms(np.ones_like(full, dtype=np.float32))
            depth_vals = np.linspace(near, far, self.ndepths, dtype=np.float32)
        elif depth_ms is None:
            warnings.warn("Depth missing and random_depth=False; returning None for depth keys.")

        return {
            "imgs": imgs,
            "imgs_aug": imgs_aug,
            "proj_matrices": proj_ms,
            "depth": depth_ms,
            "depth_values": depth_vals,
            "mask": mask_ms,
            "sup_imgs": sup_imgs,
            "sup_proj_matrices": sup_proj_ms,
        }

    # small inline PFM reader (grayscale)
    @staticmethod
    def _read_pfm(path):
        with open(path, "rb") as f:
            head = f.readline().decode().rstrip()
            if head not in ("Pf", "PF"):
                raise ValueError("Not a PFM file")
            w, h = map(int, f.readline().decode().split())
            scale = float(f.readline().decode().rstrip())
            endian = "<" if scale < 0 else ">"
            data = np.fromfile(f, endian + "f")
            return np.flipud(data.reshape(h, w))


# ---------------------------------------------------------------------------
#                                quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open('lists/coolant/train.txt', 'r') as f:
        scene_list = [line.strip() for line in f.readlines()]
    ds = MVSDataset(
        root="/home/works/coolant-dataset/dataset/",
        scene_list=scene_list,
        nviews=5,
        random_view=False,
        sup_view_selection="topk",
    )
    print("Dataset size:", len(ds))
    item = ds[0]
    for k, v in item.items():
        if isinstance(v, dict):
            print(k, {kk: vv.shape for kk, vv in v.items()})
        elif isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, type(v))
