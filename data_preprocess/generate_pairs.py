#!/usr/bin/env python3
"""
Generate COLMAP‑style pairs.txt for every scene in a dataset tree.

A *scene* is assumed to have the layout

    <root>/<scene>/
        images/            # colour images (unused by this script)
        sparse/
            cameras.txt
            images.txt
            points3D.txt   # optional but speeds‑up overlap computation

The script parses **images.txt** to recover which 3D points each image
observes, computes the number of shared points for every image pair, and
keeps the top‑K neighbours (largest overlap) for each reference image.

It then writes         <scene>/sparse/pairs.txt      in the simple format
expected by `ColmapMVSDataset._read_pairs_txt`::

    ref_id  src_id  score\n
One line per (reference, source) pair.  *score* is just the raw overlap
count but can be any float.

Usage examples
--------------
    python generate_pairs.py  /path/to/dataset_root   --topk 10 --ext .txt

    # if you only want certain scenes:
    python generate_pairs.py  /root  --scenes sceneA sceneB sceneC  --topk 5
"""
import argparse, os, itertools, multiprocessing as mp
from collections import defaultdict
from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# ---------- minimal parsers for COLMAP TXT exports ---------------------------
# -----------------------------------------------------------------------------
def read_images_txt(path):
    """Return dict[id] -> set(point3D_ids) (excludes -1)"""
    imgs = {}
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.lstrip().startswith("#") or line.strip()=="":
                continue
            elems = line.split()
            img_id = int(elems[0])
            # skip qvec tvec camera_id image_name
            pts_line = f.readline().split()
            pt_ids = np.array(list(map(int, pts_line[2::3])))
            imgs[img_id] = set(pt_ids[pt_ids!=-1])
    return imgs

# -----------------------------------------------------------------------------
# ---------------------- helper: pair overlap calculation ---------------------
# -----------------------------------------------------------------------------
def compute_topk_pairs(img_pts: dict[int,set[int]], k: int):
    img_ids = list(img_pts.keys())
    n = len(img_ids)
    pairs = defaultdict(list)  # ref -> list[(src,score)]

    # brute‑force all pairs (could be optimised with indexing but fine for <2k imgs)
    for i in range(n):
        id_i = img_ids[i]
        pts_i = img_pts[id_i]
        for j in range(i+1, n):
            id_j = img_ids[j]
            ov = len(pts_i & img_pts[id_j])
            if ov == 0:
                continue
            pairs[id_i].append((id_j, ov))
            pairs[id_j].append((id_i, ov))

    # pick top‑k per reference (fall back to any image if not enough)
    for ref in img_ids:
        srcs = pairs[ref]
        srcs.sort(key=lambda t: t[1], reverse=True)  # by score
        if len(srcs) < k:
            # pad with remaining images (score 0) to satisfy length
            missing = [s for s in img_ids if s!=ref and s not in [x[0] for x in srcs]]
            srcs.extend([(m, 0.0) for m in missing[:k-len(srcs)]])
        pairs[ref] = srcs[:k]
    return pairs

# -----------------------------------------------------------------------------
# ----------------------------- main routine ----------------------------------
# -----------------------------------------------------------------------------

def process_scene(args):
    scene_path, topk, ext = args
    sparse = scene_path / "sparse"
    images_txt = sparse / f"images{ext}"
    if not images_txt.exists():
        print(f"[WARN] No images{ext} in {sparse}")
        return
    img_pts = read_images_txt(images_txt)
    pairs = compute_topk_pairs(img_pts, topk)

    out_path = sparse / "pairs.txt"
    with open(out_path, "w") as f:
        for ref, lst in pairs.items():
            for src, score in lst:
                f.write(f"{ref} {src} {score}\n")
    print(f" Wrote {out_path.relative_to(scene_path)}  (|imgs|={len(img_pts)})")


def main():
    p = argparse.ArgumentParser(description="Generate pairs.txt for COLMAP scenes")
    p.add_argument("root", type=Path, help="Dataset root containing scene folders")
    p.add_argument("--topk", type=int, default=10, help="number of source views per reference")
    p.add_argument("--scenes", nargs="*", default=None, help="optional explicit list of scene directory names")
    p.add_argument("--ext", choices=[".txt", ".bin"], default=".txt", help="extension of COLMAP model files inside sparse/")
    p.add_argument("--workers", type=int, default=mp.cpu_count())
    args = p.parse_args()

    if args.scenes:
        scene_dirs = [args.root / s for s in args.scenes]
    else:
        scene_dirs = [d for d in args.root.iterdir() if d.is_dir()]

    todo = []
    for scene in scene_dirs:
        if not (scene/"sparse").exists():
            print(f"[SKIP] {scene} has no sparse/")
            continue
        todo.append((scene, args.topk, args.ext))

    if not todo:
        print("Nothing to process.")
        return

    print(f"Processing {len(todo)} scene(s) with top‑k={args.topk} using {args.workers} workers")
    with mp.Pool(processes=args.workers) as pool:
        pool.map(process_scene, todo)

if __name__ == "__main__":
    main()
