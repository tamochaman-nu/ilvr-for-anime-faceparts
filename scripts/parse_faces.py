"""
切り出し済みの実写の顔 (scripts/align_faces.py の出力) を、部位ごとに塗り分けるラベル画像にする
(部位別のストローク化 face_input_transform.stroke_parts の前段)。顔パーシングは ~/anime-face-alignment の
util/face_parsing.py (BiSeNet、CelebAMask-HQ の 19 クラス) をそのまま使う。

このプロジェクトのイメージからは anime-face-alignment のコードと重みが見えないので、anime-face-alignment の
イメージで実行する:

  docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v anime-face-alignment_face_parsing_cache:/home/app/.cache/face_parsing \
    -v "$HOME/anime-face-alignment":/afa:ro -v "$PWD":/work -w /work \
    --entrypoint python anime-face-alignment:latest scripts/parse_faces.py \
    --input_dir input/sample04-12119910_aligned256_head_rotffhq --afa_root /afa

出力 (--out_dir、既定は <input_dir>_parse):
  <入力名の拡張子を除いたもの>.png   ラベル画像 (8 bit のグレー、値 = クラス番号 0〜18。入力と同じ大きさ)
  (<out_dir>_preview.png               クラスを色分けした確認用の画像。--preview 枚)
クラス: 0 背景, 1 肌, 2/3 眉, 4/5 目, 6 眼鏡, 7/8 耳, 9 耳飾り, 10 鼻, 11 口, 12/13 唇, 14 首, 15 首飾り,
        16 服, 17 髪, 18 帽子
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

EXTS = (".png", ".jpg", ".jpeg")
PALETTE = np.array([
    [255, 255, 255], [255, 204, 170], [120, 60, 0], [150, 80, 0], [0, 0, 255], [0, 120, 255], [128, 128, 128],
    [255, 150, 0], [255, 180, 0], [255, 0, 255], [255, 120, 120], [180, 0, 0], [255, 0, 0], [200, 0, 60],
    [0, 200, 0], [0, 255, 200], [100, 100, 255], [40, 40, 40], [160, 0, 160]], dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--afa_root", required=True, help="anime-face-alignment のリポジトリ (util/face_parsing.py)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--preview", type=int, default=8)
    args = p.parse_args()

    sys.path.insert(0, args.afa_root)
    import torch
    from util.face_parsing import parse_class_map_batch

    out_dir = args.out_dir or args.input_dir.rstrip("/") + "_parse"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(args.input_dir) if n.lower().endswith(EXTS))
    prev = []
    for k in range(0, len(names), args.batch_size):
        chunk = names[k:k + args.batch_size]
        imgs = [np.asarray(Image.open(os.path.join(args.input_dir, n)).convert("RGB")) for n in chunk]
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
        cls = parse_class_map_batch(x, args.device).cpu().numpy().astype(np.uint8)
        for n, im, c in zip(chunk, imgs, cls):
            Image.fromarray(c).save(os.path.join(out_dir, os.path.splitext(n)[0] + ".png"))
            if len(prev) < args.preview:
                prev.append(np.concatenate([im, PALETTE[c]], 0))
    print(f"{len(names)} 枚 -> {out_dir}")
    if prev:
        Image.fromarray(np.concatenate(prev, 1)).save(out_dir.rstrip("/") + "_preview.png")


if __name__ == "__main__":
    main()
