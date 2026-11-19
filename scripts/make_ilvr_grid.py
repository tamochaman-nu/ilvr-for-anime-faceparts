"""
複数の実行結果 (scripts/ilvr_sample_eye.py の出力ディレクトリ) を 1 枚のグリッド画像に並べる。
列は参照画像 (--ref_list の順)、行は 1 行目が参照画像、以降は 実行 x シード。各行の左に見出しを付ける。

使い方の例:
  python scripts/make_ilvr_grid.py --ref_list lists/real_eye_grid16.txt --seeds 0 \
    --runs "uncond=outputs/x/uncond,N8 rt20=outputs/x/N8_rt20" --out outputs/x/grid.png
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

PAD, LABEL_W = 2, 120


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_list", required=True)
    p.add_argument("--runs", required=True, help="見出し=ディレクトリ をカンマ区切りで (見出しを省くとディレクトリ名)")
    p.add_argument("--seeds", default="0")
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.ref_list) as f:
        names = [line.strip() for line in f if line.strip()]
    stems = [os.path.splitext(n)[0] for n in names]
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = []
    for item in args.runs.split(","):
        label, _, d = item.rpartition("=")
        runs.append((label or os.path.basename(d.rstrip("/")), d))

    def load(path):
        return np.array(Image.open(path).convert("RGB"))

    rows = [("reference", [load(os.path.join(runs[0][1], "refs", f"{s}.png")) for s in stems])]
    for label, d in runs:
        with open(os.path.join(d, "run.json")) as f:
            run_seeds = {o["seed"] for o in json.load(f)["outputs"]}
        for seed in seeds:
            if seed in run_seeds:
                rows.append((f"{label} s{seed}" if len(seeds) > 1 else label,
                             [load(os.path.join(d, "samples", f"{s}_seed{seed}.png")) for s in stems]))

    h, w = rows[0][1][0].shape[:2]
    cell_h, cell_w = (h + PAD) * args.scale, (w + PAD) * args.scale
    canvas = Image.new("RGB", (LABEL_W + len(stems) * cell_w + PAD, len(rows) * cell_h + PAD), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (label, imgs) in enumerate(rows):
        y = PAD + r * cell_h
        draw.text((4, y + cell_h // 2 - 6), label, fill="black")
        for c, im in enumerate(imgs):
            tile = Image.fromarray(im).resize((w * args.scale, h * args.scale), Image.NEAREST)
            canvas.paste(tile, (LABEL_W + PAD + c * cell_w, y))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    canvas.save(args.out)
    print(f"-> {args.out} ({len(rows)} rows x {len(stems)} cols)")


if __name__ == "__main__":
    main()
