"""
アニメの目 (学習データ) と実写の目 (ILVR の参照画像) の違いを、ILVR が合わせ込む低周波で比べる。

  - 統計: 平均輝度、中央 (目の付近) と上部 1/4 (眉・影) の輝度、暗い画素 (輝度 < 0.25) の割合、彩度、
    ラプラシアンの分散、RGB の平均
  - dataset_lowpass.png: 行 = アニメ / アニメ 8x8 / アニメ 4x4 / 実写 / 実写 8x8 / 実写 4x4
    (縮小して条件付けと同じ Resizer で拡大し直したもの。ILVR が出力に写す成分)
  - dataset_mean.png: アニメと実写の平均画像

使い方: docker compose run --rm shell python diagnostics/dataset_lowfreq.py
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch as th
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from guided_diffusion.ilvr_eye import load_image, make_resizers  # noqa: E402
from guided_diffusion.ilvr_metrics import style_stats  # noqa: E402


def read_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def stats(x):
    x01 = (x + 1) / 2
    lum = 0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]
    s = {k: v.mean().item() for k, v in style_stats(x).items()}
    s.update(luminance_center=lum[:, 12:20, 8:24].mean().item(), luminance_top_quarter=lum[:, :8].mean().item(),
             dark_frac=(lum < 0.25).float().mean().item(), rgb_mean=x01.mean((0, 2, 3)).tolist())
    return s


def save_grid(rows, path, scale=3):
    h = 34
    g = np.full((len(rows) * h + 2, max(len(r) for r in rows) * h + 2, 3), 255, np.uint8)
    for r, row in enumerate(rows):
        for c, x in enumerate(row):
            g[2 + r * h:2 + r * h + 32, 2 + c * h:2 + c * h + 32] = (
                ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy())
    Image.fromarray(g).resize((g.shape[1] * scale, g.shape[0] * scale), Image.NEAREST).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anime_dir", default="/data/anime_eye")
    p.add_argument("--anime_list", default="lists/anime_eye_kidref10000.txt")
    p.add_argument("--real_dir", default="/data/real_eye")
    p.add_argument("--real_list", default="lists/real_eye_eval1000.txt")
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "diagnostics", "outputs", "dataset_lowfreq"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = random.Random(0)
    anime_names = rng.sample(read_list(args.anime_list), args.num)
    real_names = rng.sample(read_list(args.real_list), min(args.num, len(read_list(args.real_list))))
    anime = th.stack([load_image(os.path.join(args.anime_dir, n), 32) for n in anime_names])
    real = th.stack([load_image(os.path.join(args.real_dir, n), 32) for n in real_names])

    report = {"anime": stats(anime), "real": stats(real), "num_anime": len(anime), "num_real": len(real)}
    lp = {}
    for N in (4, 8):
        down, up = make_resizers(32, 32, N, "resizer", "cpu")
        lp[N] = (up(down(anime[:16])), up(down(real[:16])))
    save_grid([list(anime[:16]), list(lp[4][0]), list(lp[8][0]), list(real[:16]), list(lp[4][1]), list(lp[8][1])],
              os.path.join(args.out_dir, "dataset_lowpass.png"))
    save_grid([[anime.mean(0), real.mean(0)]], os.path.join(args.out_dir, "dataset_mean.png"), scale=6)
    with open(os.path.join(args.out_dir, "dataset_lowfreq.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
