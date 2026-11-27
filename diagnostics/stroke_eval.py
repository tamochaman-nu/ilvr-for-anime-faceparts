"""
scripts/sdedit_sample.py の実行結果 (整列後の 256px) をまとめて評価し、比較の画像を作る
(diagnostics/run_sample04_stroke.sh の結果の比較用)。

指標 (実行ごと、全入力の平均):
  anime        出力のアニメらしさ (guided_diffusion/anime_score.py のロジット)
  anime_cond   モデルに渡した入力 (refs_cond、無ければ refs) のアニメらしさ
  lf16_lum     輝度を画像ごとに標準化し、16 画素の平均プーリング後の RMSE を元の入力 (refs) と比べたもの
  saturation, laplacian_var   ilvr_metrics.style_stats
  瞳の色のばらつき (--parse_dir があるとき): 目の領域 (パーシングの目 4, 5 を重心を中心に横 1.2・縦 1.8 倍に
  広げて 2px 膨らませたもの) のうち L* < --eye_l_max の画素の a*, b* の中央値を画像ごとに求め、
  その画素が --eye_min_px 以上ある画像について、ビュー間の標準偏差 (eye_a_std, eye_b_std) と
  平均の彩度 (eye_chroma)、対象の画像の数 (eye_n) を出す
出力: <out>_metrics.json、<out>_<名前>.jpg (--grid "名前=実行1,実行2,..." ごと。行 = 実行 (入力と出力の 2 行)、
      列 = --names)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_diffusion.anime_score import AnimeScore  # noqa: E402
from guided_diffusion.face_input_transform import _scale_region, to_lab  # noqa: E402
from guided_diffusion.ilvr_eye import load_image  # noqa: E402
from guided_diffusion.ilvr_metrics import style_stats  # noqa: E402


def lum_std(x):
    x = (x + 1) / 2
    lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return (lum - lum.mean((2, 3), keepdim=True)) / (lum.std((2, 3), keepdim=True) + 1e-6)


def eye_masks(parse_dir, stems):
    from scipy.ndimage import binary_dilation

    out = []
    for s in stems:
        lb = np.asarray(Image.open(os.path.join(parse_dir, s + ".png")))
        m = _scale_region(lb == 4, 1.2, 1.8) | _scale_region(lb == 5, 1.2, 1.8)
        out.append(binary_dilation(m, iterations=2))
    return np.stack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="名前=実行のディレクトリ をカンマ区切りで")
    p.add_argument("--out", required=True, help="出力の接頭辞 (例 diagnostics/outputs/face_align/t0)")
    p.add_argument("--parse_dir", default=None)
    p.add_argument("--names", default="Cam1_frame_00001,Cam1_frame_00008,Cam2_frame_00006,Cam3_frame_00005,"
                                      "Cam4_frame_00006,Cam5_frame_00006")
    p.add_argument("--grid", action="append", default=[], help="名前=実行の名前1,実行の名前2,... (複数回指定できる)")
    p.add_argument("--score_path", default="diagnostics/outputs/anime_score/anime_score.pt")
    p.add_argument("--eye_l_max", type=float, default=70.0)
    p.add_argument("--eye_min_px", type=int, default=5)
    args = p.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    scorer = AnimeScore(args.score_path, device)
    runs = dict(r.split("=", 1) for r in args.runs.split(","))
    res, paths = {}, {}
    for name, d in runs.items():
        with open(os.path.join(d, "config.json")) as f:
            outs = json.load(f)["outputs"]
        stems = [os.path.splitext(o["input"])[0] for o in outs]
        y = th.stack([load_image(os.path.join(d, o["sample"]), 256) for o in outs])
        x = th.stack([load_image(os.path.join(d, o["ref_png"]), 256) for o in outs])
        c = th.stack([load_image(os.path.join(d, o.get("cond_png", o["ref_png"])), 256) for o in outs])
        yd = y.to(device)
        st = style_stats(yd)
        r = {"anime": scorer.logit(yd).mean().item(), "anime_cond": scorer.logit(c.to(device)).mean().item(),
             "lf16_lum": ((F.avg_pool2d(lum_std(y), 16) - F.avg_pool2d(lum_std(x), 16)) ** 2)
             .flatten(1).mean(1).sqrt().mean().item(),
             **{k: float(v.mean()) for k, v in st.items()}}
        if args.parse_dir:
            em = eye_masks(args.parse_dir, stems)
            lab = to_lab(y).permute(0, 2, 3, 1).numpy()
            ab = []
            for li, mi in zip(lab, em):
                sel = mi & (li[..., 0] < args.eye_l_max)
                if sel.sum() >= args.eye_min_px:
                    ab.append(np.median(li[sel][:, 1:], 0))
            ab = np.array(ab).reshape(-1, 2)
            r.update(eye_n=len(ab), eye_a_std=float(ab[:, 0].std()) if len(ab) else None,
                     eye_b_std=float(ab[:, 1].std()) if len(ab) else None,
                     eye_chroma=float(np.hypot(ab[:, 0], ab[:, 1]).mean()) if len(ab) else None)
        res[name] = r
        paths[name] = {s: (os.path.join(d, o.get("cond_png", o["ref_png"])), os.path.join(d, o["sample"]))
                       for s, o in zip(stems, outs)}
        print(f"{name:22s}", " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()),
              flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + "_metrics.json", "w") as f:
        json.dump({"runs": runs, "metrics": res}, f, indent=1, ensure_ascii=False)
    names = args.names.split(",")
    for gspec in args.grid:
        gname, members = gspec.split("=", 1)
        members = members.split(",")
        g = Image.new("RGB", (258 * len(names), 258 * 2 * len(members)), "white")
        for i, m in enumerate(members):
            for j, n in enumerate(names):
                ci, yi = paths[m][n]
                g.paste(Image.open(ci).convert("RGB"), (258 * j, 258 * 2 * i))
                g.paste(Image.open(yi).convert("RGB"), (258 * j, 258 * (2 * i + 1)))
        g.resize((g.width * 3 // 4, g.height * 3 // 4)).save(f"{args.out}_{gname}.jpg", quality=90)


if __name__ == "__main__":
    main()
