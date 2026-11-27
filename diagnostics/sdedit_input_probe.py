"""
入力の加工 (guided_diffusion/face_input_transform.py: 抽象化と色の写像) の効果を、代表的な変換の設定ごとに比べる。
変換は scripts/sdedit_sample.py と同じライブラリ関数・同じ使い分けで回す (加工した入力を SDEdit の開始点と ILVR の
参照画像に使う)。乱数は CLI の既定と同じ「全入力で同じシード」。

指標 (参照画像について平均):
  lf16_lum      輝度を画像ごとに標準化してから、16 画素の平均プーリング後の RMSE を、元の入力と比べる。
                色の写像で明るさや色味が変わっても、構図・顔の向きの保持を比べられるようにする
  lf16_cond     加工した入力 (モデルに渡したもの) との 16 画素の平均プーリング後の RMSE
  anime_logit   アニメらしさ (guided_diffusion/anime_score.py)
  saturation, laplacian_var, luminance   ilvr_metrics.style_stats
出力 (--out_dir): <設定>/<入力の加工>/<入力名>.png、inputs/<入力の加工>/<入力名>.png、input_summary.json、
  grid_<設定>.png (行: 入力、列: 入力の加工ごとの出力)、grid_inputs.png (加工した入力)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from guided_diffusion.anime_score import AnimeScore  # noqa: E402
from guided_diffusion.face_input_transform import transform_input  # noqa: E402
from guided_diffusion.ilvr_eye import (  # noqa: E402
    SampleRNG,
    create_diffusion_from_cfg,
    load_image,
    load_model,
    make_resizers,
    read_cfg,
    save_png,
)
from guided_diffusion.ilvr_metrics import style_stats  # noqa: E402
from guided_diffusion.ilvr_ref_transform import ColorTransfer  # noqa: E402
from guided_diffusion.sdedit import sdedit_repeat, sdedit_t0, stop_ratio_to_range_t  # noqa: E402

SETTINGS = {
    # 名前: (t0_ratio, down_N, ilvr_stop_ratio, K)。格子状の探索 (sdedit_grid_search.py) の代表点と SDEdit 単体
    "sd_t0.25": (0.25, 0, None, 1),
    "P1_t0.5_N16_s0.1_K1": (0.5, 16, 0.1, 1),
    "P2_t0.7_N16_s0.2_K2": (0.7, 16, 0.2, 2),
    "P3_t0.7_N32_s0.2_K2": (0.7, 32, 0.2, 2),
}
ABS = dict(abstract_iters=3, abstract_sigma_s=3.0, abstract_sigma_r=8.0)
VARIANTS = {
    # 名前: (transform_input の引数, 色の写像の方式 or None)
    "none": ({}, None),
    "ct_meanstd": ({}, "lab_meanstd"),
    "ct_meanstd_l": ({}, "lab_meanstd_l"),
    "ct_quantile": ({}, "lab_quantile"),
    "abs3": (ABS, None),
    "abs3_q8": ({**ABS, "quant_levels": 8}, None),
    "abs3_xdog": ({**ABS, "xdog_strength": 0.5}, None),
    "abs3_ct_meanstd_l": (ABS, "lab_meanstd_l"),
    "abs3_q8_xdog_ct_meanstd_l": ({**ABS, "quant_levels": 8, "xdog_strength": 0.5}, "lab_meanstd_l"),
}


def quantize(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round() / 127.5 - 1


def lum_std(x):
    x01 = (x + 1) / 2
    lum = 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]
    mu = lum.mean((2, 3), keepdim=True)
    sd = lum.std((2, 3), keepdim=True) + 1e-6
    return (lum - mu) / sd


def lf16(a, b):
    return ((F.avg_pool2d(a, 16) - F.avg_pool2d(b, 16)) ** 2).flatten(1).mean(1).sqrt()


def u8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def tile(rows):
    h = len(rows)
    w = max(len(r) for r in rows)
    g = np.full((h * 258, w * 258, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            g[i * 258:i * 258 + 256, j * 258:j * 258 + 256] = im
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--base_samples", required=True)
    p.add_argument("--ref_list", required=True)
    p.add_argument("--color_stats", default="lists/face_color_stats.json")
    p.add_argument("--score_path", default="diagnostics/outputs/anime_score/anime_score.pt")
    p.add_argument("--out_dir", default="diagnostics/outputs/sdedit_input")
    p.add_argument("--timestep_respacing", default="100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--settings", default=",".join(SETTINGS))
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--grid_refs", type=int, default=4)
    args = p.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    cfg = read_cfg(args.cfg)
    model = load_model(cfg, args.model_path, device)
    diffusion = create_diffusion_from_cfg(cfg, args.timestep_respacing)
    T = diffusion.num_timesteps
    with open(args.ref_list) as f:
        names = [line.strip() for line in f if line.strip()]
    stems = [os.path.splitext(n)[0] for n in names]
    x0 = th.stack([load_image(os.path.join(args.base_samples, n), 256) for n in names])
    scorer = AnimeScore(args.score_path, device)
    settings = args.settings.split(",")
    variants = args.variants.split(",")

    conds = {}
    for v in variants:
        kw, ct = VARIANTS[v]
        c = transform_input(x0, color_transfer=ColorTransfer(args.color_stats, ct) if ct else None, device=device,
                            **kw)
        conds[v] = quantize(c)
        d = os.path.join(args.out_dir, "inputs", v)
        os.makedirs(d, exist_ok=True)
        for s, ci in zip(stems, conds[v]):
            save_png(ci, os.path.join(d, s + ".png"))
    x0d = x0.to(device)
    rows = []
    in_stats = {v: {k: float(t.mean()) for k, t in style_stats(c).items()} for v, c in conds.items()}
    for v in variants:
        in_stats[v]["anime_logit"] = scorer.logit(conds[v]).mean().item()
    outs = {}
    for sname in settings:
        r0, n, stop, k = SETTINGS[sname]
        t0 = sdedit_t0(r0, T)
        resizers = make_resizers(256, 256, n, device=device) if n > 0 else None
        range_t = stop_ratio_to_range_t(stop, T) if stop is not None else 0
        for v in variants:
            t1 = time.time()
            c = conds[v].to(device)
            rng = SampleRNG([args.seed] * len(names), device)
            y = quantize(sdedit_repeat(diffusion, model, c, repeats=k, t0=t0, resizers=resizers, range_t=range_t,
                                       rng=rng))
            sec = time.time() - t1
            d = os.path.join(args.out_dir, sname, v)
            os.makedirs(d, exist_ok=True)
            for s, yi in zip(stems, y):
                save_png(yi, os.path.join(d, s + ".png"))
            outs[(sname, v)] = y.cpu()
            st = style_stats(y)
            r = {"setting": sname, "variant": v, "sec": round(sec, 1),
                 "lf16_lum": lf16(lum_std(y), lum_std(x0d)).mean().item(),
                 "lf16_cond": lf16((y + 1) / 2, (c + 1) / 2).mean().item(),
                 "anime_logit": scorer.logit(y).mean().item(),
                 **{k_: float(t.mean()) for k_, t in st.items()}}
            rows.append(r)
            print(f"{sname} {v}: lf16_lum {r['lf16_lum']:.4f} lf16_cond {r['lf16_cond']:.4f} "
                  f"anime {r['anime_logit']:+.2f} sat {r['saturation']:.3f} lap {r['laplacian_var']:.4f} "
                  f"lum {r['luminance']:.3f} {sec:.0f}s", flush=True)

    with open(os.path.join(args.out_dir, "input_summary.json"), "w") as f:
        json.dump({"args": vars(args), "settings": {s: SETTINGS[s] for s in settings},
                   "variants": {v: {"transform": VARIANTS[v][0], "color_transfer": VARIANTS[v][1]} for v in variants},
                   "input_stats": in_stats, "rows": rows}, f, indent=2, ensure_ascii=False)
    n_show = min(args.grid_refs, len(names))
    Image.fromarray(tile([[u8(conds[v][i]) for v in variants] for i in range(n_show)])).save(
        os.path.join(args.out_dir, "grid_inputs.png"))
    for sname in settings:
        Image.fromarray(tile([[u8(x0[i])] + [u8(outs[(sname, v)][i]) for v in variants] for i in range(n_show)])).save(
            os.path.join(args.out_dir, f"grid_{sname}.png"))
    with open(os.path.join(args.out_dir, "grid.txt"), "w") as f:
        f.write("grid_inputs.png の列: " + ", ".join(variants) + "\n")
        f.write("grid_<設定>.png の列: 入力, " + ", ".join(variants) + "\n")


if __name__ == "__main__":
    main()
