"""
SDEdit の t0 を振って、複数のモデルを同じ入力・同じシードで比べる。変換は scripts/sdedit_sample.py を subprocess で呼ぶ。

  --models  名前=チェックポイント:設定ファイル をカンマ区切りで
  --extra   scripts/sdedit_sample.py にそのまま渡す追加の引数 (既定は SDEdit 単体: "--use_ilvr False")

指標 (画像ごと → 平均):
  rmse_to_input, lpips_to_input  入力との距離 (LPIPS は AlexNet、256px のまま)
  saturation, laplacian_var      彩度 (HSV の S) と輝度のラプラシアンの分散 (線のくっきりさ)。ilvr_metrics.style_stats
--anime_dir を渡すと、そこから --anime_n 枚 (名前順に等間隔) を学習時と同じ前処理で読み、同じ指標の参考値を出す。

出力 (--out_dir): <モデル名>/t<比>/ (CLI の出力)、sweep.csv、sweep_summary.json、grid.png
  grid.png: 入力ごとに、モデルごとの行 [入力, t0 の小さい順の出力 ...]
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import torch as th
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from guided_diffusion.ilvr_eye import load_image  # noqa: E402
from guided_diffusion.ilvr_metrics import style_stats  # noqa: E402


def read_png(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1
    return th.from_numpy(a.transpose(2, 0, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--t0_ratios", default="0.1,0.15,0.2,0.25,0.3,0.5")
    p.add_argument("--base_samples", required=True)
    p.add_argument("--ref_list", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep_respacing", default="100")
    p.add_argument("--extra", default="--use_ilvr False")
    p.add_argument("--grid_refs", type=int, default=4)
    p.add_argument("--anime_dir", default="")
    p.add_argument("--anime_n", type=int, default=64)
    args = p.parse_args()

    models = {}
    for item in args.models.split(","):
        name, rest = item.split("=", 1)
        ckpt, cfg = rest.split(":", 1)
        models[name] = (ckpt, cfg)
    ratios = [float(r) for r in args.t0_ratios.split(",")]
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    import lpips

    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    rows, runs = [], {}
    for name, (ckpt, cfg) in models.items():
        for r in ratios:
            d = os.path.join(args.out_dir, name, f"t{r:.2f}")
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "sdedit_sample.py"), "--model_path", ckpt,
                   "--cfg", cfg, "--base_samples", args.base_samples, "--ref_list", args.ref_list,
                   "--timestep_respacing", args.timestep_respacing, "--seed", str(args.seed),
                   "--use_sdedit", "--t0_ratio", str(r), "--out_dir", d, *args.extra.split()]
            print("$ " + " ".join(cmd[1:]), flush=True)
            subprocess.run(cmd, check=True)
            with open(os.path.join(d, "config.json")) as f:
                c = json.load(f)
            runs[(name, r)] = (d, c)
            x = th.stack([read_png(os.path.join(d, o["sample"])) for o in c["outputs"]]).to(device)
            ref = th.stack([read_png(os.path.join(d, o["ref_png"])) for o in c["outputs"]]).to(device)
            with th.no_grad():
                lv = lp(x, ref).flatten()
            rm = (((x - ref) / 2) ** 2).flatten(1).mean(1).sqrt()
            st = style_stats(x)
            s = c["derived"]["sdedit_start"]
            for i, o in enumerate(c["outputs"]):
                rows.append({"model": name, "t0_ratio": r, "start_index": s["start_index"],
                             "t_original": s["t_original"], "sqrt_alpha_bar": s["sqrt_alpha_bar"], "snr": s["snr"],
                             "input": o["input"], "rmse_to_input": rm[i].item(), "lpips_to_input": lv[i].item(),
                             "saturation": st["saturation"][i].item(), "laplacian_var": st["laplacian_var"][i].item()})

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    keys = ["rmse_to_input", "lpips_to_input", "saturation", "laplacian_var"]
    summary = []
    for name in models:
        for r in ratios:
            rs = [x for x in rows if x["model"] == name and x["t0_ratio"] == r]
            d = {k: rs[0][k] for k in ("model", "t0_ratio", "start_index", "t_original", "sqrt_alpha_bar", "snr")}
            d["n"] = len(rs)
            d.update({k: float(np.mean([x[k] for x in rs])) for k in keys})
            summary.append(d)
    # 入力 (実写) の参考値
    any_run = next(iter(runs.values()))
    refs = th.stack([read_png(os.path.join(any_run[0], o["ref_png"])) for o in any_run[1]["outputs"]]).to(device)
    ref_stats = {k: float(v.mean()) for k, v in style_stats(refs).items()}
    anime_stats = None
    if args.anime_dir:
        names = sorted(f for f in os.listdir(args.anime_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        pick = names[::max(1, len(names) // args.anime_n)][:args.anime_n]
        a = th.stack([load_image(os.path.join(args.anime_dir, n), 256) for n in pick]).to(device)
        anime_stats = {k: float(v.mean()) for k, v in style_stats(a).items()}
        anime_stats["n"] = len(pick)
    with open(os.path.join(args.out_dir, "sweep_summary.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary, "input_style": ref_stats, "anime_style": anime_stats},
                  f, indent=2, ensure_ascii=False)
    print("model\tt0\tidx\tt\tsqrt_ab\tsnr\trmse\tlpips\tsat\tlapvar")
    for s in summary:
        print(f"{s['model']}\t{s['t0_ratio']}\t{s['start_index']}\t{s['t_original']}\t{s['sqrt_alpha_bar']:.3f}\t"
              f"{s['snr']:.3f}\t{s['rmse_to_input']:.4f}\t{s['lpips_to_input']:.4f}\t{s['saturation']:.4f}\t"
              f"{s['laplacian_var']:.5f}")
    print("input", ref_stats)
    print("anime", anime_stats)

    # grid
    n_show = min(args.grid_refs, len(any_run[1]["outputs"]))
    tiles = []
    for k in range(n_show):
        for name in models:
            d0, c0 = runs[(name, ratios[0])]
            row = [os.path.join(d0, c0["outputs"][k]["ref_png"])]
            row += [os.path.join(runs[(name, r)][0], runs[(name, r)][1]["outputs"][k]["sample"]) for r in ratios]
            tiles.append(row)
    w = max(len(r) for r in tiles)
    g = Image.new("RGB", (w * 258, len(tiles) * 258), "white")
    for i, r in enumerate(tiles):
        for j, f in enumerate(r):
            g.paste(Image.open(f).convert("RGB"), (j * 258, i * 258))
    g.save(os.path.join(args.out_dir, "grid.png"))


if __name__ == "__main__":
    main()
