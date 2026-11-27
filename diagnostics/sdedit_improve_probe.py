"""
SDEdit の改善の余地を探る予備実験。条件ごとに scripts/sdedit_sample.py を subprocess で呼び、同じ指標で比べる。

条件 (CONDITIONS): t0、追加の引数、入力の幾何変換 (zoom) の組。
  zoom: 入力を (0.5, ANCHOR_Y) を中心に s 倍に縮めてから変換し、出力を 1/s 倍に戻してから元の入力と比べる。
        アニメ顔のデータ (anime-face-alignment の head 設定) は顎先の高さを FFHQ と同じ 0.906 にそろえ、
        目の高さは 0.570 (FFHQ は 0.469)。顎先を中心に 0.769 倍にすると、目と顎先の高さの両方がアニメの
        データにそろう (横幅や目の間隔はそろわない)。はみ出した外側は端の画素で埋める。
指標 (元の入力と比べる。zoom の条件は戻した後で):
  lpips_to_input   入力との LPIPS
  lf_rmse16        16 画素の平均プーリング後の RMSE (構図・顔の向きがどれだけ保たれたか)
  saturation, laplacian_var  ilvr_metrics.style_stats (彩度、線のくっきりさ)
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from guided_diffusion.ilvr_eye import load_image, save_png  # noqa: E402
from guided_diffusion.ilvr_metrics import style_stats  # noqa: E402

ANCHOR_Y = 0.906  # 顎先の高さ (FFHQ とアニメ顔のデータで共通)
ZOOM = 0.336 / 0.437  # (顎先 - 目) の高さの比: アニメ 0.906-0.570 / FFHQ 0.906-0.469

ILVR32 = "--use_ilvr --down_N 32 --ilvr_stop_ratio 0.2"
ILVR16 = "--use_ilvr --down_N 16 --ilvr_stop_ratio 0.2"
CONDITIONS = [
    # 名前, t0_ratio, 追加の引数, zoom
    ("base_t0.25", 0.25, "--use_ilvr False", None),
    ("base_t0.50", 0.50, "--use_ilvr False", None),
    ("steps1000_t0.25", 0.25, "--use_ilvr False --timestep_respacing 1000", None),
    ("K2_t0.25", 0.25, "--use_ilvr False --sdedit_repeats 2", None),
    ("ilvrN32_t0.50", 0.50, ILVR32, None),
    ("ilvrN32_t0.70", 0.70, ILVR32, None),
    ("ilvrN16_t0.70", 0.70, ILVR16, None),
    ("zoom_t0.25", 0.25, "--use_ilvr False", ZOOM),
    ("zoom_t0.35", 0.35, "--use_ilvr False", ZOOM),
    ("zoom_ilvrN32_t0.70", 0.70, ILVR32, ZOOM),
]


def scale_about(x, s, anchor_y=ANCHOR_Y):
    """[N x 3 x H x W] を (中央, anchor_y) を中心に s 倍する (s < 1 で縮小)。外側は端の画素。"""
    ay = anchor_y * 2 - 1
    theta = th.tensor([[1 / s, 0, 0], [0, 1 / s, ay * (1 - 1 / s)]], dtype=x.dtype, device=x.device)
    grid = F.affine_grid(theta[None].expand(x.shape[0], -1, -1), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bicubic", padding_mode="border", align_corners=False).clamp(-1, 1)


def read_png(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1
    return th.from_numpy(a.transpose(2, 0, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--base_samples", required=True)
    p.add_argument("--ref_list", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default="", help="この名前の条件だけ (カンマ区切り)")
    p.add_argument("--grid_refs", type=int, default=4)
    args = p.parse_args()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    import lpips

    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    with open(args.ref_list) as f:
        names = [line.strip() for line in f if line.strip()]
    stems = [os.path.splitext(n)[0] for n in names]
    x_in = th.stack([load_image(os.path.join(args.base_samples, n), 256) for n in names]).to(device)

    # 縮めた入力 (zoom の条件用)
    zoom_dirs = {}
    for _, _, _, z in CONDITIONS:
        if z is not None and z not in zoom_dirs:
            d = os.path.join(args.out_dir, f"inputs_zoom{z:.3f}")
            os.makedirs(d, exist_ok=True)
            for n, x in zip(names, scale_about(x_in, z)):
                save_png(x, os.path.join(d, os.path.splitext(n)[0] + ".png"))
            zoom_dirs[z] = d
    zoom_list = os.path.join(args.out_dir, "ref_list_png.txt")
    with open(zoom_list, "w") as f:
        f.write("".join(s + ".png\n" for s in stems))

    only = set(args.only.split(",")) if args.only else None
    results, outs = [], {}
    for name, r, extra, z in CONDITIONS:
        if only and name not in only:
            continue
        d = os.path.join(args.out_dir, name)
        base = zoom_dirs[z] if z is not None else args.base_samples
        ref_list = zoom_list if z is not None else args.ref_list
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "sdedit_sample.py"), "--model_path", args.model_path,
               "--cfg", args.cfg, "--base_samples", base, "--ref_list", ref_list, "--seed", str(args.seed),
               "--use_sdedit", "--t0_ratio", str(r), "--out_dir", d, *extra.split()]
        print("$ " + " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True)
        with open(os.path.join(d, "config.json")) as f:
            c = json.load(f)
        y = th.stack([read_png(os.path.join(d, o["sample"])) for o in c["outputs"]]).to(device)
        if z is not None:
            y = scale_about(y, 1 / z)  # 元の入力の位置に戻す
            for s, yi in zip(stems, y):
                os.makedirs(os.path.join(d, "unzoomed"), exist_ok=True)
                save_png(yi, os.path.join(d, "unzoomed", s + ".png"))
        outs[name] = y
        with th.no_grad():
            lv = lp(y, x_in).flatten()
        a01, b01 = (y + 1) / 2, (x_in + 1) / 2
        lf = ((F.avg_pool2d(a01, 16) - F.avg_pool2d(b01, 16)) ** 2).flatten(1).mean(1).sqrt()
        st = style_stats(y)
        results.append({"name": name, "t0_ratio": r, "extra": extra, "zoom": z,
                        "sqrt_alpha_bar": c["derived"]["sdedit_start"]["sqrt_alpha_bar"],
                        "num_steps": c["derived"]["start_index"] + 1,
                        "elapsed_sec": c["elapsed_sec"],
                        "lpips_to_input": lv.mean().item(), "lf_rmse16": lf.mean().item(),
                        "saturation": st["saturation"].mean().item(),
                        "laplacian_var": st["laplacian_var"].mean().item()})
    st_in = style_stats(x_in)
    summary = {"input": {"saturation": st_in["saturation"].mean().item(),
                         "laplacian_var": st_in["laplacian_var"].mean().item()},
               "zoom": ZOOM, "anchor_y": ANCHOR_Y, "results": results}
    with open(os.path.join(args.out_dir, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("name\tsqrt_ab\tsteps\tlpips\tlf16\tsat\tlapvar\tsec")
    for r in results:
        print(f"{r['name']}\t{r['sqrt_alpha_bar']:.3f}\t{r['num_steps']}\t{r['lpips_to_input']:.4f}\t"
              f"{r['lf_rmse16']:.4f}\t{r['saturation']:.4f}\t{r['laplacian_var']:.5f}\t{r['elapsed_sec']}")
    print("input", summary["input"])

    # grid: 入力ごとに 1 行 [入力, 縮めた入力, 各条件 (zoom は戻した後)]
    def u8(x):
        return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()

    n = min(args.grid_refs, len(names))
    z0 = next(iter(zoom_dirs.values())) if zoom_dirs else None
    cols = [u8(x_in[k]) for k in range(n)]
    g_rows = []
    for k in range(n):
        row = [u8(x_in[k])]
        if z0:
            row.append(np.asarray(Image.open(os.path.join(z0, stems[k] + ".png")).convert("RGB")))
        row += [u8(outs[name][k]) for name in outs]
        g_rows.append(row)
    del cols
    w = len(g_rows[0])
    g = np.full((n * 258, w * 258, 3), 255, np.uint8)
    for i, row in enumerate(g_rows):
        for j, im in enumerate(row):
            g[i * 258:i * 258 + 256, j * 258:j * 258 + 256] = im
    Image.fromarray(g).save(os.path.join(args.out_dir, "grid.png"))
    with open(os.path.join(args.out_dir, "grid.txt"), "w") as f:
        f.write("列: 入力" + (", 縮めた入力" if z0 else "") + ", " + ", ".join(outs) + "\n")


if __name__ == "__main__":
    main()
