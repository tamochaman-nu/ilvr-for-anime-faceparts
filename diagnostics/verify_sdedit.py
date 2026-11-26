"""
scripts/sdedit_sample.py の動作確認 (受け入れ条件 1〜4)。CLI を subprocess で呼び、出力の PNG を比べる。

  1. 回帰: 全てのオプションをオフ (ILVR だけ) にした出力が、現行 ILVR と一致するか
     (a) --seed_per_file True の出力 と scripts/ilvr_sample_eye.py の出力 (--a_dir の N<down_N>/run.json の seed 0)
     (b) 既定 (全入力で同じノイズ列) の出力 と、同じプロセスで ilvr_eye.ilvr_sample を SampleRNG([seed] * N) で回した結果
  2. SDEdit 単体 (--use_ilvr False): t0_ratio ∈ T0_RATIOS で、入力との距離が t0 とともに増えるか。
     2b: --t0_sqrt_alpha に t0_ratio=0.2 の開始時の √ᾱ を渡すと、同じ番号から始まり出力も一致するか
  3. 組み合わせ (1 枚): SDEdit+ILVR (N16, stop 0.2)、それに --use_mask (--blend_mask_dir)、--use_color_fix、
     全部オン、--t_ref t-1 を足したもの。マスクの外が入力と一致するか (色の補正を含む全部オンでも)
  4. 同じシードで 2 回実行した出力が一致するか (全部オンの設定)

例 (Docker):
  docker compose run --rm -v <FFHQ>:/data/real_face:ro shell python diagnostics/verify_sdedit.py \
      --model_path ddpm/anime-face256-mp128-ema_0.9995_100000.pt --cfg ddpm/anime-face256-mp128.cfg \
      --base_samples /data/real_face --ref_list diagnostics/outputs/stability/ref_list16.txt \
      --a_dir diagnostics/outputs/stability/A --out_dir diagnostics/outputs/sdedit_verify
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

from guided_diffusion.ilvr_eye import (  # noqa: E402
    SampleRNG,
    create_diffusion_from_cfg,
    ilvr_sample,
    load_image,
    load_model,
    make_resizers,
    read_cfg,
)


def read_u8(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def max_diff_u8(a, b):
    return int(np.abs(read_u8(a) - read_u8(b)).max())


def to_u8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(th.int16).permute(1, 2, 0).cpu().numpy()


def run_cli(common, out_dir, extra):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "sdedit_sample.py"), *common, "--out_dir", out_dir, *extra]
    print("$ " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)
    with open(os.path.join(out_dir, "config.json")) as f:
        return json.load(f)


def make_masks(base_samples, names, out_dir, image_size):
    """背景除去済みの白背景の入力から、人物の領域 (白くない画素を 4px 膨らませたもの) を白 = 編集する領域にしたマスク"""
    os.makedirs(out_dir, exist_ok=True)
    for n in names:
        x = (load_image(os.path.join(base_samples, n), image_size) + 1) / 2
        m = (x.min(0, keepdim=True).values < 0.94).float()[None]
        m = F.max_pool2d(m, 9, stride=1, padding=4)[0, 0]
        Image.fromarray((m.numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, n))


T0_RATIOS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7)


def make_grid(args, names, r2, r3, path):
    """1〜4 行目: 入力、現行 ILVR、SDEdit 単体 (T0_RATIOS)。5 行目: マスク、3a〜3e"""
    od = lambda name: os.path.join(args.out_dir, name)  # noqa: E731
    rows = []
    for k, n in enumerate(names):
        st = os.path.splitext(n)[0]
        rows.append([os.path.join(od("1_regress_shared"), "refs", st + ".png"),
                     os.path.join(od("1_regress_shared"), "samples", f"{st}_xT_ilvrN{args.down_N}_rt{args.range_t}"
                                  f"_seed{args.seed}.png")]
                    + [r2[f"t0_ratio={r}"]["outputs"][k] for r in T0_RATIOS])
    rows.append([os.path.join(od("mask_src"), names[0])] + [v["outputs"][0] for v in r3.values()
                                                             if isinstance(v, dict) and "outputs" in v])
    w = max(len(r) for r in rows)
    g = Image.new("RGB", (w * 258, len(rows) * 258), "white")
    for i, r in enumerate(rows):
        for j, f in enumerate(r):
            g.paste(Image.open(f).convert("RGB"), (j * 258, i * 258))
    g.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--base_samples", required=True)
    p.add_argument("--ref_list", required=True)
    p.add_argument("--a_dir", default="diagnostics/outputs/stability/A")
    p.add_argument("--out_dir", default="diagnostics/outputs/sdedit_verify")
    p.add_argument("--down_N", type=int, default=16)
    p.add_argument("--range_t", type=int, default=20)
    p.add_argument("--timestep_respacing", default="100")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(args.ref_list) as f:
        names = [line.strip() for line in f if line.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    lists = {}
    for k in (1, 4):
        lists[k] = os.path.join(args.out_dir, f"ref_list{k}.txt")
        with open(lists[k], "w") as f:
            f.write("".join(n + "\n" for n in names[:k]))
    common = ["--model_path", args.model_path, "--cfg", args.cfg, "--base_samples", args.base_samples,
              "--timestep_respacing", args.timestep_respacing, "--seed", str(args.seed)]
    od = lambda name: os.path.join(args.out_dir, name)  # noqa: E731
    result = {}

    # ---- 1. 回帰
    ilvr = ["--down_N", str(args.down_N), "--range_t", str(args.range_t)]
    c_pf = run_cli(common, od("1_regress_per_file"), ["--ref_list", args.ref_list, "--seed_per_file", *ilvr])
    a_out = {}
    with open(os.path.join(args.a_dir, f"N{args.down_N}", "run.json")) as f:
        a_run = json.load(f)
    assert a_run["args"]["range_t"] == args.range_t and a_run["args"]["down_N"] == args.down_N
    for o in a_run["outputs"]:
        if o["seed"] == args.seed:
            a_out[o["ref"]] = os.path.join(args.a_dir, f"N{args.down_N}", o["sample"])
    diffs = [max_diff_u8(os.path.join(od("1_regress_per_file"), o["sample"]), a_out[o["input"]])
             for o in c_pf["outputs"]]
    result["1a_per_file_vs_ilvr_sample_eye"] = {"n": len(diffs), "max_abs_diff_u8": max(diffs),
                                                "per_image": diffs}

    c_sh = run_cli(common, od("1_regress_shared"), ["--ref_list", lists[4], *ilvr])
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    cfg = read_cfg(args.cfg)
    model = load_model(cfg, args.model_path, device)
    diffusion = create_diffusion_from_cfg(cfg, args.timestep_respacing)
    size = int(cfg["IMAGE_SIZE"])
    x0 = th.stack([load_image(os.path.join(args.base_samples, n), size) for n in names[:4]]).to(device)
    ref_out = ilvr_sample(diffusion, model, x0, make_resizers(size, size, args.down_N, device=device),
                          range_t=args.range_t, rng=SampleRNG([args.seed] * 4, device))
    diffs = [int(np.abs(read_u8(os.path.join(od("1_regress_shared"), o["sample"])) - to_u8(r)).max())
             for o, r in zip(c_sh["outputs"], ref_out)]
    result["1b_shared_vs_ilvr_eye_loop"] = {"n": len(diffs), "max_abs_diff_u8": max(diffs), "per_image": diffs,
                                            "noise_seeds": sorted({o["noise_seed"] for o in c_sh["outputs"]})}

    # ---- 2. SDEdit 単体
    import lpips

    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def dist_to_input(run_dir, c):
        rm, lv = [], []
        for o in c["outputs"]:
            a = th.from_numpy(read_u8(os.path.join(run_dir, o["sample"]))).float()
            b = th.from_numpy(read_u8(os.path.join(run_dir, o["ref_png"]))).float()
            rm.append(((a - b) / 255).pow(2).mean().sqrt().item())
            to_t = lambda u: (u.permute(2, 0, 1)[None] / 127.5 - 1).to(device)  # noqa: E731
            with th.no_grad():
                lv.append(lp(to_t(a), to_t(b)).item())
        return float(np.mean(rm)), float(np.mean(lv))

    r2 = {}
    for r in T0_RATIOS:
        d = od(f"2_sdedit_only_t{r:.2f}")
        c = run_cli(common, d, ["--ref_list", lists[4], "--use_sdedit", "--use_ilvr", "False", "--t0_ratio", str(r)])
        rmse, lpv = dist_to_input(d, c)
        r2[f"t0_ratio={r}"] = {"t0_steps": c["derived"]["sdedit_t0_steps"], **c["derived"]["sdedit_start"],
                               "rmse_to_input": rmse, "lpips_to_input": lpv,
                               "outputs": [os.path.join(d, o["sample"]) for o in c["outputs"]]}
    rows = [v for v in r2.values()]
    r2["lpips_increases_with_t0"] = bool(all(a["lpips_to_input"] < b["lpips_to_input"] for a, b in zip(rows, rows[1:])))
    r2["rmse_increases_with_t0"] = bool(all(a["rmse_to_input"] < b["rmse_to_input"] for a, b in zip(rows, rows[1:])))
    # 2b. --t0_sqrt_alpha: t0_ratio=0.2 の開始時の √ᾱ を渡すと、同じ番号から始まり、出力も一致するはず
    ref_r = r2["t0_ratio=0.2"]
    d = od("2b_sqrt_alpha")
    c = run_cli(common, d, ["--ref_list", lists[4], "--use_sdedit", "--use_ilvr", "False",
                            "--t0_sqrt_alpha", f"{ref_r['sqrt_alpha_bar']:.6f}", "--t0_ratio", "0.9"])
    diffs = [max_diff_u8(os.path.join(d, o["sample"]), p) for o, p in zip(c["outputs"], ref_r["outputs"])]
    r2["2b_t0_sqrt_alpha"] = {"given": round(ref_r["sqrt_alpha_bar"], 6), "derived": c["derived"]["sdedit_start"],
                              "from": c["derived"]["sdedit_t0_from"],
                              "same_start_as_ratio0.2": c["derived"]["start_index"] == ref_r["start_index"],
                              "max_abs_diff_u8_vs_ratio0.2": max(diffs)}
    result["2_sdedit_only"] = r2

    # ---- 3. 組み合わせ (1 枚)
    mask_dir = od("mask_src")
    make_masks(args.base_samples, names[:1], mask_dir, size)
    combo = ["--ref_list", lists[1], "--use_sdedit", "--use_ilvr", "--down_N", "16", "--ilvr_stop_ratio", "0.2"]
    blend = ["--use_mask", "--blend_mask_dir", mask_dir, "--mask_feather", "4"]
    cases = {"3a_sdedit_ilvr": combo,
             "3b_sdedit_ilvr_mask": combo + blend,
             "3c_sdedit_ilvr_cf": combo + ["--use_color_fix"],
             "3d_all_on": combo + blend + ["--use_color_fix", "--save_intermediate"],
             "3e_sdedit_ilvr_tref_t-1": combo + ["--t_ref", "t-1"]}
    r3 = {}
    for name, extra in cases.items():
        c = run_cli(common, od(name), extra)
        paths = [os.path.join(od(name), o["sample"]) for o in c["outputs"]]
        r3[name] = {"saved": all(os.path.exists(pp) for pp in paths), "outputs": paths,
                    "range_t_effective": c["derived"]["range_t_effective"],
                    "ilvr_applied_indices": c["derived"]["ilvr_applied_indices"],
                    "t_ref": c["args"]["t_ref"]}
    # マスクの外 (ぼかした後のマスクが 0 の画素) は入力のまま残るはず。色の補正を含む 3d でも
    stem = os.path.splitext(names[0])[0]
    for name in ("3b_sdedit_ilvr_mask", "3d_all_on"):
        o = read_u8(r3[name]["outputs"][0])
        ref = read_u8(os.path.join(od(name), "refs", stem + ".png"))
        outside = read_u8(os.path.join(od(name), "blend_masks", stem + ".png"))[..., 0] == 0
        r3[name]["outside_mask_max_diff_u8"] = int(np.abs(o - ref)[outside].max()) if outside.any() else None
        r3[name]["outside_mask_ratio"] = float(outside.mean())
    r3["3e_differs_from_3a_max_u8"] = max_diff_u8(r3["3a_sdedit_ilvr"]["outputs"][0],
                                                  r3["3e_sdedit_ilvr_tref_t-1"]["outputs"][0])
    result["3_combinations"] = r3

    # ---- 4. 同じシードで 2 回
    c = run_cli(common, od("4_seed_repeat"), cases["3d_all_on"])
    a = os.path.join(od("4_seed_repeat"), c["outputs"][0]["sample"])
    result["4_same_seed_twice"] = {"run1": r3["3d_all_on"]["outputs"][0], "run2": a,
                                   "max_abs_diff_u8": max_diff_u8(r3["3d_all_on"]["outputs"][0], a)}

    make_grid(args, names[:4], r2, r3, od("grid.png"))
    with open(od("verify.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
