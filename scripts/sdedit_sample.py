"""
SDEdit / ILVR / マスク合成 / 色の補正を、CLI のオプションで個別にオン・オフして変換する。
ループは guided_diffusion/sdedit.py (ステップ番号の意味と処理順はそちらを参照)。

既定 (--use_sdedit False, --use_ilvr True, --use_mask False, --use_color_fix False) は、現行の ILVR
(scripts/ilvr_sample_eye.py と同じ down_N / range_t / t_ref の計算) と同じ。
bool のオプションは値を省略できる (--use_sdedit は --use_sdedit True と同じ)。

  --use_sdedit         x_T の代わりに、入力を q_sample で t0 までノイズ化した状態から逆過程を始める
  --t0_ratio           t0 = max(1, int(t0_ratio * T)) (T は間引き後のステップ数)。番号 t0-1 から t0 ステップ回す
  --t0_sqrt_alpha      指定すると、√ᾱ_i がこの値に最も近い i を開始の番号にする (--t0_ratio より優先)。
                       どちらの指定でも、開始の番号・元の過程の t・√ᾱ・SNR を config.json の derived.sdedit_start に残す
  --sdedit_repeats     K 回繰り返す。2 回目以降は前回の出力を再ノイズ化して始める (参照画像・マスクの x0 は元の入力)
  --use_ilvr           ILVR の低周波の置き換え (--down_N, --range_t, --t_ref)。--down_N 0 でも無効になる
  --ilvr_stop_ratio    指定すると、i >= int(ratio * T) の間だけ ILVR を使う (range_t = int(ratio * T) - 1 に読み替える)
  --use_mask           各ステップの後に x = m*x + (1-m)*q_sample(x0, i-1)、最後は x = m*x + (1-m)*x0。
                       マスクは --blend_mask_dir/<入力と同じファイル名> (白 = 1 = 編集する領域)。入力と同じ前処理
                       (center_crop_arr) で 256px にして [0, 1] にする。--mask_feather で境界をぼかす (半径 px)。
                       (scripts/ilvr_sample_eye.py の --mask_dir は「マスク付き ILVR」で、意味が違う)
  --use_color_fix      後処理: CIELAB の a*, b* の低周波 (sigma = --color_fix_sigma px) を入力のものに置き換える。
                       --use_mask と併用すると、補正の後に out = m*out + (1-m)*x0 をかけ直す (マスクの外は入力のまま)
  入力の加工 (guided_diffusion/face_input_transform.py。加工した入力を SDEdit の開始点と ILVR の参照画像に使い、
  マスク合成のマスクの外と色の補正には元の入力を使う。加工した入力は refs_cond/ に保存する):
  --abstract_iters     双方向フィルタ (CIELAB) の回数 (0 で無効)。--abstract_sigma_s (px), --abstract_sigma_r (ΔE)
  --quant_levels       L* の段階化の段数 (0 で無効)。--quant_sharpness
  --xdog_strength      XDoG の輪郭線を描き足す濃さ (0 で無効)。--xdog_sigma
  --stroke             均一な色の塗りの集まり (ストローク画) にする (none / generic / parts)。
                       generic: SLIC の超画素 + 色の近い隣どうしの統合 (--stroke_segments, --stroke_compactness,
                       --stroke_merge ΔE)。parts: 顔パーシングのラベル --stroke_parse_dir/<入力名>.png
                       (scripts/parse_faces.py) の部位ごとの平均色。--stroke_shade_levels (2 以上で影の段)、
                       --stroke_eye_scale_w / --stroke_eye_scale_h (目の領域を広げる倍率)、--stroke_eye_style
                       (dark: 目を髪の色で塗る / iris: 白目・瞳 (--stroke_iris_rgb の一定の色)・上まぶたの線)
  --color_transfer     前景の色をアニメ顔のデータの分布に写す (none / lab_meanstd / lab_meanstd_l / lab_quantile /
                       lab_mean)。統計は --color_stats (scripts/fit_face_color_stats.py)、強さは --color_strength
  --seed               乱数のシード。既定では全ての入力で同じノイズ列を使う (視点が違っても同じノイズ)。
                       --seed_per_file True にすると scripts/ilvr_sample_eye.py と同じく、シードとファイル名から
                       入力ごとのシードを決める (ilvr_eye.sample_seed。回帰確認用)
  --save_intermediate  開始時のノイズ画像と、--intermediate_every ステップおきの pred_xstart を保存する

出力 (--out_dir):
  samples/<入力名>_<オプションの要約>_seed<S>.png   例: 00000_sdedit_t0.50_K1_ilvrN16_stop0.20_blendmask_cf_seed0.png
                                                     (--t0_sqrt_alpha のときは sdedit_sa0.80_K1_... の形)
  refs/<入力名>.png                                  前処理した入力
  blend_masks/<入力名>.png                           使ったマスク (ぼかした後。--use_mask のとき)
  intermediate/<入力名>_<要約>_seed<S>/*.png         --save_intermediate のとき
  config.json                                        全てのオプション、導出した値 (t0, range_t など)、出力の一覧、時間
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_diffusion.ilvr_eye import (  # noqa: E402
    RESIZER_CHOICES,
    SampleRNG,
    create_diffusion_from_cfg,
    list_images,
    load_image,
    load_model,
    make_resizers,
    read_cfg,
    sample_seed,
    save_png,
)
from guided_diffusion.face_input_transform import EYE_STYLES, STROKE_CHOICES, transform_input  # noqa: E402
from guided_diffusion.ilvr_ref_transform import COLOR_TRANSFER_CHOICES, ColorTransfer  # noqa: E402
from guided_diffusion.image_datasets import center_crop_arr  # noqa: E402
from guided_diffusion.script_util import str2bool  # noqa: E402
from guided_diffusion.sdedit import (  # noqa: E402
    T_REF_CHOICES,
    color_fix,
    gaussian_feather,
    sdedit_repeat,
    sdedit_t0,
    sdedit_t0_from_sqrt_alpha,
    start_info,
    stop_ratio_to_range_t,
)


def load_mask(path, image_size):
    """白 = 1 のマスク PNG -> [1 x H x W] in [0, 1] (入力と同じ center_crop_arr で image_size にする)"""
    with open(path, "rb") as f:
        im = Image.open(f)
        im.load()
    arr = center_crop_arr(im.convert("L"), image_size).astype(np.float32) / 255.0
    return th.from_numpy(arr)[None]


def option_tag(args, num_timesteps):
    parts = []
    if args.use_sdedit and args.t0_sqrt_alpha is not None:
        parts.append(f"sdedit_sa{args.t0_sqrt_alpha:.2f}_K{args.sdedit_repeats}")
    elif args.use_sdedit:
        parts.append(f"sdedit_t{args.t0_ratio:.2f}_K{args.sdedit_repeats}")
    else:
        parts.append("xT")
    if args.use_ilvr and args.down_N > 0:
        parts.append(f"ilvrN{args.down_N}")
        parts.append(f"stop{args.ilvr_stop_ratio:.2f}" if args.ilvr_stop_ratio is not None else f"rt{args.range_t}")
        if args.t_ref == "t-1":
            parts.append("tm1")  # 参照画像のノイズを t-1 の番号で乗せる (既定の t は付けない)
    else:
        parts.append("noilvr")
    if args.use_mask:
        parts.append("blendmask" + (f"f{args.mask_feather}" if args.mask_feather > 0 else ""))
    if args.use_color_fix:
        parts.append("cf" + (f"{args.color_fix_sigma:g}" if args.color_fix_sigma != 8.0 else ""))
    if args.abstract_iters > 0:
        parts.append(f"abs{args.abstract_iters}")
    if args.stroke == "generic":
        parts.append(f"stg{args.stroke_segments}m{args.stroke_merge:g}")
    elif args.stroke == "parts":
        e = (f"e{args.stroke_eye_scale_w:g}x{args.stroke_eye_scale_h:g}"
             if (args.stroke_eye_scale_w, args.stroke_eye_scale_h) != (1.0, 1.0) else "")
        if args.stroke_eye_style == "iris":
            e += "iris" + "-".join(str(v) for v in args.stroke_iris_rgb)
        parts.append(f"stp{args.stroke_shade_levels}{e}")
    if args.quant_levels > 0:
        parts.append(f"q{args.quant_levels}")
    if args.xdog_strength > 0:
        parts.append(f"xdog{args.xdog_strength:g}")
    if args.color_transfer != "none":
        parts.append(f"ct-{args.color_transfer.replace('lab_', '')}"
                     + (f"{args.color_strength:g}" if args.color_strength != 1.0 else ""))
    return "_".join(parts)


def main():
    args = create_argparser().parse_args()
    assert args.t_ref in T_REF_CHOICES and args.resizer in RESIZER_CHOICES
    assert args.sdedit_repeats >= 1
    assert args.sdedit_repeats == 1 or args.use_sdedit, "--sdedit_repeats > 1 には --use_sdedit が要る"
    assert not args.use_mask or args.blend_mask_dir, "--use_mask には --blend_mask_dir が要る"
    assert args.t0_sqrt_alpha is None or 0 < args.t0_sqrt_alpha < 1, "--t0_sqrt_alpha は 0 と 1 の間"
    device = th.device(args.device)

    cfg = read_cfg(args.cfg)
    model = load_model(cfg, args.model_path, device)
    diffusion = create_diffusion_from_cfg(cfg, args.timestep_respacing)
    image_size = int(cfg["IMAGE_SIZE"])
    T = diffusion.num_timesteps

    if args.ref_list:
        with open(args.ref_list) as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = list_images(args.base_samples)
        if args.num_refs > 0:
            names = names[:args.num_refs]
    stems = [os.path.splitext(n)[0] for n in names]
    x0_all = th.stack([load_image(os.path.join(args.base_samples, n), image_size) for n in names])

    t0 = None
    if args.use_sdedit:
        t0 = (sdedit_t0_from_sqrt_alpha(diffusion, args.t0_sqrt_alpha) if args.t0_sqrt_alpha is not None
              else sdedit_t0(args.t0_ratio, T))
    use_ilvr = args.use_ilvr and args.down_N > 0
    range_t = stop_ratio_to_range_t(args.ilvr_stop_ratio, T) if args.ilvr_stop_ratio is not None else args.range_t
    resizers = make_resizers(image_size, image_size, args.down_N, args.resizer, device) if use_ilvr else None
    start = (t0 - 1) if t0 is not None else T - 1
    ilvr_steps = [i for i in range(start, -1, -1) if i > range_t] if use_ilvr else []

    masks_all = None
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "refs"), exist_ok=True)
    ct = (ColorTransfer(args.color_stats, args.color_transfer, args.color_strength)
          if args.color_transfer != "none" else None)
    parse_all = None
    if args.stroke == "parts":
        assert args.stroke_parse_dir, "--stroke parts には --stroke_parse_dir が要る"
        parse_all = th.stack([th.from_numpy(np.asarray(Image.open(os.path.join(args.stroke_parse_dir, s + ".png")),
                                                       dtype=np.int64)) for s in stems])
        assert tuple(parse_all.shape[1:]) == tuple(x0_all.shape[2:]), (parse_all.shape, x0_all.shape)
    cond_all = transform_input(x0_all, abstract_iters=args.abstract_iters, abstract_sigma_s=args.abstract_sigma_s,
                               abstract_sigma_r=args.abstract_sigma_r, quant_levels=args.quant_levels,
                               quant_sharpness=args.quant_sharpness, xdog_strength=args.xdog_strength,
                               xdog_sigma=args.xdog_sigma, color_transfer=ct, device=device,
                               stroke=args.stroke, stroke_segments=args.stroke_segments,
                               stroke_compactness=args.stroke_compactness, stroke_merge=args.stroke_merge,
                               parse=parse_all, stroke_shade_levels=args.stroke_shade_levels,
                               stroke_eye_scale_w=args.stroke_eye_scale_w, stroke_eye_scale_h=args.stroke_eye_scale_h,
                               stroke_eye_style=args.stroke_eye_style, stroke_iris_rgb=args.stroke_iris_rgb)
    has_cond = cond_all is not x0_all
    if has_cond:
        os.makedirs(os.path.join(args.out_dir, "refs_cond"), exist_ok=True)
        for stem, c in zip(stems, cond_all):
            save_png(c, os.path.join(args.out_dir, "refs_cond", f"{stem}.png"))
    for stem, x in zip(stems, x0_all):
        save_png(x, os.path.join(args.out_dir, "refs", f"{stem}.png"))
    if args.use_mask:
        masks_all = gaussian_feather(
            th.stack([load_mask(os.path.join(args.blend_mask_dir, n), image_size) for n in names]), args.mask_feather)
        os.makedirs(os.path.join(args.out_dir, "blend_masks"), exist_ok=True)
        for stem, m in zip(stems, masks_all):
            save_png(m.expand(3, -1, -1) * 2 - 1, os.path.join(args.out_dir, "blend_masks", f"{stem}.png"))

    tag = option_tag(args, T)
    outputs = []
    if device.type == "cuda":
        th.cuda.reset_peak_memory_stats(device)
        th.cuda.synchronize(device)
    t_start = time.time()
    for st in range(0, len(names), args.batch_size):
        idx = list(range(st, min(st + args.batch_size, len(names))))
        x0 = x0_all[idx].to(device)
        c0 = cond_all[idx].to(device)
        mask = masks_all[idx].to(device) if masks_all is not None else None
        seeds = [sample_seed(args.seed, names[i]) if args.seed_per_file else args.seed for i in idx]
        rng = SampleRNG(seeds, device)
        inter = [] if args.save_intermediate else None
        # 加工した入力 (c0) を SDEdit の開始点と ILVR の参照画像に、元の入力 (x0) をマスク合成と色の補正に使う
        x = sdedit_repeat(diffusion, model, c0, repeats=args.sdedit_repeats, t0=t0, resizers=resizers,
                          range_t=range_t, t_ref=args.t_ref, rng=rng, mask=mask,
                          x_blend=x0 if has_cond else None, clip_denoised=args.clip_denoised,
                          intermediate=inter, intermediate_every=args.intermediate_every, progress=args.progress)
        if args.use_color_fix:
            x = color_fix(x, x0, args.color_fix_sigma, mask=mask)
        for j, i in enumerate(idx):
            out = os.path.join("samples", f"{stems[i]}_{tag}_seed{args.seed}.png")
            save_png(x[j], os.path.join(args.out_dir, out))
            o = {"input": names[i], "ref_png": os.path.join("refs", f"{stems[i]}.png"), "sample": out,
                 "noise_seed": int(seeds[j])}
            if has_cond:
                o["cond_png"] = os.path.join("refs_cond", f"{stems[i]}.png")
            if inter is not None:
                d = os.path.join("intermediate", f"{stems[i]}_{tag}_seed{args.seed}")
                os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)
                for name, v in inter:
                    save_png(v[j], os.path.join(args.out_dir, d, f"{name}.png"))
                o["intermediate"] = d
            outputs.append(o)
    if device.type == "cuda":
        th.cuda.synchronize(device)
    elapsed = time.time() - t_start

    config = {
        "args": vars(args),
        "derived": {
            "num_timesteps": T,
            "option_tag": tag,
            "sdedit_t0_steps": t0,
            "sdedit_t0_from": (None if t0 is None else
                               "t0_sqrt_alpha" if args.t0_sqrt_alpha is not None else "t0_ratio"),
            "sdedit_start": start_info(diffusion, start) if t0 is not None else None,
            "start_index": start,
            "input_transformed": has_cond,
            "use_ilvr_effective": use_ilvr,
            "range_t_effective": range_t if use_ilvr else None,
            "ilvr_applied_indices": [ilvr_steps[0], ilvr_steps[-1]] if ilvr_steps else None,
            "ilvr_num_steps": len(ilvr_steps),
            "timestep_map_at_start": int(diffusion.timestep_map[start]) if hasattr(diffusion, "timestep_map")
            else start,
        },
        "train_cfg": cfg,
        "outputs": outputs,
        "elapsed_sec": round(elapsed, 3),
        "peak_vram_mb": round(th.cuda.max_memory_allocated(device) / 2 ** 20, 1) if device.type == "cuda" else None,
        "torch": th.__version__,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"{len(outputs)} samples ({tag}) -> {args.out_dir} ({elapsed:.1f}s)")


def create_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    def flag(name, default, help_):
        p.add_argument(f"--{name}", type=str2bool, nargs="?", const=True, default=default, help=help_)

    p.add_argument("--model_path", required=True)
    p.add_argument("--cfg", required=True, help="学習時の train.cfg (ddpm/*.cfg)")
    p.add_argument("--base_samples", required=True, help="入力画像のディレクトリ")
    p.add_argument("--ref_list", default="", help="入力のファイル名を 1 行ずつ書いたテキスト")
    p.add_argument("--num_refs", type=int, default=0, help="ref_list が無いとき、名前順の先頭から使う枚数 (0 なら全部)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--timestep_respacing", default="100")
    # SDEdit
    flag("use_sdedit", False, "入力を t0 までノイズ化した状態から始める")
    p.add_argument("--t0_ratio", type=float, default=0.5)
    p.add_argument("--t0_sqrt_alpha", type=float, default=None,
                   help="開始時の √ᾱ。指定すると最も近い番号から始める (--t0_ratio より優先)")
    p.add_argument("--sdedit_repeats", type=int, default=1)
    # ILVR (既存と同じ意味)
    flag("use_ilvr", True, "ILVR の低周波の置き換え")
    p.add_argument("--down_N", type=int, default=8, help="縮小の倍率 (256/N px に縮小)。0 なら ILVR なし")
    p.add_argument("--range_t", type=int, default=20, help="i > range_t のステップだけ ILVR (i は間引き後の番号)")
    p.add_argument("--ilvr_stop_ratio", type=float, default=None, help="指定時は i >= int(ratio * T) の間だけ ILVR")
    p.add_argument("--t_ref", default="t", choices=T_REF_CHOICES, help="ILVR の参照画像のノイズの番号 (既定は公式と同じ t)")
    p.add_argument("--resizer", default="resizer", choices=RESIZER_CHOICES)
    # マスク合成
    flag("use_mask", False, "マスク合成")
    p.add_argument("--blend_mask_dir", default=None, help="マスク合成のマスク (白 = 編集する領域)")
    p.add_argument("--mask_feather", type=int, default=0)
    # 色の補正
    flag("use_color_fix", False, "a*b* の低周波を入力に合わせる後処理")
    p.add_argument("--color_fix_sigma", type=float, default=8.0)
    # 入力の加工 (guided_diffusion/face_input_transform.py)
    p.add_argument("--abstract_iters", type=int, default=0, help="双方向フィルタの回数 (0 で無効)")
    p.add_argument("--abstract_sigma_s", type=float, default=3.0)
    p.add_argument("--abstract_sigma_r", type=float, default=8.0, help="色の差の sigma (CIELAB の ΔE)")
    p.add_argument("--quant_levels", type=int, default=0, help="L* の段階化の段数 (0 で無効)")
    p.add_argument("--quant_sharpness", type=float, default=3.0)
    p.add_argument("--xdog_strength", type=float, default=0.0, help="XDoG の輪郭線の濃さ (0 で無効)")
    p.add_argument("--xdog_sigma", type=float, default=1.0)
    p.add_argument("--stroke", default="none", choices=STROKE_CHOICES, help="ストローク化 (none で無効)")
    p.add_argument("--stroke_segments", type=int, default=200, help="generic: SLIC の超画素の数")
    p.add_argument("--stroke_compactness", type=float, default=10.0, help="generic: SLIC の compactness")
    p.add_argument("--stroke_merge", type=float, default=8.0, help="generic: 隣どうしをまとめる色の差 (ΔE)")
    p.add_argument("--stroke_parse_dir", default=None, help="parts: パーシングのラベル (scripts/parse_faces.py)")
    p.add_argument("--stroke_shade_levels", type=int, default=1, help="parts: 肌・首・髪・服の影の段数")
    p.add_argument("--stroke_eye_scale_w", type=float, default=1.0, help="parts: 目の領域の横の倍率")
    p.add_argument("--stroke_eye_scale_h", type=float, default=1.0, help="parts: 目の領域の縦の倍率")
    p.add_argument("--stroke_eye_style", default="dark", choices=EYE_STYLES, help="parts: 目の塗り方")
    p.add_argument("--stroke_iris_rgb", type=lambda v: tuple(int(c) for c in v.split(",")), default=(90, 60, 50),
                   help="parts + iris: 瞳の色 (R,G,B)")
    p.add_argument("--color_transfer", default="none", choices=COLOR_TRANSFER_CHOICES)
    p.add_argument("--color_stats", default="lists/face_color_stats.json")
    p.add_argument("--color_strength", type=float, default=1.0)
    # 乱数・その他
    p.add_argument("--seed", type=int, default=0)
    flag("seed_per_file", False, "入力ごとにシードを変える (scripts/ilvr_sample_eye.py と同じ乱数)")
    flag("save_intermediate", False, "開始時のノイズ画像と pred_xstart を保存する")
    p.add_argument("--intermediate_every", type=int, default=10)
    flag("clip_denoised", True, "")
    flag("progress", False, "")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main()
