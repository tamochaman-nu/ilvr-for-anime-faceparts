"""
実写の目のクロップを参照画像にして、目DDPM (MPUNetModel) から ILVR でアニメ調の目を生成する。

出力 (--out_dir):
  refs/<参照画像名>.png             学習時と同じ前処理をした参照画像 (加工前。評価の対応の指標はこれと比べる)
  refs_cond/<参照画像名>.png        色を写した参照画像 (--color_transfer を指定したときだけ。モデルに渡したもの)
  masks/<参照画像名>.png            補正に掛けたマスク (--mask_dir を指定したときだけ)
  samples/<参照画像名>_seed<S>.png  生成結果
  run.json                          設定 (引数と train.cfg)、参照画像と出力の対応、実行時間、VRAM のピーク

乱数はサンプルごとの Generator から引き、そのシードは --seeds の値と参照画像のファイル名から決める
(ilvr_eye.sample_seed)。そのため --batch_size や参照画像の組を変えても、同じ参照画像・同じシードの結果は同じ
(GPU の演算の非決定性による浮動小数点の誤差は除く)。
--down_N 0 なら条件付けしない (無条件生成。評価の比較基準に使う)。

Step 3 の改善案 (どれも既定では無効で、無効なら Phase 3 と同じ結果になる):
  --down_size S       縮小後の一辺を直接指定する (down_N より優先。6 や 3 など 32 を割り切らない大きさ)
  --color_transfer M  参照画像の色をアニメの色の分布に写してから渡す (lab_meanstd / lab_quantile)。
                      統計は --color_stats (scripts/fit_color_stats.py)、強さは --color_strength
  --mask_dir D        D/<参照画像名> のマスク (実写の目の eye_mask) で、目の部分だけを補正する。
                      --mask_dilate で膨らませる。補正の式は --mask_mode (ilvr_eye.ilvr_sample を参照)
"""

import argparse
import json
import os
import time

import torch as th

from guided_diffusion.ilvr_eye import (
    MASK_MODE_CHOICES,
    RESIZER_CHOICES,
    T_REF_CHOICES,
    SampleRNG,
    create_diffusion_from_cfg,
    list_images,
    load_image,
    load_model,
    make_resizers,
    read_cfg,
    sample_seed,
    save_png,
    select_refs,
    ilvr_sample,
)
from guided_diffusion.ilvr_ref_transform import COLOR_TRANSFER_CHOICES, ColorTransfer, load_masks
from guided_diffusion.script_util import add_dict_to_argparser


def main():
    args = create_argparser().parse_args()
    assert args.t_ref in T_REF_CHOICES, f"--t_ref は {T_REF_CHOICES} のどれか"
    assert args.resizer in RESIZER_CHOICES, f"--resizer は {RESIZER_CHOICES} のどれか"
    assert args.mask_mode in MASK_MODE_CHOICES, f"--mask_mode は {MASK_MODE_CHOICES} のどれか"
    assert args.color_transfer in COLOR_TRANSFER_CHOICES, f"--color_transfer は {COLOR_TRANSFER_CHOICES} のどれか"
    device = th.device(args.device)

    model_path = args.ema_model_path if args.use_ema else args.model_path
    assert model_path, "--use_ema True には --ema_model_path が要る" if args.use_ema else "--model_path が要る"
    cfg = read_cfg(args.cfg)
    model = load_model(cfg, model_path, device)
    diffusion = create_diffusion_from_cfg(cfg, args.timestep_respacing)
    image_size = int(cfg["IMAGE_SIZE"])

    if args.ref_list:
        with open(args.ref_list) as f:
            ref_names = [line.strip() for line in f if line.strip()]
    elif args.num_refs > 0:
        ref_names = select_refs(args.base_samples, args.num_refs, args.ref_seed, args.balance_lr)
    else:
        ref_names = list_images(args.base_samples)
    refs = th.stack([load_image(os.path.join(args.base_samples, n), image_size) for n in ref_names])
    height, width = refs.shape[-2:]
    resizers = make_resizers(height, width, args.down_N, args.resizer, device, args.down_size)

    os.makedirs(os.path.join(args.out_dir, "refs"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    stems = [os.path.splitext(n)[0] for n in ref_names]
    for stem, ref in zip(stems, refs):
        save_png(ref, os.path.join(args.out_dir, "refs", f"{stem}.png"))
    cond = refs
    if args.color_transfer != "none":
        cond = ColorTransfer(args.color_stats, args.color_transfer, args.color_strength)(refs)
        os.makedirs(os.path.join(args.out_dir, "refs_cond"), exist_ok=True)
        for stem, c in zip(stems, cond):
            save_png(c, os.path.join(args.out_dir, "refs_cond", f"{stem}.png"))
    masks = None
    if args.mask_dir:
        masks = load_masks(args.mask_dir, ref_names, image_size, args.mask_dilate)
        os.makedirs(os.path.join(args.out_dir, "masks"), exist_ok=True)
        for stem, m in zip(stems, masks):
            save_png(m.expand(3, -1, -1) * 2 - 1, os.path.join(args.out_dir, "masks", f"{stem}.png"))

    seeds = [int(s) for s in str(args.seeds).split(",")]
    outputs = []
    if device.type == "cuda":
        th.cuda.reset_peak_memory_stats(device)
        th.cuda.synchronize(device)
    t0 = time.time()
    for seed in seeds:
        for start in range(0, len(refs), args.batch_size):
            ref = cond[start:start + args.batch_size].to(device)
            mask = masks[start:start + args.batch_size].to(device) if masks is not None else None
            rng = SampleRNG([sample_seed(seed, n) for n in ref_names[start:start + args.batch_size]], device)
            sample = ilvr_sample(
                diffusion, model, ref, resizers, range_t=args.range_t, t_ref=args.t_ref,
                clip_denoised=args.clip_denoised, progress=args.progress, rng=rng, mask=mask,
                mask_mode=args.mask_mode,
            )
            for j, x in enumerate(sample):
                stem = stems[start + j]
                out = os.path.join("samples", f"{stem}_seed{seed}.png")
                save_png(x, os.path.join(args.out_dir, out))
                o = {"ref": ref_names[start + j], "ref_png": os.path.join("refs", f"{stem}.png"),
                     "seed": seed, "sample": out}
                if args.color_transfer != "none":
                    o["cond_png"] = os.path.join("refs_cond", f"{stem}.png")
                if masks is not None:
                    o["mask_png"] = os.path.join("masks", f"{stem}.png")
                outputs.append(o)
    if device.type == "cuda":
        th.cuda.synchronize(device)
    elapsed = time.time() - t0

    run = {
        "args": vars(args),
        "model_path": model_path,
        "train_cfg": cfg,
        "num_timesteps": diffusion.num_timesteps,
        "image_shape": [3, height, width],
        "outputs": outputs,
        "elapsed_sec": round(elapsed, 3),
        "sec_per_sample": round(elapsed / max(len(outputs), 1), 4),
        "peak_vram_mb": round(th.cuda.max_memory_allocated(device) / 2 ** 20, 1) if device.type == "cuda" else None,
        "torch": th.__version__,
        "device": th.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    with open(os.path.join(args.out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
    print(f"{len(outputs)} samples -> {args.out_dir} ({elapsed:.1f}s, peak VRAM {run['peak_vram_mb']} MB)")


def create_argparser():
    defaults = dict(
        model_path="",
        ema_model_path="",
        use_ema=False,
        cfg="",  # 学習時に保存された train.cfg
        base_samples="",  # 参照画像 (実写の目のクロップ) のディレクトリ
        ref_list="",  # 参照画像のファイル名を 1 行ずつ書いたテキスト (指定時は num_refs より優先)
        num_refs=16,  # 0 なら base_samples の全画像
        ref_seed=0,
        balance_lr=True,
        down_N=8,  # 0 なら条件付けしない (無条件生成)
        down_size=0,  # > 0 なら縮小後の一辺を直接指定する (down_N より優先)
        range_t=20,
        t_ref="t",
        resizer="resizer",
        color_transfer="none",
        color_stats="lists/color_stats.json",
        color_strength=1.0,
        mask_dir="",
        mask_dilate=0,
        mask_mode="composite",
        timestep_respacing="100",
        seeds="0",
        batch_size=64,
        clip_denoised=True,
        progress=False,
        out_dir="",
        device="cuda" if th.cuda.is_available() else "cpu",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
