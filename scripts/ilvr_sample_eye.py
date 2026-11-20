"""
実写の目のクロップを参照画像にして、目DDPM (MPUNetModel) から ILVR でアニメ調の目を生成する。

出力 (--out_dir):
  refs/<参照画像名>.png             学習時と同じ前処理をした参照画像 (加工前。評価の対応の指標はこれと比べる)
  refs_cond/<参照画像名>.png        加工した参照画像 (--color_transfer か --proxy を指定したときだけ。モデルに渡したもの)
  masks/<参照画像名>.png            補正に掛けたマスク (--mask_dir を指定したときだけ)
  samples/<参照画像名>_seed<S>.png  生成結果
  run.json                          設定 (引数と train.cfg)、参照画像と出力の対応、実行時間、VRAM のピーク

乱数はサンプルごとの Generator から引き、そのシードは --seeds の値と参照画像のファイル名から決める
(ilvr_eye.sample_seed)。そのため --batch_size や参照画像の組を変えても、同じ参照画像・同じシードの結果は同じ
(GPU の演算の非決定性による浮動小数点の誤差は除く)。
--down_N 0 なら条件付けしない (無条件生成。評価の比較基準に使う)。

左右の目を別々に学習したモデル: --model_path_l / --model_path_r を両方渡すと、参照画像の名前の左右
(<元の名前>_l.png / _r.png。画像の左側の目が l) に応じてモデルを使い分ける (--model_path は使わない)。
--base_samples やマスクのディレクトリに {side} を含めると、左右別のフォルダ (eye_{side}_masked/<元の名前>.png)
から読む (ilvr_eye.image_path)。参照画像の名前は左右別でも <元の名前>_l.png の形のままなので、乱数も変わらない。

Step 3 の改善案 (どれも既定では無効で、無効なら Phase 3 と同じ結果になる):
  --down_size S       縮小後の一辺を直接指定する (down_N より優先。6 や 3 など 32 を割り切らない大きさ)
  --color_transfer M  参照画像の色をアニメの色の分布に写してから渡す (lab_meanstd / lab_quantile)。
                      統計は --color_stats (scripts/fit_color_stats.py)、強さは --color_strength
  --mask_dir D        D/<参照画像名> のマスク (実写の目の eye_mask) で、目の部分だけを補正する。
                      --mask_dilate で膨らませる。補正の式は --mask_mode (ilvr_eye.ilvr_sample を参照)
  --proxy ellipse     参照画像の代わりに、アニメの目の形をした代理画像 (guided_diffusion/ilvr_proxy.py) を渡す。
                      目の楕円は --proxy_mask_dir のマスクから求める。--proxy_blend (0〜1) で、色を写した実写の
                      参照画像と混ぜる (0 なら代理画像を使わない従来の方式と同じ)。型の寸法は --proxy_iris_scale,
                      --proxy_lash_px。色の写像と併用すると、色を写した後の参照画像から代理画像の色を取る
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
    image_path,
    list_images,
    load_image,
    load_model,
    make_resizers,
    read_cfg,
    ref_side,
    sample_seed,
    save_png,
    select_refs,
    ilvr_sample,
)
from guided_diffusion.ilvr_proxy import PROXY_CHOICES, make_proxies
from guided_diffusion.ilvr_ref_transform import COLOR_TRANSFER_CHOICES, ColorTransfer, load_masks
from guided_diffusion.script_util import add_dict_to_argparser


def main():
    args = create_argparser().parse_args()
    assert args.t_ref in T_REF_CHOICES, f"--t_ref は {T_REF_CHOICES} のどれか"
    assert args.resizer in RESIZER_CHOICES, f"--resizer は {RESIZER_CHOICES} のどれか"
    assert args.mask_mode in MASK_MODE_CHOICES, f"--mask_mode は {MASK_MODE_CHOICES} のどれか"
    assert args.color_transfer in COLOR_TRANSFER_CHOICES, f"--color_transfer は {COLOR_TRANSFER_CHOICES} のどれか"
    assert args.proxy in PROXY_CHOICES, f"--proxy は {PROXY_CHOICES} のどれか"
    assert args.proxy == "none" or args.proxy_mask_dir, "--proxy には --proxy_mask_dir (目のマスク) が要る"
    device = th.device(args.device)

    cfg = read_cfg(args.cfg)
    if args.model_path_l or args.model_path_r:
        assert args.model_path_l and args.model_path_r, "--model_path_l と --model_path_r は両方要る"
        model_paths = {"l": args.model_path_l, "r": args.model_path_r}
    else:
        model_path = args.ema_model_path if args.use_ema else args.model_path
        assert model_path, "--use_ema True には --ema_model_path が要る" if args.use_ema else "--model_path が要る"
        model_paths = {None: model_path}
    models = {side: load_model(cfg, p, device) for side, p in model_paths.items()}
    diffusion = create_diffusion_from_cfg(cfg, args.timestep_respacing)
    image_size = int(cfg["IMAGE_SIZE"])

    if args.ref_list:
        with open(args.ref_list) as f:
            ref_names = [line.strip() for line in f if line.strip()]
    elif args.num_refs > 0:
        ref_names = select_refs(args.base_samples, args.num_refs, args.ref_seed, args.balance_lr)
    else:
        ref_names = list_images(args.base_samples)
    refs = th.stack([load_image(image_path(args.base_samples, n), image_size) for n in ref_names])
    # モデルごとに参照画像の番号をまとめる (左右別のモデルなら名前の左右で分ける)
    groups = {side: [i for i, n in enumerate(ref_names) if side is None or ref_side(n) == side] for side in models}
    assert sum(len(g) for g in groups.values()) == len(ref_names), "左右の分からない参照画像の名前がある"
    height, width = refs.shape[-2:]
    resizers = make_resizers(height, width, args.down_N, args.resizer, device, args.down_size)

    os.makedirs(os.path.join(args.out_dir, "refs"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    stems = [os.path.splitext(n)[0] for n in ref_names]
    for stem, ref in zip(stems, refs):
        save_png(ref, os.path.join(args.out_dir, "refs", f"{stem}.png"))
    cond = refs
    if args.color_transfer != "none":
        color_masks = load_masks(args.color_mask_dir, ref_names, image_size) if args.color_mask_dir else None
        cond = ColorTransfer(args.color_stats, args.color_transfer, args.color_strength)(refs, color_masks)
    proxy_params = None
    if args.proxy != "none":
        cond, proxy_params = make_proxies(cond, load_masks(args.proxy_mask_dir, ref_names, image_size),
                                          blend=args.proxy_blend, iris_scale=args.proxy_iris_scale,
                                          lash_px=args.proxy_lash_px)
    has_cond = args.color_transfer != "none" or args.proxy != "none"
    if has_cond:
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
        for side, idx_all in groups.items():
            for start in range(0, len(idx_all), args.batch_size):
                idx = idx_all[start:start + args.batch_size]
                ref = cond[idx].to(device)
                mask = masks[idx].to(device) if masks is not None else None
                rng = SampleRNG([sample_seed(seed, ref_names[i]) for i in idx], device)
                sample = ilvr_sample(
                    diffusion, models[side], ref, resizers, range_t=args.range_t, t_ref=args.t_ref,
                    clip_denoised=args.clip_denoised, progress=args.progress, rng=rng, mask=mask,
                    mask_mode=args.mask_mode,
                )
                for i, x in zip(idx, sample):
                    stem = stems[i]
                    out = os.path.join("samples", f"{stem}_seed{seed}.png")
                    save_png(x, os.path.join(args.out_dir, out))
                    o = {"ref": ref_names[i], "ref_png": os.path.join("refs", f"{stem}.png"),
                         "seed": seed, "sample": out}
                    if side is not None:
                        o["model"] = os.path.basename(model_paths[side])
                    if has_cond:
                        o["cond_png"] = os.path.join("refs_cond", f"{stem}.png")
                    if proxy_params is not None:
                        o["proxy"] = {k: (round(v, 4) if isinstance(v, float) else [round(c, 4) for c in v])
                                      for k, v in proxy_params[i].items()}
                    if masks is not None:
                        o["mask_png"] = os.path.join("masks", f"{stem}.png")
                    outputs.append(o)
    if device.type == "cuda":
        th.cuda.synchronize(device)
    elapsed = time.time() - t0

    run = {
        "args": vars(args),
        "model_path": "+".join(model_paths.values()),
        "model_paths": {str(k): v for k, v in model_paths.items()},
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
        model_path_l="",  # 左右別のモデル (両方渡すと参照画像の左右で使い分ける)
        model_path_r="",
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
        color_mask_dir="",  # 色を写す部分のマスク (マスク済みの画像で白い背景を写さないため)
        mask_dir="",
        mask_dilate=0,
        mask_mode="composite",
        proxy="none",
        proxy_mask_dir="",
        proxy_blend=1.0,
        proxy_iris_scale=1.1,
        proxy_lash_px=1.6,
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
