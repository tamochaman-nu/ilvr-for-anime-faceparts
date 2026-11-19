"""
Phase 1: ILVR 公式実装 (このリポジトリの guided_diffusion) を目DDPM に流用できるかを判定する。

公式と学習フォークはどちらもパッケージ名が guided_diffusion なので、それぞれ別名
(gd_official / gd_fork) で読み込んで同じプロセス内で比較する (両方とも相対 import のみ)。

  C1 ロード互換性: 公式の create_model で UNet を作り、目DDPM の state_dict を strict=True で読む。
     失敗したら missing / unexpected / shape 不一致を列挙し、(a) 接頭辞だけの違い か (b) アーキテクチャの違い かに分類。
  C2 出力の一致: C1 が通ったときだけ、同じ x_t, t で公式側とフォーク側のモデル出力の最大絶対誤差を測る (fp32, 1e-4 以下で合格)。
  C3 拡散過程の一致: betas / alphas_cumprod / timestep_map (respacing なし・100) と、
     同じシード・同じ初期ノイズで数ステップ回した p_sample の結果。
  C4 動作確認: 公式側の拡散過程で無条件サンプルを 16 枚生成し、フォーク側の 16 枚と彩度・ラプラシアン分散を比べる。

使い方 (リポジトリのルートで):
  python diagnostics/check_ilvr_compat.py --fork_root ~/guided-diffusion-ilvr-eye \
      --model_path ddpm/anime-eye32-mp64-200000.pt --cfg ddpm/anime-eye32-mp64.cfg \
      --data_dir ~/danbooru_work/parts-32/eye --out_dir diagnostics/outputs/phase1
"""

import argparse
import importlib
import importlib.util
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_pkg(alias, root):
    """root/guided_diffusion を alias という名前のパッケージとして読み込む。"""
    pkg_dir = os.path.join(root, "guided_diffusion")
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(pkg_dir, "__init__.py"), submodule_search_locations=[pkg_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return importlib.import_module(f"{alias}.script_util")


def read_cfg(path):
    """scripts/load_cfg.sh の save_cfg が書く KEY="VALUE" 形式を読む。"""
    cfg = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'^\s*([A-Z_][A-Z0-9_]*)=(.*)$', line.split("#")[0].rstrip())
            if m:
                cfg[m.group(1)] = m.group(2).strip().strip('"')
    return cfg


def as_bool(s):
    return str(s).lower() in ("true", "1", "yes")


def model_kwargs_from_cfg(cfg):
    """学習フォークの run_train.sh が create_model に渡すのと同じ引数 (推論なので fp32)。"""
    return dict(
        image_size=int(cfg["IMAGE_SIZE"]),
        num_channels=int(cfg["NUM_CHANNELS"]),
        num_res_blocks=int(cfg["NUM_RES_BLOCKS"]),
        channel_mult=cfg.get("CHANNEL_MULT", ""),
        learn_sigma=as_bool(cfg["LEARN_SIGMA"]),
        class_cond=as_bool(cfg["CLASS_COND"]),
        use_checkpoint=False,
        attention_resolutions=cfg["ATTENTION_RESOLUTIONS"],
        num_heads=int(cfg["NUM_HEADS"]),
        num_head_channels=int(cfg["NUM_HEAD_CHANNELS"]),
        num_heads_upsample=-1,
        use_scale_shift_norm=as_bool(cfg["USE_SCALE_SHIFT_NORM"]),
        dropout=float(cfg["DROPOUT"]),
        resblock_updown=as_bool(cfg["RESBLOCK_UPDOWN"]),
        use_fp16=False,
        use_new_attention_order=as_bool(cfg["USE_NEW_ATTENTION_ORDER"]),
    )


def diffusion_kwargs_from_cfg(cfg, timestep_respacing):
    # run_train.sh は diffusion_steps / noise_schedule 以外を渡さないので、残りは script_util の既定値
    return dict(
        steps=int(cfg["DIFFUSION_STEPS"]),
        learn_sigma=as_bool(cfg["LEARN_SIGMA"]),
        sigma_small=False,
        noise_schedule=cfg["NOISE_SCHEDULE"],
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing=timestep_respacing,
    )


def unwrap_state_dict(sd):
    """(a) に当たる違いを吸収する: EMA 等の入れ子の辞書と module. / _orig_mod. の接頭辞。"""
    for key in ("ema", "model_ema", "state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    out = {}
    for k, v in sd.items():
        for prefix in ("module.", "_orig_mod."):
            while k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


def diff_keys(model, sd):
    own = model.state_dict()
    missing = sorted(set(own) - set(sd))
    unexpected = sorted(set(sd) - set(own))
    mismatched = sorted(
        (k, tuple(sd[k].shape), tuple(own[k].shape))
        for k in set(own) & set(sd)
        if tuple(sd[k].shape) != tuple(own[k].shape)
    )
    return missing, unexpected, mismatched


def check_c1(sc_off, cfg, raw_sd):
    res = {}
    model = sc_off.create_model(**model_kwargs_from_cfg(cfg))
    try:
        model.load_state_dict(raw_sd, strict=True)
        res.update(passed=True, cause=None)
        return res, model
    except RuntimeError as e:
        res["error_head"] = str(e).splitlines()[0]
    missing, unexpected, mismatched = diff_keys(model, raw_sd)
    res.update(
        model_class=type(model).__name__,
        n_model_keys=len(model.state_dict()),
        n_ckpt_keys=len(raw_sd),
        missing=missing,
        unexpected=unexpected,
        shape_mismatch=mismatched,
    )
    # (a) か (b) か: 接頭辞・入れ子を外したうえで読めるか
    fixed = unwrap_state_dict(raw_sd)
    m2, u2, s2 = diff_keys(model, fixed)
    if not (m2 or u2 or s2):
        model.load_state_dict(fixed, strict=True)
        res.update(passed=True, cause="(a) キー名の接頭辞・入れ子のみ (付け替えで解決)")
        return res, model
    res.update(
        passed=False,
        cause="(b) アーキテクチャの違い (接頭辞を外しても一致するキーが無い)" if not (set(fixed) & set(model.state_dict()))
        else "(b) アーキテクチャの違い (接頭辞を外しても一部のキー・形が一致しない)",
        after_unwrap=dict(n_missing=len(m2), n_unexpected=len(u2), n_shape_mismatch=len(s2),
                          n_common_keys=len(set(fixed) & set(model.state_dict()))),
    )
    return res, None


def check_c2(model_off, model_fork, image_size, device):
    if model_off is None:
        return dict(passed=False, skipped=True, reason="C1 不合格のため公式側のモデルが作れず比較できない")
    model_off.to(device).eval()
    g = th.Generator(device="cpu").manual_seed(0)
    x = th.randn(4, 3, image_size, image_size, generator=g).to(device)
    out = {}
    with th.no_grad():
        for t in (0, 10, 100, 500, 999):
            ts = th.full((x.shape[0],), t, device=device, dtype=th.long)
            out[t] = (model_off(x, ts) - model_fork(x, ts)).abs().max().item()
    return dict(passed=max(out.values()) <= 1e-4, max_abs_err_by_t=out)


def schedule_tensors(diffusion):
    return dict(
        betas=np.asarray(diffusion.betas, dtype=np.float64),
        alphas_cumprod=np.asarray(diffusion.alphas_cumprod, dtype=np.float64),
        timestep_map=np.asarray(getattr(diffusion, "timestep_map", list(range(diffusion.num_timesteps)))),
    )


def run_p_sample(diffusion, model, noise, n_steps, seed):
    """p_sample_loop_progressive と同じ順序で n_steps (None なら全部) 回す。"""
    th.manual_seed(seed)
    img = noise.clone()
    indices = list(range(diffusion.num_timesteps))[::-1]
    if n_steps is not None:
        indices = indices[:n_steps]
    with th.no_grad():
        for i in indices:
            t = th.full((img.shape[0],), i, device=img.device, dtype=th.long)
            img = diffusion.p_sample(model, img, t, clip_denoised=True, model_kwargs={})["sample"]
    return img


def check_c3(sc_off, sc_fork, cfg, model_fork, device, respacing, n_steps=5, seed=0):
    res = {"schedule": {}, "p_sample": {}}
    passed = True
    for rs in ("", respacing):
        d_off = sc_off.create_gaussian_diffusion(**diffusion_kwargs_from_cfg(cfg, rs))
        d_fork = sc_fork.create_gaussian_diffusion(**diffusion_kwargs_from_cfg(cfg, rs))
        a, b = schedule_tensors(d_off), schedule_tensors(d_fork)
        entry = {"num_timesteps": [d_off.num_timesteps, d_fork.num_timesteps]}
        for k in a:
            same_shape = a[k].shape == b[k].shape
            entry[k + "_max_abs_diff"] = float(np.abs(a[k] - b[k]).max()) if same_shape else "shape mismatch"
            passed &= same_shape and float(np.abs(a[k] - b[k]).max()) == 0.0
        res["schedule"][rs or "none(1000)"] = entry

        size = int(cfg["IMAGE_SIZE"])
        noise = th.randn(4, 3, size, size, generator=th.Generator().manual_seed(seed)).to(device)
        x_off = run_p_sample(d_off, model_fork, noise, n_steps, seed)
        x_fork = run_p_sample(d_fork, model_fork, noise, n_steps, seed)
        err = (x_off - x_fork).abs().max().item()
        res["p_sample"][rs or "none(1000)"] = {"n_steps": n_steps, "max_abs_diff": err}
        passed &= err <= 1e-6
    res["passed"] = bool(passed)
    return res


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(th.uint8)


def image_stats(x01):
    """x01: [N,3,H,W] in [0,1]。HSV の彩度の平均と、グレースケールのラプラシアン分散。"""
    mx, mn = x01.max(1).values, x01.min(1).values
    sat = th.where(mx > 0, (mx - mn) / mx.clamp(min=1e-8), th.zeros_like(mx)).mean((1, 2))
    gray = (0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2])[:, None]
    k = th.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x01.dtype, device=x01.device)[None, None]
    lap = F.conv2d(gray, k).flatten(1).var(1)
    return dict(
        saturation_mean=float(sat.mean()), saturation_std=float(sat.std()),
        laplacian_var_mean=float(lap.mean()), laplacian_var_std=float(lap.std()),
    )


def save_grid(x, path, nrow=8, scale=4):
    arr = to_uint8(x).permute(0, 2, 3, 1).cpu().numpy()
    n, h, w, _ = arr.shape
    rows = (n + nrow - 1) // nrow
    grid = np.full((rows * (h + 2) + 2, nrow * (w + 2) + 2, 3), 255, np.uint8)
    for i, im in enumerate(arr):
        r, c = divmod(i, nrow)
        grid[2 + r * (h + 2): 2 + r * (h + 2) + h, 2 + c * (w + 2): 2 + c * (w + 2) + w] = im
    Image.fromarray(grid).resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST).save(path)


def load_data_samples(data_dir, n, size, seed=0):
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    files = random.Random(seed).sample(files, min(n, len(files)))
    arr = []
    for f in files:
        im = Image.open(os.path.join(data_dir, f)).convert("RGB")
        assert im.size == (size, size), f"{f}: {im.size}"
        arr.append(np.asarray(im, np.float32) / 255.0)
    return th.tensor(np.stack(arr)).permute(0, 3, 1, 2)


def check_c4(sc_off, sc_fork, cfg, model_fork, device, respacing, out_dir, data_dir, n=16, seed=1):
    size = int(cfg["IMAGE_SIZE"])
    noise = th.randn(n, 3, size, size, generator=th.Generator().manual_seed(seed)).to(device)
    res = {"timestep_respacing": respacing, "num_samples": n}
    samples = {}
    for name, sc in (("official", sc_off), ("fork", sc_fork)):
        d = sc.create_gaussian_diffusion(**diffusion_kwargs_from_cfg(cfg, respacing))
        t0 = time.time()
        samples[name] = run_p_sample(d, model_fork, noise, None, seed)
        res[name] = dict(image_stats((samples[name] + 1) / 2), sec=round(time.time() - t0, 2))
        save_grid(samples[name], os.path.join(out_dir, f"c4_{name}_uncond.png"))
    res["max_abs_diff_official_vs_fork"] = (samples["official"] - samples["fork"]).abs().max().item()
    if data_dir:
        real = load_data_samples(data_dir, 256, size)
        res["training_data_256"] = image_stats(real)
        save_grid(real[:n] * 2 - 1, os.path.join(out_dir, "c4_training_data.png"))
    s_o, s_f = res["official"], res["fork"]
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-8)
    res["passed"] = bool(
        rel(s_o["saturation_mean"], s_f["saturation_mean"]) < 0.1
        and rel(s_o["laplacian_var_mean"], s_f["laplacian_var_mean"]) < 0.1
    )
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official_root", default=REPO_ROOT)
    p.add_argument("--fork_root", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--cfg", required=True, help="学習時に保存された train.cfg (KEY=\"VALUE\")")
    p.add_argument("--data_dir", default="", help="C4 で統計を比べる学習データ (省略可)")
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "diagnostics", "outputs", "phase1"))
    p.add_argument("--timestep_respacing", default="100")
    p.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 比較を厳密にするため TF32 を切る
    th.backends.cuda.matmul.allow_tf32 = False
    th.backends.cudnn.allow_tf32 = False

    sc_off = load_pkg("gd_official", args.official_root)
    sc_fork = load_pkg("gd_fork", args.fork_root)
    cfg = read_cfg(args.cfg)
    raw_sd = th.load(args.model_path, map_location="cpu", weights_only=True)

    report = {"model_path": args.model_path, "cfg": args.cfg, "cfg_values": cfg,
              "torch": th.__version__, "device": args.device}

    # フォーク側のモデル (学習時と同じ create_model の arch=mp 分岐、推論なので fp32)
    model_fork = sc_fork.create_model(
        **model_kwargs_from_cfg(cfg),
        use_bf16=False,
        arch=cfg["ARCH"],
        mp_clip_act=float(cfg["MP_CLIP_ACT"]),
        attr_cond=as_bool(cfg["ATTR_COND"]),
        diffusion_steps=int(cfg["DIFFUSION_STEPS"]),
        noise_schedule=cfg["NOISE_SCHEDULE"],
    )
    model_fork.load_state_dict(raw_sd, strict=True)
    model_fork.to(args.device).eval()
    report["fork_model"] = {"class": type(model_fork).__name__, "strict_load": True,
                            "n_params": sum(p.numel() for p in model_fork.parameters())}

    report["C1"], model_off = check_c1(sc_off, cfg, raw_sd)
    report["C2"] = check_c2(model_off, model_fork, int(cfg["IMAGE_SIZE"]), args.device)
    report["C3"] = check_c3(sc_off, sc_fork, cfg, model_fork, args.device, args.timestep_respacing)
    report["C4"] = check_c4(sc_off, sc_fork, cfg, model_fork, args.device, args.timestep_respacing,
                            args.out_dir, args.data_dir)

    c = {k: report[k]["passed"] for k in ("C1", "C2", "C3", "C4")}
    if all(c.values()):
        verdict = "Plan A"
    elif not c["C1"] and report["C1"]["cause"].startswith("(b)"):
        verdict = "Plan B (C1 がアーキテクチャの違いで不合格)"
    else:
        verdict = "境界的: 要判断"
    report["summary"] = {**c, "verdict": verdict}

    path = os.path.join(args.out_dir, "check_ilvr_compat.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # 長いキー一覧は JSON にだけ残し、画面には要約を出す
    c1 = report["C1"]
    print(f"[C1] passed={c1['passed']} cause={c1.get('cause')}")
    if not c1["passed"]:
        print(f"     model={c1['model_class']} keys model/ckpt={c1['n_model_keys']}/{c1['n_ckpt_keys']} "
              f"missing={len(c1['missing'])} unexpected={len(c1['unexpected'])} "
              f"shape_mismatch={len(c1['shape_mismatch'])} after_unwrap={c1['after_unwrap']}")
        print(f"     missing[:5]={c1['missing'][:5]}")
        print(f"     unexpected[:5]={c1['unexpected'][:5]}")
    print(f"[C2] {report['C2']}")
    print(f"[C3] passed={report['C3']['passed']}")
    print(json.dumps({k: report['C3'][k] for k in ('schedule', 'p_sample')}, indent=1))
    print(f"[C4] {json.dumps(report['C4'], indent=1)}")
    print(f"verdict: {verdict}\nreport: {path}")


if __name__ == "__main__":
    main()
