"""
ILVR の自作ループ (guided_diffusion/ilvr_eye.py) の検証。

  V1 間引き (SpacedDiffusion) の対応付け: 間引き後のステップ番号 i について
     - 間引き後の alphas_cumprod[i] が元の alphas_cumprod[timestep_map[i]] と一致する
     - p_sample がモデルに渡すタイムステップが timestep_map[i] になる
     - q_sample(y, i) が元の拡散過程の q_sample(y, timestep_map[i]) と一致する
  V2 公式実装との一致: 同じシード・同じ初期ノイズ・t_ref="t"・Resizer で、自作ループと公式の
     p_sample_loop(resizers=..., range_t=...) (MPUNetModel は DropRefImg で包む) の結果が一致する
  V3 条件付けの効き: down_N で縮小した画像どうしの参照画像との L2 が、ILVR では無条件生成より小さい。
     t_ref="t-1" と interpolate も動き、既定 (t, Resizer) と結果が変わる
  V4 (--fork_root 指定時) create_model_from_cfg のモデルと学習フォークの create_model のモデルの出力が一致する
  V5 Resizer1px (resizer_1px.py): 縮小後が 2 画素以上 (down_N=2..16) では公式の Resizer と重み・出力が一致し、
     1 画素 (down_N=32) でも動いて、up(down(x)) が射影 (2 回かけても変わらない) に近い
  V6 サンプルごとの乱数 (SampleRNG): 同じ参照画像・同じシードなら、バッチの分け方や順序を変えても結果が同じ
     (GPU の演算の非決定性による誤差の範囲)
  V7 Step 3 の追加機能:
     - マスク付き ILVR: マスクがすべて 1 なら composite / multiply とも通常の ILVR と一致し、すべて 0 なら
       出力が参照画像によらない (参照画像を入れ替えても同じ)
     - down_size: 割り切れる大きさ (4 = down_N 8) なら down_N と一致し、割り切れない 6 でも up(down(x)) の
       冪等性の誤差 (2 回かけたときの変化) が down_N=8 と同程度 (1.5 倍以内。公式の Resizer は 3 次補間なので
       整数倍でも厳密な射影ではなく、down_N=8 でも最大 0.1〜0.2 程度の誤差がある)
     - 色の写像: strength=0 なら恒等写像 (CIELAB との往復の誤差の範囲)
  V8 代理画像 (ilvr_proxy.py): blend=0 なら参照画像がそのまま (従来の方式と同じ) で、楕円の推定 (マスクの
     モーメント) が合成した楕円の半径・傾きを復元する

使い方: docker compose run --rm verify-ilvr
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch as th

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from guided_diffusion.ilvr_eye import (  # noqa: E402
    DropRefImg,
    create_diffusion_from_cfg,
    create_model_from_cfg,
    ilvr_sample,
    load_image,
    SampleRNG,
    load_model,
    make_resizers,
    read_cfg,
    sample_seed,
    select_refs,
)
from guided_diffusion.resizer_1px import Resizer1px  # noqa: E402
from resizer import Resizer  # noqa: E402


class RecordingModel(th.nn.Module):
    """受け取ったタイムステップを記録して 0 を返す (learn_sigma なので 6 チャネル)。"""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x, timesteps, **kwargs):
        self.seen.append(timesteps.detach().cpu().clone())
        return th.zeros(x.shape[0], 6, *x.shape[2:], device=x.device)


def check_v1(cfg, device, respacings=("100", "50", "25")):
    base = create_diffusion_from_cfg(cfg, "")
    res, passed = {}, True
    y = th.rand(2, 3, 32, 32, generator=th.Generator().manual_seed(0), device="cpu").to(device) * 2 - 1
    for rs in respacings:
        sp = create_diffusion_from_cfg(cfg, rs)
        tmap = np.asarray(sp.timestep_map)
        ac_err = float(np.abs(sp.alphas_cumprod - base.alphas_cumprod[tmap]).max())

        rec = RecordingModel().to(device)
        ts_ok = True
        q_err = 0.0
        for i in range(sp.num_timesteps):
            t = th.full((2,), i, device=device, dtype=th.long)
            sp.p_sample(rec, y, t, clip_denoised=True, model_kwargs={})
            ts_ok &= bool((rec.seen[-1] == int(tmap[i])).all())
            eps = th.randn_like(y)
            q_sp = sp.q_sample(y, t, eps)
            q_base = base.q_sample(y, th.full_like(t, int(tmap[i])), eps)
            q_err = max(q_err, (q_sp - q_base).abs().max().item())
        ok = ac_err < 1e-10 and ts_ok and q_err < 1e-5
        passed &= ok
        res[rs] = {"num_timesteps": sp.num_timesteps, "timestep_map_head": tmap[:3].tolist(),
                   "timestep_map_tail": tmap[-3:].tolist(), "alphas_cumprod_max_abs_err": ac_err,
                   "model_receives_timestep_map": ts_ok, "q_sample_max_abs_err": q_err, "passed": ok}
    res["passed"] = bool(passed)
    return res


def check_v2(cfg, model, refs, device, cases, seed=0):
    diffusion = create_diffusion_from_cfg(cfg, "100")
    shape = tuple(refs.shape)
    res, passed = {}, True
    for down_N, range_t in cases:
        # 公式 scripts/ilvr_sample.py と同じく、バッチを含む形で Resizer を作る
        down_off = Resizer(shape, 1 / down_N).to(device)
        up_off = Resizer((shape[0], 3, shape[2] // down_N, shape[3] // down_N), down_N).to(device)
        noise = th.randn(*shape, generator=th.Generator().manual_seed(seed + 1000)).to(device)

        th.manual_seed(seed)
        x_off = diffusion.p_sample_loop(
            DropRefImg(model), shape, noise=noise.clone(), clip_denoised=True,
            model_kwargs={"ref_img": refs}, resizers=(down_off, up_off), range_t=range_t,
        )
        th.manual_seed(seed)
        x_own = ilvr_sample(diffusion, model, refs, make_resizers(shape[2], shape[3], down_N, "resizer", device),
                            range_t=range_t, t_ref="t", noise=noise.clone())
        err = (x_off - x_own).abs().max().item()
        passed &= err == 0.0
        res[f"N{down_N}_range{range_t}"] = {"max_abs_diff": err}
    res["passed"] = bool(passed)
    return res


def lowfreq_l2(x, ref, down_N):
    down, _ = make_resizers(x.shape[2], x.shape[3], down_N, "resizer", x.device)
    return ((down(x) - down(ref)) ** 2).flatten(1).mean(1).sqrt()


def check_v3(cfg, model, refs, device, down_N=8, range_t=0, seed=0):
    diffusion = create_diffusion_from_cfg(cfg, "100")
    shape = tuple(refs.shape)
    noise = th.randn(*shape, generator=th.Generator().manual_seed(seed + 1000)).to(device)
    th.manual_seed(seed)
    x_uncond = diffusion.p_sample_loop(model, shape, noise=noise.clone(), clip_denoised=True)
    xs = {}
    for t_ref in ("t", "t-1"):
        for method in ("resizer", "interpolate"):
            th.manual_seed(seed)
            xs[(t_ref, method)] = ilvr_sample(diffusion, model, refs,
                                              make_resizers(shape[2], shape[3], down_N, method, device),
                                              range_t=range_t, t_ref=t_ref, noise=noise.clone())
    l2_uncond = lowfreq_l2(x_uncond, refs, down_N).mean().item()
    res = {"down_N": down_N, "range_t": range_t, "lowfreq_l2_uncond": l2_uncond}
    passed = True
    for (t_ref, method), x in xs.items():
        l2 = lowfreq_l2(x, refs, down_N).mean().item()
        diff = (x - xs[("t", "resizer")]).abs().max().item()
        res[f"{t_ref}_{method}"] = {"lowfreq_l2": l2, "max_abs_diff_vs_default": diff}
        passed &= l2 < 0.5 * l2_uncond
        if (t_ref, method) != ("t", "resizer"):
            passed &= diff > 0
    res["passed"] = bool(passed)
    return res, x_uncond, xs


def load_fork_script_util(root):
    pkg_dir = os.path.join(root, "guided_diffusion")
    spec = importlib.util.spec_from_file_location(
        "gd_fork", os.path.join(pkg_dir, "__init__.py"), submodule_search_locations=[pkg_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gd_fork"] = mod
    spec.loader.exec_module(mod)
    return importlib.import_module("gd_fork.script_util")


def check_v4(cfg, model, model_path, fork_root, device):
    sc = load_fork_script_util(fork_root)
    b = lambda k: cfg[k].lower() == "true"  # noqa: E731
    fork_model = sc.create_model(
        int(cfg["IMAGE_SIZE"]), int(cfg["NUM_CHANNELS"]), int(cfg["NUM_RES_BLOCKS"]),
        channel_mult=cfg["CHANNEL_MULT"], learn_sigma=b("LEARN_SIGMA"), class_cond=b("CLASS_COND"),
        attention_resolutions=cfg["ATTENTION_RESOLUTIONS"], num_heads=int(cfg["NUM_HEADS"]),
        num_head_channels=int(cfg["NUM_HEAD_CHANNELS"]), use_scale_shift_norm=b("USE_SCALE_SHIFT_NORM"),
        dropout=float(cfg["DROPOUT"]), resblock_updown=b("RESBLOCK_UPDOWN"),
        use_new_attention_order=b("USE_NEW_ATTENTION_ORDER"), arch=cfg["ARCH"],
        mp_clip_act=float(cfg["MP_CLIP_ACT"]), diffusion_steps=int(cfg["DIFFUSION_STEPS"]),
        noise_schedule=cfg["NOISE_SCHEDULE"],
    )
    fork_model.load_state_dict(th.load(model_path, map_location="cpu", weights_only=True), strict=True)
    fork_model.to(device).eval()
    x = th.randn(4, 3, 32, 32, generator=th.Generator().manual_seed(0)).to(device)
    errs = {}
    with th.no_grad():
        for t in (0, 10, 100, 500, 999):
            ts = th.full((4,), t, device=device, dtype=th.long)
            errs[t] = (model(x, ts) - fork_model(x, ts)).abs().max().item()
    return {"max_abs_err_by_t": errs, "passed": max(errs.values()) == 0.0}


def check_v5(device):
    x = th.rand(4, 3, 32, 32, generator=th.Generator().manual_seed(0)).to(device) * 2 - 1
    res, passed = {}, True
    for down_N in (2, 4, 8, 16):
        err_w, err_out = 0.0, 0.0
        for in_shape, scale in (((1, 3, 32, 32), 1 / down_N), ((1, 3, 32 // down_N, 32 // down_N), down_N)):
            a, b = Resizer(in_shape, scale).to(device), Resizer1px(in_shape, scale).to(device)
            for pa, pb in zip(list(a.weights) + list(a.field_of_view), list(b.weights) + list(b.field_of_view)):
                err_w = max(err_w, float("inf") if pa.shape != pb.shape else (pa - pb).abs().max().item())
            xin = x if scale < 1 else x[:, :, :32 // down_N, :32 // down_N]
            err_out = max(err_out, (a(xin) - b(xin)).abs().max().item())
        passed &= err_w == 0.0 and err_out == 0.0
        res[f"N{down_N}"] = {"max_abs_diff_weights": err_w, "max_abs_diff_output": err_out}
    down, up = make_resizers(32, 32, 32, "resizer", device)
    d = down(x)
    proj = up(d)
    idem = (up(down(proj)) - proj).abs().max().item()
    const = (proj - proj.mean((2, 3), keepdim=True)).abs().max().item()
    ok = tuple(d.shape) == (4, 3, 1, 1) and idem < 1e-4 and const < 1e-5
    passed &= ok
    res["N32"] = {"down_shape": list(d.shape), "up_down_constant_max_dev": const, "idempotence_max_abs_err": idem,
                  "max_abs_diff_vs_pixel_mean": (d - x.mean((2, 3), keepdim=True)).abs().max().item(), "passed": ok}
    res["passed"] = bool(passed)
    return res


def check_v6(cfg, model, refs, names, device, down_N=8, range_t=20, seed=0):
    diffusion = create_diffusion_from_cfg(cfg, "100")
    resizers = make_resizers(refs.shape[2], refs.shape[3], down_N, "resizer", device)
    seeds = [sample_seed(seed, n) for n in names]

    def run(idx):
        return ilvr_sample(diffusion, model, refs[idx], resizers, range_t=range_t,
                           rng=SampleRNG([seeds[i] for i in idx], device))

    n = len(names)
    full = run(list(range(n)))
    half = th.cat([run(list(range(0, n // 2))), run(list(range(n // 2, n)))])
    rev = run(list(range(n))[::-1]).flip(0)
    err_half = (full - half).abs().max().item()
    err_rev = (full - rev).abs().max().item()
    # 別のシードなら当然結果は変わる (乱数が実際にシードで決まっていることの確認)
    other = ilvr_sample(diffusion, model, refs, resizers, range_t=range_t,
                        rng=SampleRNG([sample_seed(seed + 1, nm) for nm in names], device))
    diff_seed = (full - other).abs().mean().item()
    ok = err_half < 1e-3 and err_rev < 1e-3 and diff_seed > 0.05
    return {"down_N": down_N, "range_t": range_t, "max_abs_diff_split_batches": err_half,
            "max_abs_diff_reversed_order": err_rev, "mean_abs_diff_other_seed": diff_seed, "passed": ok}


def check_v7(cfg, model, refs, names, device, color_stats="lists/color_stats.json", seed=0):
    from guided_diffusion.ilvr_ref_transform import ColorTransfer

    diffusion = create_diffusion_from_cfg(cfg, "100")
    h, w = refs.shape[2:]
    resizers = make_resizers(h, w, 8, "resizer", device)
    seeds = [sample_seed(seed, n) for n in names]

    def run(ref, res=resizers, **kw):
        return ilvr_sample(diffusion, model, ref, res, range_t=20, rng=SampleRNG(seeds, device), **kw)

    base = run(refs)
    ones = th.ones_like(refs[:, :1])
    r = {"mask_ones_composite": (run(refs, mask=ones, mask_mode="composite") - base).abs().max().item(),
         "mask_ones_multiply": (run(refs, mask=ones, mask_mode="multiply") - base).abs().max().item()}
    zeros = th.zeros_like(ones)
    r["mask_zeros_ref_swapped"] = (run(refs, mask=zeros) - run(refs.flip(0), mask=zeros)).abs().max().item()
    r["down_size4_vs_down_N8"] = (run(refs, res=make_resizers(h, w, 0, "resizer", device, down_size=4))
                                  - base).abs().max().item()
    down6, up6 = make_resizers(h, w, 0, "resizer", device, down_size=6)
    p1 = up6(down6(refs))
    r["down_size6_shape"] = list(down6(refs).shape)
    r["down_size6_idempotence"] = (up6(down6(p1)) - p1).abs().max().item()
    down8, up8 = resizers
    p8 = up8(down8(refs))
    r["down_N8_idempotence"] = (up8(down8(p8)) - p8).abs().max().item()
    ok = (r["mask_ones_composite"] < 1e-3 and r["mask_ones_multiply"] < 1e-3 and r["mask_zeros_ref_swapped"] < 1e-3
          and r["down_size4_vs_down_N8"] < 1e-3 and r["down_size6_shape"][-2:] == [6, 6]
          and r["down_size6_idempotence"] < 1.5 * r["down_N8_idempotence"])
    if os.path.exists(color_stats):
        x = refs.cpu()
        r["color_strength0_identity"] = (ColorTransfer(color_stats, "lab_meanstd", 0.0)(x) - x).abs().max().item()
        ok = ok and r["color_strength0_identity"] < 1e-3
    r["passed"] = ok
    return r


def check_v8(refs):
    from guided_diffusion.ilvr_proxy import ellipse_from_mask, make_proxies

    r = {}
    # 合成した楕円 (32px の 8 倍の格子で描いて平均で縮小し、縁を滑らかにしたもの)
    errs = []
    for cx, cy, a, b, deg in [(15.5, 15.5, 10.0, 5.0, 0.0), (14.0, 17.0, 11.0, 4.5, 12.0), (16.5, 15.0, 9.0, 6.0, -20.0)]:
        S = 8
        Y, X = np.mgrid[:32 * S, :32 * S] / S + (0.5 / S - 0.5)
        t = np.radians(deg)
        u, v = (X - cx) * np.cos(t) + (Y - cy) * np.sin(t), -(X - cx) * np.sin(t) + (Y - cy) * np.cos(t)
        m = ((u / a) ** 2 + (v / b) ** 2 <= 1).astype(np.float64).reshape(32, S, 32, S).mean((1, 3))
        ecx, ecy, ea, eb, et = ellipse_from_mask(m)
        errs.append(max(abs(ecx - cx), abs(ecy - cy), abs(ea - a), abs(eb - b), abs(np.degrees(et) - deg)))
    r["ellipse_max_err"] = float(max(errs))  # 画素 (位置・半径) と度 (傾き) の最大
    x = refs.cpu()
    m = th.zeros(len(x), 1, 32, 32)
    m[:, :, 12:20, 6:26] = 1
    p0, _ = make_proxies(x, m, blend=0.0)
    p1, _ = make_proxies(x, m, blend=1.0)
    r["blend0_max_abs_diff"] = (p0 - x).abs().max().item()
    r["blend1_differs"] = (p1 - x).abs().mean().item()
    r["passed"] = r["ellipse_max_err"] < 0.5 and r["blend0_max_abs_diff"] == 0.0 and r["blend1_differs"] > 0.05
    return r


def save_grid(rows, path, scale=4):
    """rows: [[3 x H x W] in [-1, 1], ...] のリストのリスト"""
    from PIL import Image

    h, w = rows[0][0].shape[1:]
    ncol = max(len(r) for r in rows)
    grid = np.full((len(rows) * (h + 2) + 2, ncol * (w + 2) + 2, 3), 255, np.uint8)
    for r, row in enumerate(rows):
        for c, x in enumerate(row):
            im = ((x.clamp(-1, 1) + 1) * 127.5).round().to(th.uint8).permute(1, 2, 0).cpu().numpy()
            grid[2 + r * (h + 2): 2 + r * (h + 2) + h, 2 + c * (w + 2): 2 + c * (w + 2) + w] = im
    Image.fromarray(grid).resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--base_samples", required=True, help="実写の目クロップのディレクトリ")
    p.add_argument("--num_refs", type=int, default=8)
    p.add_argument("--fork_root", default="", help="学習フォーク (指定時のみ V4)")
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "diagnostics", "outputs", "verify_ilvr_eye"))
    p.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ビット単位で比べるので TF32 を切る
    th.backends.cuda.matmul.allow_tf32 = False
    th.backends.cudnn.allow_tf32 = False

    cfg = read_cfg(args.cfg)
    model = load_model(cfg, args.model_path, args.device)
    names = select_refs(args.base_samples, args.num_refs, seed=0)
    refs = th.stack([load_image(os.path.join(args.base_samples, n), int(cfg["IMAGE_SIZE"])) for n in names])
    refs = refs.to(args.device)

    report = {"model_path": args.model_path, "refs": names}
    report["V1"] = check_v1(cfg, args.device)
    report["V2"] = check_v2(cfg, model, refs, args.device, [(4, 0), (8, 20), (16, 10), (16, 40)])
    report["V3"], x_uncond, xs = check_v3(cfg, model, refs, args.device)
    if args.fork_root:
        report["V4"] = check_v4(cfg, model, args.model_path, args.fork_root, args.device)
    report["V5"] = check_v5(args.device)
    report["V6"] = check_v6(cfg, model, refs, names, args.device)
    report["V7"] = check_v7(cfg, model, refs, names, args.device)
    report["V8"] = check_v8(refs)

    # 行: 参照画像 / 無条件生成 / ILVR (t, Resizer) / (t, interpolate) / (t-1, Resizer) / (t-1, interpolate)
    save_grid([list(refs), list(x_uncond)] + [list(xs[k]) for k in
              [("t", "resizer"), ("t", "interpolate"), ("t-1", "resizer"), ("t-1", "interpolate")]],
              os.path.join(args.out_dir, "v3_grid.png"))

    keys = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")
    report["summary"] = {k: report[k]["passed"] for k in keys if k in report}
    with open(os.path.join(args.out_dir, "verify_ilvr_eye.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    for k in keys:
        if k in report:
            print(f"[{k}] {json.dumps(report[k], ensure_ascii=False)}")
    print(f"summary: {report['summary']}")


if __name__ == "__main__":
    main()
