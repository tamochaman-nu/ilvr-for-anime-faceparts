# ILVR 公式実装 (jychoi118/ilvr_adm) からの変更の記録

公式実装の既存ファイルは変更していない。目DDPM 用の処理はすべて新規ファイルで追加している。

## 追加したファイル

| ファイル | 内容 |
|---|---|
| `guided_diffusion/unet_mp.py` | 学習フォーク [tamochaman-nu/guided-diffusion](https://github.com/tamochaman-nu/guided-diffusion) の `dev/detect-resblock-overflow` (`5bb6a5c48289241638cdc7bb3f794d54c67c7e81`) の同名ファイルをそのままコピー (sha256 `19031385ba502c7f5c668bd7a5f7d73e1ed535b5d90750844d1d956d84a18319`)。**このファイルだけ CC BY-NC-SA 4.0** (NVlabs/edm2 由来、ファイル冒頭の表記を参照) |
| `guided_diffusion/ilvr_eye.py` | train.cfg からの MPUNetModel / 拡散過程の構築、ILVR ループ (t_ref・リサイズ方法・サンプルごとの乱数を選べる)、参照画像の読み込み |
| `guided_diffusion/resizer_1px.py` | 縮小後が 1 画素でも動く Resizer のサブクラス (下記) |
| `guided_diffusion/ilvr_ref_transform.py`, `scripts/fit_color_stats.py` | Step 3: 参照画像の色の写像 (CIELAB でアニメの色の分布へ) と、その統計 (`lists/color_stats.json`)、目のマスクの読み込み |
| `guided_diffusion/ilvr_metrics.py` | 評価指標 (低周波の RMSE、LPIPS、CIELAB の色差、彩度・線のくっきりさ、KID) |
| `scripts/ilvr_sample_eye.py`, `scripts/run_ilvr_eye.sh` | 変換の実行 (参照画像名と出力の対応、設定、時間、VRAM を run.json に残す) |
| `scripts/make_ref_lists.py`, `lists/` | 参照画像の選別 (眼鏡・閉じた目・隠れた目・横向きを除く) と、評価・グリッド・KID 用のリスト |
| `scripts/eval_ilvr_eye.py`, `scripts/make_ilvr_grid.py` | 実行結果の評価 (CSV) とグリッド画像 |
| `scripts/run_phase3.sh`, `scripts/run_step3.sh` | Phase 3 (down_N x range_t) と Step 3 (改善案) の一括実行 |
| `diagnostics/dataset_lowfreq.py` | アニメと実写の目の低周波の統計の比較 |
| `diagnostics/check_ilvr_compat.py` | Phase 1 の流用可否の判定 |
| `diagnostics/verify_ilvr_eye.py` | 自作 ILVR ループの検証 (間引きの対応付け、公式の p_sample_loop との一致、Step 3 の追加機能) |
| `Dockerfile`, `docker-compose.yml`, `.env.example`, `.gitignore` | 実行環境 |

## 公式実装との違い (既存ファイルを変えずに対応したもの)

- `UNetModel.forward` は `ref_img` を受け取るが `MPUNetModel.forward` は受け取らない
  → 自作ループは `ref_img` をモデルに渡さない。公式ループと比べるときは `ilvr_eye.DropRefImg` で包む。
- `scripts/ilvr_sample.py` の `utils.save_image(..., range=(-1, 1))` は torchvision 0.19 では使えない
  → 自作スクリプトは PIL で保存する (torchvision の `normalize=True, value_range=(-1, 1)` と同じ丸め)。

- `resizer.py` は縮小後の一辺が 1 画素 (32px で down_N=32) になると `np.squeeze` で出力側の次元が潰れて
  IndexError になる → `resizer.py` は変えず、`guided_diffusion/resizer_1px.py` の `Resizer1px` (サブクラス) で
  `contributions` だけを squeeze しない版に差し替えた。`ilvr_eye.make_resizers` は常にこれを使う。
  down_N=2, 4, 8, 16 では公式の Resizer と重み・出力が完全に一致し (差 0.0)、down_N=32 では縮小結果が
  画素の平均と一致することを `diagnostics/verify_ilvr_eye.py` の V5 で確認している。
- 公式の `p_sample` は乱数を `th.randn_like` (グローバルな乱数) で引くので、サンプルごとの乱数にするために
  `ilvr_eye.ilvr_sample` の中で `p_mean_variance` から同じ計算をしている。グローバルな乱数を使う場合は
  公式のループと結果が完全に一致する (V2)。

- Step 3 で ILVR ループに加えたもの (どれも既定では無効で、無効なら公式と同じ計算):
  マスク付き補正 (`mask`, `mask_mode`)、縮小後の大きさの直接指定 (`make_resizers(..., down_size=)`。
  `Resizer` に `output_shape` を渡す。整数倍では `scale_factor` を渡した場合と出力が一致する: V7)。

## 未適用の修正案

- `proposed-resizer-1px-output.diff`: 上の 1 画素の問題を `resizer.py` 自体で直す場合の diff (`Resizer1px` と同じ修正)。
  サブクラスで対応したので適用していない。適用するなら: リポジトリのルートで
  `git apply patches/proposed-resizer-1px-output.diff`。
