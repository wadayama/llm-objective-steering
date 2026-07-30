# LLM Objective Steering — 対話デモ

LLM に**目的関数と制約を宣言させ**、最適化器がそれを認証付きで執行する電力配分システムの
対話デモ。対象は N=8 の並列 QPSK-AWGN チャネルで、相互情報量（MI）をモンテカルロ推定
しながら射影勾配法で電力配分 $P_i = \lambda_i^2$ を最適化する。

- **System 1**（最適化器）: MI の勾配上昇 + 射影。PHR 拡張ラグランジュで制約を強制し、
  KKT 残差から状態カテゴリ（TRANSIENT / CONVERGED / INFEASIBLE / DISTURBED）と
  シャドープライス μ を算出する
- **System 2**（LLM）: 自然言語のポリシーを「目的関数ファミリ + 制約宣言」に翻訳する。
  数値の逐次調節はせず、宣言だけを行う

LLM が選べる目的関数ファミリ:

| ファミリ | 目的関数 | 用途 |
|---|---|---|
| `weighted_sum` | $\sum_i w_i \mathrm{MI}_i$ | スループット最大化・チャネル優先付け |
| `alpha_fair` | $\alpha$-フェア効用 | MI の公平化（強度可変） |
| `soft_min` | ソフト最小値 | max-min 型の公平化 |
| `power_target` | 目標電力への二次罰則 | 電力均等化・特定チャネルの停止 |
| `min_power` | $-\sum_i P_i$ | 総電力最小化（制約と併用が必須） |

宣言できる制約: `sum_rate >= τ`（総レート下限）、`min_channel_mi >= τ_ch`（各チャネルの MI 下限）。

LLM の出力はクランプ・正規化・射影を通るため、**何を出力しても実行可能性は構成的に保証**される。
達成不能な制約を宣言した場合は、システムが INFEASIBLE を*証明して*返す。

本リポジトリはこのデモの実行に必要なファイルのみを収録している。

```
exp05-integrated-demo/
├── run.py        UI + LLM コントローラ + 最適化ループ
├── core.py       System 1（MI 推定・目的関数・射影・拡張ラグランジュ・KKT）
└── steering.py   System 2（プロンプト構築・応答パース・制御履歴）
```

## 動作環境

- **Python 3.13 以上** + [uv](https://docs.astral.sh/uv/)
- **GPU 不要**（PyTorch を CPU で使用）
- **OpenAI 互換 API のローカルサーバ**（`localhost:1234`）。
  [LM Studio](https://lmstudio.ai/) で動作確認済み。
  検証に使用したモデルは `gpt-oss-20b`（MXFP4-Q8、MLX ランタイム、約 12 GB）で、
  相応の RAM が必要（24 GB のマシンで動作）
- **matplotlib の GUI バックエンド**が利用できること（下記参照）

## セットアップ

```bash
git clone <this-repo>
cd <this-repo>
uv sync
```

### GUI バックエンドについて

デモは matplotlib の自動バックエンド選択に任せている
（候補順: `macosx → qtagg → gtk4agg → gtk3agg → tkagg → wxagg → agg`）。

- **macOS**: 追加作業なしで `macosx` が選ばれる
- **Linux / Windows**: Qt か Tk が必要。どちらも無い環境では非対話の `agg` に落ちてしまい、
  ウィンドウが開かず即終了する。その場合は GUI ツールキットを追加する:
  ```bash
  uv add pyqt6          # または OS 側で python3-tk を導入
  ```
- バックエンドを明示したい場合は環境変数で指定できる:
  ```bash
  MPLBACKEND=qtagg uv run python run.py
  ```

### LLM サーバの準備（LM Studio）

GUI から起動してもよいが、CLI で完結する。

```bash
lms server start                  # サーバ起動（ポート 1234）
lms load openai/gpt-oss-20b       # モデルをロード（lms ls で一覧）
lms ps                            # ロード済みモデルの確認
```

疎通確認:

```bash
curl -s http://localhost:1234/v1/models
```

モデル ID が返れば準備完了。**サーバ起動とモデルのロードは別**で、両方必要である。
デモ側はモデル名をハードコードしておらず、`/v1/models` の**先頭のモデル**を自動的に使う。

## 実行

```bash
cd exp05-integrated-demo
uv run python run.py
```

画面の構成:

- **左上**: チャネル別 MI [bits]。制約 `sum_rate >= τ` が宣言されると赤破線で表示
- **右上**: 実効重み $\partial J / \partial \mathrm{MI}_i$ — いま目的関数が何を重視しているか
- **中段**: チャネル別電力、総電力 / 予算キャップ
- **下段のバッジ**:
  ```
  Optimizer: CONVERGED | constraint sum_rate>=10.0 | gap <= 0.043 bits | shadow price mu = 0.118
  ```
  背景色が状態を表す — 灰 TRANSIENT / **緑 CONVERGED** / **赤 INFEASIBLE** / 橙 DISTURBED
- **`LLM Status`**: LLM 呼び出しの結果（`OK [min_power]` など）と reasoning

操作:

- テキストボックスにポリシーを英語で入力し **Set**（または Enter）で LLM に投げる
- 8 本のスライダでチャネルゲインを変更できる（外乱の注入）
- **Random** / **Equal** でゲインを一括変更、**Clear** でポリシー解除、**Reset** で初期化
- 終了はウィンドウを閉じるか `Ctrl-C`

試すと面白いポリシー:

| ポリシー | 期待される挙動 |
|---|---|
| `Minimize total transmit power while keeping the total data rate above 10 bits` | `min_power` + 制約宣言。電力が自律降下し境界で CONVERGED |
| `Keep the total data rate above 15 bits with minimal power` | 予算 40 では達成不能 → μ 急騰 → INFEASIBLE → LLM が予算を引き上げて回復 |
| `minimize total power while keeping MI larger than 1.0 bit for all the channels` | チャネル別制約（ベクトル乗数）の宣言 |
| `Ensure all channels achieve similar data rates` | `soft_min` |
| `Equalize transmit power across all channels` | `power_target`（均等） |
| `Shut down channels 0 to 3` | `power_target`（先頭 4 本を 0） |
| `Prioritize channels 6 and 7` | `weighted_sum`（重み付け） |

ゲインスライダを大きく動かすと DISTURBED（橙）になり、自律的に再収束する。

## トラブルシューティング

**`RuntimeError: No models loaded in LM Studio`**
サーバは起動しているがモデルが未ロード。`lms load <model>` を実行する。

**画面に `LLM Status: Error: Connection error...` と出る**
LM Studio が落ちている。デモはこの状態でもクラッシュせず、System 1 は直近の目的関数で
最適化を続ける。呼び出しは 1 秒間隔でリトライされるので、**デモを起動したまま LM Studio を
立ち上げれば自動復帰する**（デモの再起動は不要）。

**ウィンドウをリサイズすると `AttributeError: 'ResizeEvent' object has no attribute 'inaxes'`**
matplotlib 3.11.0 側の既知の不具合（`TextBox._resize` が `resize_event` に対して
`LocationEvent` 前提のデコレータを通る）。例外は matplotlib 内部で捕捉されるため
**デモは落ちない**。ログが煩わしい場合は import 直後に以下を挟むと回避できる。

```python
matplotlib.widgets.TextBox._resize = lambda self, event: self.stop_typing()
```

**ウィンドウが開かずすぐ終了する**
GUI バックエンドが無く `agg` が選ばれている。上記「GUI バックエンドについて」を参照。
現在のバックエンドは次で確認できる。

```bash
uv run python -c "import matplotlib.pyplot; import matplotlib; print(matplotlib.get_backend())"
```

**GUI なしで動作確認したい**
`DEMO_HEADLESS=1` を付けると `agg` で import できる（スモークテスト用）。

```bash
cd exp05-integrated-demo && DEMO_HEADLESS=1 uv run python -c "import run; print('OK')"
```

## ライセンス

MIT License（[LICENSE](LICENSE) 参照）
