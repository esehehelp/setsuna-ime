# Setsuna IME（刹那）

**CPU 上で 1 文あたり約 1 ms で動作する、非自己回帰ニューラルかな漢字変換（CTC-NAT）ベースの日本語入力メソッド（IME）です。**

*Setsuna (刹那) is a fast Japanese input method powered by a non-autoregressive neural kana-kanji converter (CTC-NAT) with pure-Rust CPU inference, supporting Windows (TSF), Linux (fcitx5), and Android.*

> **開発状況**: 研究・開発段階のプロジェクトです。公開モデル（Apache-2.0）と辞書（CC BY-SA 4.0）はインストーラーから自動でダウンロードでき、そのまま試用できます。

## 特長

- **高速なライブ変換**: Rust 製の独自推論エンジンにより、ONNX Runtime などの外部ランタイムなしで CPU のみで 1 文あたり数ミリ秒のリアルタイム変換を実現します。
- **実用的な入力支援**: 文節編集、英日混在入力、誤入力補正（ルール＋読み補正モデル）、学習・予測変換、ユーザー辞書、日付・計算・絵文字変換を備えています。
- **クロスプラットフォーム**: Windows・Linux・Android で共通の変換コア（`ime-core`）と設定画面（WebView）を共有しています。
- **任意の拡張機能**: ローカルの LM リランカー（llama.cpp）や、LLM による文章の書き直し機能（OpenAI / ChatGPT プラン / OpenAI 互換 API）を追加できます。

| プラットフォーム | フロントエンド | 対応環境 | ステータス |
|---|---|---|---|
| **Windows** | TSF | Windows 10 / 11（x64） | 利用可能 |
| **Linux** | fcitx5 | x86_64・fcitx5 5.1 以上・glibc 2.39 以上 | 利用可能 |
| **Android** | Android IME | arm64・Android 8（API 26）以降 | 開発中 |

---

## インストール

### Windows（TSF）

PowerShell 5.1 または PowerShell 7（`pwsh`）で以下を実行します。`regsvr32` による IME 登録時に UAC ダイアログが表示されます。

```powershell
# 標準インストール（既定の変換モデル・読み補正モデル・読み辞書・形態素解析器を自動取得）
irm https://github.com/esehehelp/setsuna-ime/releases/latest/download/setup.ps1 | iex
```

モデルを変更したい場合や、辞書なしで軽量に試したい場合はオプションを指定できます。

```powershell
# 変換モデルを Diphylleia-v1.0-flash に変更
& ([scriptblock]::Create((irm https://github.com/esehehelp/setsuna-ime/releases/latest/download/setup.ps1))) -Model Diphylleia-v1.0-flash

# 辞書・形態素解析器を入れずにモデルのみ導入
& ([scriptblock]::Create((irm https://github.com/esehehelp/setsuna-ime/releases/latest/download/setup.ps1))) -NoDictionaries
```

- **手動インストール**: リリース zip（`setsuna-ime-<version>-windows-x64.zip`）を展開し、`install.cmd` を実行しても同じ手順で導入できます。Visual C++ ランタイム（14.44 以上）がない場合は自動でインストールされます。
- **設定画面**: タスクバーの「あ / A」アイコンを右クリックし、「設定...」から開けます。変更は次の入力から反映されます。
- **更新時の反映**: 軽量シム（`setsuna_tsf_shim.dll`）が本体 DLL をホットスワップし旧エンジンプロセスも自動停止するため、起動中のアプリを再起動せずに更新が反映されます。
- **アンインストール**: `%ProgramFiles%\setsuna-ime\uninstall.ps1` を実行します。

<details>
<summary>インストール先と管理者権限アプリでの動作について</summary>

DLL・エンジン・モデル・辞書は `%ProgramFiles%\setsuna-ime` に配置され、インストールログとダウンロードキャッシュは `%LOCALAPPDATA%\setsuna-ime` に保存されます。

IME はシステム全体に登録され管理者権限で実行されたアプリにも読み込まれるため、実行ファイルとデータファイルは管理者のみ書き込み可能な `%ProgramFiles%\setsuna-ime` に置かれます。管理者権限アプリ上の IME は読み込みパスを同ディレクトリ内の `installed.toml` のみから参照し、`config.toml` や環境変数によるパス指定は無視されます。

</details>

### Linux（fcitx5）

Arch Linux・Fedora 40+・Ubuntu 24.04+ など、fcitx5 5.1 以上・glibc 2.39 以上の x86_64 環境に対応しています。

```sh
# 標準インストール（既定モデル・読み補正モデル・辞書を取得し、fcitx5 へ登録・再起動）
curl -fsSL https://github.com/esehehelp/setsuna-ime/releases/latest/download/install.sh | sudo bash
```

オプションを指定してカスタマイズすることもできます。

```sh
# 変換モデルを Diphylleia-v1.0-flash に変更
curl -fsSL https://github.com/esehehelp/setsuna-ime/releases/latest/download/install.sh | sudo bash -s -- --model Diphylleia-v1.0-flash

# 辞書なし／fcitx5 への自動登録を行わない場合
curl -fsSL https://github.com/esehehelp/setsuna-ime/releases/latest/download/install.sh | sudo bash -s -- --no-dictionaries --no-register
```

- **配置先**: アドオンは fcitx5 のアドオンディレクトリへ、エンジン・モデル・辞書は `/usr/lib/setsuna-ime` へインストールされます。WebKitGTK 4.1 がある環境では `/usr/bin/setsuna-settings` も配置されます。
- **設定画面**: fcitx5 の設定画面・ステータスメニュー・アプリメニューの「Setsuna の設定」から開けます。
- **アンインストール**: `sudo /usr/lib/setsuna-ime/uninstall.sh` を実行します。

### Android（開発中）

Gboard 風の日本語 QWERTY キーボード（数字行付き）に加え、数字パッド・テキスト編集パネル・クリップボード履歴を備えた Android 向け IME です。

1. リリース版の `setsuna-ime-<version>-android-arm64.apk`（現在はデバッグ鍵で署名）をインストールするか、ソースからビルドします。
2. アプリを開き、「セットアップ」画面から変換モデル・辞書をダウンロードします。
3. 同画面の「入力方法の設定」で Setsuna を有効化し、「切り替え」からキーボードを選択します。

---

## 主な機能

### 変換・編集
- **ライブ変換**: 入力と並行してリアルタイムに変換結果を表示します。表示中の文字をそのまま 1 文字ずつ削除する Backspace や、`Esc` 1 回でかなに戻し 2 回目で取り消す MS-IME 互換操作にも対応しています。
- **文節編集**: CTC アラインメントと形態素解析器によって文節を区切り、文節移動・伸縮や選択文節のみの字種変換（`F6`〜`F10`）ができます。
- **読み辞書による候補補完**: 語の連接コストで不自然な文字単位の誤変換を除外しつつ、モデルが出力しなかった同音異義語や単漢字候補（1〜2 文字のかな入力時）を補完します。

### 入力支援・誤入力補正
- **英日混在入力**: 同梱のかな／英字 n-gram モデルにより、日本語入力のまま英単語・URL・メールアドレス・パス・`@` メンションを自動判別して保持します。固有名詞・略語の自動大文字化（例: `github` → `GitHub`、`pc` → `PC`）や英文の文頭大文字化にも対応します。
- **AI ツール向けコマンド透過**: 行頭や空白直後の `/コマンド`・`$スキル`・`@ファイル` を変換せずアプリへ直接送れます（設定で有効化）。
- **ローマ字・かな配列**: 標準ローマ字・AZIK・JIS かな入力に加え、Mozc 形式のカスタムローマ字表ファイルを読み込めます。
- **誤入力補正**:
  - **読み補正モデル**: 打ち間違いを含むかな列をニューラルモデルで自動補正し、入力したままの読みの候補と併せて提示します。
  - **タイポ・かな補正**: 隣接キー・余分なキー・文字入れ替え（例: `kaisga` → 会社）、「ず／づ」の取り違え（例: `tuzuku` → 続く）、英大文字直後の Shift 離し遅れ（例: `IMENo` → `IMEの`）を補正します。
- **ユーティリティ変換**:
  - **日付・時刻**: `きょう`・`いま`・`らいしゅうのすいようび`・`30ぷんまえ`・`2しゅうかんご` などを変換エンジン未起動時でも展開します。
  - **数字・和暦・計算・文字コード**: `1001` → `10月1日` / `10:01`、`1234` → `1,234` / 漢数字、`2025ねん` ⇔ `令和7年`、`1+2=` → `3`、`U+3042` → `あ`。
  - **絵文字・顔文字・記号**: `にこにこ` → `😊`、`えもじ` / `かおもじ` による一覧表示、記号辞書（`やじるし` → `→`、`あるふぁ` → `α`）に対応します。

### 学習・ユーザー辞書・カスタマイズ
- **学習と候補非表示**: 選択した候補を `learning.jsonl` に記録して全アプリ間で共有します。文単位・単語単位・文節編集時のいずれにも反映され（文節編集で組み立てた文のように変換モデルの候補に無い選択も、次から候補に入ります）、変換中に `Ctrl+Delete` で学習取り消しや候補の非表示化ができます。
- **予測変換**: 確定した変換履歴を記憶し、読みの途中から `Tab` キーで予測候補を表示します（入力中に候補ウィンドウは自動表示しません）。
- **ユーザー辞書**: Google 日本語入力 / Mozc 互換形式（`user_dictionary.txt`）を設定 GUI から編集できます。Mozc・Microsoft IME・ATOK・Gboard のエクスポート形式のインポート、部分一致、品詞「短縮よみ」「抑制単語」、英字・記号読み（例: `w` → `(笑)`）に対応します。
- **設定 GUI・アプリ別設定**: 変換挙動・記号の全角半角・候補ウィンドウ外観（Windows）・アプリごとのオン／オフや機能切替を `setsuna-settings` から設定できます。

### 拡張機能（任意）
- **LLM 書き直し（Windows / Linux）**: 変換時に OpenAI（Sign in with ChatGPT / API キー）または OpenAI 互換 API へ読み・変換結果・直前文脈を送り、自然な日本語に整えた候補を先頭に表示します。
  - **プライバシー保護**: URL・メール・パス・API キー／トークン・長い数字・「パスワード」などの機密語を含む文脈や入力、およびパスワード欄・機密入力欄（InputScope / fcitx5 Sensitive）の内容は送信しません。資格情報は Windows では DPAPI 保護の `credentials.bin`、Linux / macOS では権限 `0600` の `credentials.json` に保存され、`config.toml` には記録しません。
- **LM リランカー**: llama.cpp（GGUF）の causal LM（`setsuna-rerank`）で上位候補の対数尤度を加算して並べ替えます。リランク完了までライブ変換の更新を待つ設定にも対応します。
- **文字 n-gram**: 出現頻度の低い不自然な文字連結（例: `博紙`）のスコアを下げます（`[ngram] table`）。

---

## 基本操作と設定ファイル

### 主なキー操作

| 操作 | キー（既定） |
|---|---|
| IME オン／オフ | `半角/全角`、`Alt+半角/全角`、`` Alt+` ``、`` ` ``（Linux のみ）、`IME オン` / `IME オフ`（Windows のみ） |
| 変換・候補選択 | `Space` / `↓` / `↑` / `PageUp` / `PageDown` / 数字キー |
| 文節の移動・伸縮 | `←` / `→`（移動）、`Shift+←` / `Shift+→`（伸縮） |
| 字種変換（全体／選択文節） | `F6` ひらがな、`F7` カタカナ、`F8` 半角カタカナ、`F9` 全角英数、`F10` 半角英数（`Ctrl+U/I/O/P/T` でも可） |
| かなに戻す／キャンセル | `Esc`（1 回目でかなに戻し、2 回目で入力を取り消し） |
| 予測候補を開く | `Tab` |
| 候補の学習取り消し／非表示 | 変換中に `Ctrl+Delete` |

※ オン／オフ切替キーは設定画面または `config.toml` の `toggle_keys` / `on_keys` / `off_keys` で変更できます。Linux の `` ` `` は US 配列で `半角/全角` の位置にあるキーに対応します。

### 設定ファイルとログの保存先

| OS | 設定ファイル | ログ |
|---|---|---|
| **Windows** | `%APPDATA%\setsuna-ime\config.toml` | `%APPDATA%\setsuna-ime\tsf.log` |
| **Linux** | `~/.config/setsuna-ime/config.toml` | `~/.local/state/setsuna-ime/fcitx5.log`（`$XDG_STATE_HOME` 優先） |
| **Android** | `<app files>/config/setsuna-ime/config.toml` | `<app files>/state/setsuna-ime/` |

> **ログのプライバシー**: ログ（4 MB ごとに `.1` へローテーション）には既定で入力文字や変換文字列を残さず、文字数のみを記録します。不具合調査などで内容の記録が必要な場合のみ、`config.toml` に `[diagnostics]` と `log_typed_text = true` を設定してください（パスワード欄は常に記録されません）。

---

## 公開モデルとベンチマーク

すべての公開ファイルは [crates/tsf/models.json](crates/tsf/models.json) に Hugging Face のリビジョン・サイズ・SHA-256 を固定しており、ログイン不要でハッシュ検証付きダウンロードが行われます。既存の検証済みファイルがある場合はオフラインでも再利用します。

### 変換モデル（Apache-2.0）

| モデル | サイズ | 1 位一致<br>（`wiki_fresh_v4` / `AJIMEE` / `probe_v6`） | 上位 5 位一致<br>（`wiki_fresh_v4` / `AJIMEE` / `probe_v6`） | 速度<br>（1 スレッド中央値） | 特徴 |
|---|---:|---:|---:|---:|---|
| [**Cymbidium-v1.2-xsmall**](https://huggingface.co/esehe/Cymbidium-v1.2-xsmall)（既定） | 約 23 MB | 83.3% / 79.5% / 84.8% | 88.9% / 88.4% / 88.7% | 1.1 ms | 高速・高精度の標準モデル |
| [**Diphylleia-v1.0-flash**](https://huggingface.co/esehe/Diphylleia-v1.0-flash) | 約 25 MB | 83.1% / 77.5% / 85.8% | 92.7% / 92.5% / 93.5% | 2.7〜2.8 ms | 浅い自己回帰デコーダー付き、上位 5 候補のカバー率重視 |
| *参考: [zenz-v3.2-small](https://huggingface.co/Miwa-Keita/zenz-v3.2-small-gguf)（Q5_K_M）* | — | *74.0% / 76.6% / 81.6%* | *78.6% / 83.9% / 86.7%* | *約 990 ms* | *自己回帰モデルとの比較用* |

<details>
<summary>評価セットと計測条件の詳細</summary>

- **評価セット（2026-11-12 時点の標準セット）**:
  - `wiki_fresh_v4`: 変換モデルの学習データ以降に作成された Wikipedia 日本語版の記事文 4,998 件（正解の表記揺れを許容）
  - `AJIMEE`（AJIMEE-Bench JWTD v2 `v2-fix2`）: 628 件（読みの誤りを修正済み）
  - `probe_v6`: 数字・人名・技術用語・同音異義語などの分野別 309 件
- **計測条件**:
  - 一致率は変換エンジン単体（ビーム幅 20、内蔵推論エンジン）での完全一致率です。実際の IME ではこれに読み辞書による候補補完が加わります。
  - `Diphylleia-v1.0-flash` は自己回帰デコーダーの探索打ち切り幅 8（既定値）で計測しています。
  - サイズはエンジンが読み込む重みファイルの合計容量です（内蔵推論用の重みのみで ONNX グラフは含みません）。
  - 速度は AJIMEE の 200 文を 1 スレッドで変換したときの中央値（2 回計測の範囲）です。
  - 参考の `zenz-v3.2-small` は llama.cpp で 1 スレッド・ビーム幅 5・直前文脈 40 文字までの条件で計測しており、速度は AJIMEE 1 件あたりの中央値です（計測条件が異なります）。

</details>

### 付属コンポーネント

標準インストール時に変換モデルと併せて自動取得されます。

| コンポーネント | 配布物 | ライセンス | サイズ | 役割 |
|---|---|---|---:|---|
| 読み補正モデル | [**Nandina-v1.0-tiny**](https://huggingface.co/esehe/Nandina-v1.0-tiny) | Apache-2.0 | 約 4 MB | 打ち間違いを含む読みを変換前に補正 |
| 読み辞書 | `lexicon-v2.lexicon`（[setsuna-ime-dict](https://huggingface.co/esehe/setsuna-ime-dict)） | CC BY-SA 4.0 | 約 28 MB | 2 位以下の並べ替え・同音語や単漢字候補の補完・漢字列の区切り判定 |
| 形態素解析器 | `nid-ma-v2-600k-tb-packed`（[setsuna-ime-dict](https://huggingface.co/esehe/setsuna-ime-dict)） | CC BY-SA 4.0 | 約 7 MB | 文節編集時の文節区切り |

- **読み補正モデルの調整**: 補正の確信度がしきい値（既定値 `4`）を超えた場合に補正後の変換を先頭に表示し、入力したままの読みの候補も併せて提示します。感度は設定画面の「読み補正モデル」または `[engine] corrector_threshold` で変更できます。
- **除外・無効化**: インストール時に除外する場合は、Windows（`install.ps1`）では `-NoCorrector` / `-NoDictionaries`、Linux（`install.sh`）では `--no-corrector` / `--no-dictionaries` を指定します。読み補正モデルを無効化したい場合は `config.toml` の `[engine] corrector_manifest` を空にしてください。

---

## アーキテクチャ

クレート間の依存は **`artifact` → `runtime` → `platform`** の一方向に保たれています。各フロントエンドは推論エンジンを直接リンクせず、子プロセス（stdio JSONL）または共有サーバー経由で通信します。

| Crate | 層 | 役割 |
|---|---|---|
| `crates/artifact` | artifact | 各モデルマニフェスト（`*-export-v1`）の読み込み・検証 |
| `crates/engine` | runtime | CTC-NAT 推論エンジン `setsuna-engine`（library / CLI / daemon / Windows serve） |
| `crates/rerank` | runtime | llama.cpp による LM リランカー `setsuna-rerank` |
| `crates/analyzer` | runtime | vibrato / vibrato-packed 形態素解析器（`setsuna-engine --analyzer` が保持） |
| `crates/pipe` | runtime | Windows 共有エンジン・リランカー用の名前付きパイプ実装 |
| `crates/ime-core` | platform | 各フロントエンド共通の IME コア（ローマ字・状態機械・英日混在・学習・辞書・後処理・クライアント） |
| `crates/tsf` / `tsf-shim` / `tsf-reg` | platform | Windows TSF IME DLL・ホットスワップ用シム・登録補助 |
| `crates/linux` | platform | fcitx5 アドオン（Rust ライブラリ + `fcitx5/` の C++ shim） |
| `crates/android` | platform | Android IME（Rust ライブラリ + `app/` の Kotlin キーボード・サービス） |
| `crates/rewrite` | platform | LLM 書き直しデーモン `setsuna-rewrite` |
| `crates/settings` | platform | WebView 設定 GUI `setsuna-settings`（デスクトップ・Android 共用） |
| `crates/bench` | tool | 変換精度・速度の評価ツールと評価セット |

### 実行モデルとマニフェスト

- **プロセス共有（Windows）**: 通常アプリはログインごとに 1 つの共有サーバー（`setsuna-engine serve` / `setsuna-rerank serve`）へ名前付きパイプで接続し、モデルや解析器辞書のメモリを共有します（無接続 10 分で終了、リランカーの KV キャッシュは 60 秒で解放）。AppContainer（スタートメニュー検索・ストアアプリ）は既存共有エンジンへの接続のみ行い、管理者権限アプリは専用子プロセスを使用します。
- **マニフェスト駆動ロード**: すべてのモデルは `*.manifest.json` 経由で読み込みます（例は `tests/fixtures/`）。
  - **変換モデル（`neo-ime-ctc-nat-export-v1`）**: 内蔵推論エンジン用の重み（`proposal.native`、AR デコーダー付きモデルでは `ar_decoder.native`）・語彙・special token ID・`decode_defaults`・任意の CTC-CRF 遷移表（`crf`）やリスコア表（`rescore`）を持ちます。推論は Rust 独自実装（AVX-512 VNNI / AVX-VNNI / AVX2 / AVX / 汎用フォールバック、全経路で同一の整数演算結果）で動作し、ONNX グラフは読み込みません。
  - **リランカー（`neo-ime-lm-scorer-export-v1`）**: GGUF 重み、`score_defaults`、任意の SentencePiece トークナイザー。
  - **解析器（`neo-ime-analyzer-export-v1`）**: `vibrato-zstd`（通常の vibrato 辞書 + 任意の別読み表 `alternates`）または省メモリな `vibrato-packed-zstd`（`vendor/vibrato-packed` + 符号表 `packed`）。

---

## ソースからのビルド

### 共通要件

- Rust（stable）
- CMake・C/C++ コンパイラ（`crates/rerank` が llama.cpp をソースビルドするために必要）

```sh
cargo build --release
cargo test
```

> モデルファイルが手元になくても、ビルドとユニットテストはすべて実行できます。

### Windows

```powershell
# ビルド
cargo build --release

# 公開モデル・辞書を自動取得してインストール（手元のモデルは -Manifest <path> 等で指定）
pwsh crates/tsf/install.ps1

# リリース用 zip・setup.ps1 の作成（リランカーも同梱する場合は -WithRerank）
pwsh crates/tsf/package.ps1

# ダウンローダー回帰テスト（オフライン実行可）
pwsh tests/model-download.ps1
```

主な `install.ps1` オプション: `-Model <name>`, `-Manifest <path>`, `-RerankManifest <path>`, `-Lexicon <path>`, `-Analyzer <path>`, `-Corrector <path>`, `-NameDictionary <tsv>`, `-NoDictionaries`, `-NoCorrector`。モデル単体取得は `pwsh crates/tsf/fetch-model.ps1` を使用します。

### Linux（fcitx5）

追加要件: fcitx5 開発パッケージ、WebKitGTK 4.1 開発パッケージ（設定 GUI 用）、OpenMP 対応 C/C++ コンパイラ、`curl`、`jq`。

```sh
# 開発ビルド
cmake -B target/fcitx5 -DCMAKE_INSTALL_PREFIX=/usr crates/linux/fcitx5
cmake --build target/fcitx5 && sudo cmake --install target/fcitx5

# リリース用 tar.gz・install.sh の作成
crates/linux/package.sh
sudo bash target/dist/install.sh --archive target/dist/setsuna-ime-*.tar.gz

# ダウンローダー回帰テスト（オフライン実行可）
bash tests/model-download.sh
```

開発中は `-DSETSUNA_MODULE_DIR=<repo>/target/release` を渡すと `target/release` のエンジンと `local/models/onnx` のモデルを参照します。モデル単体取得は `crates/linux/fetch-model.sh` を使用します。

### Android

追加要件: Android SDK（platform 35 + NDK、`ANDROID_HOME`）、JDK 17+（`JAVA_HOME`）、`rustup target add aarch64-linux-android`、`cargo install cargo-ndk`。

```powershell
# arm64 ネイティブライブラリと APK をビルドし adb でインストール（-Release でリリースビルド）
pwsh crates/android/build.ps1 -Install

# （任意・debug ビルド時）PC 側のモデル・辞書を端末へ転送
pwsh crates/android/push-models.ps1
```

### 設定 GUI の単体起動

```sh
cargo run --release -p setsuna-settings
```

---

## ライセンス

- **Setsuna IME 本体**: MIT License（[LICENSE](LICENSE)）。`crates/tsf` の一部は karukan-tsf（MIT）を基にしています（[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）。
- **外部コンポーネント**:
  - [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT）
  - [vibrato](https://github.com/daac-tools/vibrato)（MIT / Apache-2.0、`vendor/vibrato-packed` に辞書レイアウト変更を加えて同梱。詳細は [vendor/vibrato-packed/VENDORED.md](vendor/vibrato-packed/VENDORED.md)）
  - 英日混在入力モデル（[crates/ime-core/data/mixed/README.md](crates/ime-core/data/mixed/README.md)）: FineWeb2-edu-japanese / FineWeb 由来、ODC-By 1.0
  - その他の Rust クレートは `Cargo.lock` および [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
- **公開モデル・辞書**: 変換モデル（Cymbidium / Diphylleia）および読み補正モデル（Nandina）は Apache-2.0、配布辞書（[esehe/setsuna-ime-dict](https://huggingface.co/esehe/setsuna-ime-dict)）は CC BY-SA 4.0 です。モデルの重みおよび辞書ファイル本体はこのリポジトリに含まれません。
