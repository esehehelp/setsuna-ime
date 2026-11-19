# Setsuna IME（刹那）

Setsuna (刹那) is a Japanese input method built around a non-autoregressive neural kana-kanji converter (CTC-NAT). It provides fast CPU live conversion with frontends for **Windows (TSF)**, **Linux (fcitx5)**, and **Android** (in development), plus an optional LM reranker, LLM rewriter, and settings GUI.

**Setsuna（刹那）** は、非自己回帰ニューラルかな漢字変換（CTC-NAT）を中心とした日本語入力メソッド（IME）です。CPU 上で 1 文あたり数ミリ秒の高速なライブ変換を実現します。

> **ステータス**: 研究・開発段階のプロジェクトです。公開変換モデル（Apache-2.0）と辞書（CC BY-SA 4.0）はインストーラーから自動取得してそのまま試用できます。

---

## インストール

### Windows（TSF / x64 Windows 10・11）

Windows PowerShell 5.1 または PowerShell 7 (`pwsh`) で実行します（`regsvr32` 登録時に UAC が表示されます）。

```powershell
# 標準インストール（既定モデル Cymbidium-v1.2-xsmall + 読み辞書・解析器を自動取得）
irm https://github.com/esehehelp/setsuna-ime/releases/latest/download/setup.ps1 | iex

# 変換モデルを切り替える例（Bletilla-v0.1-small）
& ([scriptblock]::Create((irm https://github.com/esehehelp/setsuna-ime/releases/latest/download/setup.ps1))) -Model Bletilla-v0.1-small

# 辞書・解析器を取得せずモデルのみ入れる場合
& ([scriptblock]::Create((irm https://github.com/esehehelp/setsuna-ime/releases/latest/download/setup.ps1))) -NoDictionaries
```

- リリース zip（`setsuna-ime-<version>-windows-x64.zip`）を展開して `install.cmd` を実行しても同じです。Visual C++ ランタイム（14.44 以上）が無い場合は自動導入します。
- **設定 GUI**: タスクバーの「あ / A」アイコンを右クリック →「設定...」から開けます。変更は次の入力から反映されます。
- **起動中アプリへの更新反映**: TSF に登録した軽量シム（`setsuna_tsf_shim.dll`）が新バージョンの本体 DLL をホットスワップし、旧エンジンプロセスも自動停止するため、アプリを再起動せずに更新が届きます。
- **インストール先**: DLL・エンジン・モデル・辞書は `%ProgramFiles%\setsuna-ime` に入ります。IME はマシン全体に登録され、管理者として実行したアプリにも読み込まれるため、読み込むコードとファイルは管理者だけが書き換えられる場所に置き、管理者として実行したアプリの IME はそれらのパスを同じ場所の `installed.toml` からだけ取ります（`config.toml` と環境変数のパスは使いません）。インストールのログとダウンロードのキャッシュは `%LOCALAPPDATA%\setsuna-ime` です。
- **アンインストール**: `%ProgramFiles%\setsuna-ime\uninstall.ps1` を実行します。

### Linux（fcitx5 / x86_64）

fcitx5 5.1 以上・glibc 2.39 以上（Arch Linux、Fedora 40+、Ubuntu 24.04+ など）に対応しています。

```sh
# 標準インストール（既定モデル + 読み辞書・解析器を自動取得し fcitx5 へ登録・再起動）
curl -fsSL https://github.com/esehehelp/setsuna-ime/releases/latest/download/install.sh | sudo bash

# 変換モデルを切り替える例
curl -fsSL https://github.com/esehehelp/setsuna-ime/releases/latest/download/install.sh | sudo bash -s -- --model Bletilla-v0.1-small

# 辞書・解析器なし / fcitx5 への登録を行わない例
curl -fsSL https://github.com/esehehelp/setsuna-ime/releases/latest/download/install.sh | sudo bash -s -- --no-dictionaries --no-register
```

- アドオンは fcitx5 のアドオンディレクトリへ、エンジン・モデル・辞書は `/usr/lib/setsuna-ime` へ入ります（WebKitGTK 4.1 がある環境では `/usr/bin/setsuna-settings` も配置されます）。
- **設定 GUI**: fcitx5 の設定画面・状態表示メニュー・アプリメニューの「Setsuna の設定」から開けます。
- **アンインストール**: `sudo /usr/lib/setsuna-ime/uninstall.sh` を実行します。

### Android（開発中 / arm64・Android 8 以降）

Gboard 風の日本語 QWERTY（数字行付き）＋数字パッド・テキスト編集・クリップボード履歴を備えた Android IME です。アプリ内の設定画面（「この端末」）から変換モデルや辞書を Hugging Face 経由でダウンロードできます。現在はソースからのビルドで導入します（[Android のビルド](#android) を参照）。

---

## 主な機能

### 変換・編集
- **ライブ変換**: 入力と並行してリアルタイムに変換を表示します。表示中の文字をそのまま 1 文字ずつ消す Backspace（`backspace_deletes_shown`）や、`Esc` 1 回でかなに戻し 2 回目で入力を消す MS-IME 互換挙動（`escape_to_kana`）に対応しています。
- **文節編集**: CTC アラインメントと形態素解析器（`--analyzer`）で文節に区切り、`←` / `→` で文節移動、`Shift+←` / `Shift+→` で文節伸縮ができます。選択文節のみの字種変換（`F6`–`F10`）も可能です。
- **読み辞書による候補補完・フィルタ**: 読み辞書（`[lexicon]`）がある場合、語の連接コストで不自然な文字単位の誤変換を除外しつつ、モデルが出さなかった同音異義語や単漢字候補（1〜2 文字のかな入力時）を補完します。

### 英日混在入力・誤入力補正
- **英日混在入力**: 同梱のかな / 英字 n-gram モデルにより、日本語入力のまま英単語・URL・メールアドレス・パス・`@` メンションを自動で切り分けて保持します。固有名詞・略語の大文字化（例: `github` → `GitHub`、`pc` → `PC`、`name_case`）や英文の文頭大文字化にも対応します。
- **AI ツール向けコマンド透過**: `pass_through_commands = true` にすると、行頭や空白直後の `/コマンド`・`$スキル`・`@ファイル` を変換せずアプリへ直接送ります。
- **入力方式・ローマ字表**: 標準・AZIK・JIS かな入力・Mozc 形式のカスタム表（`romaji_table`）を選べます。
- **誤入力補正（`typo_correction`）**:
  - かなとして読めない打ち間違い（隣接キー・余計なキー・2 文字入れ替え、例: `kaisga` → 会社）の補正候補を提示します。
  - `ず` / `づ` の取り違えをかなモデルで補正します（例: `tuzuku` → 続く、両方使われる `きずく` は両候補を提示）。
  - 英大文字の直後に Shift を離し遅れた入力（例: `IMENo`、`IMEDemo`、`PCWokatta`）に対し、元の英語読みと並行して `IMEの`・`IMEでも`・`PCをかった` などの別読み候補を提示します。

### 学習・予測・ユーザー辞書
- **学習**: 選択した候補を `learning.jsonl` に記録し、全アプリ間で共有します。文全体だけでなく単語単位・文節編集時の候補リストにも適用されます。変換中に `Ctrl+Delete` で学習取り消し・候補の非表示化ができます。
- **予測変換（`prediction`）**: 確定した変換を記憶し、読みの途中から `Tab` で予測候補を展開します（入力中に自動で窓は出しません）。
- **ユーザー辞書**: `user_dictionary.txt`（Google 日本語入力 / Mozc 互換形式）を設定 GUI から編集でき、Mozc・MS-IME・ATOK・Gboard のエクスポート形式をインポートできます。部分一致、品詞「短縮よみ」「抑制単語」、`w` → `(笑)` のような英字・記号読みにも対応します。

### ユーティリティ変換
- **日付・時刻（`date_time_candidates`）**: `きょう`、`あした`、`いま`、`にちじ`、`らいしゅうのすいようび`、`らいげつ`、`30ぷんまえ`、`2しゅうかんご` などをモデル未起動でも展開します。
- **数字・計算・文字コード（`number_candidates`）**: `1001` → `10月1日` / `10:01`、`1234` → `1,234` / 漢数字、`2025ねん` ⇔ `令和7年`、`1+2=` → `3`、`U+3042` → `あ`。
- **絵文字・顔文字・記号**: `にこにこ` → `😊`、`えもじ` / `かおもじ` で一覧表示、記号辞書（`やじるし` → `→`、`あるふぁ` → `α`）に対応します。

### 拡張機能（任意）
- **LLM 書き直し（Windows / Linux）**: `[rewriter]` を有効にすると、変換時に OpenAI（Sign in with ChatGPT / API キー）または OpenAI 互換 API へ読み・変換結果・直前文脈を送り、自然な日本語に書き直した候補を先頭へ表示します。
  - **プライバシー保護**: URL・メール・パス・API キー/トークン・長い数字・「パスワード」等の語を含む文脈や入力、およびパスワード/機密入力欄（InputScope / fcitx5 Sensitive）の内容は送信しません。資格情報は Windows では DPAPI 保護の `credentials.bin`、Linux / macOS では権限 `0600` の `credentials.json` に保存され、`config.toml` には書きません。
- **LM リランカ（任意）**: llama.cpp（GGUF）による causal LM（`setsuna-rerank`）で上位候補の対数尤度を加算して並べ替えます。`[rerank] defer_ms` でライブ変換表示をリランク結果まで待たせることも可能です。
- **文字 n-gram（任意）**: `[ngram] table` を指定すると珍しい文字連結（例: `博紙`）のスコアを下げます。
- **設定 GUI・アプリ別設定**: `setsuna-settings` から変換挙動・記号の全角半角・候補ウィンドウ外観（Windows）・アプリごとのオン/オフや機能切替（`[[apps]]`）を設定できます。

---

## 公開モデル・辞書

すべての公開ファイルは [crates/tsf/models.json](crates/tsf/models.json) に Hugging Face のリビジョン・サイズ・SHA-256 を固定しており、ログイン不要で検証つき取得が行われます。既定のインストールではダウンロード用キャッシュからハードリンクで配置した後にキャッシュを削除し、再実行時は既存の検証済みファイルをオフラインでも再利用します。

### 変換モデル（Apache-2.0）

| モデル | サイズ | 1 位一致（wiki_fresh_v4 / AJIMEE v2-fix2 / probe_v6） | 上位 5 位一致（同） | 速度（1 スレッド・中央値） | 用途 |
|---|---:|---:|---:|---:|---|
| [**Cymbidium-v1.2-xsmall**](https://huggingface.co/esehe/Cymbidium-v1.2-xsmall)（既定） | 約 23 MB | 83.2% / 79.1% / 84.8% | 88.9% / 88.1% / 88.7% | 2.3〜2.4 ms | 高速・高精度の標準モデル |
| [**Cymbidium-v1.1-xsmall**](https://huggingface.co/esehe/Cymbidium-v1.1-xsmall) | 約 23 MB | 82.3% / 79.1% / 85.1% | 88.4% / 87.4% / 88.7% | 2.4 ms | 前の既定 |
| [**Cymbidium-v1.0-xsmall**](https://huggingface.co/esehe/Cymbidium-v1.0-xsmall) | 約 14 MB | — | — | — | より軽量な旧版（このセットでは未計測） |
| [**Bletilla-v0.1-small**](https://huggingface.co/esehe/Bletilla-v0.1-small) | 約 130 MB | 81.1% / 78.0% / 84.1% | 89.3% / 92.5% / 89.6% | 7.6〜8.4 ms | 上位 5 候補のカバー率重視 |
| 参考: [zenz-v3.2-small](https://huggingface.co/Miwa-Keita/zenz-v3.2-small-gguf)（Q5_K_M） | — | 74.0% / 76.6% / 81.6% | 78.6% / 83.9% / 86.7% | 約 990 ms（AJIMEE 1 件） | 自己回帰モデルとの比較用 |

- 評価セットは 2026-11-12 時点の標準のもの: wiki_fresh_v4（変換モデルの学習データより後に作られた Wikipedia 日本語版の記事の文 4,998 件。正解の表記の揺れを許す）、AJIMEE-Bench JWTD v2 の v2-fix2（628 件。読みの誤りを直したもの）、probe_v6（数字・人名・技術用語・同音異義語などの分野別 309 件）。
- 数値は変換エンジン単体（ビーム幅 20）の完全一致率。IME ではこれに読み辞書が候補を足します。Bletilla は並べ替えの表（`pairs.bin` / `words.tsv`）を使ったときの値です。
- 速度は AJIMEE の 200 文を 1 スレッドで変換したときの中央値（2 回の計測の幅）。
- zenz-v3.2-small は llama.cpp で 1 スレッド、ビーム幅 5、前の文脈 40 文字までで動かした値で、速度は AJIMEE の 1 件あたりの中央値（計測の仕方が上のモデルとは違います）。

### 辞書・解析器（CC BY-SA 4.0 / [esehe/setsuna-ime-dict](https://huggingface.co/esehe/setsuna-ime-dict)）

| コンポーネント | サイズ | 用途 |
|---|---:|---|
| `lexicon` (`lexicon-v2.lexicon`) | 約 28 MB | 読み辞書（変換モデルの 1 位はそのままに、2 位以下の並べ替え、モデルが出さない語・単漢字の候補の追加、漢字の並びの区切り） |
| `lexicon-small` (`lexicon-v1-small.lexicon`) | 約 3.6 MB | Android 向けの軽量読み辞書（よく使う語とすべての単漢字の読み。候補を足すだけで並べ替えない） |
| `analyzer` (`nid-ma-v2-600k-tb-packed`) | 約 7 MB | 形態素解析器（文節編集の区切り） |

---

## 基本操作・設定ファイル

### 主なキー操作

| 操作 | キー（既定） |
|---|---|
| IME オン / オフ（Windows / Linux） | `半角/全角`、`Alt+半角/全角`、`` Alt+` ``、`IME オン` / `IME オフ`（Windows のみ）。`toggle_keys` / `on_keys` / `off_keys` で変更可 |
| 変換・次候補 | `Space` / `↓` / `↑` / `PageUp` / `PageDown` / 数字キー |
| 文節移動・伸縮 | `←` / `→`（移動）、`Shift+←` / `Shift+→`（伸縮） |
| 字種変換（全体 / 選択文節） | `F6` ひらがな、`F7` カタカナ、`F8` 半角カタカナ、`F9` 全角英数、`F10` 半角英数（`Ctrl+U/I/O/P/T`） |
| かなに戻す / 入力キャンセル | `Esc`（1 回目でかな戻し、2 回目でキャンセル） |
| 予測候補を開く | `Tab` |
| 候補の学習取消 / 非表示 | 変換中に `Ctrl+Delete` |

### 設定・ログの保存先

| OS | 設定ファイル | ログ |
|---|---|---|
| **Windows** | `%APPDATA%\setsuna-ime\config.toml` | `%APPDATA%\setsuna-ime\tsf.log` |
| **Linux** | `~/.config/setsuna-ime/config.toml` | `~/.local/state/setsuna-ime/fcitx5.log`（`$XDG_STATE_HOME` 優先） |
| **Android** | `<app files>/config/setsuna-ime/config.toml` | `<app files>/state/setsuna-ime/` |

> **ログのプライバシー**: ログ（4 MB で `.1` にローテーション）には既定で打鍵文字や変換文字列を残さず、文字数のみを記録します。手元で不具合を調査したい場合のみ `config.toml` に `[diagnostics]` と `log_typed_text = true` を設定してください（パスワード欄は常に変換対象外のため記録されません）。

---

## アーキテクチャ・構成

依存は **artifact → runtime → platform** の一方向です。各フロントエンドは推論エンジンを直接リンクせず、子プロセス（stdio JSONL）または共有サーバー経由で通信します。

| Crate | 層 | 役割 |
|---|---|---|
| `crates/artifact` | artifact | 各モデルマニフェスト（`*-export-v1`）の読み込み・検証 |
| `crates/engine` | runtime | CTC-NAT 推論エンジン `setsuna-engine`（library / CLI / daemon / Windows serve） |
| `crates/rerank` | runtime | llama.cpp による LM リランカ `setsuna-rerank` |
| `crates/analyzer` | runtime | vibrato / vibrato-packed 形態素解析器（`setsuna-engine --analyzer` が保持） |
| `crates/pipe` | runtime | Windows 共有エンジン・リランカ用の名前付きパイプ実装 |
| `crates/ime-core` | platform | 各フロントエンド共通の IME ロジック（ローマ字・状態機械・英日混在・学習・辞書・後処理・クライアント） |
| `crates/tsf` / `tsf-shim` / `tsf-reg` | platform | Windows TSF IME DLL、ホットスワップ用シム、登録補助 |
| `crates/linux` | platform | fcitx5 アドオン（Rust ライブラリ + `fcitx5/` の C++ shim） |
| `crates/android` | platform | Android IME（Rust ライブラリ + `app/` の Kotlin キーボード・サービス） |
| `crates/rewrite` | platform | LLM 書き直しデーモン `setsuna-rewrite`（Sign in with ChatGPT / OpenAI / 互換 API） |
| `crates/settings` | platform | WebView 設定 GUI `setsuna-settings`（デスクトップ・Android 共用ページ） |

### 実行モデルとマニフェスト

- **プロセス共有（Windows）**: 通常アプリはログインごとに 1 つの共有サーバー（`setsuna-engine serve` / `setsuna-rerank serve`）へ名前付きパイプで接続し、モデルや解析器辞書のメモリを共有します（無接続 10 分で終了、リランカの KV キャッシュは 60 秒で解放）。AppContainer（スタートメニュー検索・ストアアプリ）は既存共有エンジンへ接続のみ行い、管理者権限アプリは専用子プロセスを使用します。
- **マニフェスト駆動ロード**: モデルは必ず `*.manifest.json` 経由で読み込みます（例は `tests/fixtures/`）。
  - **変換モデル**（`neo-ime-ctc-nat-export-v1`）: ONNX proposal グラフ・語彙・special token ID・`decode_defaults`・任意の CTC-CRF 遷移表 (`crf`) や rescore 表 (`rescore`)。カスタム op（`neo_ime` ドメインの `TokenMoE` / `AddLayerNorm` / `LinearQ4G` / `TokenMoEQ4G`）は AVX-512 VNNI・AVX-VNNI・AVX2・ARM dotprod（および汎用 Rust フォールバック）で実行されます。
  - **リランカ**（`neo-ime-lm-scorer-export-v1`）: GGUF 重み、`score_defaults`、任意の SentencePiece トークナイザ。
  - **解析器**（`neo-ime-analyzer-export-v1`）: `vibrato-zstd`（通常の vibrato 辞書 + 任意の別読み表 `alternates`）または省メモリな `vibrato-packed-zstd`（`vendor/vibrato-packed` + 符号表 `packed`、`examples/analyzer.rs build` / `usage` で生成）。

---

## ソースからのビルド

### 共通要件

- Rust（stable）
- CMake・C/C++ コンパイラ（`crates/rerank` が llama.cpp をソースビルドするため必要）
- ONNX Runtime **1.28.0**（エンジンは実行時に共有ライブラリを動的ロードします。探索順: `--ort-dylib` → 環境変数 `SETSUNA_ORT_DYLIB` → エンジンと同一ディレクトリ）

```sh
cargo build --release
cargo test
```

> モデルファイルが無くてもビルドとユニットテストは通ります。なお ONNX Runtime のバージョン変更は int8 演算の数値に影響するため、既定版の変更はベンチマーク再計測とセットで行います。

### Windows

```powershell
# ビルドと ONNX Runtime 取得
cargo build --release
pwsh crates/engine/fetch-onnxruntime.ps1

# 公開モデル・辞書を自動取得してインストール（手元のモデルは -Manifest <path> 等を指定）
pwsh crates/tsf/install.ps1

# リリース用 zip・setup.ps1 の作成（リランカも同梱する場合は -WithRerank）
pwsh crates/tsf/package.ps1

# ダウンローダー回帰テスト（オフライン実行可）
pwsh tests/model-download.ps1
```

主な `install.ps1` オプション: `-Model <name>`、`-Manifest <path>`、`-RerankManifest <path>`、`-Lexicon <path>`、`-Analyzer <path>`、`-NameDictionary <tsv>`、`-NoDictionaries`。モデル単体取得は `pwsh crates/tsf/fetch-model.ps1`。

### Linux（fcitx5）

追加要件: fcitx5 開発パッケージ、WebKitGTK 4.1 開発パッケージ（設定 GUI 用）、OpenMP 対応 C/C++ コンパイラ、`curl`、`jq`。

```sh
# ONNX Runtime 取得と開発ビルド
crates/engine/fetch-onnxruntime.sh
cmake -B target/fcitx5 -DCMAKE_INSTALL_PREFIX=/usr crates/linux/fcitx5
cmake --build target/fcitx5 && sudo cmake --install target/fcitx5

# リリース用 tar.gz・install.sh の作成
crates/linux/package.sh
sudo bash target/dist/install.sh --archive target/dist/setsuna-ime-*.tar.gz

# ダウンローダー回帰テスト（オフライン実行可）
bash tests/model-download.sh
```

開発中は `-DSETSUNA_MODULE_DIR=<repo>/target/release` を渡すと `target/release` のエンジンと `local/models/onnx` のモデルを参照します。モデル単体取得は `crates/linux/fetch-model.sh`。

### Android

追加要件: Android SDK（platform 35 + NDK、`ANDROID_HOME`）、JDK 17+（`JAVA_HOME`）、`rustup target add aarch64-linux-android`、`cargo install cargo-ndk`。

```powershell
# arm64 ネイティブライブラリと APK をビルドし adb でインストール（-Release でリリースビルド）
pwsh crates/android/build.ps1 -Install

# （任意・debug ビルド時）PC 側のモデル・辞書を端末へ転送
pwsh crates/android/push-models.ps1
```

インストール後、アプリの「この端末」画面からモデルをダウンロードし、「入力方法の設定」で Setsuna を有効化してキーボードを切り替えます（リリース APK も現在は debug 鍵で署名されています）。

### 設定 GUI の単体起動

```sh
cargo run --release -p setsuna-settings
```

---

## ライセンス

- **Setsuna IME 本体**: MIT License（[LICENSE](LICENSE)）。`crates/tsf` の一部は karukan-tsf（MIT）を基にしています（[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）。
- **外部コンポーネント**:
  - [ONNX Runtime](https://github.com/microsoft/onnxruntime)（MIT）
  - [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT）
  - [vibrato](https://github.com/daac-tools/vibrato)（MIT / Apache-2.0、`vendor/vibrato-packed` に辞書レイアウト変更を加えて同梱。詳細は [vendor/vibrato-packed/VENDORED.md](vendor/vibrato-packed/VENDORED.md)）
  - 英日混在入力モデル（[crates/ime-core/data/mixed/README.md](crates/ime-core/data/mixed/README.md)）: FineWeb2-edu-japanese / FineWeb 由来、ODC-By 1.0
  - その他の Rust クレートは `Cargo.lock` および [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
- **モデル・辞書**: 公開変換モデル（Cymbidium / Bletilla）は Apache-2.0、配布辞書（[esehe/setsuna-ime-dict](https://huggingface.co/esehe/setsuna-ime-dict)）は CC BY-SA 4.0 です。モデルの重みおよび辞書ファイル本体はこのリポジトリには含みません。
