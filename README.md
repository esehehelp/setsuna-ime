# Setsuna IME

Setsuna (刹那) is a Japanese input method built around a non-autoregressive
neural kana-kanji converter (CTC-NAT). This repository holds the inference engine, the IME
frontends for Windows (TSF) and Linux (fcitx5), an optional LM reranker, a
settings GUI and a browser demo. **Research stage: no trained model is
distributed yet**, so the IME cannot convert out of the box (see
[Status](#状態)).

Setsuna（刹那）は、非自己回帰のニューラルかな漢字変換（CTC-NAT）を中心にした
日本語入力メソッド。推論エンジン、
Windows（TSF）と Linux（fcitx5）の IME、言語モデルによるリランカ、設定 GUI、
ブラウザで動くデモを含む。

## 状態

研究段階のプロジェクトで、**学習済みモデルと辞書はまだ配布していない**。

- エンジンと IME は変換モデル Aster-v1.2 を前提に調整している（記号分割、
  前の文脈 64 トークン、長い読みは 80 文字ごとに区切って変換）。配布の予定が
  決まり次第ここに書く。
- 辞書データは、生成元は公開せず辞書ファイルとして配布し、エンジンと IME は
  それを読んで使う予定。今読むのは記号辞書（`[frontend] symbol_dictionary_file`）
  と英字の名前の辞書（`name_dictionary_file`）で、読みの辞書はこれから使う。

モデルが無くてもビルドとテストは通り、日付・時間候補のようにモデルを使わない
部分は動く。

## 特徴

- **推論専用エンジン** — CTC-NAT の proposal（ONNX）→ CTC beam search →
  候補。モデルは manifest 経由でのみ読み、推奨 decode 設定も manifest に載る。
  stdio の JSONL protocol で動く daemon として IME から使う。
- **文節編集** — CTC アラインメント（候補の各文字の読み位置）から文節を作り、
  変換中に ← / → / Shift+← / Shift+→ で文節ごとに編集できる。
- **リランク（任意）** — 入力が止まったときに causal LM（llama.cpp、GGUF）で
  上位候補の対数尤度を足して並べ替える。
- **学習** — 選んだ候補を覚えて並べ替える。読みの文字 bigram のコサイン
  類似度で似た読みにも適用し、変えた単語単位でも学習する。
  変換中に Ctrl+Delete で、選んでいる候補の学習を取り消す（学習で上がった
  候補でなければ、その読みでは表示しないようにする。設定 GUI で戻せる）。
- **候補の後処理** — 記号の全角半角、句読点の種類、中黒の統一、任意置換、
  読みに無い記号を含む候補の除外。
- **英日混在入力（任意）** — ローマ字の中の英単語を打ったまま残す
  （kana / Latin の n-gram モデルによる分割。モデルは IME に同梱：
  `crates/ime-core/data/mixed`、インストールで有効になり、`mixed_model = ""`
  でオフ）。英単語の後ろの助詞
  （`Ollamade` → Ollamaで）から日本語に戻る。英単語が 2 語以上の英文は
  文頭を大文字にし、`.` `,` も英文のまま残す（`sentence_case`）。小文字で
  打った英語の表記は `name_case` で選ぶ：`proper` は英字の名前の辞書
  （`name_dictionary_file`）の固有名詞をその表記に（`github` → GitHub、
  `iphone` → iPhone）、`acronyms`（既定）はさらに、英単語として読めない
  つづりの略語を大文字に（`pc` → PC、`usb` → USB、`ok` → OK）。`word` や
  `apple` のように普通の語でもある名前はどちらでもそのまま。設定 GUI の
  「英日混在入力」でも変えられる。
- **ユーザー辞書** — 設定 GUI で語を登録する（`user_dictionary.txt`、Google
  日本語入力 / Mozc と同じ形式）。読み全体のほか、長い読みの一部に一致しても
  その部分を登録した語にした候補を出す。品詞「短縮よみ」は読み全体だけ、
  「抑制単語」はその語を出さない。Google 日本語入力 / Mozc、Microsoft IME、
  ATOK、Gboard の書き出したファイルを読み込める。
- **日付・時間候補**、**数字の候補**（1234 → 1,234・千二百三十四・壱阡弐百参拾四・
  12:34、じゅうに → 12、2025ねん → 令和7年、れいわ7ねん → 2025年、`1+2=` → 3、
  `U+3042` → あ。`number_candidates`）、**絵文字・顔文字**（にこにこ → 😊、
  えもじ / かおもじ で一覧。`emoji_candidates`）、**記号辞書**（`やじるし` → `→`
  など）。
- **書き直し（任意、Windows）** — Space で変換すると、OpenAI のモデルが入力を
  自然な日本語に書き直した候補を先頭に出す。
  ChatGPT プラン（Sign in with ChatGPT）、OpenAI の API キー、OpenAI 互換 API
  のどれでも使える（[書き直し](#書き直し)）。

## 構成

依存は一方向: artifact → runtime → platform。

| crate | 層 | 内容 |
|---|---|---|
| `crates/artifact` | artifact | manifest reader |
| `crates/engine` | runtime | 推論エンジン（`setsuna-engine`、library / CLI / daemon） |
| `crates/rerank` | runtime | LM リランカ（`setsuna-rerank`、llama.cpp） |
| `crates/ime-core` | platform | 各 frontend 共通の IME ロジック（ローマ字合成、状態遷移、後処理、学習、設定、daemon client） |
| `crates/tsf` | platform | Windows TSF IME（DLL） |
| `crates/linux` | platform | fcitx5 addon（Rust 部分 + `fcitx5/` の C++ shim） |
| `crates/rewrite` | platform | 書き直し（`setsuna-rewrite`、Sign in with ChatGPT、OpenAI / 互換 API の client） |
| `crates/settings` | platform | 設定 GUI `setsuna-settings`（システムの WebView に出す HTML 画面。Windows / macOS / Linux） |
| `crates/demo` | platform | ブラウザで IME を動かすローカルデモ |

frontend はエンジンをリンクせず、`setsuna-engine daemon` / `setsuna-rerank
daemon` / `setsuna-rewrite daemon` を子プロセスとして起動し stdio で話す。

## ビルド

Rust（stable）が要る。`crates/rerank` は llama.cpp をソースから build するので
CMake と C/C++ コンパイラも要る。

```sh
cargo build --release
cargo test
```

エンジンは ONNX Runtime をリンクせず、実行時に共有ライブラリを読む
（`--ort-dylib` > 環境変数 `SETSUNA_ORT_DYLIB` > エンジンと同じディレクトリ）。
Windows では公式 build を取ってくるスクリプトがある:

```powershell
pwsh crates/engine/fetch-onnxruntime.ps1   # ONNX Runtime 1.28.0 を target/release へ
```

Linux / macOS では `libonnxruntime.so` / `.dylib` をエンジンの隣に置くか
`SETSUNA_ORT_DYLIB` で指す。ORT の版が変わると int8 の数値が変わり候補も
変わるので、既定版（1.28.0）の変更は bench の取り直しとセットで行う。

### Windows（TSF）

```powershell
cargo build --release
pwsh crates/engine/fetch-onnxruntime.ps1
pwsh crates/tsf/install.ps1 -Manifest <path to *.manifest.json>
# リランカも入れる場合: -RerankManifest <path to LM scorer manifest>
# 英日混在入力の名前の辞書も入れる場合: -NameDictionary <names tsv>
```

`%LOCALAPPDATA%\setsuna-ime` に DLL・エンジン・モデルを置き、`regsvr32` で登録する
（UAC が出る）。設定は `%APPDATA%\setsuna-ime\config.toml` で、タスクバーの
「あ / A」アイコンの右クリックメニュー「設定...」から設定 GUI を開ける。アンインストールは
`%LOCALAPPDATA%\setsuna-ime\uninstall.ps1`。

### Linux（fcitx5）

fcitx5 の開発パッケージ（`Fcitx5Core` 5.1 以上）と CMake が要る。設定 GUI の build には
WebKitGTK 4.1 の開発パッケージ（`libwebkit2gtk-4.1-dev` など）も要る。

```sh
cmake -B build -DCMAKE_INSTALL_PREFIX=/usr crates/linux/fcitx5
cmake --build build && sudo cmake --install build
```

エンジンとモデルは `config.toml`（`~/.config/setsuna-ime/`）、環境変数
`SETSUNA_ENGINE_EXE` / `SETSUNA_MANIFEST` / `SETSUNA_MODEL_DIR`、最後に
`SETSUNA_MODULE_DIR`（既定 `<libdir>/setsuna-ime`）の順に探す。開発中は
`-DSETSUNA_MODULE_DIR=<repo>/target/release` とすると `target/release` の
エンジンと `<repo>/local/models/onnx` のモデルを使う。

### デモ / 設定 GUI

```sh
cargo run --release -p setsuna-demo -- --port 8765   # http://127.0.0.1:8765/
cargo run --release -p setsuna-settings
```

デモは fcitx5 frontend と同じ session をブラウザのキー入力で動かし、エンジンの
所要時間・スコア・アラインメントを表示する。エンジンとモデルは IME と同じ
`config.toml` から読む。

## モデルと manifest

エンジンは manifest（`*.manifest.json`）だけを入口にモデルを読む。

- 変換モデル: schema `neo-ime-ctc-nat-export-v1`。proposal の ONNX、語彙、
  special token id、`decode_defaults`（beam 幅、記号の扱いなど）を持つ。
- リランカ: schema `neo-ime-lm-scorer-export-v1`。GGUF の重みと
  `score_defaults` を持つ。

例は `tests/fixtures/` にある（中身は空の placeholder）。

## 日付・時間候補

`crates/ime-core/src/date_time.rs` が端末のローカル日時から候補を生成する。
Windows TSF / Linux fcitx5 共通で、モデル未起動でも Space で利用できる。
通常の変換候補・かな候補は残し、その後に日付・時間候補を追加する。

- `きのう` / `きょう` / `あした` / `あさって` / `おととい`：日付。
  `YYYY/MM/DD`、`YYYY-MM-DD`、`YYYY年M月D日`、曜日付きなど。
- `いま` / `じこく` / `にちじ`：現在時刻、日時、タイムゾーン付きの日時。
- `すいようび` など：今日を含む次の該当曜日。
  `らいしゅうのすいようび` / `こんしゅうのげつようび` も指定できる。
- `らいしゅう` / `こんしゅう` / `せんしゅう` / `さらいしゅう`：月曜〜日曜の
  日付範囲と開始日。`らいげつ` などは月の範囲、`らいねん` などは年。
- `1じかんご` / `いちじかんご` / `30ぷんまえ` / `2しゅうかんご` など：相対指定。
  秒・分・時間は経過時間、日・週・月・年はカレンダー上の日付として計算する。
  月・年の加減算で存在しない日は月末に合わせる。

読み全体、または編集中の文節全体が一致した場合だけ生成する。
Space ごとに時計を読み直し、開いている候補の日時は後から届くモデル結果や
リランク結果で変わらない。既定で有効。設定 GUI、または
`[frontend]` の `date_time_candidates = false` で無効にできる。

## 書き直し

書き直しがオンなら、Space で変換すると、読み・変換結果（と、設定でオンなら
カーソルの前の文章）をモデルに送り、自然な日本語に書き直した文を候補の先頭に
出す（変換候補の窓の上に「書き直し中…」と出て、届くと先頭が入れ替わる）。
Enter で確定、↓ でいつもの変換候補を選べる。変換のたびに送るので、そのぶん
使用量や料金がかかる。

- **有効にする** — 設定 GUI の「書き直し」をオンにし、接続先とモデルを選んで
  保存する（`[rewriter]`）。IME を有効にし直すと使える（タスクバーの表示が
  「直」になる）。Windows のみ。
- **ChatGPT プラン** — 「Continue with ChatGPT」でブラウザからサインインし、
  ChatGPT プランの使用量で書き直す（OpenAI の Sign in with ChatGPT。ローカルで
  動くオープンソースのアプリとして登録する）。使用量は ChatGPT の設定で管理する。
- **API キー / 互換 API** — OpenAI の API キー、または OpenAI 互換 API（ローカルの
  LLM サーバーなど）のベース URL と、必要ならその API キー。互換 API は
  Chat Completions と Responses のどちらでも話せる。
- **送らないもの** — カーソルの前の文章に URL、メールアドレス、パス、API キーや
  トークン、長い数字（カード番号や電話番号）、「パスワード」「暗証番号」のような
  語があれば、文章は送らない。読みや変換結果がそういうものなら書き直さない。
  パスワード欄など入力欄が非公開のとき（Windows の InputScope、fcitx5 の
  Password / Sensitive）は、文章も読まず、書き直さず、学習もしない。変換モデルは
  文章のうち最後に見つかった箇所より後ろだけを読む。
- **資格情報** — サインインのトークンと API キーは設定と同じディレクトリの
  `credentials.bin` に、Windows の DPAPI でこのユーザーだけが読める形で保存する
  （Linux / macOS では権限 0600 の `credentials.json`）。`config.toml` には入らない。

## License

MIT（[LICENSE](LICENSE)）。`crates/tsf` の一部は karukan-tsf（MIT）から流用して
おり、その著作権表示は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) にある。

実行時に読む / build 時に取り込む外部コンポーネントはそれぞれのライセンスに従う:
[ONNX Runtime](https://github.com/microsoft/onnxruntime)（MIT）、
[llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT）、
その他の Rust crate は `Cargo.lock` を参照。英日混在入力のモデル
（`crates/ime-core/data/mixed`）は FineWeb2-edu-japanese と FineWeb から数えた
出現回数で、ODC-By 1.0（出典は同じディレクトリの README.md と
THIRD_PARTY_NOTICES.md）。辞書データとかな漢字変換モデルの重みは
このリポジトリに含まない。配布する辞書ファイルのライセンスと出典表示は、
それぞれの辞書ファイルに同梱する。
