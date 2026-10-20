# Setsuna IME

Setsuna (刹那) is a Japanese input method built around a non-autoregressive
neural kana-kanji converter (CTC-NAT). This repository holds the inference engine, the IME
frontends for Windows (TSF) and Linux (fcitx5), an optional LM reranker, a
settings GUI and a browser demo. **Research stage**: a public conversion model
is available, and the Windows installer downloads it automatically (see
[Status](#状態)).

Setsuna（刹那）は、非自己回帰のニューラルかな漢字変換（CTC-NAT）を中心にした
日本語入力メソッド。推論エンジン、
Windows（TSF）と Linux（fcitx5）の IME、言語モデルによるリランカ、設定 GUI、
ブラウザで動くデモを含む。

## 状態

研究段階のプロジェクト。変換モデル
[Aster-v1.2-small](https://huggingface.co/esehe/Aster-v1.2-small)
（Apache-2.0）を公開しており、Windows・Linux のインストーラーは既定で int8 版を自動取得する。

- エンジンと IME はこの変換モデルを前提に調整している（記号分割、
  前の文脈 64 トークン、長い読みは 80 文字ごとに区切って変換）。
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
  上位候補の対数尤度を足して並べ替える。`[rerank] defer_ms` を設定すると、
  ライブ変換の表示を並べ替えの結果まで待たせて、先頭が二度入れ替わらないように
  する（Mozkey の deferred display と同じ考え方）。
- **学習** — 選んだ候補を覚えて並べ替える。読みの文字 bigram のコサイン
  類似度で似た読みにも適用し、変えた単語単位でも学習する。
  確定した変換を覚えて、読みの始めを打つと予測として候補の窓に出す（Tab で
  選ぶ。直前に確定した語のあとによく打ったものを上げる。`prediction`）。
  変換中に Ctrl+Delete で、選んでいる候補の学習を取り消す（学習で上がった
  候補でなければ、その読みでは表示しないようにする。設定 GUI で戻せる）。
- **候補の後処理** — 記号の全角半角、句読点の種類、中黒の統一、任意置換、
  読みに無い記号を含む候補の除外。
- **候補の窓** — テーマ（自動は Windows のアプリのモードに合わせてライト・
  ダーク）、大きさ、文字の大きさと太さ、角の丸み、不透明度（`[ui.candidate_window]`、
  設定 GUI の「候補の窓」）。
- **アプリ別の設定** — 実行ファイル名ごとに、開始時のオン・オフ、ライブ変換、
  学習、書き直し、英日混在、予測を変える（`[[apps]]`、設定 GUI の「アプリ別の
  設定」）。
- **入力方式** — ローマ字表を選べる（`romaji_table`）：標準、AZIK、かな入力
  （JIS 配列）、Mozc 形式の表のファイル。設定 GUI の「入力方式」でも変えられる。
  ローマ字として読めない文字があると、隣のキー・余計なキー・入れ替えを直した
  読みの候補も出す（kaisga → 会社。`typo_correction`）。
- **英日混在入力（任意）** — ローマ字の中の英単語を打ったまま残す
  （kana / Latin の n-gram モデルによる分割。モデルは IME に同梱：
  `crates/ime-core/data/mixed`、インストールで有効になり、`mixed_model = ""`
  でオフ）。英単語の後ろの助詞
  （`Ollamade` → Ollamaで）から日本語に戻る。小文字で打ったまれな語や名前が
  「よくある英単語＋する」とも読めるときは文を取る（`dlsite` → dlして、助詞が
  続けば `dlsitede` → DLsiteで。名前の表記どおり Shift で打てばそのまま）。
  英単語が 2 語以上の英文は文頭を大文字にする。英単語の直後の `.` `,` `!` `?`
  は、その文に日本語がなければ英文のまま（`this is a pen.`、英文を確定した後の
  `.`）、日本語の文の中なら `、。`（`korehagoogle.` → これはgoogle。）
  （`sentence_case`）。小文字で
  打った英語の表記は `name_case` で選ぶ：`proper` は英字の名前の辞書
  （`name_dictionary_file`）の固有名詞をその表記に（`github` → GitHub、
  `iphone` → iPhone）、`acronyms`（既定）はさらに、英単語として読めない
  つづりの略語を大文字に（`pc` → PC、`usb` → USB、`ok` → OK）。`word` や
  `apple` のように普通の語でもある名前はどちらでもそのまま。設定 GUI の
  「英日混在入力」でも変えられる。
  URL（`https://…`）、メールアドレス、パス（`C:\…`、`~/…`）、@メンションは
  途中で切らず打ったまま残す。`pass_through_commands` をオンにすると、行の先頭や
  空白の後の `/コマンド`・`$スキル`・`@ファイル` を変換せずにアプリへ送る（AI
  エージェントの補完が使える。アプリの文章を読めるとき）。
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

Linux では `crates/engine/fetch-onnxruntime.sh` が同じことをする（x64 / aarch64）。
macOS では `libonnxruntime.dylib` をエンジンの隣に置くか
`SETSUNA_ORT_DYLIB` で指す。ORT の版が変わると int8 の数値が変わり候補も
変わるので、既定版（1.28.0）の変更は bench の取り直しとセットで行う。

### Windows（TSF）

```powershell
cargo build --release
pwsh crates/engine/fetch-onnxruntime.ps1
pwsh crates/tsf/install.ps1                  # 公開モデルの int8 版を自動取得してインストール
# fp32 版: pwsh crates/tsf/install.ps1 -ModelPrecision fp32
# 取得済み・独自のモデル（ネット接続不要）:
# pwsh crates/tsf/install.ps1 -Manifest <path to *.manifest.json>
# リランカも入れる場合: -RerankManifest <path to LM scorer manifest>
# 英日混在入力の名前の辞書も入れる場合: -NameDictionary <names tsv>
```

`%LOCALAPPDATA%\setsuna-ime` に DLL・エンジン・モデルを置き、`regsvr32` で登録する
（UAC が出る）。設定は `%APPDATA%\setsuna-ime\config.toml` で、タスクバーの
「あ / A」アイコンの右クリックメニュー「設定...」から設定 GUI を開ける。アンインストールは
`%LOCALAPPDATA%\setsuna-ime\uninstall.ps1`。

モデルはログイン不要で Hugging Face から取得する。取得元・commit・各ファイルの
サイズと SHA-256 は `crates/tsf/models.json` に固定し、manifest と参照先の ONNX・語彙を
検証してから配置する。キャッシュは `%LOCALAPPDATA%\setsuna-ime\cache\models`。
再実行では検証済みファイルを再利用するので、取得完了後はオフラインでも使える。
通信・検証に失敗するとインストールを中止する（終了コード 30）。`-Manifest` と
`-ModelPrecision` は併用できない。ビルドと ONNX Runtime の取得は上記の手順が別途必要。

モデルだけ取得する場合（標準出力にローカル manifest のパスを返す）:

```powershell
pwsh crates/tsf/fetch-model.ps1 -Precision int8 -CacheDir local/models/downloads
# ダウンローダーの回帰テスト（ネット接続不要）:
pwsh tests/model-download.ps1
```

### Linux（fcitx5）

fcitx5（5.1 以上）が入った x86_64 の環境（glibc 2.39 以上: Arch、Fedora 40、Ubuntu 24.04 以降）なら:

```sh
curl -fsSL https://github.com/esehehelp/setsuna-ime/releases/latest/download/install.sh | sudo bash
# fp32 版: ... | sudo bash -s -- --precision fp32
# アンインストール: sudo /usr/lib/setsuna-ime/uninstall.sh
```

`install.sh`（`crates/linux/install.sh`）は root で動く前提で、中で権限昇格はしない。
自分と組になった release アーカイブ（SHA-256 固定）と、そのアーカイブが固定する
変換モデル（`crates/tsf/models.json` と同じ。`/var/cache/setsuna-ime/models` に取得・検証し、
検証済みならオフラインで再利用）を取ってきて入れる。addon は fcitx5 の addon
ディレクトリに、エンジン・書き直しの daemon・ONNX Runtime 1.28.0・モデル・英日混在の
モデルは `/usr/lib/setsuna-ime`（addon に埋め込んだ場所なので `config.toml` は要らない）に、
設定 GUI は WebKitGTK 4.1 があれば `/usr/bin/setsuna-settings` に入る。最後に sudo
（または pkexec）した利用者の fcitx5 の入力メソッドに Setsuna を足し、fcitx5 を
再起動して読み込ませる（KDE Plasma の Wayland では `fcitx5-wayland-launcher` も
起動し直させる）。`--no-register` ならファイルを置くだけ。

release は `crates/linux/package.sh` が作る（build に要るもの: CMake、fcitx5 と
WebKitGTK 4.1 の開発パッケージ、curl、jq）:

```sh
crates/linux/package.sh        # target/dist/ に tar.gz、install.sh、SHA256SUMS
sudo bash target/dist/install.sh --archive target/dist/setsuna-ime-*.tar.gz   # 手元で試す
gh release create v<version> target/dist/*                                     # 公開
```

```sh
# 開発中に手で build する場合（モデルも入れるなら -DSETSUNA_MODEL_MANIFEST=<manifest>、
# ONNX Runtime は crates/engine/fetch-onnxruntime.sh、モデルは crates/linux/fetch-model.sh で取る）:
cmake -B target/fcitx5 -DCMAKE_INSTALL_PREFIX=/usr crates/linux/fcitx5
cmake --build target/fcitx5 && sudo cmake --install target/fcitx5
# ダウンローダーの回帰テスト（ネット接続不要）:
bash tests/model-download.sh
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
  公開モデル Aster-v1.2-small には int8 版（約 29 MB）と fp32 版（約 107 MB）がある。
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
  URL・メール・数字・パスワードなどの入力欄（Windows の InputScope）では IME を
  通さず直接入力になり、非公開の入力欄（InputScope の private、fcitx5 の
  Password / Sensitive）では文章も読まず、書き直さず、学習もしない。変換モデルは
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
THIRD_PARTY_NOTICES.md）。かな漢字変換モデルは
[Aster-v1.2-small](https://huggingface.co/esehe/Aster-v1.2-small)（Apache-2.0）。
辞書データとかな漢字変換モデルの重みはこのリポジトリに含まない。
配布する辞書ファイルのライセンスと出典表示は、
それぞれの辞書ファイルに同梱する。
