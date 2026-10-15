# Third-party notices

## karukan-tsf

Parts of `crates/tsf` (the DLL exports, COM / TSF registration, the class
factory, the text input processor, edit sessions, the composition sink and
the display attributes) are adapted from karukan-tsf, part of
<https://github.com/anosatsuk124/karukan> (`karukan-tsf/`), which is dual
licensed under MIT or Apache-2.0. It is used here under the MIT License:

```
MIT License

Copyright (c) togatoga
Copyright (c) Satsuki Akiba <anosatsuk124@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Mozc and Mozkey

Parts of Setsuna follow the design, rules and data of Mozc
(<https://github.com/google/mozc>) and of Mozkey, a Mozc fork
(<https://github.com/koyasi777/mozkey>), both under the BSD 3-Clause
License: the context privacy rules (`crates/ime-core/src/privacy.rs`, from
Mozkey's `src/session/zenz_text_privacy_analyzer.cc`), the private input
scopes (`crates/tsf`, as Mozc's `win32/tip`), Ctrl+Delete on a candidate
(Mozc's DeleteSelectedCandidate and suppression dictionary) and the user
dictionary's text format and parts of speech (`crates/ime-core/src/user_dict.rs`).
`crates/ime-core/data/emoji.tsv` is Mozc's emoji and emoticon data
(`src/data/emoji/emoji_data.tsv`, `src/data/emoticon/emoticon.tsv`), its
hiragana readings only.

```
Copyright 2010-2018, Google Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the
    distribution.
  * Neither the name of Google Inc. nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## azooKey

`crates/ime-core/data/romaji/azik.tsv` is the AZIK table (the extended romaji
scheme by Kiyoshi Kimura) of azooKey's AzooKeyKanaKanjiConverter
(<https://github.com/azooKey/AzooKeyKanaKanjiConverter>,
`InputManagement/InputTables/defaultAzik.swift`), under the MIT License:

```
MIT License

Copyright (c) 2023 Miwa / Ensan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Unicode CLDR

Part of the emoji readings in `crates/ime-core/data/emoji.tsv` come, through
Mozc's emoji data, from the Unicode Common Locale Data Repository
(<https://cldr.unicode.org/>):

```
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
```

## ChatGPT logo

The settings GUI's "Continue with ChatGPT" button carries OpenAI's ChatGPT
logo as OpenAI provides it for Sign in with ChatGPT buttons
(<https://developers.openai.com/siwc/website>). ChatGPT and its logo are
trademarks of OpenAI; they are used only to mark that button.

## Mixed input model data

`crates/ime-core/data/mixed` (kana n-gram and Latin word counts, no text) is
derived from:

- FineWeb2-edu-japanese, <https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese>,
  derived from FineWeb-2, <https://huggingface.co/datasets/HuggingFaceFW/fineweb-2>:
  Open Data Commons Attribution License (ODC-By) v1.0; its use is also
  subject to the Common Crawl Terms of Use, <https://commoncrawl.org/terms-of-use>.
- FineWeb, <https://huggingface.co/datasets/HuggingFaceFW/fineweb> (`sample/10BT`):
  ODC-By v1.0; Common Crawl Terms of Use.
- The readings were made with Sudachi and SudachiDict
  (<https://github.com/WorksApplications/SudachiDict>, Apache-2.0).

The counts are distributed under ODC-By v1.0,
<https://opendatacommons.org/licenses/by/1-0/>.
