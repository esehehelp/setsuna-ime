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
