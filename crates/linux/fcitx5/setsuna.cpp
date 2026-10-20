/*
 * Setsuna fcitx5 addon: turns fcitx5 events into calls on the Rust session
 * (../include/setsuna.h) and renders the view it returns with fcitx5's panel.
 *
 * One session serves every input context: only the focused one composes,
 * and the composition is committed when it loses focus.
 */
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <fcitx-config/configuration.h>
#include <fcitx-config/option.h>
#include <fcitx-config/rawconfig.h>
#include <fcitx-utils/event.h>
#include <fcitx-utils/eventdispatcher.h>
#include <fcitx-utils/keysym.h>
#include <fcitx-utils/key.h>
#include <fcitx-utils/misc.h>
#include <fcitx-utils/trackableobject.h>
#include <fcitx-utils/utf8.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addoninstance.h>
#include <fcitx/action.h>
#include <fcitx/addonmanager.h>
#include <fcitx/candidatelist.h>
#include <fcitx/event.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputmethodengine.h>
#include <fcitx/inputpanel.h>
#include <fcitx/instance.h>
#include <fcitx/statusarea.h>
#include <fcitx/text.h>
#include <fcitx/userinterface.h>
#include <fcitx/userinterfacemanager.h>

#include "setsuna.h"

namespace {

using namespace fcitx;

class SetsunaEngine;

/* A candidate of the panel; a click picks it by its index in the list. */
class SetsunaCandidate : public CandidateWord {
public:
    SetsunaCandidate(SetsunaEngine *engine, size_t index, const char *text,
                    const char *comment)
        : CandidateWord(Text(text)), engine_(engine), index_(index) {
        if (comment && *comment) {
            setComment(Text(comment));
        }
    }
    void select(InputContext *ic) const override;

private:
    SetsunaEngine *engine_;
    size_t index_;
};

/* A row that only keeps the panel's height. */
class BlankCandidate : public CandidateWord {
public:
    BlankCandidate() : CandidateWord(Text("\u3000")) {}
    void select(InputContext *) const override {}
};

/* The panel's list, as fcitx5-mozc builds it: only the page on screen
 * (the session owns the selection and the paging), the cursor an index
 * into that page (-1 for the suggestions, which have none), "1. " labels
 * where the digit keys pick. The page arrows, the scroll wheel and fcitx5's
 * cursor keys go to the session as the keys they stand for. */
class SetsunaCandidateList final : public CandidateList,
                                  public PageableCandidateList,
                                  public CursorMovableCandidateList {
public:
    SetsunaCandidateList(SetsunaEngine *engine, bool hasPrev, bool hasNext, int cursor)
        : engine_(engine), hasPrev_(hasPrev), hasNext_(hasNext), cursor_(cursor) {
        setPageable(this);
        setCursorMovable(this);
    }
    void add(std::unique_ptr<CandidateWord> word, std::string label) {
        words_.push_back(std::move(word));
        labels_.emplace_back(std::move(label));
    }
    /* Blank rows up to `rows`: the panel then has the same height on every
     * page and while suggesting, and the app does not flip it between
     * below and above the caret as its height changes. */
    void padTo(int rows) {
        while (size() < rows) {
            add(std::make_unique<BlankCandidate>(), "");
        }
    }

    const Text &label(int idx) const override { return labels_.at(idx); }
    const CandidateWord &candidate(int idx) const override { return *words_.at(idx); }
    int size() const override { return static_cast<int>(words_.size()); }
    int cursorIndex() const override { return cursor_; }
    CandidateLayoutHint layoutHint() const override { return CandidateLayoutHint::Vertical; }

    bool hasPrev() const override { return hasPrev_; }
    bool hasNext() const override { return hasNext_; }
    void prev() override;
    void next() override;
    bool usedNextBefore() const override { return true; }
    void prevCandidate() override;
    void nextCandidate() override;

private:
    SetsunaEngine *engine_;
    bool hasPrev_, hasNext_;
    int cursor_;
    std::vector<std::unique_ptr<CandidateWord>> words_;
    std::vector<Text> labels_;
};

/* The status area's mode button: あ on, A off; a click toggles. */
class SetsunaModeAction : public Action {
public:
    explicit SetsunaModeAction(SetsunaEngine *engine) : engine_(engine) {}
    std::string shortText(InputContext *) const override;
    std::string longText(InputContext *) const override;
    std::string icon(InputContext *) const override;
    void activate(InputContext *ic) override;

private:
    SetsunaEngine *engine_;
};

/* What fcitx5's settings show for Setsuna: a button opening its own
 * settings GUI (setSubConfig), which edits config.toml. */
FCITX_CONFIGURATION(SetsunaConfig,
                    ExternalOption settings{this, "Settings", "Setsuna の設定",
                                            "fcitx://config/addon/setsuna/settings"};);

void openSettings() { startProcess({SETSUNA_SETTINGS_EXE}); }

class SetsunaEngine final : public InputMethodEngineV2 {
public:
    explicit SetsunaEngine(Instance *instance)
        : instance_(instance), modeAction_(this) {
        dispatcher_.attach(&instance->eventLoop());
        session_ = setsuna_session_new(SETSUNA_MODULE_DIR, &SetsunaEngine::notify, this);
        instance->userInterfaceManager().registerAction("setsuna-mode", &modeAction_);
        settingsAction_.setShortText("設定");
        settingsAction_.setIcon("configure");
        settingsAction_.connect<SimpleAction::Activated>([](InputContext *) { openSettings(); });
        instance->userInterfaceManager().registerAction("setsuna-settings", &settingsAction_);
    }

    ~SetsunaEngine() override {
        instance_->userInterfaceManager().unregisterAction(&settingsAction_);
        instance_->userInterfaceManager().unregisterAction(&modeAction_);
        // Joins the engine client's threads: no notify follows, so nothing
        // is scheduled on the dispatcher after this.
        setsuna_session_free(session_);
    }

    void activate(const InputMethodEntry &, InputContextEvent &event) override {
        auto *ic = event.inputContext();
        // the settings of this application ([[apps]])
        bool enabled = setsuna_session_enabled(session_);
        setsuna_session_configure_app(session_, ic->program().c_str());
        if (enabled != setsuna_session_enabled(session_)) {
            modeChanged(ic);
        }
        ic_ = ic->watch();
        ic->statusArea().addAction(StatusGroup::InputMethod, &modeAction_);
        ic->statusArea().addAction(StatusGroup::InputMethod, &settingsAction_);
    }

    const Configuration *getConfig() const override { return &config_; }
    void setSubConfig(const std::string &path, const RawConfig &) override {
        if (path == "settings") {
            openSettings();
        }
    }

    /* Focus out or input method switch: commit what is composed. On focus
     * out fcitx5 already commits the client preedit (itself, or the client
     * does with ClientUnfocusCommit), so only a server-side preedit is
     * committed here. */
    void deactivate(const InputMethodEntry &, InputContextEvent &event) override {
        auto *ic = event.inputContext();
        setsuna_session_flush(session_);
        bool committed = event.type() == EventType::InputContextFocusOut &&
                         ic->capabilityFlags().test(CapabilityFlag::Preedit);
        render(ic, !committed);
    }

    /* The app reset the context (moved the caret, cleared the field). */
    void reset(const InputMethodEntry &, InputContextEvent &event) override {
        setsuna_session_reset(session_);
        render(event.inputContext());
    }

    void keyEvent(const InputMethodEntry &, KeyEvent &event) override {
        if (event.isRelease()) {
            return;
        }
        auto *ic = event.inputContext();
        ic_ = ic->watch();
        // Unnormalized: normalize() drops Shift from printable keys, and
        // Shift+letter is the temporary alphabet mode.
        const Key &key = event.rawKey();
        bool enabled = setsuna_session_enabled(session_);
        std::string before;
        bool known = beforeCaret(ic, before);
        if (setsuna_session_key(session_, key.sym(), key.code(), mods(key.states()), field(ic),
                                known ? before.c_str() : nullptr)) {
            event.filterAndAccept();
            render(ic);
        }
        if (enabled != setsuna_session_enabled(session_)) {
            modeChanged(ic);
        }
    }

    bool enabled() const { return setsuna_session_enabled(session_); }

    /* あ on, 直 on and rewriting, A off: the TSF language bar's glyph. */
    std::string modeLabel() const {
        if (!enabled()) {
            return "A";
        }
        return setsuna_session_rewrite_mode(session_) ? "直" : "あ";
    }

    /* The tray icon: あ on, A off (icons/). */
    std::string subModeIconImpl(const InputMethodEntry &, InputContext &) override {
        return modeIcon();
    }
    std::string modeIcon() const {
        return enabled() ? "setsuna-hiragana" : "setsuna-direct";
    }

    /* The mode button was clicked. */
    void toggle(InputContext *ic) {
        setsuna_session_toggle(session_);
        render(ic);
        modeChanged(ic);
    }

    /* A key the panel stands in for (a page arrow, the scroll wheel). Run
     * from the event loop: the panel's list is replaced while rendering, so
     * not from inside its own callback. */
    void deferKey(KeySym sym) {
        dispatcher_.schedule([this, sym] {
            auto *ic = ic_.get();
            if (ic && setsuna_session_key(session_, sym, 0, 0, SETSUNA_FIELD_NORMAL, nullptr)) {
                render(ic);
            }
        });
    }

    /* あ while on (直 rewriting), A while ` / Zenkaku_Hankaku has turned it off. */
    std::string subModeLabelImpl(const InputMethodEntry &, InputContext &) override {
        return modeLabel();
    }
    std::string subMode(const InputMethodEntry &, InputContext &) override {
        return setsuna_session_enabled(session_) ? "" : "Direct";
    }

    /* A click on a candidate: pick and commit it, as its digit would. */
    void select(InputContext *ic, size_t index) {
        if (setsuna_session_select(session_, index)) {
            setsuna_session_key(session_, FcitxKey_Return, 0, 0, SETSUNA_FIELD_NORMAL, nullptr);
            render(ic);
        }
    }

private:
    void modeChanged(InputContext *ic) {
        modeAction_.update(ic);
        ic->updateUserInterface(UserInterfaceComponent::StatusArea);
        // only over an empty panel, as fcitx5-mozc
        const auto &panel = ic->inputPanel();
        if (panel.clientPreedit().empty() && panel.preedit().empty() &&
            !panel.candidateList()) {
            instance_->showCustomInputMethodInformation(
                ic, !enabled()                          ? "A  直接入力"
                    : setsuna_session_rewrite_mode(session_) ? "直  書き直し"
                                                             : "あ  ひらがな");
        }
    }

    /* Called from the engine client's reader thread. */
    static void notify(void *user) {
        auto *self = static_cast<SetsunaEngine *>(user);
        self->dispatcher_.schedule([self] { self->applyAsync(); });
    }

    void applyAsync() {
        if (!setsuna_session_apply_async(session_)) {
            return;
        }
        if (auto *ic = ic_.get()) {
            render(ic);
        }
        armDeferred();
    }

    /* A deferred preedit ([rerank] defer_ms): shown anyway at its deadline. */
    void armDeferred() {
        uint32_t ms = setsuna_session_deferred_ms(session_);
        if (ms == 0) {
            deferredTimer_.reset();
            return;
        }
        deferredTimer_ = instance_->eventLoop().addTimeEvent(
            CLOCK_MONOTONIC, now(CLOCK_MONOTONIC) + uint64_t(ms) * 1000, 0,
            [this](EventSourceTime *, uint64_t) {
                if (setsuna_session_tick(session_)) {
                    if (auto *ic = ic_.get()) {
                        render(ic);
                    }
                }
                return true;
            });
    }

    /* What the field is, from the client's hints, as the TSF DLL reads input
     * scopes (Mozc's list): a URL, an e-mail address, a number, a phone
     * number, a date or time, alphanumerics or a password take direct input;
     * a sensitive one (a private window) is private. */
    static uint32_t field(InputContext *ic) {
        const auto caps = ic->capabilityFlags();
        for (auto flag : {CapabilityFlag::Url, CapabilityFlag::Email, CapabilityFlag::Digit,
                          CapabilityFlag::Number, CapabilityFlag::Dialable,
                          CapabilityFlag::Date, CapabilityFlag::Time, CapabilityFlag::Alpha,
                          CapabilityFlag::Password}) {
            if (caps.test(flag)) {
                return SETSUNA_FIELD_DIRECT;
            }
        }
        return caps.test(CapabilityFlag::Sensitive) ? SETSUNA_FIELD_PRIVATE
                                                     : SETSUNA_FIELD_NORMAL;
    }

    /* The app's text before the caret, at most kContextChars of it, when the
     * client tells it (surrounding text); false when it does not. */
    static bool beforeCaret(InputContext *ic, std::string &out) {
        if (!ic->capabilityFlags().test(CapabilityFlag::SurroundingText)) {
            return false;
        }
        const auto &surrounding = ic->surroundingText();
        if (!surrounding.isValid()) {
            return false;
        }
        const std::string &text = surrounding.text();
        size_t cursor = std::min<size_t>(surrounding.cursor(), utf8::length(text));
        size_t start = cursor > kContextChars ? cursor - kContextChars : 0;
        auto from = utf8::ncharByteLength(text.begin(), start);
        auto len = utf8::ncharByteLength(text.begin() + from, cursor - start);
        out = text.substr(from, len);
        return true;
    }
    /* Chars of context read, as the TSF DLL's 160 UTF-16 units: the bridge
     * keeps the last 80. */
    static constexpr size_t kContextChars = 160;

    static uint32_t mods(KeyStates states) {
        uint32_t m = 0;
        if (states.test(KeyState::Shift)) m |= SETSUNA_MOD_SHIFT;
        if (states.test(KeyState::Ctrl)) m |= SETSUNA_MOD_CTRL;
        if (states.test(KeyState::Alt)) m |= SETSUNA_MOD_ALT;
        if (states.test(KeyState::Super) || states.test(KeyState::Super2)) {
            m |= SETSUNA_MOD_SUPER;
        }
        return m;
    }

    void render(InputContext *ic, bool commit = true) {
        const SetsunaView *v = setsuna_session_view(session_);
        for (size_t i = 0; commit && i < v->n_commits; i++) {
            ic->commitString(v->commits[i]);
        }

        auto &panel = ic->inputPanel();
        panel.reset();
        if (v->n_preedit > 0) {
            // the segment being converted highlighted, the rest underlined
            Text preedit;
            for (size_t i = 0; i < v->n_preedit; i++) {
                bool focused = v->focus >= 0 && static_cast<size_t>(v->focus) == i;
                preedit.append(v->preedit[i], focused ? TextFormatFlag::HighLight
                                                      : TextFormatFlag::Underline);
            }
            preedit.setCursor(static_cast<int>(v->cursor));
            if (ic->capabilityFlags().test(CapabilityFlag::Preedit)) {
                panel.setClientPreedit(preedit);
            } else {
                panel.setPreedit(preedit);
            }
        }
        // One aux line above the list, as fcitx5-mozc: where the selection
        // is while converting, the hint while suggesting.
        if (v->n_candidates > 0) {
            auto page = setsuna_session_page_size(session_);
            size_t first = v->selected / page * page;
            size_t last = std::min(first + page, v->n_candidates);
            auto list = std::make_unique<SetsunaCandidateList>(
                this, first > 0, last < v->n_candidates,
                static_cast<int>(v->selected - first));
            for (size_t i = first; i < last; i++) {
                list->add(std::make_unique<SetsunaCandidate>(this, i, v->candidates[i],
                                                            v->comments[i]),
                          std::to_string((i - first + 1) % 10) + ". ");
            }
            list->padTo(static_cast<int>(page));
            panel.setCandidateList(std::move(list));
            std::string aux =
                "[" + std::to_string(v->selected + 1) + "/" + std::to_string(v->n_candidates) + "]";
            // after it, as on Windows: the rewrite's progress or failure
            if (v->status && *v->status) {
                aux += "  ";
                aux += v->status;
            }
            panel.setAuxUp(Text(aux));
        } else if (v->n_suggestions > 0) {
            auto list = std::make_unique<SetsunaCandidateList>(this, false, false, -1);
            for (size_t i = 0; i < v->n_suggestions; i++) {
                list->add(std::make_unique<SetsunaCandidate>(this, i, v->suggestions[i],
                                                            v->suggestion_comments[i]),
                          "");
            }
            list->padTo(static_cast<int>(setsuna_session_page_size(session_)));
            panel.setCandidateList(std::move(list));
            panel.setAuxUp(Text("[Tabキーで選択]"));
        }
        ic->updatePreedit();
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
    }

    Instance *instance_;
    SetsunaModeAction modeAction_;
    SimpleAction settingsAction_;
    SetsunaConfig config_;
    EventDispatcher dispatcher_;
    std::unique_ptr<EventSourceTime> deferredTimer_;
    SetsunaSession *session_ = nullptr;
    TrackableObjectReference<InputContext> ic_;
};

void SetsunaCandidate::select(InputContext *ic) const { engine_->select(ic, index_); }

void SetsunaCandidateList::prev() { engine_->deferKey(FcitxKey_Page_Up); }
void SetsunaCandidateList::next() { engine_->deferKey(FcitxKey_Page_Down); }
void SetsunaCandidateList::prevCandidate() { engine_->deferKey(FcitxKey_Up); }
void SetsunaCandidateList::nextCandidate() { engine_->deferKey(FcitxKey_Down); }

std::string SetsunaModeAction::shortText(InputContext *) const {
    return engine_->modeLabel();
}
std::string SetsunaModeAction::longText(InputContext *) const {
    return engine_->enabled() ? "ひらがな (` で直接入力)" : "直接入力 (` でひらがな)";
}
std::string SetsunaModeAction::icon(InputContext *) const { return engine_->modeIcon(); }
void SetsunaModeAction::activate(InputContext *ic) { engine_->toggle(ic); }

class SetsunaFactory : public AddonFactory {
public:
    AddonInstance *create(AddonManager *manager) override {
        return new SetsunaEngine(manager->instance());
    }
};

} // namespace

FCITX_ADDON_FACTORY_V2(setsuna, SetsunaFactory);
