//! CTC decoding: greedy collapse and prefix beam search.
//!
//! Prefix beam search keeps, per prefix, the log-probability mass of the CTC
//! paths ending in blank (`pb`) and in a non-blank (`pnb`), merges paths with
//! logsumexp, collapses repeats only across a blank, and bounds each step by
//! the top-K tokens of that frame. A hypothesis' score is
//! `logsumexp(pb, pnb)`, the total CTC probability of its token sequence.
//!
//! Each hypothesis also carries the frame that emitted each of its tokens,
//! taken from the most probable path merged into it. The input is one frame
//! per input token, so the frames say which reading characters a token was
//! produced from (the alignment a host segments a conversion with). They
//! never change the scores or the order.
//!
//! Prefixes live in a trie (a prefix is a node id; extending it is a child
//! lookup) and a hypothesis' token frames in a persistent list (extending
//! it is one push), so a step copies no token sequence. Over the 1603 rows of
//! the research general dev set this returned the same top-20 lists, frames
//! included, as the `HashMap<Vec<u32>, _>` implementation it replaced
//! (kept as `tests::reference`), 2.5x faster.
//!
//! What a search works in (the trie, its child table, the frame lists, the
//! beam) is a [`BeamScratch`], kept between searches: a thread's own for the
//! functions that take none, so a search after the first allocates only the
//! hypotheses it returns. A frame whose tokens come best first (as the
//! proposal gives them) is stepped in that order, each prefix stopping at the
//! first token the cut is known to drop (`BeamScratch::step_ordered`); the
//! paths merged, and their order, are those of the plain step.

use std::cell::RefCell;
use std::cmp::Ordering;

use crate::pipeline::crf::CrfTables;
use crate::pipeline::proposal::{FrameLogProbs, ProposalOutput};

const NEG_INF: f32 = f32::NEG_INFINITY;

#[inline]
fn logsumexp(a: f32, b: f32) -> f32 {
    if a == NEG_INF {
        return b;
    }
    if b == NEG_INF {
        return a;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let d = lo - hi;
    if d < negligible_below(hi) {
        return hi;
    }
    hi + d.exp().ln_1p()
}

/// A gap `d` below which `hi + ln_1p(exp(d))` rounds back to `hi`: with `hi`
/// normal, 2^e <= |hi| < 2^(e+1), `exp(d) < 2^(e-27)` keeps the added term
/// (`ln_1p(x) <= x`, exp within an ulp) under 2^(e-26), less than half the
/// spacing of the floats next to `hi` on either side (2^(e-24) toward zero
/// when |hi| is a power of two, 2^(e-23) else). -inf (no early return) for a
/// zero, subnormal, infinite or NaN `hi`. The beam took two transcendental
/// calls a merge (about half of its time).
#[inline(always)]
fn negligible_below(hi: f32) -> f32 {
    let biased = (hi.to_bits() >> 23) & 0xff;
    if biased == 0 || biased == 0xff {
        return NEG_INF;
    }
    (biased as i32 - 127 - 27) as f32 * std::f32::consts::LN_2
}

#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    /// logsumexp(pb, pnb): the CTC probability mass of `tokens`.
    pub score: f32,
    /// The frame that emitted each token (first frame of its run), along
    /// the most probable path; one per token.
    pub frames: Vec<u32>,
}

/// A prefix. It also holds where its state is in the step being built and in
/// the beam, each stamped with the step that wrote it, so a step clears
/// neither.
#[derive(Debug, Clone, Copy)]
struct Node {
    parent: u32,
    token: u32,
    /// The prefix's length in tokens.
    depth: u32,
    next_mark: u32,
    next_slot: u32,
    beam_mark: u32,
    beam_slot: u32,
}

/// The empty prefix (node 0).
const ROOT: Node = Node {
    parent: 0,
    token: u32::MAX,
    depth: 0,
    next_mark: 0,
    next_slot: 0,
    beam_mark: 0,
    beam_slot: 0,
};

/// A slot of the trie's child table: taken when `epoch` is the search's.
#[derive(Debug, Clone, Copy, Default)]
struct ChildSlot {
    key: u64,
    node: u32,
    epoch: u32,
}

/// The child table's first size (a power of two, as every size of it).
const CHILD_TABLE_MIN: usize = 1024;

/// Prefix trie: node 0 is the empty prefix. (parent, token) -> node is an
/// open-addressed table whose slots carry the epoch of the search that took
/// them: a new search bumps the epoch, which empties the table without
/// touching it.
#[derive(Debug, Default)]
struct Trie {
    nodes: Vec<Node>,
    children: Vec<ChildSlot>,
    epoch: u32,
    used: usize,
}

impl Trie {
    fn reset(&mut self) {
        self.nodes.clear();
        self.nodes.push(ROOT);
        self.used = 0;
        if self.children.is_empty() {
            self.children.resize(CHILD_TABLE_MIN, ChildSlot::default());
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.children.fill(ChildSlot::default());
            self.epoch = 1;
        }
    }

    #[inline(always)]
    fn home(key: u64, mask: usize) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask
    }

    #[inline]
    fn child(&mut self, node: u32, token: u32) -> u32 {
        let key = ((node as u64) << 32) | token as u64;
        let mask = self.children.len() - 1;
        let mut at = Self::home(key, mask);
        loop {
            let slot = self.children[at];
            if slot.epoch != self.epoch {
                break;
            }
            if slot.key == key {
                return slot.node;
            }
            at = (at + 1) & mask;
        }
        let id = self.nodes.len() as u32;
        let depth = self.nodes[node as usize].depth + 1;
        self.nodes.push(Node {
            parent: node,
            token,
            depth,
            ..ROOT
        });
        self.children[at] = ChildSlot {
            key,
            node: id,
            epoch: self.epoch,
        };
        self.used += 1;
        // at most half full: a probe always ends
        if self.used * 2 > self.children.len() {
            self.grow();
        }
        id
    }

    #[cold]
    fn grow(&mut self) {
        let len = self.children.len() * 2;
        let old = std::mem::replace(&mut self.children, vec![ChildSlot::default(); len]);
        let mask = len - 1;
        for slot in old {
            if slot.epoch != self.epoch {
                continue;
            }
            let mut at = Self::home(slot.key, mask);
            while self.children[at].epoch == self.epoch {
                at = (at + 1) & mask;
            }
            self.children[at] = slot;
        }
    }

    #[inline]
    fn last(&self, node: u32) -> Option<u32> {
        (node != 0).then(|| self.nodes[node as usize].token)
    }

    fn tokens(&self, mut node: u32) -> Vec<u32> {
        let mut out = vec![0u32; self.nodes[node as usize].depth as usize];
        for slot in out.iter_mut().rev() {
            let n = &self.nodes[node as usize];
            *slot = n.token;
            node = n.parent;
        }
        out
    }
}

/// The order of the token sequences of two prefixes (as `Vec<u32>` orders
/// them), read off the trie: where the two first part, the tokens they part
/// with; a prefix of the other first.
fn cmp_tokens(nodes: &[Node], a: u32, b: u32) -> Ordering {
    let (da, db) = (nodes[a as usize].depth, nodes[b as usize].depth);
    let (mut x, mut y) = (a, b);
    for _ in db..da {
        x = nodes[x as usize].parent;
    }
    for _ in da..db {
        y = nodes[y as usize].parent;
    }
    if x == y {
        return da.cmp(&db);
    }
    loop {
        let (nx, ny) = (&nodes[x as usize], &nodes[y as usize]);
        if nx.parent == ny.parent {
            return nx.token.cmp(&ny.token);
        }
        (x, y) = (nx.parent, ny.parent);
    }
}

/// A prefix in the beam: CTC masses ending in blank / non-blank, and its
/// token frames (`frames`: a node of the persistent frame list) from the most
/// probable single path merged into it (`frames_logp`).
#[derive(Debug, Clone, Copy)]
struct State {
    node: u32,
    pb: f32,
    pnb: f32,
    frames: u32,
    frames_logp: f32,
    /// `logsumexp(pb, pnb)`, the prefix's score: set once for every state of
    /// a step when the step is done (the cut compares it, and the next step
    /// reads it for each path out of the prefix instead of taking the
    /// log-sum-exp again).
    rank: f32,
}

/// The beam's order: best score first, ties by token sequence.
#[inline]
fn by_rank(nodes: &[Node], a: &State, b: &State) -> Ordering {
    // two scores apart, neither a NaN, are in the order of their values: the rest below
    if a.rank > b.rank {
        return Ordering::Less;
    }
    if a.rank < b.rank {
        return Ordering::Greater;
    }
    order_key(b.rank)
        .total_cmp(&order_key(a.rank))
        .then_with(|| cmp_tokens(nodes, a.node, b.node))
}

/// Add a path ending in blank to `node`'s state of the step `mark` (`next`,
/// merged by prefix). Its frames replace the state's when this single
/// contribution is the most probable one seen for the prefix.
#[inline(always)]
fn merge_pb(next: &mut Vec<State>, nodes: &mut [Node], mark: u32, node: u32, pb: f32, frames: u32) {
    let n = &mut nodes[node as usize];
    if n.next_mark == mark {
        let s = &mut next[n.next_slot as usize];
        s.pb = logsumexp(s.pb, pb);
        if pb > s.frames_logp {
            s.frames = frames;
            s.frames_logp = pb;
        }
    } else {
        n.next_mark = mark;
        n.next_slot = next.len() as u32;
        next.push(State {
            node,
            pb,
            pnb: NEG_INF,
            frames,
            frames_logp: pb,
            rank: 0.0,
        });
    }
}

/// [`merge_pb`] for a path ending in a non-blank; its frames are built only
/// when they are taken.
#[inline(always)]
fn merge_pnb(
    next: &mut Vec<State>,
    nodes: &mut [Node],
    mark: u32,
    node: u32,
    pnb: f32,
    frames: impl FnOnce() -> u32,
) {
    let n = &mut nodes[node as usize];
    if n.next_mark == mark {
        let s = &mut next[n.next_slot as usize];
        s.pnb = logsumexp(s.pnb, pnb);
        if pnb > s.frames_logp {
            s.frames = frames();
            s.frames_logp = pnb;
        }
    } else {
        n.next_mark = mark;
        n.next_slot = next.len() as u32;
        next.push(State {
            node,
            pb: NEG_INF,
            pnb,
            frames: frames(),
            frames_logp: pnb,
            rank: 0.0,
        });
    }
}

/// Put `v` at `k` of the min-heap `heap` (whose value there is dropped) and
/// let it sink to its place.
#[inline(always)]
fn sink(heap: &mut [f32], mut k: usize, v: f32) {
    let len = heap.len();
    loop {
        let mut down = 2 * k + 1;
        if down >= len {
            break;
        }
        if down + 1 < len && heap[down + 1] < heap[down] {
            down += 1;
        }
        if heap[down] >= v {
            break;
        }
        heap[k] = heap[down];
        k = down;
    }
    heap[k] = v;
}

/// Make a min-heap of `values` (`values[0]` the smallest). No NaN among them.
#[inline]
fn heapify(values: &mut [f32]) {
    for k in (0..values.len() / 2).rev() {
        sink(values, k, values[k]);
    }
}

/// Keep the `width` largest of the values offered, in a min-heap (`top[0]`
/// the smallest kept). No NaN is offered.
#[inline(always)]
fn offer(top: &mut Vec<f32>, width: usize, v: f32) {
    if top.len() < width {
        let mut k = top.len();
        top.push(v);
        while k > 0 {
            let up = (k - 1) / 2;
            if top[up] <= v {
                break;
            }
            top[k] = top[up];
            k = up;
        }
        top[k] = v;
    } else if v > top[0] {
        sink(top, 0, v);
    }
}

/// Neither a NaN nor +inf.
#[inline(always)]
fn under_inf(x: f32) -> bool {
    x < f32::INFINITY
}

/// One frame as a step reads it.
struct Frame<'a> {
    /// The step's stamp (the frame's index + 1: 0 is no step).
    mark: u32,
    index: u32,
    /// The frame's tokens the step expands, (id, log-prob).
    candidates: &'a [(u32, f32)],
    blank: f32,
    blank_id: u32,
}

/// No place among a frame's candidates.
const NO_PLACE: u32 = u32::MAX;

/// The most candidates a frame may have for the ordered step (a bit each in a prefix's mask).
const ORDERED_MAX_CANDIDATES: usize = 64;

/// The token ids the ordered step keeps a table over (a frame with an id past it takes the plain step).
const CANDIDATE_TABLE_MAX: usize = 1 << 20;

/// What a prefix beam search works in: the trie and its child table, the
/// persistent frame lists, the beam and the step being built, and each step's
/// working lists. A search leaves it holding nothing the next one reads, only
/// the memory, so one kept between searches (or lent by the thread, for the
/// functions that take none) saves a search its allocations: after the first,
/// only the hypotheses returned are allocated.
#[derive(Debug, Default)]
pub struct BeamScratch {
    trie: Trie,
    /// Persistent frame lists: (frame, previous); 0 is the empty list.
    frame_list: Vec<(u32, u32)>,
    beam: Vec<State>,
    next: Vec<State>,
    // the plain step: each beam state's tokens whose child prefix is also in the beam (`child_tokens` from
    // `child_start[i]` to `child_start[i + 1]`), and the bounds
    child_start: Vec<u32>,
    child_tokens: Vec<u32>,
    bounds: Vec<f32>,
    // the ordered step: each token's place among the frame's candidates ((epoch, place), read when the epoch is
    // the frame's), per beam state the places of its children in the beam (a bit each) and of its last token, the
    // largest bounds, and the states with paths yet to offer
    candidate_at: Vec<(u32, u32)>,
    candidate_epoch: u32,
    child_mask: Vec<u64>,
    repeat_at: Vec<u32>,
    top: Vec<f32>,
    live: Vec<u32>,
}

impl BeamScratch {
    /// The state before the first frame: the empty prefix alone.
    fn start(&mut self) {
        self.trie.reset();
        self.frame_list.clear();
        self.frame_list.push((0, 0));
        self.beam.clear();
        self.beam.push(State {
            node: 0,
            pb: 0.0,
            pnb: NEG_INF,
            frames: 0,
            frames_logp: 0.0,
            rank: 0.0,
        });
        self.next.clear();
    }

    /// Stamp each beam state's node with its place in the beam.
    #[inline]
    fn mark_beam(nodes: &mut [Node], beam: &[State], mark: u32) {
        for (i, s) in beam.iter().enumerate() {
            let node = &mut nodes[s.node as usize];
            node.beam_mark = mark;
            node.beam_slot = i as u32;
        }
    }

    /// A path out of `s` that emits the new token `c`: into the child prefix, its frames the state's and this
    /// frame.
    #[inline(always)]
    fn extend(
        trie: &mut Trie,
        next: &mut Vec<State>,
        frame_list: &mut Vec<(u32, u32)>,
        frame: &Frame,
        s: &State,
        c: u32,
        pnb: f32,
    ) {
        let child = trie.child(s.node, c);
        merge_pnb(next, &mut trie.nodes, frame.mark, child, pnb, || {
            frame_list.push((frame.index, s.frames));
            frame_list.len() as u32 - 1
        });
    }

    /// One step of the search, for any frame.
    ///
    /// A bound for the cut that never drops a prefix the cut keeps (2026-11-19: 20 x 16 merges a frame were the
    /// 23 MB CTC model's 0.44 ms of beam, p50): each beam prefix's score is at least its blank path's (and its
    /// repeat path's), and a prefix new to the beam is reached by one path alone (its parent is a beam prefix, the
    /// token its own), its score that path's. These are distinct prefixes; when more than `beam_width`, a new
    /// prefix whose path scores below the `beam_width`-th largest of them ranks below at least `beam_width`
    /// others whatever else merges, so the cut drops it: it is not built. The paths kept merge in the same order.
    fn step_plain(&mut self, frame: &Frame, beam_width: usize, crf: Option<&CrfTables>) {
        let Self {
            trie,
            frame_list,
            beam,
            next,
            child_start,
            child_tokens,
            bounds,
            ..
        } = self;
        let candidates = frame.candidates;
        let n = beam.len();
        let mark = frame.mark;
        let thr = if n * (candidates.len() + 1) > beam_width && !cut_bound_off() {
            Self::mark_beam(&mut trie.nodes, beam, mark);
            // a beam state's place when its parent prefix is in the beam too, and its token
            let child_of = |nodes: &[Node], s: &State| -> Option<(usize, u32)> {
                if s.node == 0 {
                    return None;
                }
                let node = &nodes[s.node as usize];
                let parent = &nodes[node.parent as usize];
                (parent.beam_mark == mark).then_some((parent.beam_slot as usize, node.token))
            };
            // the lists, one after another: counted (two places on), summed to each list's start (one place on),
            // which filling moves to its end, the start of the next
            child_start.clear();
            child_start.resize(n + 2, 0);
            for s in beam.iter() {
                if let Some((p, _)) = child_of(&trie.nodes, s) {
                    child_start[p + 2] += 1;
                }
            }
            for i in 2..n + 2 {
                child_start[i] += child_start[i - 1];
            }
            child_tokens.clear();
            child_tokens.resize(child_start[n + 1] as usize, 0);
            for s in beam.iter() {
                if let Some((p, token)) = child_of(&trie.nodes, s) {
                    child_tokens[child_start[p + 1] as usize] = token;
                    child_start[p + 1] += 1;
                }
            }
            bounds.clear();
            for (i, s) in beam.iter().enumerate() {
                let (pb, pnb, whole) = (s.pb, s.pnb, s.rank);
                let children = &child_tokens[child_start[i] as usize..child_start[i + 1] as usize];
                let last = trie.last(s.node);
                let mut own = whole + frame.blank;
                for &(c, c_logp) in candidates {
                    if c == frame.blank_id {
                        continue;
                    }
                    let psi = crf.map_or(0.0, |t| t.log_psi(last, c));
                    if last == Some(c) {
                        own = own.max(pnb + c_logp);
                    }
                    if !children.contains(&c) {
                        bounds.push(if last == Some(c) {
                            pb + c_logp + psi
                        } else {
                            whole + c_logp + psi
                        });
                    }
                }
                bounds.push(own);
            }
            if bounds.len() > beam_width {
                bounds.select_nth_unstable_by(beam_width - 1, |a, b| b.total_cmp(a));
                bounds[beam_width - 1]
            } else {
                NEG_INF
            }
        } else {
            NEG_INF
        };
        for i in 0..n {
            let s = beam[i];
            let (pb, pnb, whole) = (s.pb, s.pnb, s.rank);
            // Emit blank: the prefix is unchanged.
            merge_pb(
                next,
                &mut trie.nodes,
                mark,
                s.node,
                whole + frame.blank,
                s.frames,
            );
            let last = trie.last(s.node);
            for &(c, c_logp) in candidates {
                if c == frame.blank_id {
                    continue;
                }
                let psi = crf.map_or(0.0, |t| t.log_psi(last, c));
                // (a bound means the lists were made)
                if thr > NEG_INF
                    && !child_tokens[child_start[i] as usize..child_start[i + 1] as usize]
                        .contains(&c)
                {
                    let path = if last == Some(c) {
                        pb + c_logp + psi
                    } else {
                        whole + c_logp + psi
                    };
                    if path < thr {
                        // a new prefix the cut drops (above); a repeat's own mass still merges, in its place
                        if last == Some(c) {
                            merge_pnb(next, &mut trie.nodes, mark, s.node, pnb + c_logp, || {
                                s.frames
                            });
                        }
                        continue;
                    }
                }
                if last == Some(c) {
                    // A new occurrence of `c` needs a blank in between.
                    Self::extend(trie, next, frame_list, frame, &s, c, pb + c_logp + psi);
                    // Repeat collapse: same prefix, more non-blank mass.
                    merge_pnb(next, &mut trie.nodes, mark, s.node, pnb + c_logp, || {
                        s.frames
                    });
                } else {
                    Self::extend(trie, next, frame_list, frame, &s, c, whole + c_logp + psi);
                }
            }
        }
    }

    /// [`Self::step_plain`] for a frame whose candidates come best first, without a CRF: the same paths merged in
    /// the same order, found without going through every (prefix, token) pair. False, with no path merged, for a
    /// frame it does not take (the plain step does): no bound to compute, candidates out of order, twice in the
    /// frame, too many or with ids past the table, or a value that is NaN or +inf (which the plain step's order of
    /// the bounds places its own way).
    ///
    /// A path into a new prefix scores `whole + logp` (the state's score and the token's log-prob), or less for the
    /// state's last token (`pb + logp`, pb <= whole), and adding is monotone, so along the candidates a state's
    /// paths only fall: past the first that is under the bound, every later one is. The bound itself is the
    /// `beam_width`-th largest of the plain step's bounds, kept in a heap of that size, a state's paths offered
    /// until one cannot enter it; the step then stops each state at its first path under the bound and visits,
    /// of the candidates after it, only the ones the bound never drops (the tokens of the state's children in the
    /// beam) and the state's last token (whose repeat mass merges whatever the bound).
    fn step_ordered(&mut self, frame: &Frame, beam_width: usize) -> bool {
        let Self {
            trie,
            frame_list,
            beam,
            next,
            candidate_at,
            candidate_epoch,
            child_mask,
            repeat_at,
            top,
            live,
            ..
        } = self;
        let candidates = frame.candidates;
        let (n, m) = (beam.len(), candidates.len());
        let mark = frame.mark;
        if plain_steps_only()
            || cut_bound_off()
            || n * (m + 1) <= beam_width
            || m > ORDERED_MAX_CANDIDATES
            || !under_inf(frame.blank)
        {
            return false;
        }

        // each candidate's place by its id
        *candidate_epoch = candidate_epoch.wrapping_add(1);
        if *candidate_epoch == 0 {
            candidate_at.fill((0, 0));
            *candidate_epoch = 1;
        }
        let epoch = *candidate_epoch;
        let mut blank_at = NO_PLACE;
        let mut above = f32::INFINITY;
        for (j, &(c, logp)) in candidates.iter().enumerate() {
            // best first and under +inf, neither of which a NaN is
            let in_order = logp <= above && under_inf(logp);
            if !in_order || c as usize >= CANDIDATE_TABLE_MAX {
                return false;
            }
            above = logp;
            if c as usize >= candidate_at.len() {
                candidate_at.resize(c as usize + 1, (0, 0));
            }
            let at = &mut candidate_at[c as usize];
            if at.0 == epoch {
                return false;
            }
            *at = (epoch, j as u32);
            if c == frame.blank_id {
                blank_at = j as u32;
            }
        }

        // each state's last token among the candidates, and its children in the beam among them
        for s in beam.iter() {
            if !(under_inf(s.pb) && under_inf(s.pnb) && under_inf(s.rank)) {
                return false;
            }
        }
        Self::mark_beam(&mut trie.nodes, beam, mark);
        child_mask.clear();
        child_mask.resize(n, 0);
        repeat_at.clear();
        repeat_at.resize(n, NO_PLACE);
        for (i, s) in beam.iter().enumerate() {
            if s.node == 0 {
                continue;
            }
            let node = &trie.nodes[s.node as usize];
            let at = match candidate_at.get(node.token as usize) {
                Some(&(e, at)) if e == epoch && at != blank_at => at,
                _ => continue,
            };
            repeat_at[i] = at;
            let parent = &trie.nodes[node.parent as usize];
            if parent.beam_mark == mark {
                child_mask[parent.beam_slot as usize] |= 1u64 << at;
            }
        }

        // the bound: the `beam_width`-th largest of each state's own score's bound and its paths into new prefixes
        // (the beam holds at most `beam_width` states: every own bound is kept)
        top.clear();
        for (i, s) in beam.iter().enumerate() {
            let mut own = s.rank + frame.blank;
            if repeat_at[i] != NO_PLACE {
                own = own.max(s.pnb + candidates[repeat_at[i] as usize].1);
            }
            top.push(own);
        }
        heapify(top);
        // the paths candidate by candidate, every state's best before any state's second (near the order of
        // their values, so that few are kept only to be pushed out): a state is left once a path of it cannot
        // enter, as none after it can
        live.clear();
        live.extend(0..n as u32);
        for (j, &(_, logp)) in candidates.iter().enumerate() {
            let mut kept = 0;
            for at in 0..live.len() {
                let i = live[at] as usize;
                let s = &beam[i];
                let path = s.rank + logp;
                if top.len() == beam_width && path <= top[0] {
                    continue;
                }
                live[kept] = i as u32;
                kept += 1;
                if j as u32 == blank_at || child_mask[i] >> j & 1 != 0 {
                    continue;
                }
                let path = if j as u32 == repeat_at[i] {
                    s.pb + logp
                } else {
                    path
                };
                offer(top, beam_width, path);
            }
            if kept == 0 {
                break;
            }
            live.truncate(kept);
        }
        // with exactly `beam_width` bounds the plain step has none (-inf); the smallest of them, under which no
        // path is, cuts as little
        let thr = if top.len() == beam_width {
            top[0]
        } else {
            NEG_INF
        };
        let bounded = thr > NEG_INF;

        for i in 0..n {
            let s = beam[i];
            let (pb, pnb, whole) = (s.pb, s.pnb, s.rank);
            // Emit blank: the prefix is unchanged.
            merge_pb(
                next,
                &mut trie.nodes,
                mark,
                s.node,
                whole + frame.blank,
                s.frames,
            );
            let (mask, repeat) = (child_mask[i], repeat_at[i]);
            // the plain step's paths write `+ psi`, 0.0 here: kept, as it makes a -0.0 a 0.0
            let psi = 0.0f32;
            let mut j = 0;
            while j < m {
                let (c, logp) = candidates[j];
                if j as u32 == blank_at {
                    j += 1;
                    continue;
                }
                let is_repeat = j as u32 == repeat;
                if bounded && mask >> j & 1 == 0 {
                    if whole + logp < thr {
                        // this path and every later one into a new prefix is under the bound
                        break;
                    }
                    if is_repeat && pb + logp + psi < thr {
                        // a new prefix the cut drops; the repeat's own mass still merges, in its place
                        merge_pnb(next, &mut trie.nodes, mark, s.node, pnb + logp, || s.frames);
                        j += 1;
                        continue;
                    }
                }
                if is_repeat {
                    // A new occurrence of `c` needs a blank in between.
                    Self::extend(trie, next, frame_list, frame, &s, c, pb + logp + psi);
                    // Repeat collapse: same prefix, more non-blank mass.
                    merge_pnb(next, &mut trie.nodes, mark, s.node, pnb + logp, || s.frames);
                } else {
                    Self::extend(trie, next, frame_list, frame, &s, c, whole + logp + psi);
                }
                j += 1;
            }
            if j == m {
                continue;
            }
            // after the stop: the children in the beam (no bound on them), and the last token's repeat mass
            let repeat_bit = if repeat == NO_PLACE {
                0
            } else {
                1u64 << repeat
            };
            let mut rest = (mask | repeat_bit) >> j << j;
            while rest != 0 {
                let p = rest.trailing_zeros();
                rest &= rest - 1;
                let (c, logp) = candidates[p as usize];
                if mask >> p & 1 == 0 {
                    // the last token, its new occurrence under the bound
                    merge_pnb(next, &mut trie.nodes, mark, s.node, pnb + logp, || s.frames);
                } else if p == repeat {
                    Self::extend(trie, next, frame_list, frame, &s, c, pb + logp + psi);
                    merge_pnb(next, &mut trie.nodes, mark, s.node, pnb + logp, || s.frames);
                } else {
                    Self::extend(trie, next, frame_list, frame, &s, c, whole + logp + psi);
                }
            }
        }
        true
    }

    /// Score the step's states and keep the best `beam_width` of them as the beam.
    fn cut(&mut self, beam_width: usize) {
        let Self {
            trie, beam, next, ..
        } = self;
        for s in next.iter_mut() {
            s.rank = logsumexp(s.pb, s.pnb);
        }
        if next.len() > beam_width {
            let nodes = &trie.nodes;
            next.select_nth_unstable_by(beam_width - 1, |a, b| by_rank(nodes, a, b));
            next.truncate(beam_width);
        }
        std::mem::swap(beam, next);
    }

    /// The beam as hypotheses, best first.
    fn finish(&mut self) -> Vec<BeamHypothesis> {
        let Self {
            trie,
            frame_list,
            beam,
            ..
        } = self;
        let nodes = &trie.nodes;
        // distinct prefixes: no two compare equal
        beam.sort_unstable_by(|a, b| by_rank(nodes, a, b));
        beam.iter()
            .map(|s| {
                let tokens = trie.tokens(s.node);
                let mut frames = Vec::with_capacity(tokens.len());
                let mut at = s.frames;
                while at != 0 {
                    frames.push(frame_list[at as usize].0);
                    at = frame_list[at as usize].1;
                }
                frames.reverse();
                BeamHypothesis {
                    tokens,
                    score: s.rank,
                    frames,
                }
            })
            .collect()
    }

    /// Give the memory back after a search far past a conversion's (a wide beam over a long input), so that a
    /// thread does not hold it for good.
    fn release_if_large(&mut self) {
        const KEEP: usize = 1 << 16;
        if self.trie.nodes.capacity() > KEEP || self.frame_list.capacity() > KEEP {
            *self = Self::default();
        }
    }
}

thread_local! {
    /// The scratch of the searches that are given none.
    static SCRATCH: RefCell<BeamScratch> = RefCell::new(BeamScratch::default());
}

/// Run prefix beam search over the proposal's frames. Returns up to
/// `beam_width` hypotheses, best first; ties break by token sequence.
pub fn prefix_beam_search(
    steps: &[FrameLogProbs],
    blank_id: u32,
    beam_width: usize,
    top_k_per_step: usize,
) -> Vec<BeamHypothesis> {
    prefix_beam_search_crf(steps, blank_id, beam_width, top_k_per_step, None)
}

#[cfg(test)]
thread_local! {
    /// Tests only: the beam without the cut's bound (every path merged), to compare against it.
    static NO_CUT_BOUND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests only: every frame through the plain step, to compare the ordered step against it.
    static PLAIN_STEPS_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn cut_bound_off() -> bool {
    NO_CUT_BOUND.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn cut_bound_off() -> bool {
    false
}

#[cfg(test)]
fn plain_steps_only() -> bool {
    PLAIN_STEPS_ONLY.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn plain_steps_only() -> bool {
    false
}

/// [`prefix_beam_search`] with a CTC-CRF's transitions: a path that emits a
/// new token `c` after the prefix's last token (the start symbol for the
/// first) adds log ψ(last, c), so a hypothesis' score is its CTC mass times
/// the transitions of its tokens (as the research side's reference decoder).
///
/// It works in the thread's scratch (its own, were that in use).
pub fn prefix_beam_search_crf(
    steps: &[FrameLogProbs],
    blank_id: u32,
    beam_width: usize,
    top_k_per_step: usize,
    crf: Option<&CrfTables>,
) -> Vec<BeamHypothesis> {
    let search = |scratch: &mut BeamScratch| {
        prefix_beam_search_crf_with_scratch(
            steps,
            blank_id,
            beam_width,
            top_k_per_step,
            crf,
            scratch,
        )
    };
    let on_thread = SCRATCH.try_with(|cell| {
        let mut scratch = cell.try_borrow_mut().ok()?;
        let hyps = search(&mut scratch);
        scratch.release_if_large();
        Some(hyps)
    });
    match on_thread {
        Ok(Some(hyps)) => hyps,
        // the thread's scratch in use, or gone with the thread's end
        _ => search(&mut BeamScratch::default()),
    }
}

/// [`prefix_beam_search_crf`] working in `scratch`, which a caller keeps
/// between searches (see [`BeamScratch`]); the result does not depend on what
/// the scratch was used for before.
pub fn prefix_beam_search_crf_with_scratch(
    steps: &[FrameLogProbs],
    blank_id: u32,
    beam_width: usize,
    top_k_per_step: usize,
    crf: Option<&CrfTables>,
    scratch: &mut BeamScratch,
) -> Vec<BeamHypothesis> {
    assert!(beam_width > 0, "beam_width must be positive");
    let top_k = top_k_per_step.max(1);
    scratch.start();
    for (t, step) in steps.iter().enumerate() {
        let frame = Frame {
            mark: t as u32 + 1,
            index: t as u32,
            candidates: &step.top[..top_k.min(step.top.len())],
            blank: step.blank,
            blank_id,
        };
        scratch.next.clear();
        if !(crf.is_none() && scratch.step_ordered(&frame, beam_width)) {
            scratch.step_plain(&frame, beam_width, crf);
        }
        scratch.cut(beam_width);
    }
    scratch.finish()
}

/// A score as the beam orders by it: a NaN as the worst, so that the order
/// is a total one (with NaN taken as equal to all, the sort's comparison
/// was no order, which a sort may panic on).
fn order_key(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

/// CTC greedy collapse: drop blanks, drop consecutive non-blank repeats.
/// Blank acts as a separator: [7, blank, 7] -> [7, 7] but [7, 7] -> [7].
pub fn collapse_frame_ids(frame_ids: &[u32], blank_id: u32) -> Vec<u32> {
    collapse_with_frames(frame_ids, blank_id).0
}

/// `collapse_frame_ids` plus the frame each kept token was emitted at (the
/// first frame of its run).
pub fn collapse_with_frames(frame_ids: &[u32], blank_id: u32) -> (Vec<u32>, Vec<u32>) {
    let mut decoded: Vec<u32> = Vec::with_capacity(frame_ids.len());
    let mut frames: Vec<u32> = Vec::with_capacity(frame_ids.len());
    let mut last: Option<u32> = None;
    for (t, &token) in frame_ids.iter().enumerate() {
        if token != blank_id && Some(token) != last {
            decoded.push(token);
            frames.push(t as u32);
        }
        last = Some(token);
    }
    (decoded, frames)
}

/// Decode the proposal: a true CTC greedy collapse when `beam_width <= 1`
/// (per-frame argmax, then collapse; a width-1 prefix beam is NOT
/// equivalent), prefix beam search otherwise (with the CTC-CRF's
/// transitions when `crf` is given). The greedy hypothesis scores 0.
///
/// The beam works in the thread's scratch ([`prefix_beam_search_crf`]).
pub fn hypotheses(
    proposal: &ProposalOutput,
    blank_id: u32,
    beam_width: usize,
    top_k_per_step: usize,
    crf: Option<&CrfTables>,
) -> Vec<BeamHypothesis> {
    if beam_width <= 1 {
        greedy_hypothesis(proposal, blank_id)
    } else {
        prefix_beam_search_crf(&proposal.steps, blank_id, beam_width, top_k_per_step, crf)
    }
}

/// [`hypotheses`] with the beam working in `scratch`.
pub fn hypotheses_with_scratch(
    proposal: &ProposalOutput,
    blank_id: u32,
    beam_width: usize,
    top_k_per_step: usize,
    crf: Option<&CrfTables>,
    scratch: &mut BeamScratch,
) -> Vec<BeamHypothesis> {
    if beam_width <= 1 {
        greedy_hypothesis(proposal, blank_id)
    } else {
        prefix_beam_search_crf_with_scratch(
            &proposal.steps,
            blank_id,
            beam_width,
            top_k_per_step,
            crf,
            scratch,
        )
    }
}

fn greedy_hypothesis(proposal: &ProposalOutput, blank_id: u32) -> Vec<BeamHypothesis> {
    let (tokens, frames) = proposal.collapsed_with_frames(blank_id);
    vec![BeamHypothesis {
        tokens,
        score: 0.0,
        frames,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A NaN score is the worst in the beam's order, which stays a total
    /// one: a sort over NaN and finite scores neither panics nor puts a NaN
    /// before a score.
    #[test]
    fn a_nan_score_orders_last() {
        let mut scores = vec![
            -1.0f32,
            f32::NAN,
            -0.5,
            f32::NEG_INFINITY,
            f32::NAN,
            -3.0,
            0.0,
        ];
        for _ in 0..50 {
            scores.sort_by(|a, b| order_key(*b).total_cmp(&order_key(*a)));
        }
        assert_eq!(&scores[..4], &[0.0, -0.5, -1.0, -3.0]);
        assert!(scores[4..]
            .iter()
            .all(|s| s.is_nan() || *s == f32::NEG_INFINITY));
    }

    /// `logsumexp` with its early return gives the bits of the plain formula: random pairs over the scores' range,
    /// pairs whose gap sits around the early return's bound, powers of two, zero, subnormals and infinities.
    #[test]
    fn logsumexp_early_return_keeps_the_bits() {
        let plain = |a: f32, b: f32| -> f32 {
            if a == NEG_INF {
                return b;
            }
            if b == NEG_INF {
                return a;
            }
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            hi + (lo - hi).exp().ln_1p()
        };
        let mut seed = 3u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut pairs = Vec::new();
        for _ in 0..200_000 {
            pairs.push((-(rnd() * 200.0), -(rnd() * 200.0)));
        }
        for e in -20..12 {
            for &sign in &[1.0f32, -1.0] {
                let hi = sign * 2f32.powi(e);
                let bound = negligible_below(hi);
                for k in -64..64 {
                    let d = bound + k as f32 * 0.05;
                    pairs.push((hi, hi + d));
                    pairs.push((hi * (1.0 + rnd() * 0.99), hi + d));
                }
            }
        }
        for &v in &[
            0.0f32,
            -0.0,
            1e-40,
            -1e-40,
            f32::INFINITY,
            -1.0e-3,
            -3.0e-38,
        ] {
            for &w in &[-150.0f32, -60.0, -1.0, 0.0, NEG_INF] {
                pairs.push((v, w));
                pairs.push((w, v));
            }
        }
        for (a, b) in pairs {
            assert_eq!(
                logsumexp(a, b).to_bits(),
                plain(a, b).to_bits(),
                "logsumexp({a}, {b})"
            );
        }
    }
    use crate::pipeline::lse;
    use crate::pipeline::proposal::{log_probs_from_logits, log_probs_with, FrameReduce};
    use ndarray::{arr2, Array2};

    /// The `HashMap<Vec<u32>, _>` prefix beam search the trie version
    /// replaced, over a full (T, V) log-prob matrix: the reference the
    /// property tests hold the trie version to.
    mod reference {
        use super::super::{logsumexp, BeamHypothesis, NEG_INF};
        use ndarray::Array2;
        use std::collections::HashMap;

        struct Cell {
            pb: f32,
            pnb: f32,
            frames: Vec<u32>,
            frames_logp: f32,
        }

        fn extend(prefix: &[u32], c: u32) -> Vec<u32> {
            let mut extended = prefix.to_vec();
            extended.push(c);
            extended
        }

        fn merge(
            beam: &mut HashMap<Vec<u32>, Cell>,
            prefix: Vec<u32>,
            pb: f32,
            pnb: f32,
            frames: impl FnOnce() -> Vec<u32>,
        ) {
            let contribution = logsumexp(pb, pnb);
            match beam.get_mut(&prefix) {
                Some(cell) => {
                    cell.pb = logsumexp(cell.pb, pb);
                    cell.pnb = logsumexp(cell.pnb, pnb);
                    if contribution > cell.frames_logp {
                        cell.frames = frames();
                        cell.frames_logp = contribution;
                    }
                }
                None => {
                    beam.insert(
                        prefix,
                        Cell {
                            pb,
                            pnb,
                            frames: frames(),
                            frames_logp: contribution,
                        },
                    );
                }
            }
        }

        pub fn prefix_beam_search(
            log_probs: &Array2<f32>,
            blank_id: u32,
            beam_width: usize,
            top_k_per_step: usize,
        ) -> Vec<BeamHypothesis> {
            let (t_len, vocab) = (log_probs.shape()[0], log_probs.shape()[1]);
            let top_k = top_k_per_step.min(vocab).max(1);
            let mut step_topk: Vec<Vec<(u32, f32)>> = Vec::with_capacity(t_len);
            for t in 0..t_len {
                let row = log_probs.row(t);
                let mut idx: Vec<usize> = (0..vocab).collect();
                idx.select_nth_unstable_by(top_k - 1, |&a, &b| {
                    row[b]
                        .partial_cmp(&row[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut top: Vec<(u32, f32)> =
                    idx[..top_k].iter().map(|&i| (i as u32, row[i])).collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                step_topk.push(top);
            }
            let mut beam: HashMap<Vec<u32>, Cell> = HashMap::new();
            beam.insert(
                Vec::new(),
                Cell {
                    pb: 0.0,
                    pnb: NEG_INF,
                    frames: Vec::new(),
                    frames_logp: 0.0,
                },
            );
            for t in 0..t_len {
                let mut next_beam: HashMap<Vec<u32>, Cell> = HashMap::new();
                let blank_logp = log_probs[[t, blank_id as usize]];
                let frame = t as u32;
                for (prefix, cell) in &beam {
                    let (pb, pnb) = (cell.pb, cell.pnb);
                    merge(
                        &mut next_beam,
                        prefix.clone(),
                        logsumexp(pb, pnb) + blank_logp,
                        NEG_INF,
                        || cell.frames.clone(),
                    );
                    for &(c, c_logp) in &step_topk[t] {
                        if c == blank_id {
                            continue;
                        }
                        if prefix.last().copied() == Some(c) {
                            merge(
                                &mut next_beam,
                                extend(prefix, c),
                                NEG_INF,
                                pb + c_logp,
                                || extend(&cell.frames, frame),
                            );
                            merge(
                                &mut next_beam,
                                prefix.clone(),
                                NEG_INF,
                                pnb + c_logp,
                                || cell.frames.clone(),
                            );
                        } else {
                            let new_pnb = logsumexp(pb, pnb) + c_logp;
                            merge(&mut next_beam, extend(prefix, c), NEG_INF, new_pnb, || {
                                extend(&cell.frames, frame)
                            });
                        }
                    }
                }
                let mut scored: Vec<_> = next_beam
                    .into_iter()
                    .map(|(p, cell)| {
                        let score = logsumexp(cell.pb, cell.pnb);
                        (p, cell, score)
                    })
                    .collect();
                scored.sort_by(|a, b| {
                    b.2.partial_cmp(&a.2)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                scored.truncate(beam_width);
                beam = scored.into_iter().map(|(p, cell, _)| (p, cell)).collect();
            }
            let mut hyps: Vec<BeamHypothesis> = beam
                .into_iter()
                .map(|(tokens, cell)| BeamHypothesis {
                    tokens,
                    score: logsumexp(cell.pb, cell.pnb),
                    frames: cell.frames,
                })
                .collect();
            hyps.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.tokens.cmp(&b.tokens))
            });
            hyps
        }

        /// The full log-softmax the engine used to build: the row's
        /// log-sum-exp in the same order, blacklisted ids at -inf.
        pub fn log_probs(
            flat: &[f32],
            t_len: usize,
            vocab: usize,
            blacklist: &[u32],
        ) -> Array2<f32> {
            let mut out = Array2::<f32>::zeros((t_len, vocab));
            for t in 0..t_len {
                let row = &flat[t * vocab..(t + 1) * vocab];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for &v in row {
                    sum += (v - max).exp();
                }
                let lse = max + sum.ln();
                for v in 0..vocab {
                    out[[t, v]] = if blacklist.contains(&(v as u32)) {
                        f32::NEG_INFINITY
                    } else {
                        row[v] - lse
                    };
                }
            }
            out
        }
    }

    /// Frames from a log-prob matrix (blank id 0), as the proposal gives them.
    fn steps(log_probs: &Array2<f32>, k: usize) -> Vec<FrameLogProbs> {
        let (t_len, vocab) = (log_probs.shape()[0], log_probs.shape()[1]);
        let flat: Vec<f32> = log_probs.iter().copied().collect();
        log_probs_from_logits(&flat, t_len, vocab, &[], 0, k).steps
    }

    fn beam(log_probs: &Array2<f32>, beam_width: usize, k: usize) -> Vec<BeamHypothesis> {
        prefix_beam_search(&steps(log_probs, k), 0, beam_width, k)
    }

    #[test]
    fn repeats_collapse_without_a_blank() {
        // Peak at 1, 1 (repeat), 2: CTC collapses to [1, 2].
        let logits = arr2(&[
            [-3.0, -0.1, -3.0, -3.0],
            [-3.0, -0.1, -3.0, -3.0],
            [-3.0, -3.0, -0.1, -3.0],
        ]);
        let out = beam(&logits, 4, 4);
        assert_eq!(out[0].tokens, vec![1, 2]);
    }

    #[test]
    fn blank_separated_repeats_expand() {
        let logits = arr2(&[[-3.0, -0.1, -3.0], [-0.1, -3.0, -3.0], [-3.0, -0.1, -3.0]]);
        let out = beam(&logits, 4, 3);
        assert_eq!(out[0].tokens, vec![1, 1]);
    }

    #[test]
    fn scores_are_sorted_and_the_beam_is_bounded() {
        let logits = arr2(&[
            [-1.0, -1.2, -1.5, -2.0],
            [-1.1, -1.0, -1.4, -2.5],
            [-0.9, -1.3, -1.2, -2.2],
        ]);
        let out = beam(&logits, 3, 4);
        assert!(out.len() <= 3);
        assert!(out.windows(2).all(|w| w[0].score >= w[1].score));
    }

    #[test]
    fn zero_top_k_per_step_is_clamped() {
        let logits = arr2(&[[-3.0, -0.1], [-0.1, -3.0]]);
        let out = prefix_beam_search(&steps(&logits, 2), 0, 2, 0);
        assert!(!out.is_empty());
    }

    #[test]
    fn crf_at_zero_is_the_plain_beam() {
        // ψ = 1 everywhere: the same hypotheses and scores
        let logits = arr2(&[[-0.5, -1.2, -2.0], [-1.0, -0.4, -2.5], [-0.3, -2.0, -1.1]]);
        let crf = crate::pipeline::crf::CrfTables::from_tables(&[0.0; 4 * 2], &[0.0; 3 * 2], 2, 3);
        let plain = prefix_beam_search(&steps(&logits, 3), 0, 4, 3);
        let with = prefix_beam_search_crf(&steps(&logits, 3), 0, 4, 3, Some(&crf));
        assert_eq!(plain.len(), with.len());
        for (p, w) in plain.iter().zip(&with) {
            assert_eq!(p.tokens, w.tokens);
            assert!((p.score - w.score).abs() < 1e-5);
        }
    }

    #[test]
    fn crf_transition_reorders_the_beam() {
        // [1, 2] wins by the frames; rank 1 with a_1 = -20 makes every
        // token after 1 nearly impossible, so it loses
        let logits = arr2(&[[-3.0, -0.1, -3.0], [-3.0, -3.0, -0.1]]);
        let plain = prefix_beam_search(&steps(&logits, 3), 0, 4, 3);
        assert_eq!(plain[0].tokens, vec![1, 2]);
        let crf =
            crate::pipeline::crf::CrfTables::from_tables(&[0.0, -20.0, 0.0, 0.0], &[0.0; 3], 1, 3);
        let with = prefix_beam_search_crf(&steps(&logits, 3), 0, 4, 3, Some(&crf));
        assert_ne!(with[0].tokens, vec![1, 2]);
    }

    #[test]
    fn hypotheses_carry_the_frame_of_each_token() {
        // frames: 0 -> 1, 1 -> 1 (repeat), 2 -> blank, 3 -> 2
        let logits = arr2(&[
            [-3.0, -0.1, -3.0],
            [-3.0, -0.1, -3.0],
            [-0.1, -3.0, -3.0],
            [-3.0, -3.0, -0.1],
        ]);
        let out = beam(&logits, 4, 3);
        assert_eq!(out[0].tokens, vec![1, 2]);
        assert_eq!(out[0].frames, vec![0, 3]);
        assert!(out.iter().all(|h| h.frames.len() == h.tokens.len()));
        assert_eq!(
            collapse_with_frames(&[0, 1, 1, 0, 2, 2], 0),
            (vec![1, 2], vec![1, 4])
        );
    }

    #[test]
    fn collapse_frame_ids_drops_blanks_and_repeats() {
        assert_eq!(
            collapse_frame_ids(&[4, 7, 7, 9, 4, 9, 11, 11], 4),
            vec![7, 9, 9, 11]
        );
        assert_eq!(collapse_frame_ids(&[], 0), Vec::<u32>::new());
        assert_eq!(collapse_frame_ids(&[4, 4, 4], 4), Vec::<u32>::new());
        assert_eq!(collapse_frame_ids(&[7, 4, 7], 4), vec![7, 7]);
    }

    // ---- property tests against the reference implementation ----

    /// xorshift64*: deterministic, dependency-free.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    struct Case {
        logits: Vec<f32>,
        t_len: usize,
        vocab: usize,
        blank: u32,
        blacklist: Vec<u32>,
        beam_width: usize,
        k: usize,
    }

    fn run_both(c: &Case) -> (Vec<BeamHypothesis>, Vec<BeamHypothesis>) {
        let lp = reference::log_probs(&c.logits, c.t_len, c.vocab, &c.blacklist);
        let want = reference::prefix_beam_search(&lp, c.blank, c.beam_width, c.k);
        // the scalar log-sum-exp, as the reference: this compares the beams
        let proposal = log_probs_with(
            &c.logits,
            c.t_len,
            c.vocab,
            &c.blacklist,
            c.blank,
            c.k,
            FrameReduce { lse: lse::exact },
        );
        let got = prefix_beam_search(&proposal.steps, c.blank, c.beam_width, c.k);
        // A blacklisted id is -inf in the reference's matrix and can fill a
        // top-k slot when fewer than k ids are allowed, making -inf
        // hypotheses; the proposal never lists blacklisted ids.
        let finite = |h: Vec<BeamHypothesis>| -> Vec<BeamHypothesis> {
            h.into_iter().filter(|h| h.score > NEG_INF).collect()
        };
        (finite(want), finite(got))
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5 * (1.0 + a.abs().max(b.abs()))
    }

    fn assert_same(c: &Case, want: &[BeamHypothesis], got: &[BeamHypothesis], seed: u64) {
        assert_eq!(want.len(), got.len(), "seed {seed}: lengths");
        for (w, g) in want.iter().zip(got) {
            assert_eq!(
                w.tokens, g.tokens,
                "seed {seed} T={} V={}",
                c.t_len, c.vocab
            );
            assert_eq!(w.frames, g.frames, "seed {seed}: frames of {:?}", w.tokens);
            assert!(
                close(w.score, g.score),
                "seed {seed}: {} vs {}",
                w.score,
                g.score
            );
        }
    }

    fn random_case(rng: &mut Rng, scale: f32) -> Case {
        let t_len = 1 + rng.below(24);
        let vocab = 3 + rng.below(40);
        let blank = rng.below(vocab) as u32;
        let mut logits: Vec<f32> = (0..t_len * vocab)
            .map(|_| (rng.unit() + rng.unit() + rng.unit() - 1.5) * scale)
            .collect();
        // CTC-like peaky frames: often the blank or one token dominates
        for t in 0..t_len {
            if rng.below(3) == 0 {
                let peak = if rng.below(2) == 0 {
                    blank as usize
                } else {
                    rng.below(vocab)
                };
                logits[t * vocab + peak] += 8.0 + 8.0 * rng.unit();
            }
        }
        let blacklist: Vec<u32> = (0..rng.below(3))
            .map(|_| rng.below(vocab) as u32)
            .filter(|&b| b != blank)
            .collect();
        Case {
            logits,
            t_len,
            vocab,
            blank,
            blacklist,
            beam_width: [2, 3, 5, 8, 20][rng.below(5)],
            k: [1, 2, 4, 8, 16][rng.below(5)],
        }
    }

    /// The cut's bound changes nothing: the beam with it and without it (every path merged) give the same
    /// hypotheses, frames and score bits, on peaky and flat random frames with the engine's widths and others.
    #[test]
    fn the_cut_bound_drops_only_what_the_cut_drops() {
        let mut built = 0usize;
        for seed in 1..=600u64 {
            let mut rng = Rng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
            let scale = [2.0, 4.0, 12.0][rng.below(3)];
            let mut c = random_case(&mut rng, scale);
            if seed % 3 == 0 {
                (c.beam_width, c.k) = (20, 16);
            }
            let proposal = log_probs_with(
                &c.logits,
                c.t_len,
                c.vocab,
                &c.blacklist,
                c.blank,
                c.k,
                FrameReduce { lse: lse::exact },
            );
            let with = prefix_beam_search(&proposal.steps, c.blank, c.beam_width, c.k);
            NO_CUT_BOUND.with(|f| f.set(true));
            let without = prefix_beam_search(&proposal.steps, c.blank, c.beam_width, c.k);
            NO_CUT_BOUND.with(|f| f.set(false));
            assert_eq!(with.len(), without.len(), "seed {seed}");
            for (a, b) in with.iter().zip(&without) {
                assert_eq!(
                    (&a.tokens, &a.frames),
                    (&b.tokens, &b.frames),
                    "seed {seed}"
                );
                assert_eq!(
                    a.score.to_bits(),
                    b.score.to_bits(),
                    "seed {seed}: {:?}",
                    a.tokens
                );
            }
            built += with.len();
        }
        assert!(built > 1000);
    }

    #[test]
    fn matches_the_reference_on_random_logits() {
        for seed in 1..=400u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let c = random_case(&mut rng, 4.0);
            let (want, got) = run_both(&c);
            assert_same(&c, &want, &got, seed);
        }
    }

    #[test]
    fn matches_the_reference_on_extreme_logits() {
        for seed in 1..=150u64 {
            let mut rng = Rng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
            let mut c = random_case(&mut rng, 60.0);
            // saturated rows: one token at 1e4, others near -1e4, some -1e30
            for t in 0..c.t_len {
                if rng.below(2) == 0 {
                    for v in 0..c.vocab {
                        c.logits[t * c.vocab + v] = match rng.below(4) {
                            0 => -1e30,
                            1 => -1e4 + rng.unit(),
                            _ => rng.unit() * 10.0,
                        };
                    }
                    c.logits[t * c.vocab + rng.below(c.vocab)] = 1e4;
                }
            }
            let (want, got) = run_both(&c);
            assert_same(&c, &want, &got, seed);
        }
    }

    #[test]
    fn matches_the_reference_on_tied_logits() {
        // Values from a tiny set, so ranks and path masses tie exactly. With
        // top-k covering the vocabulary and a beam no step fills, neither the
        // top-k selection nor the truncation has to break a tie, and every
        // prefix's mass must agree. (Which of two equally probable paths
        // supplies the frames is merge-order dependent, so frames are not
        // compared.)
        for seed in 1..=200u64 {
            let mut rng = Rng(seed.wrapping_mul(0xA24B_AED4_963E_E407));
            let t_len = 1 + rng.below(5);
            let vocab = 2 + rng.below(3);
            let logits: Vec<f32> = (0..t_len * vocab).map(|_| rng.below(3) as f32).collect();
            let c = Case {
                logits,
                t_len,
                vocab,
                blank: 0,
                blacklist: Vec::new(),
                beam_width: 4096,
                k: vocab,
            };
            let (want, got) = run_both(&c);
            assert_eq!(want.len(), got.len(), "seed {seed}");
            let by_tokens = |h: &[BeamHypothesis]| {
                let mut v: Vec<(Vec<u32>, f32)> =
                    h.iter().map(|h| (h.tokens.clone(), h.score)).collect();
                v.sort_by(|a, b| a.0.cmp(&b.0));
                v
            };
            for ((wt, ws), (gt, gs)) in by_tokens(&want).iter().zip(&by_tokens(&got)) {
                assert_eq!(wt, gt, "seed {seed}");
                assert!(close(*ws, *gs), "seed {seed}: {ws} vs {gs}");
            }
            // and the best one is the same whenever it is unique
            if want.len() > 1 && !close(want[0].score, want[1].score) {
                assert_eq!(want[0].tokens, got[0].tokens, "seed {seed}");
            }
        }
    }

    // ---- vectorized log-sum-exp against the scalar one ----

    /// Beam over `c` with the given row log-sum-exp.
    fn beam_with(c: &Case, row_lse: fn(&[f32]) -> f32) -> Vec<BeamHypothesis> {
        let p = log_probs_with(
            &c.logits,
            c.t_len,
            c.vocab,
            &c.blacklist,
            c.blank,
            c.k,
            FrameReduce { lse: row_lse },
        );
        prefix_beam_search(&p.steps, c.blank, c.beam_width, c.k)
    }

    /// The vectorized log-sum-exp may reorder only hypotheses whose scores
    /// are within `tol`: the same hypotheses in the same order otherwise,
    /// with scores within `tol`, and membership may differ only for those
    /// within `tol` of the beam's cutoff.
    fn assert_same_up_to_near_ties(want: &[BeamHypothesis], got: &[BeamHypothesis], seed: u64) {
        let tol = |a: f32| 2e-5 * (1.0 + a.abs());
        let near = |a: f32, b: f32| a == b || (a - b).abs() <= tol(a);
        let cutoff = want.last().map_or(NEG_INF, |h| h.score);
        let score_of =
            |h: &[BeamHypothesis], t: &[u32]| h.iter().find(|x| x.tokens == t).map(|x| x.score);
        for w in want {
            match score_of(got, &w.tokens) {
                Some(g) => assert!(near(g, w.score), "seed {seed}: {g} vs {}", w.score),
                None => assert!(
                    near(w.score, cutoff),
                    "seed {seed}: {:?} ({}) dropped away from the cutoff {cutoff}",
                    w.tokens,
                    w.score
                ),
            }
        }
        // order: any pair in both lists that swapped must be a near-tie
        for (i, a) in want.iter().enumerate() {
            for b in &want[i + 1..] {
                let (Some(ia), Some(ib)) = (
                    got.iter().position(|x| x.tokens == a.tokens),
                    got.iter().position(|x| x.tokens == b.tokens),
                ) else {
                    continue;
                };
                if ia > ib {
                    assert!(
                        near(a.score, b.score),
                        "seed {seed}: {:?} ({}) and {:?} ({}) swapped",
                        a.tokens,
                        a.score,
                        b.tokens,
                        b.score
                    );
                }
            }
        }
    }

    #[test]
    fn vectorized_lse_only_reorders_near_ties() {
        for seed in 1..=400u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let c = random_case(&mut rng, 4.0);
            let want = beam_with(&c, lse::exact);
            let got = beam_with(&c, lse::row);
            assert_same_up_to_near_ties(&want, &got, seed);
        }
    }

    #[test]
    fn vectorized_lse_keeps_close_but_distinct_hypotheses_in_order() {
        // Two frames; tokens 1 and 2 each win one frame by `delta`, so the
        // hypotheses [1] and [2] differ by about `delta`. Down to a gap of
        // 1e-4 they keep the scalar path's order.
        for &delta in &[1e-1f32, 1e-2, 1e-3, 1e-4] {
            for &base in &[0.0f32, 5.0, 20.0] {
                let v = 64;
                let mut logits = vec![-3.0f32; 2 * v];
                logits[0] = base; // blank strong on frame 0
                logits[1] = base - 1.0;
                logits[2] = base - 1.0 - delta;
                logits[v] = base;
                logits[v + 1] = base - 1.0 - delta;
                logits[v + 2] = base - 1.0 + delta * 0.5;
                let c = Case {
                    logits,
                    t_len: 2,
                    vocab: v,
                    blank: 0,
                    blacklist: Vec::new(),
                    beam_width: 8,
                    k: 4,
                };
                let want = beam_with(&c, lse::exact);
                let got = beam_with(&c, lse::row);
                let order =
                    |h: &[BeamHypothesis]| h.iter().map(|x| x.tokens.clone()).collect::<Vec<_>>();
                assert_eq!(order(&want), order(&got), "delta {delta} base {base}");
            }
        }
    }

    // ---- the ordered step, the scratch and the trie against the plain step ----

    /// Hypotheses to the bit: tokens, frames, score.
    fn bits(h: &[BeamHypothesis]) -> Vec<(Vec<u32>, Vec<u32>, u32)> {
        h.iter()
            .map(|h| (h.tokens.clone(), h.frames.clone(), h.score.to_bits()))
            .collect()
    }

    /// The beam with every frame through the plain step.
    fn plain_beam(
        steps: &[FrameLogProbs],
        blank: u32,
        width: usize,
        k: usize,
    ) -> Vec<BeamHypothesis> {
        PLAIN_STEPS_ONLY.with(|f| f.set(true));
        let out = prefix_beam_search(steps, blank, width, k);
        PLAIN_STEPS_ONLY.with(|f| f.set(false));
        out
    }

    fn proposal_steps(c: &Case) -> Vec<FrameLogProbs> {
        log_probs_with(
            &c.logits,
            c.t_len,
            c.vocab,
            &c.blacklist,
            c.blank,
            c.k,
            FrameReduce { lse: lse::exact },
        )
        .steps
    }

    /// Frames like a conversion's (blank id 0): of a large vocabulary each frame's best `k`, best first, falling
    /// by small and large gaps; the blank and a few tokens come back frame after frame, so that prefixes repeat
    /// their last token and have their children in the beam.
    fn peaky_steps(rng: &mut Rng, t_len: usize, k: usize) -> Vec<FrameLogProbs> {
        (0..t_len)
            .map(|_| {
                let mut ids: Vec<u32> = Vec::new();
                while ids.len() < k {
                    let id = if rng.below(3) > 0 {
                        rng.below(12) as u32
                    } else {
                        12 + rng.below(8000) as u32
                    };
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
                let mut logp = -(rng.unit() * 0.05);
                let mut top = Vec::with_capacity(k);
                for id in ids {
                    top.push((id, logp));
                    logp -= rng.unit() * [0.5, 4.0, 12.0][rng.below(3)];
                }
                let blank = top
                    .iter()
                    .find(|&&(id, _)| id == 0)
                    .map_or(-(10.0 + rng.unit() * 20.0), |&(_, l)| l);
                FrameLogProbs { top, blank }
            })
            .collect()
    }

    /// The ordered step merges what the plain step merges, in its order: the same hypotheses, frames and score
    /// bits with it and with every frame through the plain step, on the random cases of the tests above (peaky,
    /// flat, saturated, tied) and on frames like a conversion's.
    #[test]
    fn the_ordered_step_is_the_plain_step() {
        let mut built = 0usize;
        for seed in 1..=900u64 {
            let mut rng = Rng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
            let (steps, blank, width, k) = match seed % 3 {
                0 => {
                    let t_len = 1 + rng.below(30);
                    let k = [2, 4, 16, 16, 64][rng.below(5)];
                    let width = [2, 5, 20, 20, 40][rng.below(5)];
                    (peaky_steps(&mut rng, t_len, k), 0, width, k)
                }
                1 => {
                    let scale = [2.0, 4.0, 12.0, 60.0][rng.below(4)];
                    let c = random_case(&mut rng, scale);
                    (proposal_steps(&c), c.blank, c.beam_width, c.k)
                }
                _ => {
                    // values from a tiny set: bounds, paths and scores tie
                    let t_len = 1 + rng.below(8);
                    let vocab = 2 + rng.below(5);
                    let c = Case {
                        logits: (0..t_len * vocab).map(|_| rng.below(3) as f32).collect(),
                        t_len,
                        vocab,
                        blank: rng.below(vocab) as u32,
                        blacklist: Vec::new(),
                        beam_width: [2, 3, 5, 4096][rng.below(4)],
                        k: 1 + rng.below(vocab),
                    };
                    (proposal_steps(&c), c.blank, c.beam_width, c.k)
                }
            };
            let got = prefix_beam_search(&steps, blank, width, k);
            let want = plain_beam(&steps, blank, width, k);
            assert_eq!(bits(&got), bits(&want), "seed {seed}");
            built += got.len();
        }
        assert!(built > 3000);
    }

    /// A frame the ordered step does not take is the plain step's, with nothing merged: candidates out of order or
    /// twice in the frame, a NaN or +inf, too many candidates or an id past the table, no bound to compute.
    #[test]
    fn the_ordered_step_leaves_other_frames_to_the_plain_step() {
        let step = |candidates: &[(u32, f32)], blank: f32, width: usize| -> bool {
            let mut scratch = BeamScratch::default();
            scratch.start();
            let frame = Frame {
                mark: 1,
                index: 0,
                candidates,
                blank,
                blank_id: 0,
            };
            let taken = scratch.step_ordered(&frame, width);
            assert_eq!(taken, !scratch.next.is_empty());
            taken
        };
        assert!(step(&[(1, -0.1), (2, -0.5), (3, -2.0)], -1.0, 2));
        assert!(step(&[(1, -0.1), (0, -0.5), (3, -0.5)], -0.5, 2));
        assert!(step(&[(1, -0.1), (2, NEG_INF), (3, NEG_INF)], NEG_INF, 2));
        // one state and three candidates make four bounds: none to compute for a beam of four
        assert!(!step(&[(1, -0.1), (2, -0.5), (3, -2.0)], -1.0, 4));
        assert!(!step(&[(1, -0.5), (2, -0.1), (3, -2.0)], -1.0, 2));
        assert!(!step(&[(1, -0.1), (2, -0.5), (1, -2.0)], -1.0, 2));
        assert!(!step(&[(1, -0.1), (2, f32::NAN), (3, -2.0)], -1.0, 2));
        assert!(!step(&[(1, f32::NAN), (2, -0.5), (3, -2.0)], -1.0, 2));
        assert!(!step(&[(1, f32::INFINITY), (2, -0.5), (3, -2.0)], -1.0, 2));
        assert!(!step(&[(1, -0.1), (2, -0.5), (3, -2.0)], f32::NAN, 2));
        assert!(!step(&[(1, -0.1), (2, -0.5), (3, -2.0)], f32::INFINITY, 2));
        assert!(!step(
            &[(1, -0.1), (CANDIDATE_TABLE_MAX as u32, -0.5), (3, -2.0)],
            -1.0,
            2
        ));
        let many: Vec<(u32, f32)> = (0..ORDERED_MAX_CANDIDATES as u32 + 1)
            .map(|i| (i + 1, -(i as f32)))
            .collect();
        assert!(!step(&many, -1.0, 2));
        assert!(step(&many[..ORDERED_MAX_CANDIDATES], -1.0, 2));
    }

    /// Frames out of the proposal's order or with a NaN or +inf decode as they did: the same with and without the
    /// ordered step (which leaves them), a frame's candidates in any order the same hypotheses as best first.
    #[test]
    fn unordered_and_non_finite_frames_decode_the_same() {
        for seed in 1..=300u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let c = random_case(&mut rng, 4.0);
            let steps = proposal_steps(&c);

            // every frame's candidates in another order (all of them: this `k` cuts none)
            let k =
                c.k.max(steps.iter().map(|s| s.top.len()).max().unwrap_or(0));
            let sorted = prefix_beam_search(&steps, c.blank, c.beam_width, k);
            let mut shuffled = steps.clone();
            for step in shuffled.iter_mut() {
                for i in (1..step.top.len()).rev() {
                    step.top.swap(i, rng.below(i + 1));
                }
            }
            let got = prefix_beam_search(&shuffled, c.blank, c.beam_width, k);
            assert_eq!(bits(&got), bits(&sorted), "seed {seed}: shuffled");
            assert_eq!(
                bits(&got),
                bits(&plain_beam(&shuffled, c.blank, c.beam_width, k)),
                "seed {seed}: shuffled, plain"
            );

            // a NaN, a +inf or a -inf somewhere
            let mut odd = steps.clone();
            for _ in 0..1 + rng.below(3) {
                let t = rng.below(odd.len());
                let v = [f32::NAN, f32::INFINITY, NEG_INF, 1e38][rng.below(4)];
                if rng.below(4) == 0 || odd[t].top.is_empty() {
                    odd[t].blank = v;
                } else {
                    let j = rng.below(odd[t].top.len());
                    odd[t].top[j].1 = v;
                }
            }
            let got = prefix_beam_search(&odd, c.blank, c.beam_width, c.k);
            let want = plain_beam(&odd, c.blank, c.beam_width, c.k);
            assert_eq!(bits(&got), bits(&want), "seed {seed}: non-finite");
        }
    }

    /// A search gives the same whatever its scratch held: one scratch through searches of every size (wide and
    /// narrow beams, long and short inputs, with a CRF and without) against a new one each, and against the
    /// thread's.
    #[test]
    fn a_scratch_is_reused_without_a_trace() {
        let crf = crate::pipeline::crf::CrfTables::from_tables(
            &(0..9 * 2)
                .map(|i| (i % 5) as f32 * -0.3)
                .collect::<Vec<_>>(),
            &(0..8 * 2)
                .map(|i| (i % 3) as f32 * -0.2)
                .collect::<Vec<_>>(),
            2,
            8,
        );
        let mut kept = BeamScratch::default();
        for seed in 1..=300u64 {
            let mut rng = Rng(seed.wrapping_mul(0xA24B_AED4_963E_E407));
            let (steps, blank, width, k) = if seed % 2 == 0 {
                let t_len = 1 + rng.below(40);
                let width = [2, 20, 300][rng.below(3)];
                (peaky_steps(&mut rng, t_len, 16), 0, width, 16)
            } else {
                let mut c = random_case(&mut rng, 4.0);
                if seed % 5 == 0 {
                    // the CRF's tables cover 8 tokens
                    c.vocab = c.vocab.min(8);
                    c.blank = rng.below(c.vocab) as u32;
                    c.blacklist.clear();
                    c.logits.truncate(c.t_len * c.vocab);
                }
                (proposal_steps(&c), c.blank, c.beam_width, c.k)
            };
            let crf = (seed % 5 == 0 && seed % 2 == 1).then_some(&crf);
            let got = prefix_beam_search_crf_with_scratch(&steps, blank, width, k, crf, &mut kept);
            let fresh = prefix_beam_search_crf_with_scratch(
                &steps,
                blank,
                width,
                k,
                crf,
                &mut BeamScratch::default(),
            );
            assert_eq!(bits(&got), bits(&fresh), "seed {seed}");
            let thread = prefix_beam_search_crf(&steps, blank, width, k, crf);
            assert_eq!(bits(&got), bits(&thread), "seed {seed}: the thread's");
        }
    }

    /// `hypotheses_with_scratch` is `hypotheses`, greedy and beam.
    #[test]
    fn hypotheses_with_a_scratch_are_the_hypotheses() {
        let mut rng = Rng(7);
        let c = random_case(&mut rng, 4.0);
        let proposal = log_probs_with(
            &c.logits,
            c.t_len,
            c.vocab,
            &c.blacklist,
            c.blank,
            c.k,
            FrameReduce { lse: lse::exact },
        );
        let mut scratch = BeamScratch::default();
        for width in [1, 2, 20] {
            let want = hypotheses(&proposal, c.blank, width, c.k, None);
            let got = hypotheses_with_scratch(&proposal, c.blank, width, c.k, None, &mut scratch);
            assert_eq!(bits(&got), bits(&want), "width {width}");
        }
    }

    /// The trie gives a (parent, token) one node through the growth of its table and anew after a reset, and
    /// orders two prefixes as their token sequences.
    #[test]
    fn the_trie_keeps_its_children_and_orders_prefixes() {
        let mut rng = Rng(11);
        let mut trie = Trie::default();
        for round in 0..3 {
            trie.reset();
            assert_eq!(trie.nodes.len(), 1, "round {round}");
            let mut made: Vec<(u32, u32, u32)> = Vec::new();
            // past half the first table: it grows
            while trie.nodes.len() < CHILD_TABLE_MIN * 3 {
                let parent = rng.below(trie.nodes.len()) as u32;
                let token = rng.below(6) as u32;
                let before = trie.nodes.len() as u32;
                let id = trie.child(parent, token);
                match made.iter().find(|m| (m.0, m.1) == (parent, token)) {
                    Some(m) => assert_eq!((id, trie.nodes.len() as u32), (m.2, before)),
                    None => {
                        assert_eq!(id, before);
                        made.push((parent, token, id));
                    }
                }
            }
            for &(parent, token, id) in &made {
                assert_eq!(trie.child(parent, token), id);
                let mut tokens = trie.tokens(parent);
                tokens.push(token);
                assert_eq!(trie.tokens(id), tokens);
                assert_eq!(trie.last(id), Some(token));
            }
            for _ in 0..4000 {
                let (a, b) = (
                    rng.below(trie.nodes.len()) as u32,
                    rng.below(trie.nodes.len()) as u32,
                );
                assert_eq!(
                    cmp_tokens(&trie.nodes, a, b),
                    trie.tokens(a).cmp(&trie.tokens(b)),
                    "{a} and {b}"
                );
            }
        }
    }

    /// `heapify` and `offer` keep the largest `width` values, the smallest of them first.
    #[test]
    fn offer_keeps_the_largest() {
        let mut rng = Rng(5);
        for width in [1usize, 2, 3, 20, 33] {
            for n in [0usize, 1, 19, 20, 21, 200] {
                let values: Vec<f32> = (0..n)
                    .map(|_| match rng.below(8) {
                        0 => NEG_INF,
                        1 => -(rng.below(4) as f32),
                        _ => -rng.unit() * 30.0,
                    })
                    .collect();
                // some as they come, made a heap at once, the rest offered
                let mut top: Vec<f32> = values[..n.min(width) / 2].to_vec();
                heapify(&mut top);
                for &v in &values[top.len()..] {
                    offer(&mut top, width, v);
                }
                let mut want = values.clone();
                want.sort_by(|a, b| b.total_cmp(a));
                want.truncate(width);
                assert_eq!(top.len(), want.len());
                if let Some(&least) = want.last() {
                    assert_eq!(top[0], least, "width {width} of {n}");
                }
                top.sort_by(|a, b| b.total_cmp(a));
                assert_eq!(top, want, "width {width} of {n}");
            }
        }
    }
}
