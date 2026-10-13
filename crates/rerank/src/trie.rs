//! The candidates as one token trie: each candidate is
//! `[BOS] + tokens(context + candidate)`, so they share BOS and the context
//! and often a prefix of the candidate too. Each trie node is decoded once,
//! seeing its ancestors only.

/// One trie node: a token at a depth (its position), under `parent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node {
    pub token: i32,
    pub depth: usize,
    pub parent: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Trie {
    /// In insertion order, candidate by candidate: a parent before its
    /// children, and the trie of the first k candidates is a prefix of the
    /// trie of more.
    pub nodes: Vec<Node>,
    /// Each candidate's nodes, root first.
    pub paths: Vec<Vec<usize>>,
}

impl Trie {
    pub fn build(seqs: &[Vec<i32>]) -> Self {
        let mut trie = Trie::default();
        let mut index = std::collections::HashMap::new();
        for seq in seqs {
            let mut path = Vec::with_capacity(seq.len());
            let mut parent = None;
            for (depth, &token) in seq.iter().enumerate() {
                let node = *index.entry((parent, token)).or_insert_with(|| {
                    trie.nodes.push(Node {
                        token,
                        depth,
                        parent,
                    });
                    trie.nodes.len() - 1
                });
                path.push(node);
                parent = Some(node);
            }
            trie.paths.push(path);
        }
        trie
    }

    /// For each node, the candidates whose path goes through it.
    pub fn owners(&self) -> Vec<Vec<i32>> {
        let mut owners = vec![Vec::new(); self.nodes.len()];
        for (c, path) in self.paths.iter().enumerate() {
            for &n in path {
                owners[n].push(c as i32);
            }
        }
        owners
    }
}

pub fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The first scored position: every candidate's tokens from where all of
/// them part from `[BOS] + tokens(context)` (at least 1, after BOS). A
/// context char and the candidate can merge into one token; that token is
/// then scored as the candidate's.
pub fn scoring_start(seqs: &[Vec<i32>], context: &[i32]) -> usize {
    seqs.iter()
        .map(|s| common_prefix(s, context))
        .min()
        .unwrap_or(1)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_prefixes_and_grows_as_a_prefix() {
        let seqs = vec![vec![1, 5, 6, 7], vec![1, 5, 6, 8], vec![1, 5, 9]];
        let t = Trie::build(&seqs);
        // 1 5 6 7 | 8 | 9
        assert_eq!(t.nodes.len(), 6);
        assert_eq!(t.paths[1], [0, 1, 2, 4]);
        assert_eq!(t.paths[2], [0, 1, 5]);
        assert_eq!(t.nodes[5].parent, Some(1));
        assert_eq!(t.nodes[5].depth, 2);
        let fewer = Trie::build(&seqs[..2]);
        assert_eq!(t.nodes[..fewer.nodes.len()], fewer.nodes[..]);
        assert_eq!(t.owners()[2], [0, 1]);
        assert_eq!(t.owners()[0], [0, 1, 2]);
    }

    #[test]
    fn scoring_starts_where_the_candidates_leave_the_context() {
        let context = vec![1, 5, 6];
        // the second merges the last context token with its first char
        let seqs = vec![vec![1, 5, 6, 7], vec![1, 5, 9, 7]];
        assert_eq!(scoring_start(&seqs, &context), 2);
        assert_eq!(scoring_start(&seqs[..1], &context), 3);
        // no context: after BOS
        assert_eq!(scoring_start(&[vec![1, 7]], &[1]), 1);
    }
}
