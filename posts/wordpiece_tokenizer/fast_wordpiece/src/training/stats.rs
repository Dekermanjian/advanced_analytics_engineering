use std::collections::HashMap;
use std::cmp::Ordering;

type Pair = (u32, u32);

#[derive(Default)]
pub struct Stats {
    pub(crate) token_freq: HashMap<u32, u64>,
    pub(crate) pair_freq: HashMap<Pair, u64>,
}

#[derive(Clone)]
pub struct ScoredPair {
    pub(crate) score: f64,
    pub(crate) pair: Pair,
}

impl Eq for ScoredPair {}

impl PartialEq for ScoredPair {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.pair == other.pair
    }
}

impl Ord for ScoredPair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for ScoredPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}