#[derive(Clone)]
pub struct Corpus {
    pub(crate) words: Vec<Vec<u32>>,
    pub(crate) weights: Vec<u32>,
}