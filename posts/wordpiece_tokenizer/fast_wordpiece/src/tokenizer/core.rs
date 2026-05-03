use std::collections::HashMap;
use crate::pipeline::{Normalizer, PreTokenizer, PostProcessor};

pub struct Tokenizer {
    pub(crate) token_to_id: HashMap<String, u32>,
    pub(crate) id_to_token: HashMap<u32, String>,
    pub(crate) vocab: Vec<String>,

    // pipeline components
    pub(crate) normalizer: Box<dyn Normalizer + Send + Sync>,
    pub(crate) pretokenizer: Box<dyn PreTokenizer + Send + Sync>,
    pub(crate) postprocessor: Box<dyn PostProcessor + Send + Sync>,

    // config
    pub(crate) unk_token: String,
    pub(crate) cls_token: String,
    pub(crate) sep_token: String,
    pub(crate) pad_token: String,
}