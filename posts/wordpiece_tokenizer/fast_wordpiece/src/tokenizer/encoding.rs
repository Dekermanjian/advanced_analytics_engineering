pub struct NormalizedString {
    pub(crate) normalized: String,
}

pub struct PreToken {
    pub(crate) text: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

pub struct Encoding {
    pub(crate) ids: Vec<u32>,
    pub(crate) tokens: Vec<String>,
    pub(crate) offsets: Vec<(usize, usize)>,

    pub(crate) attention_mask: Vec<u8>,
    pub(crate) special_tokens_mask: Vec<u8>,

}