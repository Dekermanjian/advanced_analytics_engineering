use crate::tokenizer::encoding::NormalizedString;

pub trait Normalizer: Send + Sync {
    fn normalize(&self, input: &str) -> NormalizedString;
}

pub struct LowercaseNormalizer;

impl Normalizer for LowercaseNormalizer {
    fn normalize(&self, input: &str) -> NormalizedString {
        NormalizedString {
            normalized: input.chars().map(|c| c.to_ascii_lowercase()).collect(),
        }
    }
}