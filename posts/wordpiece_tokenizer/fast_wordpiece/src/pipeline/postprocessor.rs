use crate::tokenizer::encoding::{Encoding};
use crate::tokenizer::core::{Tokenizer};

pub trait PostProcessor: Send + Sync {
    fn process(&self, encoding: Encoding, tokenizer: &Tokenizer) -> Encoding;
}

pub struct BertPostProcessor;

impl PostProcessor for BertPostProcessor {
    fn process(&self, mut enc: Encoding, tok: &Tokenizer) -> Encoding {
        let cls_id = tok.token_to_id[&tok.cls_token];
        let sep_id = tok.token_to_id[&tok.sep_token];

        enc.tokens.insert(0, tok.cls_token.clone());
        enc.ids.insert(0, cls_id);
        enc.offsets.insert(0, (0, 0));
        enc.special_tokens_mask.insert(0, 1);

        enc.tokens.push(tok.sep_token.clone());
        enc.ids.push(sep_id);
        enc.offsets.push((0, 0));
        enc.special_tokens_mask.push(1);

        // mark non-specials
        let mut mask: Vec<u8> = vec![0; enc.ids.len()];
        mask[0] = 1;
        mask[enc.ids.len() - 1] = 1;
        enc.special_tokens_mask = mask;

        enc
    }
}