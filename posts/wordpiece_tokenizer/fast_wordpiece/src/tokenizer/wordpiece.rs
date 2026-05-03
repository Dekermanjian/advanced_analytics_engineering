use crate::tokenizer::core::*;
use crate::tokenizer::encoding::*;
use regex::Regex;

impl Tokenizer {
    fn wordpiece(
        &self,
        token: &PreToken,
    ) -> (Vec<String>, Vec<u32>, Vec<(usize, usize)>) {
        let chars: Vec<(usize, char)> = token.text.char_indices().collect();

        let mut i = 0;
        let mut tokens = Vec::new();
        let mut ids = Vec::new();
        let mut offsets = Vec::new();

        while i < chars.len() {
            let mut end = chars.len();
            let mut found = None;

            while end > i {
                let substr: String = chars[i..end].iter().map(|(_, c)| *c).collect();

                let tok = if i == 0 {
                    substr.clone()
                } else {
                    format!("##{}", substr)
                };

                if let Some(&id) = self.token_to_id.get(&tok) {
                    let start = token.start + chars[i].0;
                    let end_b = if end < chars.len() {
                        token.start + chars[end].0
                    } else {
                        token.start + token.text.len()
                    };


                    found = Some((tok, id, end, start, end_b));
                    break;
                }

                end -= 1;
            }

            if let Some((tok, id, next_i, start, end_b)) = found {
                tokens.push(tok);
                ids.push(id);
                offsets.push((start, end_b));
                i = next_i;
            } else {
                let unk_id = *self.token_to_id
                    .get(&self.unk_token)
                    .expect("UNK missing");

                return (
                    vec![self.unk_token.clone()],
                    vec![unk_id],
                    vec![(token.start, token.end)],
                );
            }
        }

        (tokens, ids, offsets)
    }
}

impl Tokenizer {
    pub fn encode(&self, text: &str) -> Encoding {
        // 1. normalize
        let normalized = self.normalizer.normalize(text);

        // 2. pretokenize
        let pretokens = self.pretokenizer.pretokenize(&normalized);

        // 3. wordpiece
        let mut tokens = Vec::new();
        let mut ids = Vec::new();
        let mut offsets = Vec::new();

        for pt in pretokens {
            let (t, i, o) = self.wordpiece(&pt);
            tokens.extend(t);
            ids.extend(i);
            offsets.extend(o);
        }

        let mut encoding = Encoding {
            ids,
            tokens,
            offsets,
            attention_mask: vec![],
            special_tokens_mask: vec![]
        };

        // 4. post-process
        encoding = self.postprocessor.process(encoding, self);

        // 5. attention mask
        encoding.attention_mask = vec![1; encoding.ids.len()];

        encoding
    }

    pub fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> String {
        let mut decoded = String::new();

        let tokens = ids
            .into_iter()
            .filter(|id| !skip_special_tokens || *id > 4)
            .map(|id| &self.id_to_token[&id]);

        for (i, token) in tokens.enumerate() {
            if token.starts_with("##") {
                decoded.push_str(&token[2..]);
            } else {
                if i > 0 {
                    decoded.push(' ');
                }
                decoded.push_str(token);
            }
        }

        // --- Post-processing (compile once ideally; shown inline for clarity) ---
        let re_punct = Regex::new(r"\s+([.,!?;:])").unwrap();
        let re_apos = Regex::new(r"\s+'\s+").unwrap();
        let re_hyphen = Regex::new(r"\s+-\s+").unwrap();
        let re_lparen = Regex::new(r"\(\s+").unwrap();
        let re_rparen = Regex::new(r"\s+\)").unwrap();
        let re_lbracket = Regex::new(r"\s+\[").unwrap();
        let re_rbracket = Regex::new(r"\]\s+").unwrap();

        let mut decoded = re_punct.replace_all(&decoded, "$1").to_string();
        decoded = re_apos.replace_all(&decoded, "'").to_string();
        decoded = re_hyphen.replace_all(&decoded, "-").to_string();
        decoded = re_lparen.replace_all(&decoded, "(").to_string();
        decoded = re_rparen.replace_all(&decoded, ")").to_string();
        decoded = re_lbracket.replace_all(&decoded, "[").to_string();
        decoded = re_rbracket.replace_all(&decoded, "]").to_string();

        decoded.trim().to_string()
    }

}