use crate::tokenizer::encoding::{NormalizedString, PreToken};
use regex::Regex;
use std::sync::OnceLock;

pub trait PreTokenizer: Send + Sync {
    fn pretokenize(&self, input: &NormalizedString) -> Vec<PreToken>;
}

pub struct WhitespacePreTokenizer;

impl PreTokenizer for WhitespacePreTokenizer {
    fn pretokenize(&self, input: &NormalizedString) -> Vec<PreToken> {
        let text = &input.normalized;
        let mut out = Vec::new();

        let mut start = None;

        for (i, c) in text.char_indices() {
            if c.is_whitespace() {
                if let Some(s) = start {
                    out.push(PreToken {
                        text: text[s..i].to_string(),
                        start: s,
                        end: i,
                    });
                    start = None;
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }

        if let Some(s) = start {
            out.push(PreToken {
                text: text[s..].to_string(),
                start: s,
                end: text.len(),
            });
        }

        out
    }
}


pub struct BertPreTokenizer;
static BERT_REGEX: OnceLock<Regex> = OnceLock::new();

impl PreTokenizer for BertPreTokenizer {
    fn pretokenize(&self, input: &NormalizedString) -> Vec<PreToken> {
        let text = &input.normalized;
        let re = BERT_REGEX.get_or_init(|| {
            Regex::new(r"\w+|[^\w\s]").unwrap()
        });

        re.find_iter(text)
            .map(|m| PreToken {
                text: m.as_str().to_string(),
                start: m.start(),
                end: m.end(),
            })
            .collect()
    }
}