use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::tokenizer::core::Tokenizer;
use crate::pipeline::*;
use crate::training::trainer::train_wordpiece_internal;

#[pyclass]
pub struct PyTokenizer {
    tokenizer: Tokenizer,
}

#[pymethods]
impl PyTokenizer {

    #[staticmethod]
    pub fn train(
        corpus: Bound<'_, PyList>,
        vocab_size: usize,
        min_frequency: u64,
    ) -> Self {

        let mut word_counts: HashMap<String, u32> = HashMap::new();
        let normalizer = LowercaseNormalizer;
        let pretokenizer = BertPreTokenizer;

        for item in corpus.iter() {
            let text: &str = item.extract().unwrap();
            let normalized = normalizer.normalize(text);
            let tokens = pretokenizer.pretokenize(&normalized);

            for token in tokens {
                let count = word_counts.entry(token.text.clone()).or_insert(0);
                *count += 1;
            }
        }

        let (token_to_id, id_to_token, vocab) =
            train_wordpiece_internal(word_counts, vocab_size, min_frequency);

        let tokenizer = Tokenizer {
            token_to_id: token_to_id,
            id_to_token: id_to_token,
            vocab: vocab,
            normalizer: Box::new(LowercaseNormalizer),
            pretokenizer: Box::new(BertPreTokenizer),
            postprocessor: Box::new(BertPostProcessor),
            unk_token: "[UNK]".to_string(),
            cls_token: "[CLS]".to_string(),
            sep_token: "[SEP]".to_string(),
            pad_token: "[PAD]".to_string(),
        };

        Self { tokenizer }
    }

    pub fn encode(&self, py: Python, text: &str) -> PyObject {
        let enc = self.tokenizer.encode(text);

        let dict = PyDict::new(py);

        dict.set_item("ids", &enc.ids).unwrap();
        dict.set_item("tokens", &enc.tokens).unwrap();
        dict.set_item("offsets", &enc.offsets).unwrap();
        
        let attention_mask: Vec<u32> = enc.attention_mask.iter().map(|&x| x as u32).collect();
        let special_tokens_mask: Vec<u32> = enc.special_tokens_mask.iter().map(|&x| x as u32).collect();

        dict.set_item("attention_mask", attention_mask).unwrap();
        dict.set_item("special_tokens_mask", special_tokens_mask).unwrap();

        dict.into()
    }


    pub fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> String {
        self.tokenizer.decode(ids, skip_special_tokens)
    }

    pub fn get_vocab<'py>(&self, py: Python<'py>) -> PyObject {
        let dict = PyDict::new(py);

        for (k, v) in &self.tokenizer.token_to_id {
            dict.set_item(k, v).unwrap();
        }

        dict.into()
    }

}