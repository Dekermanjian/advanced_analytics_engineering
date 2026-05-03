use std::collections::{HashMap, BinaryHeap};
use crate::training::stats::{ScoredPair, Stats};
use crate::training::corpus::Corpus;

fn is_valid_wordpiece_pair(vocab: &[String], _a: u32, b: u32) -> bool {
    let b_tok = &vocab[b as usize];
    b_tok.starts_with("##")
}


fn build_stats(corpus: &Corpus) -> Stats {
    let mut stats = Stats::default();

    for (w, word) in corpus.words.iter().enumerate() {
        let weight = corpus.weights[w] as u64;

        for i in 0..word.len() {
            *stats.token_freq.entry(word[i]).or_default() += weight;

            if i + 1 < word.len() {
                let p = (word[i], word[i + 1]);
                *stats.pair_freq.entry(p).or_default() += weight;
            }
        }
    }

    stats
}

fn build_heap(stats: &Stats, vocab: &[String], min_frequency: u64) -> BinaryHeap<ScoredPair> {
    let mut heap = BinaryHeap::new();

    for (&(a, b), &freq) in &stats.pair_freq {

        if !is_valid_wordpiece_pair(vocab, a, b) {
            continue;
        }

        if freq >= min_frequency {
            let count_a = stats.token_freq[&a] as f64;
            let count_b = stats.token_freq[&b] as f64;

            let score = (freq as f64) / (count_a * count_b);

            heap.push(ScoredPair {
                score: score,
                pair: (a, b),
            });
        }
    }
    heap
}

fn merge_corpus_inplace(corpus: &mut Corpus, a: u32, b: u32, new_id: u32) {
    for word in corpus.words.iter_mut() {
        let n = word.len();
        if n < 2 { continue; }

        let mut write_idx = 0;
        let mut read_idx = 0;

        while read_idx < n {
            if read_idx < n - 1 && word[read_idx] == a && word[read_idx + 1] == b {
                word[write_idx] = new_id;
                read_idx += 2;
            } else {
                word[write_idx] = word[read_idx];
                read_idx += 1;
            }
            write_idx += 1;
        }
        word.truncate(write_idx);
    }
}

pub fn train_wordpiece_internal(
    word_counts: HashMap<String, u32>,
    vocab_size: usize,
    min_frequency: u64,
) -> (HashMap<String, u32>, HashMap<u32, String>, Vec<String>) {
    // Init Vocab and Map
    let mut vocab: Vec<String> = Vec::new();
    let mut vocab_map: HashMap<String, u32> = HashMap::new();
    let mut corpus_words = Vec::new();
    let mut weights = Vec::new();

    let special_tokens = vec!["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
    for tok in special_tokens {
        vocab_map.insert(tok.to_string(), vocab.len() as u32);
        vocab.push(tok.to_string());
    }

    // Initial encoding of strings into IDs
    for (word, count) in word_counts {
        let mut encoded = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        for (i, c) in chars.iter().enumerate() {
            let tok = if i == 0 { c.to_string() } else { format!("##{}", c) };
            let id = *vocab_map.entry(tok.clone()).or_insert_with(|| {
                let id = vocab.len() as u32;
                vocab.push(tok);
                id
            });
            encoded.push(id);
        }
        corpus_words.push(encoded);
        weights.push(count);
    }

    let mut corpus = Corpus { words: corpus_words, weights };

    // Training Loop
    while vocab.len() < vocab_size {
        let stats = build_stats(&corpus);
        let mut heap = build_heap(&stats, &vocab, min_frequency);

        let best = match heap.pop() {
            Some(p) => p,
            None => break,
        };

        let (a, b) = best.pair;

        let a_tok = &vocab[a as usize];
        let b_tok = &vocab[b as usize];

        let new_token = if a_tok.starts_with("##") {
            // continuation + continuation → continuation
            format!("##{}{}", &a_tok[2..], &b_tok[2..])
        } else {
            // start + continuation → start
            format!("{}{}", a_tok, &b_tok[2..])
        };

        if let Some(&existing_id) = vocab_map.get(&new_token) {
            // Token already exists — skip or reuse
            merge_corpus_inplace(&mut corpus, a, b, existing_id);
            continue;
        }

        let new_id = vocab.len() as u32;
        vocab.push(new_token.clone());
        vocab_map.insert(new_token, new_id);

        merge_corpus_inplace(&mut corpus, a, b, new_id);
    }

    let mut final_map = HashMap::new();
    for (i, tok) in vocab.iter().enumerate() {
        final_map.insert(tok.clone(), i as u32);
    }
    
    let mut id_to_token = HashMap::new();
    for (k, v) in final_map.iter() {
        id_to_token.insert(v.clone(), k.clone());
    }


    (final_map, id_to_token, vocab)
}
