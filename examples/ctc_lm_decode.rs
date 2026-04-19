use kenlm::Model;
use std::cmp::Ordering;
use std::collections::HashMap;

const NEG_INF: f32 = f32::NEG_INFINITY;

#[derive(Debug, Clone)]
struct Candidate {
    text: String,
    ctc_log_prob: f32,
    lm_log_prob: f32,
    total_score: f32,
}

#[derive(Debug, Clone, Copy)]
struct PrefixScore {
    blank: f32,
    non_blank: f32,
}

impl PrefixScore {
    fn total(self) -> f32 {
        log_add(self.blank, self.non_blank)
    }
}

fn main() -> Result<(), kenlm::KenlmError> {
    let model = Model::new("lm/test.arpa")?;

    // Vocabulary index 0 is the CTC blank. The remaining tokens are character
    // pieces here, but the same decoder works with BPE/wordpiece tokens if
    // `tokens_to_text` matches your tokenizer.
    let vocab = ["", "l", "o", "k", "i", "n", "g", " ", "a", "t", "e"];
    let blank_id = 0;

    // Replace this with your model output after log_softmax:
    // shape = [time_steps][vocab_size], values = natural log probabilities.
    let log_probs = toy_log_probs(vocab.len());

    let candidates = ctc_prefix_beam_search(
        &log_probs,
        &vocab,
        blank_id,
        32,   // keep this many partial prefixes per frame
        8,    // return this many final CTC candidates
        &model,
        0.45, // LM weight; tune on validation data
        0.2,  // word insertion bonus; tune on validation data
    )?;

    for candidate in &candidates {
        println!(
            "{:.3}\tctc={:.3}\tlm={:.3}\t{}",
            candidate.total_score, candidate.ctc_log_prob, candidate.lm_log_prob, candidate.text
        );
    }

    if let Some(best) = candidates.first() {
        println!("\nbest: {}", best.text);
    }

    Ok(())
}

fn ctc_prefix_beam_search(
    log_probs: &[Vec<f32>],
    vocab: &[&str],
    blank_id: usize,
    beam_size: usize,
    n_best: usize,
    lm: &Model,
    lm_weight: f32,
    word_bonus: f32,
) -> Result<Vec<Candidate>, kenlm::KenlmError> {
    let mut beam: HashMap<Vec<usize>, PrefixScore> = HashMap::new();
    beam.insert(
        Vec::new(),
        PrefixScore {
            blank: 0.0,
            non_blank: NEG_INF,
        },
    );

    for frame in log_probs {
        assert_eq!(frame.len(), vocab.len(), "frame/vocab size mismatch");

        let mut next: HashMap<Vec<usize>, PrefixScore> = HashMap::new();

        for (prefix, score) in &beam {
            for (token_id, &token_log_prob) in frame.iter().enumerate() {
                if token_id == blank_id {
                    let entry = next.entry(prefix.clone()).or_insert(PrefixScore {
                        blank: NEG_INF,
                        non_blank: NEG_INF,
                    });
                    entry.blank = log_add(entry.blank, score.total() + token_log_prob);
                    continue;
                }

                let last = prefix.last().copied();

                if Some(token_id) == last {
                    let entry = next.entry(prefix.clone()).or_insert(PrefixScore {
                        blank: NEG_INF,
                        non_blank: NEG_INF,
                    });
                    entry.non_blank = log_add(entry.non_blank, score.non_blank + token_log_prob);

                    let mut extended = prefix.clone();
                    extended.push(token_id);
                    let entry = next.entry(extended).or_insert(PrefixScore {
                        blank: NEG_INF,
                        non_blank: NEG_INF,
                    });
                    entry.non_blank = log_add(entry.non_blank, score.blank + token_log_prob);
                } else {
                    let mut extended = prefix.clone();
                    extended.push(token_id);
                    let entry = next.entry(extended).or_insert(PrefixScore {
                        blank: NEG_INF,
                        non_blank: NEG_INF,
                    });
                    entry.non_blank = log_add(entry.non_blank, score.total() + token_log_prob);
                }
            }
        }

        let mut pruned = next.into_iter().collect::<Vec<_>>();
        pruned.sort_by(|a, b| cmp_f32_desc(a.1.total(), b.1.total()));
        pruned.truncate(beam_size);
        beam = pruned.into_iter().collect();
    }

    let mut candidates = beam
        .into_iter()
        .filter_map(|(tokens, score)| {
            let text = normalize_spaces(&tokens_to_text(&tokens, vocab));
            if text.is_empty() {
                None
            } else {
                Some((text, score.total()))
            }
        })
        .collect::<Vec<_>>();

    // Multiple token paths can normalize to the same text. Keep the best CTC
    // score for each visible sentence.
    let mut deduped: HashMap<String, f32> = HashMap::new();
    for (text, ctc_log_prob) in candidates.drain(..) {
        deduped
            .entry(text)
            .and_modify(|best| *best = best.max(ctc_log_prob))
            .or_insert(ctc_log_prob);
    }

    let mut reranked = Vec::new();
    for (text, ctc_log_prob) in deduped {
        // KenLM returns log10. Convert to natural log so it combines with CTC
        // log probabilities from log_softmax.
        let lm_log_prob = lm.score(&text, true, true)? * std::f32::consts::LN_10;
        let words = text.split_whitespace().count() as f32;
        let total_score = ctc_log_prob + lm_weight * lm_log_prob + word_bonus * words;

        reranked.push(Candidate {
            text,
            ctc_log_prob,
            lm_log_prob,
            total_score,
        });
    }

    reranked.sort_by(|a, b| cmp_f32_desc(a.total_score, b.total_score));
    reranked.truncate(n_best);
    Ok(reranked)
}

fn tokens_to_text(tokens: &[usize], vocab: &[&str]) -> String {
    tokens.iter().map(|&id| vocab[id]).collect()
}

fn normalize_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn log_add(a: f32, b: f32) -> f32 {
    if a == NEG_INF {
        return b;
    }
    if b == NEG_INF {
        return a;
    }

    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

fn cmp_f32_desc(a: f32, b: f32) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

fn toy_log_probs(vocab_size: usize) -> Vec<Vec<f32>> {
    let ids = [1, 2, 2, 3, 4, 5, 6, 7, 2, 5, 7, 8, 7, 1, 4, 9, 9, 1, 10];
    ids.iter()
        .map(|&best_id| {
            let mut frame = vec![(0.01_f32).ln(); vocab_size];
            frame[best_id] = (0.82_f32).ln();
            frame[0] = (0.12_f32).ln();
            frame
        })
        .collect()
}
