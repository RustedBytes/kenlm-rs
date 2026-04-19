use kenlm::Model;
use std::env;

fn main() -> Result<(), kenlm::KenlmError> {
    let mut args = env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| "lm/test.arpa".to_string());
    let words = args.collect::<Vec<_>>();
    let words = if words.is_empty() {
        vec!["looking".to_string(), "on".to_string(), "a".to_string()]
    } else {
        words
    };

    let model = Model::new(model_path)?;
    let mut state = model.begin_sentence_state();
    let mut next = model.null_context_state();
    let mut total = 0.0;

    for word in &words {
        let word_index = model.index(word)?;
        let full = model.base_full_score(&state, word_index, &mut next)?;
        total += full.log_prob;
        println!(
            "{word}\tindex={word_index}\tlog10={:.6}\tngram_length={}",
            full.log_prob, full.ngram_length
        );
        std::mem::swap(&mut state, &mut next);
    }

    let eos = model.base_full_score(&state, model.end_sentence_index(), &mut next)?;
    total += eos.log_prob;
    println!(
        "</s>\tindex={}\tlog10={:.6}\tngram_length={}",
        model.end_sentence_index(),
        eos.log_prob,
        eos.ngram_length
    );
    println!("total: {total}");

    Ok(())
}
