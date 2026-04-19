use kenlm::Model;
use std::env;

fn main() -> Result<(), kenlm::KenlmError> {
    let mut args = env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| "lm/test.arpa".to_string());
    let sentence = args.collect::<Vec<_>>().join(" ");
    let sentence = if sentence.is_empty() {
        "looking on a little"
    } else {
        sentence.as_str()
    };

    let model = Model::new(model_path)?;
    let scores = model.full_scores(sentence, true, true)?;

    for (word, score) in sentence
        .split_whitespace()
        .chain(std::iter::once("</s>"))
        .zip(scores)
    {
        println!(
            "{word}\tlog10={:.6}\tngram_length={}\toov={}",
            score.log_prob, score.ngram_length, score.oov
        );
    }

    Ok(())
}
