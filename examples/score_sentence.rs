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
    let score = model.score(sentence, true, true)?;
    let fragment_score = model.score(sentence, false, false)?;
    let perplexity = model.perplexity(sentence)?;

    println!("sentence: {sentence}");
    println!("order: {}", model.order());
    println!("score with <s> and </s>: {score}");
    println!("fragment score: {fragment_score}");
    println!("perplexity: {perplexity}");

    Ok(())
}
