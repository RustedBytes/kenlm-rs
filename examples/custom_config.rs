use kenlm::{ArpaLoadComplain, Config, LoadMethod, Model};
use std::env;

fn main() -> Result<(), kenlm::KenlmError> {
    let model_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "lm/test.arpa".to_string());

    let config = Config {
        show_progress: false,
        arpa_complain: ArpaLoadComplain::None,
        load_method: LoadMethod::Lazy,
        ..Config::default()
    };

    let model = Model::with_config(model_path, config)?;

    for word in ["looking", "definitely-not-in-this-model", "<s>", "</s>"] {
        let index = model.index(word)?;
        println!("{word}\tindex={index}\tin_vocab={}", model.contains(word)?);
    }

    Ok(())
}
