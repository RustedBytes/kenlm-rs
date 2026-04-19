//! Safe Rust bindings for KenLM language model inference.
//!
//! The crate compiles KenLM's query-only C++ sources and exposes a small Rust
//! API for loading ARPA or binary models, scoring sentences, inspecting
//! vocabulary membership, and doing explicit stateful scoring.
//!
//! # Safety model
//!
//! The public API is safe Rust. C++ exceptions are caught in the C++ shim and
//! converted into [`KenlmError`] values. Opaque [`State`] values carry an
//! internal model identity token, so stateful scoring rejects states created by
//! a different [`Model`] before calling into KenLM. The raw KenLM model handle
//! is owned by [`Model`] and released exactly once in `Drop`.

use std::error::Error;
use std::ffi::{CStr, CString, NulError};
use std::fmt;
use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

#[cfg(any(
    feature = "tools",
    feature = "estimation",
    feature = "filter",
    feature = "interpolate"
))]
pub mod commands;

/// Crate-local result type.
pub type Result<T> = std::result::Result<T, KenlmError>;

/// KenLM vocabulary index.
pub type WordIndex = u32;

#[repr(C)]
struct RawModel {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawConfig {
    load_method: c_int,
    arpa_complain: c_int,
    probing_multiplier: c_float,
    unknown_missing_logprob: c_float,
    show_progress: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawFullScore {
    prob: c_float,
    ngram_length: u8,
    independent_left: u8,
    extend_left: u64,
    rest: c_float,
}

extern "C" {
    fn kenlm_config_default(config: *mut RawConfig);
    fn kenlm_model_load(path: *const c_char, config: *const RawConfig) -> *mut RawModel;
    fn kenlm_model_free(model: *mut RawModel);
    fn kenlm_last_error() -> *const c_char;
    fn kenlm_model_state_size(model: *const RawModel) -> usize;
    fn kenlm_model_order(model: *const RawModel) -> u8;
    fn kenlm_model_begin_sentence_write(model: *const RawModel, state: *mut c_void);
    fn kenlm_model_null_context_write(model: *const RawModel, state: *mut c_void);
    fn kenlm_model_try_index(
        model: *const RawModel,
        word: *const c_char,
        out: *mut c_uint,
    ) -> c_int;
    fn kenlm_model_begin_sentence_index(model: *const RawModel) -> c_uint;
    fn kenlm_model_end_sentence_index(model: *const RawModel) -> c_uint;
    fn kenlm_model_not_found_index(model: *const RawModel) -> c_uint;
    fn kenlm_model_try_base_score(
        model: *const RawModel,
        in_state: *const c_void,
        word: c_uint,
        out_state: *mut c_void,
        out: *mut c_float,
    ) -> c_int;
    fn kenlm_model_try_base_full_score(
        model: *const RawModel,
        in_state: *const c_void,
        word: c_uint,
        out_state: *mut c_void,
        out: *mut RawFullScore,
    ) -> c_int;
}

/// Errors returned by the KenLM bindings.
#[derive(Debug)]
pub enum KenlmError {
    /// A path or word contained an interior NUL byte and cannot cross the C ABI.
    InteriorNul(NulError),
    /// KenLM could not load the requested model.
    Load(String),
    /// A state created by one model was used with another model.
    StateModelMismatch,
}

impl fmt::Display for KenlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KenlmError::InteriorNul(error) => {
                write!(f, "string contains an interior NUL byte: {error}")
            }
            KenlmError::Load(error) => f.write_str(error),
            KenlmError::StateModelMismatch => {
                f.write_str("KenLM state was created by a different model")
            }
        }
    }
}

impl Error for KenlmError {}

impl From<NulError> for KenlmError {
    fn from(value: NulError) -> Self {
        KenlmError::InteriorNul(value)
    }
}

/// How KenLM should bring binary model data into memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum LoadMethod {
    Lazy = 0,
    PopulateOrLazy = 1,
    PopulateOrRead = 2,
    Read = 3,
    ParallelRead = 4,
}

/// How loudly KenLM should complain about loading ARPA instead of binary data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum ArpaLoadComplain {
    All = 0,
    Expensive = 1,
    None = 2,
}

/// Runtime loading options for KenLM models.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub load_method: LoadMethod,
    pub arpa_complain: ArpaLoadComplain,
    pub probing_multiplier: f32,
    pub unknown_missing_logprob: f32,
    pub show_progress: bool,
}

impl Config {
    fn as_raw(self) -> RawConfig {
        RawConfig {
            load_method: self.load_method as c_int,
            arpa_complain: self.arpa_complain as c_int,
            probing_multiplier: self.probing_multiplier,
            unknown_missing_logprob: self.unknown_missing_logprob,
            show_progress: u8::from(self.show_progress),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        let mut raw = RawConfig {
            load_method: LoadMethod::Lazy as c_int,
            arpa_complain: ArpaLoadComplain::All as c_int,
            probing_multiplier: 1.5,
            unknown_missing_logprob: -100.0,
            show_progress: 1,
        };
        // SAFETY: `raw` points to a valid, writable `RawConfig` with the same C
        // layout as `KenlmConfig`. The C++ wrapper only writes scalar fields.
        unsafe {
            kenlm_config_default(&mut raw);
        }
        Self {
            load_method: match raw.load_method {
                1 => LoadMethod::PopulateOrLazy,
                2 => LoadMethod::PopulateOrRead,
                3 => LoadMethod::Read,
                4 => LoadMethod::ParallelRead,
                _ => LoadMethod::Lazy,
            },
            arpa_complain: match raw.arpa_complain {
                1 => ArpaLoadComplain::Expensive,
                2 => ArpaLoadComplain::None,
                _ => ArpaLoadComplain::All,
            },
            probing_multiplier: raw.probing_multiplier,
            unknown_missing_logprob: raw.unknown_missing_logprob,
            show_progress: raw.show_progress != 0,
        }
    }
}

/// A KenLM model loaded from an ARPA or KenLM binary file.
pub struct Model {
    raw: NonNull<RawModel>,
    state_size: usize,
    token: Arc<ModelToken>,
}

#[derive(Debug)]
struct ModelToken;

// KenLM model scoring is read-only after construction. Callers provide separate
// state buffers for each transition, so sharing a loaded model is safe.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Load a language model with default configuration.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(path, Config::default())
    }

    /// Load a language model with explicit configuration.
    pub fn with_config(path: impl AsRef<Path>, config: Config) -> Result<Self> {
        let path = path.as_ref().as_os_str().to_string_lossy();
        let path = CString::new(path.as_bytes())?;
        let raw_config = config.as_raw();
        // SAFETY: `path` and `raw_config` are valid for the duration of the call.
        // The C++ wrapper catches exceptions and returns null on failure.
        let raw = unsafe { kenlm_model_load(path.as_ptr(), &raw_config) };
        let raw = NonNull::new(raw).ok_or_else(last_error)?;
        // SAFETY: `raw` is a non-null KenLM handle returned by
        // `kenlm_model_load` and remains owned by `Self` until `Drop`.
        let state_size = unsafe { kenlm_model_state_size(raw.as_ptr()) };
        Ok(Self {
            raw,
            state_size,
            token: Arc::new(ModelToken),
        })
    }

    /// Return the n-gram order of the model.
    pub fn order(&self) -> u8 {
        // SAFETY: `self.raw` is a live KenLM handle for the lifetime of `self`.
        unsafe { kenlm_model_order(self.raw.as_ptr()) }
    }

    /// Return true when `word` exists in the model vocabulary.
    pub fn contains(&self, word: &str) -> Result<bool> {
        Ok(self.index(word)? != self.not_found_index())
    }

    /// Return KenLM's vocabulary index for `word`, or the not-found index for OOV words.
    pub fn index(&self, word: &str) -> Result<WordIndex> {
        let word = CString::new(word)?;
        // SAFETY: `self.raw` is live and `word` is a valid NUL-terminated C
        // string for the duration of the call.
        let mut index = 0;
        let status = unsafe { kenlm_model_try_index(self.raw.as_ptr(), word.as_ptr(), &mut index) };
        if status == 0 {
            Ok(index as WordIndex)
        } else {
            Err(last_error())
        }
    }

    /// Return the index for `<s>`.
    pub fn begin_sentence_index(&self) -> WordIndex {
        // SAFETY: `self.raw` is a live KenLM handle for the lifetime of `self`.
        unsafe { kenlm_model_begin_sentence_index(self.raw.as_ptr()) as WordIndex }
    }

    /// Return the index for `</s>`.
    pub fn end_sentence_index(&self) -> WordIndex {
        // SAFETY: `self.raw` is a live KenLM handle for the lifetime of `self`.
        unsafe { kenlm_model_end_sentence_index(self.raw.as_ptr()) as WordIndex }
    }

    /// Return the vocabulary index used for out-of-vocabulary words.
    pub fn not_found_index(&self) -> WordIndex {
        // SAFETY: `self.raw` is a live KenLM handle for the lifetime of `self`.
        unsafe { kenlm_model_not_found_index(self.raw.as_ptr()) as WordIndex }
    }

    /// Score a whitespace-tokenized sentence, returning log10 probability.
    ///
    /// With `bos = true` and `eos = true`, this returns
    /// `log10 p(sentence </s> | <s>)`.
    pub fn score(&self, sentence: &str, bos: bool, eos: bool) -> Result<f32> {
        self.score_words(sentence.split_whitespace(), bos, eos)
    }

    /// Score pre-tokenized words, returning log10 probability.
    pub fn score_words<'a>(
        &self,
        words: impl IntoIterator<Item = &'a str>,
        bos: bool,
        eos: bool,
    ) -> Result<f32> {
        let mut state = self.initial_state(bos);
        let mut next = self.empty_state();
        let mut total = 0.0;

        for word in words {
            let index = self.index(word)?;
            total += self.base_score(&state, index, &mut next)?;
            std::mem::swap(&mut state, &mut next);
        }

        if eos {
            total += self.base_score(&state, self.end_sentence_index(), &mut next)?;
        }

        Ok(total)
    }

    /// Return perplexity for a complete whitespace-tokenized sentence.
    pub fn perplexity(&self, sentence: &str) -> Result<f32> {
        let words = sentence.split_whitespace().count() + 1;
        Ok(10.0_f32.powf(-self.score(sentence, true, true)? / words as f32))
    }

    /// Return per-token full scores for a whitespace-tokenized sentence.
    pub fn full_scores(&self, sentence: &str, bos: bool, eos: bool) -> Result<Vec<TokenScore>> {
        self.full_scores_words(sentence.split_whitespace(), bos, eos)
    }

    /// Return per-token full scores for pre-tokenized words.
    pub fn full_scores_words<'a>(
        &self,
        words: impl IntoIterator<Item = &'a str>,
        bos: bool,
        eos: bool,
    ) -> Result<Vec<TokenScore>> {
        let mut state = self.initial_state(bos);
        let mut next = self.empty_state();
        let mut scores = Vec::new();

        for word in words {
            let index = self.index(word)?;
            let full_score = self.base_full_score(&state, index, &mut next)?;
            scores.push(TokenScore {
                log_prob: full_score.log_prob,
                ngram_length: full_score.ngram_length,
                oov: index == self.not_found_index(),
            });
            std::mem::swap(&mut state, &mut next);
        }

        if eos {
            let full_score = self.base_full_score(&state, self.end_sentence_index(), &mut next)?;
            scores.push(TokenScore {
                log_prob: full_score.log_prob,
                ngram_length: full_score.ngram_length,
                oov: false,
            });
        }

        Ok(scores)
    }

    /// Create a state initialized to beginning-of-sentence context.
    pub fn begin_sentence_state(&self) -> State {
        let mut state = self.empty_state();
        // SAFETY: `state` is exactly `self.state_size` bytes and belongs to
        // this model. KenLM writes a POD state into the provided buffer.
        unsafe {
            kenlm_model_begin_sentence_write(self.raw.as_ptr(), state.as_mut_ptr());
        }
        state
    }

    /// Create a state initialized to null context.
    pub fn null_context_state(&self) -> State {
        let mut state = self.empty_state();
        // SAFETY: `state` is exactly `self.state_size` bytes and belongs to
        // this model. KenLM writes a POD state into the provided buffer.
        unsafe {
            kenlm_model_null_context_write(self.raw.as_ptr(), state.as_mut_ptr());
        }
        state
    }

    /// Score `word_index` from `in_state`, writing the next state into `out_state`.
    pub fn base_score(
        &self,
        in_state: &State,
        word_index: WordIndex,
        out_state: &mut State,
    ) -> Result<f32> {
        self.validate_state(in_state)?;
        self.validate_state(out_state)?;
        // Safe Rust prevents passing the exact same `State` as both `&State`
        // and `&mut State`; KenLM additionally requires distinct buffers.
        debug_assert!(!std::ptr::eq(in_state.as_ptr(), out_state.as_ptr()));
        let mut score = 0.0;
        // SAFETY: states were created by this model and have the exact byte
        // size KenLM reported. Input and output buffers are distinct. The C++
        // wrapper catches exceptions and reports them through its status code.
        let status = unsafe {
            kenlm_model_try_base_score(
                self.raw.as_ptr(),
                in_state.as_ptr(),
                word_index as c_uint,
                out_state.as_mut_ptr(),
                &mut score,
            )
        };
        if status == 0 {
            Ok(score)
        } else {
            Err(last_error())
        }
    }

    /// Return KenLM's full score metadata for a state transition.
    pub fn base_full_score(
        &self,
        in_state: &State,
        word_index: WordIndex,
        out_state: &mut State,
    ) -> Result<FullScore> {
        self.validate_state(in_state)?;
        self.validate_state(out_state)?;
        // Safe Rust prevents passing the exact same `State` as both `&State`
        // and `&mut State`; KenLM additionally requires distinct buffers.
        debug_assert!(!std::ptr::eq(in_state.as_ptr(), out_state.as_ptr()));
        let mut raw = RawFullScore {
            prob: 0.0,
            ngram_length: 0,
            independent_left: 0,
            extend_left: 0,
            rest: 0.0,
        };
        // SAFETY: states were created by this model and have the exact byte
        // size KenLM reported. Input and output buffers are distinct. The C++
        // wrapper catches exceptions and reports them through its status code.
        let status = unsafe {
            kenlm_model_try_base_full_score(
                self.raw.as_ptr(),
                in_state.as_ptr(),
                word_index as c_uint,
                out_state.as_mut_ptr(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error());
        }
        Ok(FullScore {
            log_prob: raw.prob,
            ngram_length: raw.ngram_length,
            independent_left: raw.independent_left != 0,
            extend_left: raw.extend_left,
            rest: raw.rest,
        })
    }

    fn initial_state(&self, bos: bool) -> State {
        if bos {
            self.begin_sentence_state()
        } else {
            self.null_context_state()
        }
    }

    fn empty_state(&self) -> State {
        State {
            bytes: vec![0; self.state_size],
            owner: Arc::clone(&self.token),
        }
    }

    fn validate_state(&self, state: &State) -> Result<()> {
        if state.bytes.len() != self.state_size || !Arc::ptr_eq(&state.owner, &self.token) {
            return Err(KenlmError::StateModelMismatch);
        }
        Ok(())
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: `self.raw` was returned by `kenlm_model_load` and has not
        // been freed yet. `Drop` runs exactly once for `Model`.
        unsafe {
            kenlm_model_free(self.raw.as_ptr());
        }
    }
}

/// Opaque KenLM state memory used for incremental scoring.
#[derive(Clone, Debug)]
pub struct State {
    bytes: Vec<u8>,
    owner: Arc<ModelToken>,
}

impl State {
    fn as_ptr(&self) -> *const c_void {
        self.bytes.as_ptr().cast()
    }

    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.bytes.as_mut_ptr().cast()
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner) && self.bytes == other.bytes
    }
}

impl Eq for State {}

/// Detailed score metadata for one state transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FullScore {
    pub log_prob: f32,
    pub ngram_length: u8,
    pub independent_left: bool,
    pub extend_left: u64,
    pub rest: f32,
}

/// Sentence-level per-token score, including OOV metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenScore {
    pub log_prob: f32,
    pub ngram_length: u8,
    pub oov: bool,
}

fn last_error() -> KenlmError {
    // SAFETY: `kenlm_last_error` returns a pointer to thread-local storage in
    // the C++ wrapper. It is valid until the next wrapper call on this thread.
    let message = unsafe {
        let ptr = kenlm_last_error();
        if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    if message.is_empty() {
        KenlmError::Load("unknown KenLM error".to_string())
    } else {
        KenlmError::Load(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_and_scores_test_model() {
        let config = Config {
            show_progress: false,
            ..Config::default()
        };
        let model = Model::with_config("lm/test.arpa", config).unwrap();

        assert!(model.order() > 0);
        assert!(model.contains("looking").unwrap());
        assert!(!model.contains("definitely-not-in-this-model").unwrap());

        let score = model.score("looking on a little", true, true).unwrap();
        assert!(score.is_finite());

        let full_scores = model
            .full_scores("looking on a little", true, true)
            .unwrap();
        assert_eq!(full_scores.len(), 5);
        assert!(full_scores.iter().all(|score| score.log_prob.is_finite()));
    }

    #[test]
    fn supports_stateful_scoring() {
        let config = Config {
            show_progress: false,
            ..Config::default()
        };
        let model = Model::with_config("lm/test.arpa", config).unwrap();

        let mut state = model.begin_sentence_state();
        let mut out = model.null_context_state();
        let looking = model.index("looking").unwrap();

        let score = model.base_score(&state, looking, &mut out).unwrap();
        assert!(score.is_finite());

        std::mem::swap(&mut state, &mut out);
        let full = model
            .base_full_score(&state, model.end_sentence_index(), &mut out)
            .unwrap();
        assert!(full.log_prob.is_finite());
    }

    #[test]
    fn rejects_states_from_other_models() {
        let config = Config {
            show_progress: false,
            ..Config::default()
        };
        let first = Model::with_config("lm/test.arpa", config).unwrap();
        let second = Model::with_config("lm/test.arpa", config).unwrap();

        let state = first.begin_sentence_state();
        let mut out = second.null_context_state();
        let word = second.index("looking").unwrap();

        let error = second.base_score(&state, word, &mut out).unwrap_err();
        assert!(matches!(error, KenlmError::StateModelMismatch));
    }
}
