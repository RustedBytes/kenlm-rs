//! Rust entry points for KenLM's original command-line tools.
//!
//! These functions execute the corresponding KenLM C++ `main` functions in the
//! current process and return their exit status. They are mainly intended for
//! the Cargo-provided binaries, but can also be used by applications that want
//! to embed KenLM tools.

use crate::Result;
use std::env;
use std::ffi::{CString, OsString};
use std::os::raw::{c_char, c_int};
use std::process::ExitCode;

#[cfg(feature = "tools")]
extern "C" {
    fn kenlmrs_build_binary_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
    fn kenlmrs_cat_compressed_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
    fn kenlmrs_fragment_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
    fn kenlmrs_query_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
}

#[cfg(feature = "estimation")]
extern "C" {
    fn kenlmrs_count_ngrams_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
    fn kenlmrs_dump_counts_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
    fn kenlmrs_lmplz_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
}

#[cfg(feature = "filter")]
extern "C" {
    fn kenlmrs_filter_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
    fn kenlmrs_phrase_table_vocab_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
}

#[cfg(feature = "interpolate")]
extern "C" {
    fn kenlmrs_interpolate_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
    fn kenlmrs_streaming_example_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
}

#[cfg(feature = "tools")]
pub fn build_binary(args: impl IntoIterator<Item = OsString>) -> Result<i32> {
    run(args, kenlmrs_build_binary_main)
}

#[cfg(feature = "tools")]
pub fn cat_compressed(args: impl IntoIterator<Item = OsString>) -> Result<i32> {
    run(args, kenlmrs_cat_compressed_main)
}

#[cfg(feature = "tools")]
pub fn fragment(args: impl IntoIterator<Item = OsString>) -> Result<i32> {
    run(args, kenlmrs_fragment_main)
}

#[cfg(feature = "tools")]
pub fn query(args: impl IntoIterator<Item = OsString>) -> Result<i32> {
    run(args, kenlmrs_query_main)
}

#[cfg(feature = "estimation")]
pub fn count_ngrams(args: impl IntoIterator<Item = OsString>) -> Result<i32> {
    run(args, kenlmrs_count_ngrams_main)
}

#[cfg(feature = "estimation")]
pub fn dump_counts(args: impl IntoIterator<Item = OsString>) -> Result<i32> {
    run(args, kenlmrs_dump_counts_main)
}

#[cfg(feature = "estimation")]
pub fn lmplz(args: impl IntoIterator<Item = OsString>) -> Result<i32> {
    run(args, kenlmrs_lmplz_main)
}

#[cfg(feature = "filter")]
pub fn filter(args: impl IntoIterator<Item = OsString>) -> Result<i32> {
    run(args, kenlmrs_filter_main)
}

#[cfg(feature = "filter")]
pub fn phrase_table_vocab(args: impl IntoIterator<Item = OsString>) -> Result<i32> {
    run(args, kenlmrs_phrase_table_vocab_main)
}

#[cfg(feature = "interpolate")]
pub fn interpolate(args: impl IntoIterator<Item = OsString>) -> Result<i32> {
    run(args, kenlmrs_interpolate_main)
}

#[cfg(feature = "interpolate")]
pub fn streaming_example(args: impl IntoIterator<Item = OsString>) -> Result<i32> {
    run(args, kenlmrs_streaming_example_main)
}

pub fn args() -> impl Iterator<Item = OsString> {
    env::args_os()
}

pub fn main_exit(command: impl FnOnce(std::env::ArgsOs) -> Result<i32>) -> ExitCode {
    match command(env::args_os()) {
        Ok(status) => ExitCode::from(status.clamp(0, u8::MAX as i32) as u8),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run(
    args: impl IntoIterator<Item = OsString>,
    entry: unsafe extern "C" fn(c_int, *mut *mut c_char) -> c_int,
) -> Result<i32> {
    let strings = args
        .into_iter()
        .map(|arg| CString::new(arg.to_string_lossy().as_bytes()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut pointers = strings
        .iter()
        .map(|arg| arg.as_ptr() as *mut c_char)
        .collect::<Vec<_>>();
    let argc = pointers.len() as c_int;
    pointers.push(std::ptr::null_mut());
    // SAFETY: each pointer comes from a live `CString`, is NUL-terminated, and
    // remains valid for the duration of the call. `argc` excludes the trailing
    // null sentinel, matching the C/C++ `main` convention.
    let status = unsafe { entry(argc, pointers.as_mut_ptr()) };
    Ok(status)
}
