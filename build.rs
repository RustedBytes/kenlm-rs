use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=rust/kenlm_wrapper.cc");
    println!("cargo:rerun-if-changed=rust/kenlm_wrapper.h");
    println!("cargo:rerun-if-changed=rust/bin_wrappers");
    println!("cargo:rerun-if-changed=lm");
    println!("cargo:rerun-if-changed=util");
    println!("cargo:rerun-if-env-changed=BOOST_LIB_DIR");
    println!("cargo:rerun-if-env-changed=BOOST_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=EIGEN3_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=KENLM_MAX_ORDER");
    println!("cargo:rerun-if-env-changed=KENLM_RS_PREBUILT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=KENLM_RS_PREBUILT_REQUIRED");
    println!("cargo:rerun-if-env-changed=KENLM_RS_USE_PREBUILT");
    println!("cargo:rerun-if-env-changed=KENLM_RS_BUNDLED_LIB_DIR");
    println!("cargo:rerun-if-env-changed=KENLM_RS_BUNDLED_REQUIRED");
    println!("cargo:rerun-if-env-changed=KENLM_RS_USE_BUNDLED");

    let max_order = env::var("KENLM_MAX_ORDER").unwrap_or_else(|_| "6".to_string());
    let tools = feature("TOOLS");
    let estimation = feature("ESTIMATION");
    let filter = feature("FILTER");
    let interpolate = feature("INTERPOLATE");
    let target = env::var("TARGET").expect("Cargo should set TARGET");

    if try_link_prebuilt(&target) {
        link_feature_dependencies(estimation, filter, interpolate);
        link_platform_libraries();
        return;
    }

    if feature("BUNDLED")
        || env_enabled("KENLM_RS_BUNDLED_REQUIRED")
        || env_enabled("KENLM_RS_PREBUILT_REQUIRED")
    {
        panic!(
            "bundled KenLM library requested but not found. Put {} in prebuilt/{}/ or set KENLM_RS_BUNDLED_LIB_DIR.",
            prebuilt_library_name(&target),
            target
        );
    }

    if estimation || filter || interpolate {
        require_header(
            "boost/version.hpp",
            "Boost headers are required for KenLM's estimation, filter, and interpolation features. Install Boost development headers or set BOOST_INCLUDE_DIR.",
            env::var("BOOST_INCLUDE_DIR").ok().as_deref(),
            &["/usr/include", "/usr/local/include"],
        );
    }
    if interpolate {
        require_header(
            "Eigen/Core",
            "Eigen3 headers are required for KenLM interpolation. Install Eigen3 development headers or set EIGEN3_INCLUDE_DIR.",
            env::var("EIGEN3_INCLUDE_DIR").ok().as_deref(),
            &["/usr/include/eigen3", "/usr/local/include/eigen3"],
        );
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++11")
        .include(".")
        .define("KENLM_MAX_ORDER", max_order.as_str())
        .flag_if_supported("-Wno-class-memaccess")
        .flag_if_supported("-Wno-deprecated-copy")
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-implicit-fallthrough")
        .flag_if_supported("-Wno-return-type")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-sign-compare")
        .flag_if_supported("-Wno-unused-local-typedefs");

    if let Ok(include) = env::var("BOOST_INCLUDE_DIR") {
        build.include(include);
    }

    if feature("ZLIB") {
        build.define("HAVE_ZLIB", None);
        println!("cargo:rustc-link-lib=z");
    }
    if feature("BZIP2") {
        build.define("HAVE_BZLIB", None);
        println!("cargo:rustc-link-lib=bz2");
    }
    if feature("XZ") {
        build.define("HAVE_XZLIB", None);
        println!("cargo:rustc-link-lib=lzma");
    }

    if interpolate {
        let eigen_dir =
            env::var("EIGEN3_INCLUDE_DIR").unwrap_or_else(|_| "/usr/include/eigen3".to_string());
        if Path::new(&eigen_dir).exists() {
            build.include(eigen_dir);
        }
    }

    if env::var("PROFILE").as_deref() == Ok("release") {
        build.define("NDEBUG", None);
    }

    let core_sources = [
        "rust/kenlm_wrapper.cc",
        "util/double-conversion/bignum-dtoa.cc",
        "util/double-conversion/bignum.cc",
        "util/double-conversion/cached-powers.cc",
        "util/double-conversion/double-to-string.cc",
        "util/double-conversion/fast-dtoa.cc",
        "util/double-conversion/fixed-dtoa.cc",
        "util/double-conversion/string-to-double.cc",
        "util/double-conversion/strtod.cc",
        "util/bit_packing.cc",
        "util/ersatz_progress.cc",
        "util/exception.cc",
        "util/file.cc",
        "util/file_piece.cc",
        "util/float_to_string.cc",
        "util/integer_to_string.cc",
        "util/mmap.cc",
        "util/murmur_hash.cc",
        "util/parallel_read.cc",
        "util/pool.cc",
        "util/read_compressed.cc",
        "util/scoped.cc",
        "util/spaces.cc",
        "util/string_piece.cc",
        "util/usage.cc",
        "lm/bhiksha.cc",
        "lm/binary_format.cc",
        "lm/config.cc",
        "lm/lm_exception.cc",
        "lm/model.cc",
        "lm/quantize.cc",
        "lm/read_arpa.cc",
        "lm/search_hashed.cc",
        "lm/search_trie.cc",
        "lm/sizes.cc",
        "lm/trie.cc",
        "lm/trie_sort.cc",
        "lm/value_build.cc",
        "lm/virtual_interface.cc",
        "lm/vocab.cc",
    ];

    for source in core_sources {
        build.file(source);
    }

    if tools {
        for source in [
            "rust/bin_wrappers/build_binary.cc",
            "rust/bin_wrappers/cat_compressed.cc",
            "rust/bin_wrappers/fragment.cc",
            "rust/bin_wrappers/query.cc",
        ] {
            build.file(source);
        }
    }

    if estimation || interpolate {
        for source in [
            "util/stream/chain.cc",
            "util/stream/count_records.cc",
            "util/stream/io.cc",
            "util/stream/line_input.cc",
            "util/stream/multi_progress.cc",
            "util/stream/rewindable_stream.cc",
            "lm/builder/adjust_counts.cc",
            "lm/builder/corpus_count.cc",
            "lm/builder/initial_probabilities.cc",
            "lm/builder/interpolate.cc",
            "lm/builder/output.cc",
            "lm/builder/pipeline.cc",
            "lm/common/model_buffer.cc",
            "lm/common/print.cc",
            "lm/common/renumber.cc",
            "lm/common/size_option.cc",
            "rust/bin_wrappers/count_ngrams.cc",
            "rust/bin_wrappers/dump_counts.cc",
            "rust/bin_wrappers/lmplz.cc",
        ] {
            build.file(source);
        }
        link_boost_estimation();
    }

    if filter {
        for source in [
            "lm/filter/arpa_io.cc",
            "lm/filter/phrase.cc",
            "lm/filter/vocab.cc",
            "rust/bin_wrappers/filter.cc",
            "rust/bin_wrappers/phrase_table_vocab.cc",
        ] {
            build.file(source);
        }
        link_boost_filter();
    }

    if interpolate {
        for source in [
            "lm/interpolate/backoff_reunification.cc",
            "lm/interpolate/bounded_sequence_encoding.cc",
            "lm/interpolate/merge_probabilities.cc",
            "lm/interpolate/merge_vocab.cc",
            "lm/interpolate/normalize.cc",
            "lm/interpolate/pipeline.cc",
            "lm/interpolate/split_worker.cc",
            "lm/interpolate/tune_derivatives.cc",
            "lm/interpolate/tune_instances.cc",
            "lm/interpolate/tune_weights.cc",
            "lm/interpolate/universal_vocab.cc",
            "rust/bin_wrappers/interpolate.cc",
            "rust/bin_wrappers/streaming_example.cc",
        ] {
            build.file(source);
        }
        link_boost("program_options");
    }

    build.compile("kenlmrs");

    link_platform_libraries();
}

fn feature(name: &str) -> bool {
    env::var_os(format!("CARGO_FEATURE_{name}")).is_some()
}

fn link_boost(name: &str) {
    if let Ok(dir) = env::var("BOOST_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    }
    println!("cargo:rustc-link-lib=boost_{name}");
}

fn link_boost_estimation() {
    link_boost("program_options");
    link_boost("thread");
    link_boost("system");
}

fn link_boost_filter() {
    link_boost("thread");
    link_boost("system");
}

fn link_feature_dependencies(estimation: bool, filter: bool, interpolate: bool) {
    if feature("ZLIB") {
        println!("cargo:rustc-link-lib=z");
    }
    if feature("BZIP2") {
        println!("cargo:rustc-link-lib=bz2");
    }
    if feature("XZ") {
        println!("cargo:rustc-link-lib=lzma");
    }
    if estimation || interpolate {
        link_boost_estimation();
    }
    if filter {
        link_boost_filter();
    }
    if interpolate && !(estimation) {
        link_boost("program_options");
    }
}

fn link_platform_libraries() {
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") | Ok("android") => {
            println!("cargo:rustc-link-lib=stdc++");
            if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
                println!("cargo:rustc-link-lib=rt");
            }
        }
        Ok("macos") | Ok("ios") | Ok("freebsd") | Ok("openbsd") => {
            println!("cargo:rustc-link-lib=c++");
        }
        _ => {}
    }
}

fn try_link_prebuilt(target: &str) -> bool {
    if env_disabled("KENLM_RS_USE_BUNDLED") || env_disabled("KENLM_RS_USE_PREBUILT") {
        return false;
    }

    let candidates = prebuilt_dirs(target);
    let library_name = prebuilt_library_name(target);
    if let Some(dir) = candidates
        .into_iter()
        .find(|dir| dir.join(library_name).is_file())
    {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static=kenlmrs");
        println!(
            "cargo:warning=using bundled KenLM native library from {}",
            dir.display()
        );
        return true;
    }

    false
}

fn prebuilt_dirs(target: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(dir) = env::var("KENLM_RS_BUNDLED_LIB_DIR") {
        dirs.push(PathBuf::from(dir));
    }
    if let Ok(dir) = env::var("KENLM_RS_PREBUILT_LIB_DIR") {
        dirs.push(PathBuf::from(dir));
    }
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        dirs.push(PathBuf::from(manifest_dir).join("prebuilt").join(target));
    }
    dirs
}

fn prebuilt_library_name(target: &str) -> &'static str {
    if target.contains("windows-msvc") {
        "kenlmrs.lib"
    } else {
        "libkenlmrs.a"
    }
}

fn env_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn env_disabled(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.as_str(),
                "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
            )
        })
        .unwrap_or(false)
}

fn require_header(header: &str, message: &str, env_dir: Option<&str>, defaults: &[&str]) {
    let mut dirs = Vec::new();
    if let Some(dir) = env_dir {
        dirs.push(dir);
    }
    dirs.extend(defaults.iter().copied());

    if dirs.iter().any(|dir| Path::new(dir).join(header).is_file()) {
        return;
    }

    panic!("{message}");
}
