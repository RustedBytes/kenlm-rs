# kenlm

Language model inference code by Kenneth Heafield (kenlm at kheafield.com)

The website https://kheafield.com/code/kenlm/ has more documentation.  If you're a decoder developer, please download the latest version from there instead of copying from another decoder.

*This fork of kenlm was made in order to have stable versioned pip packages
for use with [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools).
Versions are numbered as `yyyy.mm.n` where `yyyy` and `mm` is the year and
month respectively of the last commit in the main repo and `n` is the release
number in that year and month.*

## Compiling
Use cmake, see [BUILDING](BUILDING) for build dependencies and more detail.
```bash
mkdir -p build
cd build
cmake ..
make -j 4
```

## Compiling with your own build system
If you want to compile with your own build system (Makefile etc) or to use as a library, there are a number of macros you can set on the g++ command line or in util/have.hh .  

* `KENLM_MAX_ORDER` is the maximum order that can be loaded.  This is done to make state an efficient POD rather than a vector.  
* `HAVE_ICU` If your code links against ICU, define this to disable the internal StringPiece and replace it with ICU's copy of StringPiece, avoiding naming conflicts.  

ARPA files can be read in compressed format with these options:
* `HAVE_ZLIB` Supports gzip.  Link with -lz.
* `HAVE_BZLIB` Supports bzip2.  Link with -lbz2.
* `HAVE_XZLIB` Supports xz.  Link with -llzma.

Note that these macros impact only `read_compressed.cc` and `read_compressed_test.cc`.  The bjam build system will auto-detect bzip2 and xz support.  

## Estimation
lmplz estimates unpruned language models with modified Kneser-Ney smoothing.  After compiling with bjam, run
```bash
bin/lmplz -o 5 <text >text.arpa
```
The algorithm is on-disk, using an amount of memory that you specify.  See https://kheafield.com/code/kenlm/estimation/ for more.

MT Marathon 2012 team members Ivan Pouzyrevsky and Mohammed Mediani contributed to the computation design and early implementation. Jon Clark contributed to the design, clarified points about smoothing, and added logging. 

## Filtering

filter takes an ARPA or count file and removes entries that will never be queried.  The filter criterion can be corpus-level vocabulary, sentence-level vocabulary, or sentence-level phrases.  Run
```bash
bin/filter
```
and see https://kheafield.com/code/kenlm/filter/ for more documentation.

## Querying

Two data structures are supported: probing and trie.  Probing is a probing hash table with keys that are 64-bit hashes of n-grams and floats as values.  Trie is a fairly standard trie but with bit-level packing so it uses the minimum number of bits to store word indices and pointers.  The trie node entries are sorted by word index.  Probing is the fastest and uses the most memory.  Trie uses the least memory and is a bit slower.

As is the custom in language modeling, all probabilities are log base 10.

With trie, resident memory is 58% of IRST's smallest version and 21% of SRI's compact version.  Simultaneously, trie CPU's use is 81% of IRST's fastest version and 84% of SRI's fast version.  KenLM's probing hash table implementation goes even faster at the expense of using more memory.  See https://kheafield.com/code/kenlm/benchmark/.

Binary format via mmap is supported.  Run `./build_binary` to make one then pass the binary file name to the appropriate Model constructor.   

## Platforms
`murmur_hash.cc` and `bit_packing.hh` perform unaligned reads and writes that make the code architecture-dependent.  
It has been sucessfully tested on x86\_64, x86, and PPC64.  
ARM support is reportedly working, at least on the iphone.   

Runs on Linux, OS X, Cygwin, and MinGW.  

Hideo Okuma and Tomoyuki Yoshimura from NICT contributed ports to ARM and MinGW.  

## Decoder developers
- I recommend copying the code and distributing it with your decoder.  However, please send improvements upstream.  

- It's possible to compile the query-only code without Boost, but useful things like estimating models require Boost.

- Select the macros you want, listed in the previous section.  

- There are two build systems: compile.sh and cmake.  They're pretty simple and are intended to be reimplemented in your build system.  

- Use either the interface in `lm/model.hh` or `lm/virtual_interface.hh`.  Interface documentation is in comments of `lm/virtual_interface.hh` and `lm/model.hh`.  

- There are several possible data structures in `model.hh`.  Use `RecognizeBinary` in `binary_format.hh` to determine which one a user has provided.  You probably already implement feature functions as an abstract virtual base class with several children.  I suggest you co-opt this existing virtual dispatch by templatizing the language model feature implementation on the KenLM model identified by `RecognizeBinary`.  This is the strategy used in Moses and cdec.

- See `lm/config.hh` for run-time tuning options.

## Contributors
Contributions to KenLM are welcome.  Please base your contributions on https://github.com/kpu/kenlm and send pull requests (or I might give you commit access).  Downstream copies in Moses and cdec are maintained by overwriting them so do not make changes there.  

## Python module
Contributed by Victor Chahuneau.

### Installation

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

When installing pip, the `MAX_ORDER` environment variable controls the max order with which KenLM was built.

### Basic Usage
```python
import kenlm
model = kenlm.Model('lm/test.arpa')
print(model.score('this is a sentence .', bos = True, eos = True))
```
See [python/example.py](python/example.py) and [python/kenlm.pyx](python/kenlm.pyx) for more, including stateful APIs.  

### Building kenlm - Using vcpkg

You can download and install kenlm using the [vcpkg](https://github.com/Microsoft/vcpkg) dependency manager:

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install
    ./vcpkg install kenlm

The kenlm port in vcpkg is kept up to date by Microsoft team members and community contributors. If the version is out of date, please [create an issue or pull request](https://github.com/Microsoft/vcpkg) on the vcpkg repository.

---

The name was Hieu Hoang's idea, not mine.

## Rust bindings

This fork also contains a Rust crate that compiles KenLM's query-only C++
sources and exposes safe bindings for model loading, sentence scoring,
per-token full scores, vocabulary lookup, and stateful scoring.

```bash
cargo test
```

Basic use:

```rust
use kenlm::Model;

fn main() -> Result<(), kenlm::KenlmError> {
    let model = Model::new("lm/test.arpa")?;
    let score = model.score("looking on a little", true, true)?;
    let perplexity = model.perplexity("looking on a little")?;

    println!("{score} {perplexity}");
    Ok(())
}
```

Set `KENLM_MAX_ORDER` while building to override the default maximum order of
6, matching KenLM's existing build-time option.

### Cargo features

The default crate builds the safe Rust inference API only. Optional features
enable KenLM's original command-line tools:

```bash
cargo build --features tools
cargo build --features estimation
cargo build --features filter
cargo build --features interpolate
cargo build --features full
```

Available feature groups:

- `bundled`: require a bundled native library instead of compiling KenLM from
  source
- `tools`: `kenlm-build-binary`, `kenlm-query`, `kenlm-fragment`, and
  `kenlm-cat-compressed`
- `estimation`: `kenlm-lmplz`, `kenlm-count-ngrams`, and `kenlm-dump-counts`
- `filter`: `kenlm-filter` and `kenlm-phrase-table-vocab`
- `interpolate`: `kenlm-interpolate` and `kenlm-streaming-example`
- `compression`: enables `zlib`, `bzip2`, and `xz` compressed input support

Native dependencies for optional features:

- `estimation`, `filter`, and `interpolate` require Boost headers and libraries.
  Set `BOOST_INCLUDE_DIR` and `BOOST_LIB_DIR` if they are installed outside the
  system search paths.
- `interpolate` also requires Eigen3 headers. Set `EIGEN3_INCLUDE_DIR` if
  needed.
- `zlib`, `bzip2`, and `xz` require the corresponding system development
  libraries.

### Bundled native libraries

Crates.io distributes crate source. To avoid C++ compilation for users, publish
native static libraries for each target you want to support and put them in:

```text
prebuilt/<target-triple>/libkenlmrs.a
```

For MSVC targets, use:

```text
prebuilt/<target-triple>/kenlmrs.lib
```

`build.rs` automatically links a matching bundled library when it exists. If no
matching artifact is found, it falls back to compiling the vendored KenLM C++
sources.

Useful environment variables:

- `KENLM_RS_BUNDLED_LIB_DIR`: look for `libkenlmrs.a` or `kenlmrs.lib` in this
  directory instead of only `prebuilt/<target-triple>/`
- `KENLM_RS_USE_BUNDLED=0`: force source compilation even if a bundled library
  is present
- `KENLM_RS_BUNDLED_REQUIRED=1`: fail the build if no matching bundled library
  is found

For release builds where users should never compile C++, enable the Cargo
feature:

```bash
cargo build --features bundled
```

Bundled libraries must be compiled with the same Cargo feature set they will be
used with. For example, a `full` artifact must include the tool, estimation,
filter, interpolation, and compression symbols.
