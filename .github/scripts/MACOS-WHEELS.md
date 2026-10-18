# macOS wheel builds

macOS 14 hosted runners retire on 2026-11-02. Build on macOS 15 and 26,
with native arm64 and x86_64 jobs. All produce `macosx_14_0_<arch>` wheels;
a macOS 14 wheel can also be installed on the newer operating systems.
There is no need to publish redundant 15-only and 26-only variants.

Both setuptools and CMake receive `MACOSX_DEPLOYMENT_TARGET=14.0`.
Cibuildwheel stays at 3.1.4 and retains its pinned repair/test dependencies.
Python build dependencies are constrained in `../constraints/macos-build.txt`.
Xcode 26.3 is explicitly selected on all four hosts, rather than latest-stable.
Build logs record the actual OS, SDK, compiler and Python dependency versions.
`SOURCE_DATE_EPOCH` comes from the source commit.

Do not use Homebrew bottles for wheel runtime dependencies: they may require
the host's OS version. This build needs no Boost runtime library: the Python
standalone CMake path returns before the main project's Boost discovery.
It uses SDK zlib/bzip2 and builds static liblzma from the immutable XZ 5.8.1
commit, with the same deployment target. CMake excludes Homebrew prefixes;
the compiler's dependency probes receive the same include/link flags as the
extension. CMake and the extension both enable `HAVE_XZLIB` when available.

CI performs the following checks:

- PRs build CPython 3.9 and 3.14; workflow_dispatch builds 3.9 through 3.14 (including 3.14t).
- Cibuildwheel's default delocate repair checks dependency target versions.
- The wheel audit checks filename and WHEEL metadata, every Mach-O slice,
  minimum OS load commands, and bundled/system dependency paths. It rejects
  Homebrew paths, unresolved rpaths and binaries requiring an OS newer than 14.
- Each installed wheel imports and scores plain/gzip/bzip2/xz ARPA fixtures
  outside the checkout and loads the standalone dylib via ctypes.
- macOS 15 jobs rebuild CPython 3.9 from a clean build directory and compare
  SHA-256 hashes of the complete wheels, for both architectures.
- Negative regression fixtures ensure the audit rejects misleading old tags,
  newer binaries, wrong architectures and external/missing dependencies.

These checks establish binary deployment metadata and execution on macOS 15/26.
They do not replace a runtime test on a physical/self-hosted macOS 14 machine,
or prove absence of unguarded newer system symbols. The clean rebuild comparison verifies byte-for-byte reproducibility within
one job. It does not establish reproducibility across hosted image updates: runner images and
Apple SDK contents remain external inputs. Pins and version logs make builds
repeatable and changes diagnosable; retain artifacts from both SDK hosts for
comparison. No publishing is enabled in this workflow.

References:
- https://github.blog/changelog/2026-10-01-github-actions-macos-14-runner-image-retirement/
- https://cibuildwheel.pypa.io/en/v3.1.4/faq/#missing-dependencies
- https://github.com/actions/runner-images/tree/main/images/macos
