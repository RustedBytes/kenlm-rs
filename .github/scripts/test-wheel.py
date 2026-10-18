"""Exercise the installed wheel, including all supported compressed inputs."""

import bz2
import ctypes
import gzip
import lzma
import math
from pathlib import Path
import sys
import tempfile

import kenlm

source = Path(sys.argv[1])
text = source.read_bytes()
sentence = 'this is a test'
reference = kenlm.Model(str(source)).score(sentence)
assert math.isfinite(reference), reference
with tempfile.TemporaryDirectory() as directory:
    for suffix, compress in (('gz', gzip.compress), ('bz2', bz2.compress), ('xz', lzma.compress)):
        path = Path(directory) / f'model.arpa.{suffix}'
        path.write_bytes(compress(text))
        actual = kenlm.Model(str(path)).score(sentence)
        assert math.isclose(actual, reference, abs_tol=1e-6), (suffix, actual, reference)
        print(f'{suffix}: score={actual}')

# The wheel also ships a standalone shared library. Check its dependency closure.
libraries = list(Path(kenlm.__file__).parent.glob('libkenlm*.dylib'))
assert len(libraries) == 1, libraries
ctypes.CDLL(str(libraries[0]))
print(f'Installed extension: {kenlm.__file__}; standalone library: {libraries[0]}')
