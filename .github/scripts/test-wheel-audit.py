"""Regression fixtures for wheel compatibility gates (runs without macOS)."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('audit', Path(__file__).with_name('check-macos-wheels.py'))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class WheelAuditTests(unittest.TestCase):
    def check(self, *, filename_platform='macosx_14_0_arm64', metadata_platform=None,
              arch='arm64', minimum='14.0', dependency='/usr/lib/libc++.1.dylib',
              legacy=False, bundled=False):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / f'camel_kenlm-1.0-cp39-cp39-{filename_platform}.whl'
            tag = f'cp39-cp39-{metadata_platform or filename_platform}'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('camel_kenlm-1.0.dist-info/WHEEL', f'Tag: {tag}\n')
                archive.writestr('kenlm.so', b'fixture')
                archive.writestr('libkenlm.dylib', b'fixture')
                if bundled:
                    archive.writestr('.dylibs/libtest.dylib', b'fixture')
            def output(command, **kwargs):
                if command[0] == 'lipo':
                    return arch + '\n'
                target = (f'cmd LC_VERSION_MIN_MACOSX\n version {minimum}' if legacy else
                          f'cmd LC_BUILD_VERSION\n platform 1\n minos {minimum}')
                if Path(command[-1]).name == 'libtest.dylib':
                    return f'Load command 0\n {target}\n'
                return (f'Load command 0\n {target}\nLoad command 1\n'
                        f' cmd LC_LOAD_DYLIB\n name {dependency} (offset 24)\n')
            with patch.object(checker.subprocess, 'check_output', side_effect=output):
                checker.audit(wheel, 'arm64', '14.0')

    def test_valid_system_dependencies(self):
        self.check()

    def test_older_binary_is_compatible(self):
        self.check(minimum='11.0', legacy=True)

    def test_wrong_filename_tag(self):
        with self.assertRaisesRegex(ValueError, 'expected platform'):
            self.check(filename_platform='macosx_15_0_arm64')

    def test_inconsistent_metadata(self):
        with self.assertRaisesRegex(ValueError, 'inconsistent WHEEL'):
            self.check(metadata_platform='macosx_15_0_arm64')

    def test_wrong_architecture(self):
        with self.assertRaisesRegex(ValueError, 'architectures'):
            self.check(arch='x86_64')

    def test_newer_binary_under_old_tag(self):
        with self.assertRaisesRegex(ValueError, 'minimum OS'):
            self.check(minimum='15.0')

    def test_homebrew_dependency(self):
        with self.assertRaisesRegex(ValueError, 'external dependency'):
            self.check(dependency='/opt/homebrew/lib/liblzma.5.dylib')

    def test_unresolved_rpath(self):
        with self.assertRaisesRegex(ValueError, 'external dependency'):
            self.check(dependency='@rpath/libtest.dylib')

    def test_missing_bundled_dependency(self):
        with self.assertRaisesRegex(ValueError, 'missing bundled'):
            self.check(dependency='@loader_path/.dylibs/libtest.dylib')

    def test_valid_bundled_dependency(self):
        # Root binaries may load a repaired dependency relative to themselves.
        # The dependency itself only loads system libraries in real wheel output.
        self.check(dependency='@loader_path/.dylibs/libtest.dylib', bundled=True)


if __name__ == '__main__':
    unittest.main()
