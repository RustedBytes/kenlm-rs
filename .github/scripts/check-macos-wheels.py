"""Reject mislabeled wheels and native dependencies newer than the target OS."""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile


def version(value):
    parts = tuple(map(int, value.split('.')))
    return (parts + (0, 0, 0))[:3]


def audit(wheel, arch, target):
    platform = f"macosx_{target.replace('.', '_')}_{arch}"
    if wheel.stem.rsplit('-', 1)[-1] != platform:
        raise ValueError(f"{wheel.name}: expected platform {platform}")
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as tmp:
        metadata = [n for n in archive.namelist() if n.endswith('.dist-info/WHEEL')]
        if len(metadata) != 1:
            raise ValueError(f"{wheel.name}: expected one WHEEL metadata file")
        tags = re.findall(r'^Tag: (.+)$', archive.read(metadata[0]).decode(), re.M)
        filename_tag = '-'.join(wheel.stem.split('-')[-3:])
        if tags != [filename_tag]:
            raise ValueError(f"{wheel.name}: inconsistent WHEEL tags: {tags}")
        archive.extractall(tmp)
        root = Path(tmp).resolve()
        binaries = [p for p in root.rglob('*') if p.suffix in ('.so', '.dylib')]
        if not any(p.suffix == '.so' for p in binaries):
            raise ValueError(f"{wheel.name}: no extension module")
        if not any(p.name == 'libkenlm.dylib' for p in binaries):
            raise ValueError(f"{wheel.name}: no standalone libkenlm.dylib")
        for binary in binaries:
            archs = subprocess.check_output(['lipo', '-archs', str(binary)], text=True).split()
            if archs != [arch]:
                raise ValueError(f"{binary.name}: unexpected architectures {archs}")
            commands = subprocess.check_output(['otool', '-l', str(binary)], text=True)
            minimums = []
            for block in re.split(r'Load command \d+', commands)[1:]:
                if re.search(r'cmd LC_BUILD_VERSION\b', block):
                    match = re.search(r'^\s+minos ([\d.]+)', block, re.M)
                    if not re.search(r'^\s+platform (?:1|MACOS)\s*$', block, re.M):
                        raise ValueError(f"{binary.name}: non-macOS platform")
                    minimums.append(match.group(1) if match else '999')
                elif re.search(r'cmd LC_VERSION_MIN_MACOSX\b', block):
                    match = re.search(r'^\s+version ([\d.]+)', block, re.M)
                    minimums.append(match.group(1) if match else '999')
                if re.search(r'cmd LC_(?:LOAD|LOAD_WEAK|REEXPORT|LOAD_UPWARD|LAZY_LOAD)_DYLIB\b', block):
                    name = re.search(r'^\s+name (.+) \(offset \d+\)', block, re.M).group(1)
                    if name.startswith(('/usr/lib/', '/System/Library/')):
                        continue
                    if not name.startswith('@loader_path/'):
                        raise ValueError(f"{binary.name}: external dependency {name}")
                    dependency = (binary.parent / name[len('@loader_path/'):]).resolve()
                    if root not in dependency.parents or not dependency.is_file():
                        raise ValueError(f"{binary.name}: missing bundled dependency {name}")
            if not minimums or any(version(v) > version(target) for v in minimums):
                raise ValueError(f"{binary.name}: minimum OS {minimums}, target {target}")
            print(f"{wheel.name}: {binary.relative_to(root)} arch={arch} minOS={minimums}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheelhouse', type=Path)
    parser.add_argument('--arch', choices=('arm64', 'x86_64'), required=True)
    parser.add_argument('--target', default='14.0')
    args = parser.parse_args()
    wheels = sorted(args.wheelhouse.glob('*.whl'))
    if not wheels:
        parser.error('no wheels found')
    for wheel in wheels:
        audit(wheel, args.arch, args.target)


if __name__ == '__main__':
    main()
