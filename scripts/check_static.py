"""Check owned Python imports, generated images, documentation and dashboard links.

Run from the repository root after generating the Milestone 3 bundle. This is a
static integrity check, not a browser visual or interaction test.
"""
import importlib
from pathlib import Path
import re
from PIL import Image
from experiments.milestone3.documentation import source_files

root = Path.cwd()
files = source_files()
for path in files:
    compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
modules = ['.'.join(p.relative_to(root).with_suffix('').parts) for p in files if p.is_relative_to(root/'experiments')]
for module in modules:
    importlib.import_module(module)
documents = [root/'README.md', root/'experiments/README.md'] + list((root/'docs').glob('*.md'))
broken = []
for path in documents:
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
        if not target.startswith(('http:', 'https:', '#')) and not (path.parent/target.split('#')[0]).exists():
            broken.append((str(path), target))
if broken:
    raise ValueError(f'Broken documentation links: {broken}')
images = list((root/'docs/milestone3').glob('*.png'))
if len(images) != 16:
    raise ValueError(f'Expected 16 report figures, found {len(images)}')
for path in images:
    with Image.open(path) as image:
        image.verify()
html = (root/'docs/index.html').read_text(encoding='utf-8')
static = re.findall(r'(?:src|href)="([^"]+)"', html)
missing = [target for target in static if not target.startswith('#') and not (root/'docs'/target).exists()]
if missing:
    raise ValueError(f'Missing dashboard resources: {missing}')
print(f'PASS: {len(files)} Python files compiled, {len(modules)} modules imported, {len(images)} PNGs decoded; documentation/dashboard links exist.')
