#!/usr/bin/env python3
"""Build the user-supplied Experiment Viewer template as a standalone report."""
import argparse
import json
import subprocess
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
REPORT = ROOT / 'results' / 'viewer'


def script_json(value):
    return value.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blank', action='store_true', help='Build the reusable template without experiment data')
    parser.add_argument('--output', type=Path, default=REPORT / 'index.html')
    args = parser.parse_args()
    template = (SOURCE / 'index.html').read_text(encoding='utf-8')
    sample = json.loads((SOURCE / 'demo.json').read_text(encoding='utf-8'))
    replacements = {
        'STYLE': (SOURCE / 'style.css').read_text(encoding='utf-8'),
        'SAMPLE': script_json(json.dumps(sample, ensure_ascii=False, separators=(',', ':'), allow_nan=False)),
    }
    for name in ['CORE', 'LEGACY', 'NAVIGATION', 'APP']:
        source = (SOURCE / (name.lower() + '.js')).read_text(encoding='utf-8')
        if '</script' in source.lower():
            raise ValueError(f'{name} contains an HTML script end tag')
        replacements[name] = source
    for name, value in replacements.items():
        marker = f'/*__{name}__*/'
        if template.count(marker) != 1:
            raise ValueError(f'Expected exactly one marker: {marker}')
        template = template.replace(marker, value)
    if not args.blank:
        encoded = subprocess.run([
            'node', '-e',
            "const fs=require('node:fs'),path=require('node:path');"
            "const root=process.argv[1],doc=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));"
            "require(path.join(root,'core.js')).validate(doc);"
            "require(path.join(root,'navigation.js')).validate(doc);"
            "process.stdout.write(JSON.stringify(doc));",
            str(SOURCE), str(REPORT / 'experiment_data.json'),
        ], check=True, capture_output=True, text=True).stdout
        marker = '<script type="application/json" id="embedded-data">null</script>'
        if template.count(marker) != 1:
            raise ValueError('Expected one empty embedded-data script')
        template = template.replace(marker, '<script type="application/json" id="embedded-data">' + script_json(encoded) + '</script>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template, encoding='utf-8')
    print(f'Built {args.output} ({len(template.encode()):,} bytes)')


if __name__ == '__main__':
    main()
