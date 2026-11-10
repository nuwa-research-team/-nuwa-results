"""Verify and optionally restore split ZIPs. Run after downloading ALL batches."""
import argparse,hashlib,json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

ap=argparse.ArgumentParser()
ap.add_argument('--restore',action='store_true',help='Reconstruct original ZIPs after verifying all parts')
args=ap.parse_args()
root=Path(__file__).resolve().parent
def inside(rel):
    p=(root/rel).resolve();p.relative_to(root.resolve());return p

data=json.loads((root/'large_files_manifest.json').read_text(encoding='utf-8'))
for entry in data['files']:
    h=hashlib.sha256();n=0
    for piece in entry['parts']:
        p=inside(piece['path'])
        if p.stat().st_size!=piece['bytes'] or digest(p)!=piece['sha256']:
            raise SystemExit('Part checksum mismatch: '+piece['path'])
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
    if n!=entry['bytes'] or h.hexdigest()!=entry['sha256']:
        raise SystemExit('Combined checksum mismatch: '+entry['original_path'])
    print('PASS '+entry['original_path'])
    if not args.restore:continue
    dest=inside(entry['original_path'])
    if dest.exists():
        if digest(dest)==entry['sha256']:print('Already restored; kept existing file');continue
        raise SystemExit('Existing different file preserved: '+entry['original_path'])
    temp=dest.with_name(dest.name+'.assembling')
    with temp.open('xb') as out:
        for piece in entry['parts']:
            with inside(piece['path']).open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):out.write(b)
    if digest(temp)!=entry['sha256']:raise SystemExit('Restore verification failed; original not replaced')
    if dest.exists():raise SystemExit('Target appeared during restore; preserved both files')
    temp.rename(dest)
    print('RESTORED '+entry['original_path'])
print('All checks passed. '+('Original ZIPs restored.' if args.restore else 'Use --restore to reconstruct.'))
