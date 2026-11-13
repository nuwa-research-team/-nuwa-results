import gzip
import hashlib
import json
import pickle
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
KINDS = ('cnn', 'lstm', 'gru', 'sdae', 'fsnet', 'tst')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    with gzip.open(BASE / 'six_data.pkl.gz', 'rb') as f:
        return pickle.load(f)


def main():
    torch.set_num_threads(4)
    hashes = json.loads((BASE / 'verified.json').read_bytes())['hashes']
    for name in ['six_inputs.tgz', 'six_predictions.npz', *['six_' + k + '.pt' for k in KINDS]]:
        assert sha(BASE / name) == hashes[name]
    data = load_inputs()
    keys, lookup, ix = [], {}, {}
    for tag, (seqs, _) in data.items():
        order = []
        for s in seqs:
            b = np.asarray(s, dtype=np.int16).tobytes()
            if b not in lookup:
                lookup[b] = len(keys)
                keys.append(b)
            order.append(lookup[b])
        ix[tag] = np.array(order)
    buckets = {}
    for i, b in enumerate(keys):
        width = max(32, 1 << (len(b)//2 - 1).bit_length())
        buckets.setdefault(width, []).append(i)
    print('Unique sequences',len(keys),flush=True)
    saved = np.load(BASE / 'six_predictions.npz')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with tarfile.open(BASE / 'six_inputs.tgz') as tar:
            assert hashlib.sha256(tar.extractfile('six_data.pkl.gz').read()).hexdigest() == sha(BASE/'six_data.pkl.gz')
            for member in tar:
                if member.name.endswith(('.py','.pt')):
                    dest = (tmp / member.name).resolve()
                    assert dest.is_relative_to(tmp) and member.isfile()
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(tar.extractfile(member).read())
        sys.path[:0] = [str(tmp), str(tmp / 'repaired')]
        import models
        import models_repaired as mr
        from train_repaired import SDAERepaired
        for kind in KINDS:
            plain = kind in ('cnn','lstm','gru')
            net = models.build(kind,8) if plain else SDAERepaired(8) if kind=='sdae' else mr.build(kind,8)
            net.load_state_dict(torch.load(tmp / (kind+'.pt'),map_location='cpu',weights_only=True)['state'])
            if plain: net.fc = nn.Identity()
            elif kind=='sdae': net.head = nn.Identity()
            elif kind=='fsnet': net.head[-1] = nn.Identity()
            else: net.base.output_layer = nn.Identity()
            net.eval()
            head = torch.load(BASE / ('six_'+kind+'.pt'),weights_only=True,map_location='cpu')
            logits = np.empty((len(keys),6),np.float32)
            with torch.inference_mode():
                for width, idxs in buckets.items():
                    for start in range(0,len(idxs),24):
                        ids = idxs[start:start+24]
                        x = np.zeros((len(ids),width),np.float32)
                        lengths = [len(keys[i])//2 for i in ids]
                        for j,i in enumerate(ids):
                            x[j,:lengths[j]] = np.frombuffer(keys[i],np.int16)
                        x = torch.from_numpy(x)
                        z = net(x[:,None,:]) if plain else net(x/1600,torch.tensor(lengths))
                        scaled = (z-head['mean'])/head['scale']
                        logits[ids] = nn.functional.linear(scaled,head['state']['weight'],head['state']['bias']).numpy()
            out = {tag:logits[index] for tag,index in ix.items()}
            mismatches = {tag:int((mat.argmax(1)!=saved[kind+'_'+tag]).sum()) for tag,mat in out.items()}
            assert all(np.isfinite(mat).all() for mat in out.values())
            np.savez_compressed(OUT/('logits_'+kind+'.npz'),**out)
            print(kind,mismatches,flush=True)


if __name__=='__main__':
    main()
