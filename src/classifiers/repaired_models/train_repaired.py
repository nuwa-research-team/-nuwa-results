# -*- coding: utf-8 -*-
"""One bounded revision per simplified classifier: TST / FS-Net / SDAE.

Protocol (frozen before any fit):
  seed 42, same train/val/test roles as the engineering table, no test-based
  selection, supervised epochs <=15, patience 4, model selected on val accuracy.
  TST: pinned original mvts_transformer implementation (adapter only).
  FS-Net: original encoder/decoder + reconstruction loss weight 0.5.
  SDAE: 2 greedy denoising pre-training stages (<=5 epochs each, mask noise 0.15,
        MSE reconstruction), then <=15 epochs supervised fine-tuning.
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models_repaired as MR

ROOT = Path('/data1/lhy/nuwa')
SEED = 42


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class SeqDS(Dataset):
    def __init__(self, seqs, y, idx, scale):
        self.seqs, self.y, self.idx, self.scale = seqs, y, list(idx), scale

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        s = self.seqs[i]
        return torch.tensor(s, dtype=torch.float32) * self.scale, len(s), int(self.y[i])


def collate(batch):
    xs, ls, ys = zip(*batch)
    L = max(ls)
    out = torch.zeros(len(xs), L)
    for i, x in enumerate(xs):
        out[i, :len(x)] = x
    return out, torch.tensor(ls), torch.tensor(ys)


def loader(seqs, y, idx, scale, batch, shuffle, seed=SEED):
    ds = SeqDS(seqs, y, idx, scale)
    order = np.argsort([len(seqs[i]) for i in idx])
    if shuffle:
        chunk = batch * 20
        rng = np.random.default_rng(seed)
        parts = [order[i:i + chunk] for i in range(0, len(order), chunk)]
        for p in parts:
            rng.shuffle(p)
        order = np.concatenate(parts)
    return DataLoader(ds, batch_size=batch, sampler=list(order), collate_fn=collate,
                      drop_last=shuffle)


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    ys, ps = [], []
    for x, l, y in dl:
        ps.append(model(x.to(dev), l.to(dev)).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float((y == p).mean()), y, p


def sup_train(model, tr, va, dev, epochs_max, patience, lr, fsnet=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(epochs_max):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, l, y in tr:
            x, l, y = x.to(dev), l.to(dev), y.to(dev)
            opt.zero_grad()
            loss = model.supervised_loss(x, l, y) if fsnet else nn.functional.cross_entropy(model(x, l), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(y)
            n += len(y)
        acc, _, _ = evaluate(model, va, dev)
        hist.append({'epoch': ep, 'train_loss': round(tot / n, 4), 'val_acc': round(acc, 4),
                     'seconds': round(time.time() - t0, 1)})
        print(json.dumps(hist[-1]), flush=True)
        if acc > best + 1e-4:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist, best


def sdae_greedy(seqs, y, tr_idx, va_idx, scale, dev, out_dir, log):
    """Two-layer greedy denoising pre-training on the 512-padded input."""
    W = 512
    X = np.zeros((len(tr_idx), W), dtype=np.float32)
    for k, i in enumerate(tr_idx):
        s = np.asarray(seqs[i][:W], dtype=np.float32) * scale
        X[k, :len(s)] = s
    Xt = torch.tensor(X)
    dl = DataLoader(torch.utils.data.TensorDataset(Xt), batch_size=256, shuffle=True,
                    drop_last=True, generator=torch.Generator().manual_seed(SEED))

    l1 = nn.Sequential(nn.Linear(W, 256), nn.ReLU()).to(dev)
    d1 = nn.Sequential(nn.Linear(256, W)).to(dev)
    opt = torch.optim.Adam(list(l1.parameters()) + list(d1.parameters()), lr=1e-3)
    h1 = []
    for ep in range(5):
        tot, n = 0.0, 0
        for (xb,) in dl:
            xb = xb.to(dev)
            noisy = xb.masked_fill(torch.rand_like(xb) < 0.15, 0)
            out = d1(l1(noisy))
            loss = nn.functional.mse_loss(out, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(xb); n += len(xb)
        h1.append(round(tot / n, 6))
        log(f'sdae pretrain layer1 ep{ep} mse={h1[-1]}')
        if len(h1) >= 2 and abs(h1[-2] - h1[-1]) < 1e-5:
            break

    with torch.no_grad():
        H = l1(Xt.to(dev)).cpu()
    l2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU()).to(dev)
    d2 = nn.Sequential(nn.Linear(128, 256)).to(dev)
    opt = torch.optim.Adam(list(l2.parameters()) + list(d2.parameters()), lr=1e-3)
    dl2 = DataLoader(torch.utils.data.TensorDataset(H), batch_size=256, shuffle=True,
                     drop_last=True, generator=torch.Generator().manual_seed(SEED))
    h2 = []
    for ep in range(5):
        tot, n = 0.0, 0
        for (xb,) in dl2:
            xb = xb.to(dev)
            noisy = xb.masked_fill(torch.rand_like(xb) < 0.15, 0)
            out = d2(l2(noisy))
            loss = nn.functional.mse_loss(out, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(xb); n += len(xb)
        h2.append(round(tot / n, 6))
        log(f'sdae pretrain layer2 ep{ep} mse={h2[-1]}')
        if len(h2) >= 2 and abs(h2[-2] - h2[-1]) < 1e-5:
            break
    return l1, l2, {'layer1_mse': h1, 'layer2_mse': h2}


class SDAERepaired(nn.Module):
    def __init__(self, classes, W=512, l1=None, l2=None):
        super().__init__()
        self.l1 = l1 if l1 is not None else nn.Sequential(nn.Linear(W, 256), nn.ReLU())
        self.l2 = l2 if l2 is not None else nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.head = nn.Linear(128, classes)

    def forward(self, x, lengths):
        h = x.flatten(1)
        if h.size(1) < 512:
            h = nn.functional.pad(h, (0, 512 - h.size(1)))
        return self.head(self.l2(self.l1(h)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['tst', 'fsnet', 'sdae'])
    ap.add_argument('--out', default=str(ROOT / 'repaired'))
    ap.add_argument('--gpu', default='1')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED); np.random.seed(SEED)

    out = Path(args.out)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    logf = open(out / 'logs' / f'repair_{args.model}.log', 'a')

    def log(msg):
        line = time.strftime('%H:%M:%S ') + str(msg)
        print(line, flush=True)
        logf.write(line + '\n'); logf.flush()

    d = pickle.load(open(ROOT / 'work/d4_aligned/raw.pkl', 'rb'))
    seqs, y, labels = d['seqs'], d['y'], d['labels']
    tr, va, te = d['split']['train'], d['split']['val'], d['split']['test']
    n_cls = len(labels)
    scale = 1.0 / 1600.0

    log(f'start {args.model} seed={SEED} epochs<={args.epochs} patience={args.patience} scale={scale}')

    if args.model == 'sdae':
        t_pre = time.time()
        l1, l2, pre = sdae_greedy(seqs, y, tr, va, scale, dev, out, log)
        pre['pretrain_seconds'] = round(time.time() - t_pre, 1)
        model = SDAERepaired(n_cls, l1=l1, l2=l2)
    elif args.model == 'tst':
        pre = {}
        model = MR.build('tst', n_cls, seq_len=512)
    else:
        pre = {}
        model = MR.build('fsnet', n_cls)
    model.to(dev)

    trl = loader(seqs, y, tr, scale, args.batch, True)
    val = loader(seqs, y, va, scale, 512, False)
    t0 = time.time()
    model, hist, best = sup_train(model, trl, val, dev, args.epochs, args.patience,
                                  args.lr, fsnet=(args.model == 'fsnet'))
    train_s = time.time() - t0
    torch.save({'state': model.state_dict(), 'labels': labels, 'model': args.model},
               out / f'repaired_{args.model}.pt')
    meta = {'model': args.model, 'seed': SEED, 'epochs_max': args.epochs,
            'patience': args.patience, 'batch': args.batch, 'lr': args.lr,
            'input_scale': scale, 'val_acc_best': best, 'epochs_run': len(hist),
            'train_seconds': round(train_s, 1), 'device': str(dev),
            'pretraining': pre, 'history': hist,
            'stage_accounting': {'pretrain_seconds': pre.get('pretrain_seconds', 0.0),
                                 'supervised_seconds': round(train_s, 1)},
            'data_sha256': sha(ROOT / 'work/d4_aligned/raw.pkl'),
            'code_sha256': sha(Path(__file__)),
            'models_code_sha256': sha(Path(MR.__file__)),
            'declaration': 'declared migration / mechanism reproduction, not bit-level original'}
    json.dump(meta, open(out / f'repaired_{args.model}.json', 'w'), indent=2)
    log(f'done {args.model}: val_acc={best:.4f} epochs={len(hist)} {train_s:.1f}s')


if __name__ == '__main__':
    main()
