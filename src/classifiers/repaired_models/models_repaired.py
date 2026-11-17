"""Repaired classifier topologies (declared migration, NOT bit-level originals).

Source of truth:
  * FS-Net topology: WSPTTH/FS-Net model.py (encoder/decoder BiGRU + reconstruction),
    adapted in shared/models.py.
  * TST: pinned gzerveas/mvts_transformer commit, loaded and adapted to the
    PyTorch>=2 TransformerEncoderLayer signature.
  * SDAE: two-layer greedy denoising pre-training + supervised fine-tuning
    (implemented in train_repaired.py, not here).
"""
import importlib.util
import torch
from pathlib import Path
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MultiBiGRU(nn.Module):
    def __init__(self, input_dim, hidden, layers=2, dropout=.2):
        super().__init__()
        self.hidden = hidden
        self.layers = nn.ModuleList([
            nn.GRU(input_dim if i == 0 else 2 * hidden, hidden, bidirectional=True, batch_first=True)
            for i in range(layers)])
        self.initial = nn.ParameterList([nn.Parameter(torch.zeros(2, 1, hidden)) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        states, outputs = [], []
        for rnn, h0 in zip(self.layers, self.initial):
            packed = pack_padded_sequence(self.dropout(x), lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            result, h = rnn(packed, h0.expand(-1, x.shape[0], -1).contiguous())
            x, _ = pad_packed_sequence(result, batch_first=True, total_length=x.shape[1])
            outputs.append(x)
            states.append(h.transpose(0, 1).reshape(x.shape[0], -1))
        return torch.cat(states, 1), torch.cat(outputs, 2)


class FSNet(nn.Module):
    """Original FS-Net topology: BiGRU encoder, BiGRU decoder over replicated
    encoding, feature head on concat(enc, dec), plus reconstruction branch."""

    def __init__(self, classes, hidden=128, layers=2, vocab=3202, **kw):
        super().__init__()
        width = 2 * hidden * layers
        self.embedding = nn.Embedding(vocab, 16, padding_idx=0)
        self.encoder = MultiBiGRU(16, hidden, layers)
        self.decoder = MultiBiGRU(width, hidden, layers)
        self.reconstruct = nn.Sequential(nn.Linear(width, hidden), nn.SELU(),
                                         nn.Linear(hidden, vocab))
        self.head = nn.Sequential(nn.Linear(2 * width, 2 * hidden), nn.SELU(),
                                  nn.Dropout(.2), nn.Linear(2 * hidden, classes))

    def features(self, x, lengths):
        ids = (x * 1600).round().long().clamp(-1600, 1600) + 1601
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        ids = ids.masked_fill(~mask, 0)
        enc, _ = self.encoder(self.embedding(ids), lengths)
        dec, out = self.decoder(enc[:, None, :].expand(-1, x.shape[1], -1), lengths)
        return self.head(torch.cat([enc, dec], 1)), out, ids, mask

    def forward(self, x, lengths):
        return self.features(x, lengths)[0]

    def supervised_loss(self, x, lengths, y):
        logits, out, ids, mask = self.features(x, lengths)
        rec = nn.functional.cross_entropy(self.reconstruct(out)[mask], ids[mask])
        return nn.functional.cross_entropy(logits, y) + .5 * rec


def make_tst(classes, seq_len=512, **kw):
    path = Path(__file__).resolve().parent.parent / 'upstream/mvts_transformer/src/models/ts_transformer.py'
    if not path.exists():
        raise FileNotFoundError(f'pinned TST not found at {path}')
    spec = importlib.util.spec_from_file_location('nuwa_original_tst', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.TransformerBatchNormEncoderLayer.forward

    def adapted(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if is_causal:
            raise ValueError('causal attention not used')
        return original(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    module.TransformerBatchNormEncoderLayer.forward = adapted
    base = module.TSTransformerEncoderClassiregressor(
        1, seq_len, 64, 8, 3, 256, num_classes=classes, dropout=.1,
        pos_encoding='fixed', activation='gelu', norm='BatchNorm')

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base

        def forward(self, x, lengths):
            # pinned TST builds fixed positional encoding for seq_len; pad to it.
            if x.size(1) < seq_len:
                x = nn.functional.pad(x, (0, seq_len - x.size(1)))
            valid = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
            return self.base(x.unsqueeze(-1), valid)

    return Wrapper()


def build(name, classes, seq_len=512):
    if name == 'fsnet':
        return FSNet(classes)
    if name == 'tst':
        return make_tst(classes, seq_len=seq_len)
    raise ValueError(name)
