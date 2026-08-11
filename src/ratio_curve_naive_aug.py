#-*- coding: utf-8 -*-
"""Modality-level missingness-ratio curve for Naive + Aug (Table 2 / Fig. 3a).

The Naive + Aug configuration (see naive_aug.py) evaluated on the same
modality-ratio curve protocol as ratio_curve.py.
Paper protocol: --epochs 45 --patience 12, seeds {0, 1, 2, 42, 123}.
"""
import os, argparse, random, csv
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class LMVDPair(Dataset):
    def __init__(self, root, fold):
        self.root = root; self.samples = []
        with open(os.path.join(root, 'lmvd_labels.csv')) as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                if len(row) < 3 or row[2] != fold: continue
                self.samples.append((row[0], int(row[1] == 'depression')))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        sid, y = self.samples[i]
        v = np.nan_to_num(np.load(os.path.join(self.root, 'visual', sid + '_visual.npy')).astype('float32'))
        a = np.nan_to_num(np.load(os.path.join(self.root, 'audio', sid + '.npy')).astype('float32'))
        return torch.from_numpy(v), torch.from_numpy(a), y

def collate_pair(batch):
    vs, as_, ys = zip(*batch)
    V = pad_sequence(vs, batch_first=True); A = pad_sequence(as_, batch_first=True)
    mv = (V.abs().sum(-1) != 0); ma = (A.abs().sum(-1) != 0)
    return V, A, mv, ma, torch.tensor(ys)

def masked_mean(seq, mask):
    m = mask.unsqueeze(-1).float()
    return (seq * m).sum(1) / m.sum(1).clamp(min=1)

def apply_drop(V, A, mv, ma, mode):
    if mode == 'drop_v': return torch.zeros_like(V), A, torch.zeros_like(mv), ma
    if mode == 'drop_a': return V, torch.zeros_like(A), mv, torch.zeros_like(ma)
    return V, A, mv, ma

def frame_drop(X, m, p):
    drop = (torch.rand(m.shape, device=X.device) < p) & m
    X = X.clone(); m = m.clone(); X[drop] = 0; m[drop.bool()] = 0
    return X, m

def drop_ratio(V, A, mv, ma, r):
    """At test time, drop one modality (vision or audio, equal probability) per sample with probability r.
    Naive + Aug baseline for the modality-ratio curve (Table 2)."""
    B = V.size(0); dev = V.device
    drop = torch.rand(B, device=dev) < r
    which_v = torch.rand(B, device=dev) < 0.5
    dv = drop & which_v; da = drop & ~which_v
    V = V.clone(); A = A.clone(); mv = mv.clone(); ma = ma.clone()
    V[dv] = 0; mv[dv] = 0; A[da] = 0; ma[da] = 0
    return V, A, mv, ma

@torch.no_grad()
def eval_ratio(model, loader, dev, ratios):
    model.eval(); out = {}
    for r in ratios:
        ys, ps = [], []
        for V, A, mv, ma, y in loader:
            V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
            V, A, mv, ma = drop_ratio(V, A, mv, ma, r)
            logit = model(V, A, mv, ma)
            logit = logit[0] if isinstance(logit, tuple) else logit
            ps += logit.argmax(1).cpu().tolist(); ys += y.tolist()
        out[r] = f1_score(ys, ps, zero_division=0)
    return out

class AdaFuse(nn.Module):
    """Naive attention fusion: vision-led cross-modal attention + gated fusion; no completeness gate or supervision."""
    def __init__(self, hid=128, nhead=4, nclass=2):
        super().__init__()
        self.v_lstm = nn.LSTM(136, hid, batch_first=True, bidirectional=True)
        self.a_lstm = nn.LSTM(128, hid, batch_first=True, bidirectional=True)
        D = hid * 2
        self.attn = nn.MultiheadAttention(D, nhead, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(D * 2, D), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(D * 2, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, nclass))
    def forward(self, V, A, mv, ma):
        v_seq, _ = self.v_lstm(V); a_seq, _ = self.a_lstm(A)
        v_vec = masked_mean(v_seq, mv)
        # NaN-safe attention: if a sample has an all-False audio mask, softmax(-inf) yields NaN.
        # Guard: keep the first key visible (LSTM output on zero input is well-defined and near-uninformative).
        safe_ma = ma.clone()
        safe_ma[:, 0] = True
        a_ctx, _ = self.attn(v_vec.unsqueeze(1), a_seq, a_seq, key_padding_mask=~safe_ma)
        a_ctx = torch.nan_to_num(a_ctx.squeeze(1))
        g = self.gate(torch.cat([v_vec, a_ctx], -1))
        return self.fc(torch.cat([v_vec, g * a_ctx], -1))

@torch.no_grad()
def eval_frame(model, loader, dev, ps):
    """Evaluate over visual frame-drop probabilities ps; returns p -> (F1, macro-F1, acc)."""
    model.eval(); out = {}
    for p in ps:
        ys = []; preds = []
        for V, A, mv, ma, y in loader:
            V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
            V, mv = frame_drop(V, mv, p)
            preds += model(V, A, mv, ma).argmax(1).cpu().tolist(); ys += y.tolist()
        out[p] = (f1_score(ys, preds, zero_division=0),
                  f1_score(ys, preds, average='macro', zero_division=0),
                  accuracy_score(ys, preds))
    return out

@torch.no_grad()
def eval_mode(model, loader, dev, mode):
    model.eval(); ys = []; preds = []
    for V, A, mv, ma, y in loader:
        V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
        V, A, mv, ma = apply_drop(V, A, mv, ma, mode)
        preds += model(V, A, mv, ma).argmax(1).cpu().tolist(); ys += y.tolist()
    return (f1_score(ys, preds, zero_division=0),
            f1_score(ys, preds, average='macro', zero_division=0),
            accuracy_score(ys, preds))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/lmvd')
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hid', type=int, default=128)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--reserve_gb', type=float, default=0.0)
    ap.add_argument('--dump_probs', action='store_true')   # optional: export per-sample probabilities after training
    args = ap.parse_args()
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.reserve_gb > 0 and dev == 'cuda':
        _b = torch.empty(int(args.reserve_gb*(1<<30)//4), dtype=torch.float32, device=dev); del _b
    dls = {f: DataLoader(LMVDPair(args.data_dir, f), batch_size=args.batch_size,
                         shuffle=(f == 'train'), collate_fn=collate_pair, num_workers=4)
           for f in ['train', 'valid', 'test']}
    model = AdaFuse(args.hid).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    print(f"=== naive+aug (AdaFuse + full-spectrum augmentation, no gate, no L_comp) seed={args.seed} ===", flush=True)

    best_vf1, best_state, wait = -1, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for V, A, mv, ma, y in dls['train']:
            V, A, mv, ma, y = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev), y.to(dev)
            # full-spectrum augmentation schedule
            Vd, Ad, mvd, mad = V, A, mv, ma
            r = random.random()
            if r < 0.33:
                Vd, Ad, mvd, mad = apply_drop(V, A, mv, ma, random.choice(['drop_v', 'drop_a']))
            elif r < 0.66:
                p = random.random() * 0.95
                if random.random() < 0.5: Vd, mvd = frame_drop(V, mv, p)
                else: Ad, mad = frame_drop(A, ma, p)
            out = model(Vd, Ad, mvd, mad)
            loss = ce(out, y)        # classification loss only (no completeness term)
            opt.zero_grad(); loss.backward(); opt.step()
        vf1 = eval_frame(model, dls['valid'], dev, [0.0])[0.0][0]
        if vf1 > best_vf1:
            best_vf1 = vf1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if wait >= args.patience: break
    model.load_state_dict(best_state)
    fr = eval_frame(model, dls['test'], dev, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    m_full = eval_mode(model, dls['test'], dev, 'full')
    m_dv = eval_mode(model, dls['test'], dev, 'drop_v')
    m_da = eval_mode(model, dls['test'], dev, 'drop_a')
    print(f">>> [naive_aug|seed{args.seed}] "
          f"frame-level f0={fr[0.0][0]:.4f} f25={fr[0.25][0]:.4f} f50={fr[0.5][0]:.4f} f75={fr[0.75][0]:.4f} f90={fr[0.9][0]:.4f} f100={fr[1.0][0]:.4f}  "
          f"modality-level full={m_full[0]:.4f} dropV={m_dv[0]:.4f} dropA={m_da[0]:.4f}", flush=True)
    print(f">>> [audit|naive_aug|seed{args.seed}] "
          f"full F1/mac/acc={m_full[0]:.4f}/{m_full[1]:.4f}/{m_full[2]:.4f}  "
          f"dropV={m_dv[0]:.4f}/{m_dv[1]:.4f}/{m_dv[2]:.4f}  "
          f"dropA={m_da[0]:.4f}/{m_da[1]:.4f}/{m_da[2]:.4f}  "
          f"f50={fr[0.5][0]:.4f}/{fr[0.5][1]:.4f}/{fr[0.5][2]:.4f}", flush=True)
    rc = eval_ratio(model, dls['test'], dev, [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    print(f">>> [naug_ratio|seed{args.seed}] r0={rc[0.0]:.4f} r10={rc[0.1]:.4f} r25={rc[0.25]:.4f} "
          f"r50={rc[0.5]:.4f} r75={rc[0.75]:.4f} r100={rc[1.0]:.4f}", flush=True)
    if args.dump_probs:
        outp = os.path.join("./outputs", f"naive_aug_s{args.seed}.npz")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        dump = {}
        with torch.no_grad():
            model.eval()
            for split in ['valid', 'test']:
                ys = []
                for mi, mode in enumerate(['full', 'drop_v', 'drop_a']):
                    ps = []
                    for V, A, mv, ma, y in dls[split]:
                        V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
                        Vd, Ad, mvd, mad = apply_drop(V, A, mv, ma, mode)
                        ps += torch.softmax(model(Vd, Ad, mvd, mad), -1)[:, 1].cpu().tolist()
                        if mi == 0: ys += y.tolist()
                    dump[f'{split}_{mode}'] = np.array(ps, dtype='float32')
                dump[f'y_{split}'] = np.array(ys)
        np.savez(outp, **dump)
        print(f">>> probs -> {outp}", flush=True)

if __name__ == '__main__':
    main()
