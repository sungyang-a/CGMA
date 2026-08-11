#-*- coding: utf-8 -*-
"""Modality-level missingness-ratio curve for MMIN-core (Table 2 / Fig. 3a).

The MMIN-core re-implementation (see mmin_core.py) evaluated on the same
modality-ratio curve protocol as ratio_curve.py: at each ratio r, one modality
(vision or audio, equal probability) is removed from each selected sample.
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
    """Frame-level: randomly drop a fraction p of valid visual frames (intermediate states)."""
    drop = (torch.rand(m.shape, device=X.device) < p) & m
    X = X.clone(); m = m.clone(); X[drop] = 0; m[drop.bool()] = 0
    return X, m

def drop_ratio(V, A, mv, ma, r):
    """At test time, drop one modality (vision or audio, equal probability) per sample with probability r.
    MMIN-core baseline for the modality-ratio curve (Table 2 / Fig. 3a)."""
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
            ps += model(V, A, mv, ma)[0].argmax(1).cpu().tolist(); ys += y.tolist()
        out[r] = f1_score(ys, ps, zero_division=0)
    return out

class ResAEBlock(nn.Module):
    """A single residual autoencoder block: x -> enc -> dec, returns x + residual."""
    def __init__(self, dim, hid):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hid), nn.ReLU())
        self.dec = nn.Linear(hid, dim)
    def forward(self, x):
        z = self.enc(x)
        return x + self.dec(z), z

class CRA(nn.Module):
    """Cascade residual autoencoder: stacked blocks that progressively refine the joint representation; returns the final representation and the last latent."""
    def __init__(self, dim, hid, n_block=3):
        super().__init__()
        self.blocks = nn.ModuleList([ResAEBlock(dim, hid) for _ in range(n_block)])
    def forward(self, x):
        z = None
        for b in self.blocks:
            x, z = b(x)
        return x, z

class MMIN(nn.Module):
    def __init__(self, hid=128, nclass=2):
        super().__init__()
        self.v_lstm = nn.LSTM(136, hid, batch_first=True, bidirectional=True)
        self.a_lstm = nn.LSTM(128, hid, batch_first=True, bidirectional=True)
        D = hid * 2                      # per-modality vector dim = 256
        self.J = D * 2                   # joint (concat) dim = 512
        self.cra = CRA(self.J, hid, n_block=3)
        self.cls = nn.Sequential(nn.Linear(self.J, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, nclass))
    def encode(self, V, A, mv, ma):
        return masked_mean(self.v_lstm(V)[0], mv), masked_mean(self.a_lstm(A)[0], ma)
    def forward(self, V, A, mv, ma):
        v, a = self.encode(V, A, mv, ma)
        joint = torch.cat([v, a], -1)            # a missing modality's half is ~0 (masked mean)
        recon, _ = self.cra(joint)               # imagined/refined joint representation
        return self.cls(recon), joint, recon

@torch.no_grad()
def evaluate(model, loader, dev, mode=None, pv=None):
    model.eval(); ys, ps = [], []
    for V, A, mv, ma, y in loader:
        V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
        if mode: V, A, mv, ma = apply_drop(V, A, mv, ma, mode)
        if pv is not None: V, mv = frame_drop(V, mv, pv)
        ps += model(V, A, mv, ma)[0].argmax(1).cpu().tolist(); ys += y.tolist()
    return (f1_score(ys, ps, zero_division=0),
            f1_score(ys, ps, average='macro', zero_division=0),
            accuracy_score(ys, ps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/lmvd')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hid', type=int, default=128)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--lam_img', type=float, default=1.0)   # imagination (teacher-student) weight
    ap.add_argument('--lam_cyc', type=float, default=0.5)   # cycle-consistency weight
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dls = {f: DataLoader(LMVDPair(args.data_dir, f), batch_size=args.batch_size,
                         shuffle=(f == 'train'), collate_fn=collate_pair, num_workers=4)
           for f in ['train', 'valid', 'test']}
    model = MMIN(args.hid).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    print(f"=== MMIN (re-impl, shared backbone) seed={args.seed} ===", flush=True)
    best_vf1, best_state, wait = -1, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for V, A, mv, ma, y in dls['train']:
            V, A, mv, ma, y = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev), y.to(dev)
            # teacher: joint representation of the complete input (detached target)
            with torch.no_grad():
                _, joint_full, recon_full = model(V, A, mv, ma)
            # student: imagine from a randomly corrupted input
            mode = random.choice(['full', 'drop_v', 'drop_a'])
            Vd, Ad, mvd, mad = apply_drop(V, A, mv, ma, mode)
            out, joint_miss, recon_miss = model(Vd, Ad, mvd, mad)
            loss = ce(out, y)
            loss = loss + args.lam_img * F.mse_loss(recon_miss, recon_full.detach())   # imagination loss
            # cycle consistency: the imagined representation should reconstruct the joint one
            loss = loss + args.lam_cyc * F.mse_loss(recon_miss, joint_full.detach())
            opt.zero_grad(); loss.backward(); opt.step()
        vf1 = evaluate(model, dls['valid'], dev, 'full')[0]
        if vf1 > best_vf1:
            best_vf1 = vf1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if wait >= args.patience: break
    model.load_state_dict(best_state)
    ff, fm, fa = evaluate(model, dls['test'], dev, 'full')
    dv = evaluate(model, dls['test'], dev, 'drop_v')
    da = evaluate(model, dls['test'], dev, 'drop_a')
    # frame-level curve up to f90 (f100 = whole-modality drop, reported separately as dropV)
    fr = {p: evaluate(model, dls['test'], dev, pv=p)[0] for p in [0.0, 0.25, 0.5, 0.75, 0.9]}
    print(f">>> [mmin|seed{args.seed}] full={ff:.4f} dropV={dv[0]:.4f} dropA={da[0]:.4f}  "
          f"frame-level f0={fr[0.0]:.4f} f25={fr[0.25]:.4f} f50={fr[0.5]:.4f} f75={fr[0.75]:.4f} f90={fr[0.9]:.4f}  "
          f"(acc: full{fa:.3f}/dV{dv[2]:.3f} | macro: full{fm:.3f}/dV{dv[1]:.3f})", flush=True)
    rc = eval_ratio(model, dls['test'], dev, [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    print(f">>> [mmin_ratio|seed{args.seed}] r0={rc[0.0]:.4f} r10={rc[0.1]:.4f} r25={rc[0.25]:.4f} "
          f"r50={rc[0.5]:.4f} r75={rc[0.75]:.4f} r100={rc[1.0]:.4f}", flush=True)

if __name__ == '__main__':
    main()
