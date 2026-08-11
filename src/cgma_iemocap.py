#-*- coding: utf-8 -*-
"""IEMOCAP transfer probe (Table 7, Fig. 5b).

Adapts the gating-and-augmentation design to trimodal utterance-level
emotion recognition (DialogueRNN-format features; text 100 / audio 1582 /
visual 342; session-level 120/31 dialogue split). --ablate ours applies
modality-specific gates with binary utterance-availability supervision and
whole-modality / utterance-level training perturbations; --ablate naive is
the ungated, unaugmented control. Reports weighted-F1.
Paper protocol: --epochs 60 --patience 12, seeds {42, 1, 2, 3, 123}.
"""
import os, argparse, random, pickle
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

DIMS = {'text': 100, 'audio': 1582, 'visual': 342}
MODS = ['text', 'audio', 'visual']

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class IEMOCAPDS(Dataset):
    """One sample = one dialogue (utterance sequence). Returns three-modality features + labels + valid-utterance mask."""
    def __init__(self, pkl, vids, stats=None):
        self.vid, self.spk, self.lab, self.txt, self.aud, self.vis = pkl[0], pkl[1], pkl[2], pkl[3], pkl[4], pkl[5]
        self.vids = list(vids)
        self.stats = stats  # per-modality (mean, std) computed on the training set
    def __len__(self): return len(self.vids)
    def _norm(self, x, m):
        if self.stats is None: return x
        mu, sd = self.stats[m]
        return (x - mu) / sd
    def __getitem__(self, i):
        k = self.vids[i]
        t = self._norm(np.array(self.txt[k], dtype='float32'), 'text')
        a = self._norm(np.array(self.aud[k], dtype='float32'), 'audio')
        v = self._norm(np.array(self.vis[k], dtype='float32'), 'visual')
        y = np.array(self.lab[k], dtype='int64')
        return t, a, v, y

def collate(batch):
    U = max(len(b[3]) for b in batch)
    B = len(batch)
    out = {}
    for mi, m in enumerate(MODS):
        d = DIMS[m]
        X = np.zeros((B, U, d), dtype='float32')
        for bi, b in enumerate(batch):
            x = b[mi]; X[bi, :len(x)] = x
        out[m] = torch.from_numpy(X)
    Y = np.full((B, U), -1, dtype='int64')
    M = np.zeros((B, U), dtype='float32')
    for bi, b in enumerate(batch):
        y = b[3]; Y[bi, :len(y)] = y; M[bi, :len(y)] = 1
    return out, torch.from_numpy(Y), torch.from_numpy(M)

def compute_stats(pkl, trn):
    """Per-modality z-score statistics on the training set (utterances concatenated across dialogues)."""
    stats = {}
    src = {'text': pkl[3], 'audio': pkl[4], 'visual': pkl[5]}
    for m in MODS:
        arr = np.concatenate([np.array(src[m][k], dtype='float32') for k in trn], axis=0)
        mu = arr.mean(0); sd = arr.std(0); sd[sd < 1e-6] = 1.0
        stats[m] = (mu, sd)
    return stats

def drop_modal(X, M, m):
    """Whole-modality missingness: zero out the modality and clear its presence mask."""
    pres = {k: M.clone() for k in MODS}
    X = {k: v.clone() for k, v in X.items()}
    X[m] = torch.zeros_like(X[m]); pres[m] = torch.zeros_like(M)
    return X, pres

def drop_modal_frac(X, M, m, p):
    """Utterance-level missingness: randomly drop a fraction p of valid utterances for the modality; returns perturbed X and per-modality presence masks."""
    pres = {k: M.clone() for k in MODS}
    X = {k: v.clone() for k, v in X.items()}
    drop = (torch.rand_like(M) < p) * M
    X[m] = X[m] * (1 - drop).unsqueeze(-1)
    pres[m] = M * (1 - drop)
    return X, pres

class Fusion(nn.Module):
    def __init__(self, ablate='ours', hid=128, nclass=6):
        super().__init__(); self.ablate = ablate
        self.proj = nn.ModuleDict({m: nn.Linear(DIMS[m], hid) for m in MODS})
        self.lstm = nn.ModuleDict({m: nn.LSTM(hid, hid, batch_first=True, bidirectional=True) for m in MODS})
        D = hid * 2
        self.comp = nn.ModuleDict({m: nn.Sequential(nn.Linear(D, D // 2), nn.ReLU(),
                                                    nn.Linear(D // 2, 1), nn.Sigmoid()) for m in MODS})
        self.fuse_gate = nn.Sequential(nn.Linear(D * 3, D * 3), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(D * 3, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, nclass))
    def forward(self, X, pres):
        reps, ws = [], {}
        for m in MODS:
            h = torch.relu(self.proj[m](X[m]))
            seq, _ = self.lstm[m](h)                      # B,U,D
            if self.ablate == 'ours':
                w = self.comp[m](seq) * pres[m].unsqueeze(-1)   # B,U,1 completeness gate (masks missing utterances)
                reps.append(w * seq); ws[m] = w.squeeze(-1)
            else:
                reps.append(seq)                          # naive: no gating
        fused = torch.cat(reps, -1)                       # B,U,3D
        if self.ablate == 'ours':
            fused = self.fuse_gate(fused) * fused
        return self.fc(fused), ws                         # B,U,nclass

@torch.no_grad()
def evaluate(model, loader, dev, drop=None, frac=None):
    model.eval(); ys, ps = [], []
    for X, Y, M in loader:
        X = {m: v.to(dev) for m, v in X.items()}; M = M.to(dev)
        if drop is not None and frac is None:
            Xd, pres = drop_modal(X, M, drop)
        elif drop is not None:
            Xd, pres = drop_modal_frac(X, M, drop, frac)
        else:
            Xd, pres = X, {m: M for m in MODS}
        out, _ = model(Xd, pres)
        valid = M.bool().view(-1)
        pred = out.view(-1, out.size(-1)).argmax(-1)[valid].cpu().tolist()
        gt = Y.view(-1).to(dev)[valid].cpu().tolist()
        ps += pred; ys += gt
    return dict(wf1=f1_score(ys, ps, average='weighted', zero_division=0),
                acc=accuracy_score(ys, ps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./data/IEMOCAP_features.pkl')
    ap.add_argument('--ablate', choices=['ours', 'naive'], default='ours')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--hid', type=int, default=128)
    ap.add_argument('--lam_c', type=float, default=0.5)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    pkl = pickle.load(open(args.data, 'rb'), encoding='latin1')
    trainvid, testvid = list(pkl[7]), list(pkl[8])
    # hold out 10% of training dialogues for validation (deterministic)
    rng = random.Random(0); rng.shuffle(trainvid)
    nval = max(1, len(trainvid) // 10)
    valvid = trainvid[:nval]; trnvid = trainvid[nval:]
    stats = compute_stats(pkl, trnvid)
    mk = lambda vids, sh: DataLoader(IEMOCAPDS(pkl, vids, stats), batch_size=args.batch_size,
                                     shuffle=sh, collate_fn=collate)
    dl_tr, dl_va, dl_te = mk(trnvid, True), mk(valvid, False), mk(testvid, False)

    model = Fusion(args.ablate, args.hid).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    print(f"=== iemocap ablate={args.ablate} seed={args.seed} train/val/test={len(trnvid)}/{len(valvid)}/{len(testvid)} ===", flush=True)

    best_vf1, best_state, wait = -1, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for X, Y, M in dl_tr:
            X = {m: v.to(dev) for m, v in X.items()}; Y = Y.to(dev); M = M.to(dev)
            pres = {m: M for m in MODS}
            if args.ablate == 'ours':
                r = random.random()
                if r < 0.33:                              # whole-modality missingness
                    X, pres = drop_modal(X, M, random.choice(MODS))
                elif r < 0.66:                            # utterance-level partial missingness
                    X, pres = drop_modal_frac(X, M, random.choice(MODS), random.random() * 0.95)
            out, ws = model(X, pres)
            loss = ce(out.view(-1, out.size(-1)), Y.view(-1))
            if args.ablate == 'ours':                     # completeness soft supervision
                vm = M.bool()
                for m in MODS:
                    loss = loss + args.lam_c * F.binary_cross_entropy(ws[m][vm], pres[m][vm])
            opt.zero_grad(); loss.backward(); opt.step()
        vf1 = evaluate(model, dl_va, dev)['wf1']
        if vf1 > best_vf1:
            best_vf1 = vf1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if wait >= args.patience: break
    model.load_state_dict(best_state)

    full = evaluate(model, dl_te, dev)
    dt = evaluate(model, dl_te, dev, drop='text')
    da = evaluate(model, dl_te, dev, drop='audio')
    dv = evaluate(model, dl_te, dev, drop='visual')
    frac = {p: evaluate(model, dl_te, dev, drop='text', frac=p)['wf1'] for p in [0.25, 0.5, 0.75, 1.0]}
    print(f">>> [iemocap|{args.ablate}|seed{args.seed}] intact wF1={full['wf1']:.4f} acc={full['acc']:.4f} | "
          f"drop-text={dt['wf1']:.4f} drop-audio={da['wf1']:.4f} drop-visual={dv['wf1']:.4f} | "
          f"utterance-level drop-text f25={frac[0.25]:.4f} f50={frac[0.5]:.4f} f75={frac[0.75]:.4f} f100={frac[1.0]:.4f}", flush=True)

if __name__ == '__main__':
    main()
