#-*- coding: utf-8 -*-
"""Modality-level missingness-ratio curve (Table 2 / Fig. 3a).

At each ratio r, every test sample is independently selected with probability r;
if selected, one modality (vision or audio, equal probability) is removed.
--method ours    : final CGMA configuration (completeness gating + augmentation).
--method adafuse : naive attention fusion (no robustness training).
--method complete: legacy proxy + completeness variant (kept for reference).
Paper protocol: --epochs 45 --patience 12, seeds {0, 1, 2, 42, 123}.
"""
import os, argparse, random, csv
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

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

def drop_ratio(V, A, mv, ma, r):
    """At test time, drop one modality (vision or audio, equal probability) per sample with probability r."""
    B = V.size(0); dev = V.device
    drop = torch.rand(B, device=dev) < r
    which_v = torch.rand(B, device=dev) < 0.5
    dv = drop & which_v; da = drop & ~which_v
    V = V.clone(); A = A.clone(); mv = mv.clone(); ma = ma.clone()
    V[dv] = 0; mv[dv] = 0; A[da] = 0; ma[da] = 0
    return V, A, mv, ma

def apply_drop(V, A, mv, ma, mode):              # whole-modality drop (training)
    if mode == 'drop_v': return torch.zeros_like(V), A, torch.zeros_like(mv), ma
    if mode == 'drop_a': return V, torch.zeros_like(A), mv, torch.zeros_like(ma)
    return V, A, mv, ma

def frame_drop(X, m, p):                          # frame-level drop (training)
    drop = (torch.rand(m.shape, device=X.device) < p) & m
    X = X.clone(); m = m.clone(); X[drop] = 0; m[drop.bool()] = 0
    return X, m

# ---------- Naive attention fusion (no robustness training) ----------
class AdaFuse(nn.Module):
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
        a_ctx, _ = self.attn(v_vec.unsqueeze(1), a_seq, a_seq, key_padding_mask=~ma)
        a_ctx = torch.nan_to_num(a_ctx.squeeze(1))
        g = self.gate(torch.cat([v_vec, a_ctx], -1))
        return self.fc(torch.cat([v_vec, g * a_ctx], -1))

# ---------- Proxy + completeness variant ----------
class CompletenessFusion(nn.Module):
    def __init__(self, hid=128, nclass=2):
        super().__init__()
        self.v_lstm = nn.LSTM(136, hid, batch_first=True, bidirectional=True)
        self.a_lstm = nn.LSTM(128, hid, batch_first=True, bidirectional=True)
        D = hid * 2
        self.proxy_v = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.proxy_a = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.comp_v = nn.Sequential(nn.Linear(D, D // 2), nn.ReLU(), nn.Linear(D // 2, 1), nn.Sigmoid())
        self.comp_a = nn.Sequential(nn.Linear(D, D // 2), nn.ReLU(), nn.Linear(D // 2, 1), nn.Sigmoid())
        self.gate = nn.Sequential(nn.Linear(D * 2, D), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(D * 2, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, nclass))
    def encode(self, V, A, mv, ma):
        return masked_mean(self.v_lstm(V)[0], mv), masked_mean(self.a_lstm(A)[0], ma)
    def forward(self, V, A, mv, ma):
        v_vec, a_vec = self.encode(V, A, mv, ma)
        pv = self.proxy_v(a_vec); pa = self.proxy_a(v_vec)
        wv = self.comp_v(v_vec); wa = self.comp_a(a_vec)
        h_v = wv * v_vec + (1 - wv) * pv; h_a = wa * a_vec + (1 - wa) * pa
        g = self.gate(torch.cat([h_v, h_a], -1))
        return self.fc(torch.cat([h_v, g * h_a], -1)), v_vec, a_vec, wv, wa

# ---------- Final CGMA: completeness gating + full-spectrum augmentation (no proxy) ----------
class OursFusion(nn.Module):
    def __init__(self, hid=128, nclass=2):
        super().__init__()
        self.v_lstm = nn.LSTM(136, hid, batch_first=True, bidirectional=True)
        self.a_lstm = nn.LSTM(128, hid, batch_first=True, bidirectional=True)
        D = hid * 2
        self.comp_v = nn.Sequential(nn.Linear(D, D // 2), nn.ReLU(), nn.Linear(D // 2, 1), nn.Sigmoid())
        self.comp_a = nn.Sequential(nn.Linear(D, D // 2), nn.ReLU(), nn.Linear(D // 2, 1), nn.Sigmoid())
        self.gate = nn.Sequential(nn.Linear(D * 2, D), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(D * 2, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, nclass))
    def encode(self, V, A, mv, ma):
        return masked_mean(self.v_lstm(V)[0], mv), masked_mean(self.a_lstm(A)[0], ma)
    def forward(self, V, A, mv, ma):
        v_vec, a_vec = self.encode(V, A, mv, ma)
        wv = self.comp_v(v_vec); wa = self.comp_a(a_vec)
        h_v = wv * v_vec; h_a = wa * a_vec
        g = self.gate(torch.cat([h_v, h_a], -1))
        return self.fc(torch.cat([h_v, g * h_a], -1)), v_vec, a_vec, wv, wa

@torch.no_grad()
def eval_curve(model, loader, dev, method, ratios):
    model.eval(); out = {}
    for r in ratios:
        ys = []; ps = []
        for V, A, mv, ma, y in loader:
            V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
            V, A, mv, ma = drop_ratio(V, A, mv, ma, r)
            logit = model(V, A, mv, ma)[0] if method in ('complete', 'ours') else model(V, A, mv, ma)
            ps += logit.argmax(1).cpu().tolist(); ys += y.tolist()
        out[r] = f1_score(ys, ps, zero_division=0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/lmvd')
    ap.add_argument('--method', choices=['adafuse', 'complete', 'ours'], required=True)
    ap.add_argument('--drop_p', type=float, default=0.5)        # used by the 'complete' variant only
    ap.add_argument('--lam_c', type=float, default=0.5)         # completeness supervision weight (ours)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hid', type=int, default=128)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dls = {f: DataLoader(LMVDPair(args.data_dir, f), batch_size=args.batch_size,
                         shuffle=(f == 'train'), collate_fn=collate_pair, num_workers=4)
           for f in ['train', 'valid', 'test']}
    model = {'complete': CompletenessFusion, 'ours': OursFusion, 'adafuse': AdaFuse}[args.method](args.hid).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    print(f"=== curve method={args.method} drop_p={args.drop_p} seed={args.seed} ===", flush=True)
    best_vf1, best_state, wait = -1, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for V, A, mv, ma, y in dls['train']:
            V, A, mv, ma, y = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev), y.to(dev)
            if args.method == 'complete':
                v_r, a_r = model.encode(V, A, mv, ma)
                mode = random.random() < args.drop_p
                Vd, Ad, mvd, mad = drop_ratio(V, A, mv, ma, 1.0) if mode else (V, A, mv, ma)
                out, _, _, wv, wa = model(Vd, Ad, mvd, mad)
                loss = ce(out, y)
                pv = model.proxy_v(a_r.detach()); pa = model.proxy_a(v_r.detach())
                loss = loss + F.mse_loss(pv, v_r.detach()) + F.mse_loss(pa, a_r.detach())
            elif args.method == 'ours':                       # full-spectrum augmentation + completeness supervision
                mv0 = mv.float().sum(1, keepdim=True).clamp(min=1); ma0 = ma.float().sum(1, keepdim=True).clamp(min=1)
                Vd, Ad, mvd, mad = V, A, mv, ma
                rr = random.random()
                if rr < 0.33:
                    Vd, Ad, mvd, mad = apply_drop(V, A, mv, ma, random.choice(['drop_v', 'drop_a']))
                elif rr < 0.66:
                    p = random.random() * 0.95
                    if random.random() < 0.5: Vd, mvd = frame_drop(V, mv, p)
                    else: Ad, mad = frame_drop(A, ma, p)
                out, _, _, wv, wa = model(Vd, Ad, mvd, mad)
                loss = ce(out, y)
                pres_v = (mvd.float().sum(1, keepdim=True) / mv0).clamp(0, 1)
                pres_a = (mad.float().sum(1, keepdim=True) / ma0).clamp(0, 1)
                loss = loss + args.lam_c * (F.binary_cross_entropy(wv, pres_v) + F.binary_cross_entropy(wa, pres_a))
            else:
                loss = ce(model(V, A, mv, ma), y)
            opt.zero_grad(); loss.backward(); opt.step()
        # model selection on validation at r=0 (intact)
        vf1 = eval_curve(model, dls['valid'], dev, args.method, [0.0])[0.0]
        if vf1 > best_vf1:
            best_vf1 = vf1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if wait >= args.patience: break
    model.load_state_dict(best_state)
    ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    curve = eval_curve(model, dls['test'], dev, args.method, ratios)
    s = "  ".join([f"r{int(r*100)}={curve[r]:.4f}" for r in ratios])
    print(f">>> [curve|{args.method}|seed{args.seed}] {s}", flush=True)

if __name__ == '__main__':
    main()
