#-*- coding: utf-8 -*-
"""Gate calibration curve: visual gate w vs remaining-frame ratio r = 1 - p (Fig. 5a).

Trained with the same protocol as cgma.py --ablate no_proxy. At test time,
visual frames are removed at probabilities p in {0, 0.25, 0.5, 0.75, 0.9, 1.0}
and the mean visual gate output is compared with r = 1 - p.
--ablate no_proxy    : final CGMA (gate tracks retention continuously).
--ablate no_frameaug : control trained only at endpoints (gate responds mainly
                       at complete removal).
Descriptive probe over seeds {0, 1, 42}.
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

def apply_drop(V, A, mv, ma, mode):
    if mode == 'drop_v': return torch.zeros_like(V), A, torch.zeros_like(mv), ma
    if mode == 'drop_a': return V, torch.zeros_like(A), mv, torch.zeros_like(ma)
    return V, A, mv, ma

def frame_drop(X, m, p):
    drop = (torch.rand(m.shape, device=X.device) < p) & m
    X = X.clone(); m = m.clone(); X[drop] = 0; m[drop.bool()] = 0
    return X, m

class Fusion(nn.Module):
    """Same architecture as the main CGMA model (includes the unused proxy head so parameter-init RNG matches)."""
    def __init__(self, ablate='no_proxy', hid=128, nclass=2):
        super().__init__(); self.ablate = ablate
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
        if self.ablate in ('no_proxy', 'no_frameaug'):
            h_v = wv * v_vec; h_a = wa * a_vec
        elif self.ablate == 'no_comp':
            h_v = 0.5 * v_vec + 0.5 * pv; h_a = 0.5 * a_vec + 0.5 * pa
        else:
            h_v = wv * v_vec + (1 - wv) * pv; h_a = wa * a_vec + (1 - wa) * pa
        g = self.gate(torch.cat([h_v, h_a], -1))
        return self.fc(torch.cat([h_v, g * h_a], -1)), v_vec, a_vec, wv, wa

@torch.no_grad()
def eval_frame(model, loader, dev, ps):
    model.eval(); out = {}
    for p in ps:
        ys = []; preds = []
        for V, A, mv, ma, y in loader:
            V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
            V, mv = frame_drop(V, mv, p)
            preds += model(V, A, mv, ma)[0].argmax(1).cpu().tolist(); ys += y.tolist()
        out[p] = f1_score(ys, preds, zero_division=0)
    return out

@torch.no_grad()
def eval_mode(model, loader, dev, mode):
    model.eval(); ys = []; preds = []
    for V, A, mv, ma, y in loader:
        V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
        V, A, mv, ma = apply_drop(V, A, mv, ma, mode)
        preds += model(V, A, mv, ma)[0].argmax(1).cpu().tolist(); ys += y.tolist()
    return f1_score(ys, preds, zero_division=0)

@torch.no_grad()
def eval_w(model, loader, dev, mode):
    model.eval(); wvs = []; was = []
    for V, A, mv, ma, y in loader:
        V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
        V, A, mv, ma = apply_drop(V, A, mv, ma, mode)
        _, _, _, wv, wa = model(V, A, mv, ma)
        wvs += wv.squeeze(-1).cpu().tolist(); was += wa.squeeze(-1).cpu().tolist()
    return float(np.mean(wvs)), float(np.mean(was))

@torch.no_grad()
def eval_w_frame(model, loader, dev, p):
    """Mean gate output under visual frame-drop ratio p; w_a is the unperturbed control."""
    model.eval(); wvs = []; was = []
    for V, A, mv, ma, y in loader:
        V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
        V, mv = frame_drop(V, mv, p)
        _, _, _, wv, wa = model(V, A, mv, ma)
        wvs += wv.squeeze(-1).cpu().tolist(); was += wa.squeeze(-1).cpu().tolist()
    return float(np.mean(wvs)), float(np.std(wvs)), float(np.mean(was))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/lmvd')
    ap.add_argument('--ablate', choices=['full', 'no_proxy', 'no_comp', 'no_frameaug'], default='no_proxy')
    ap.add_argument('--lam_r', type=float, default=1.0)
    ap.add_argument('--lam_c', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hid', type=int, default=128)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--reserve_gb', type=float, default=0.0)  # pre-allocate GPU memory on a busy shared GPU (0 to disable)
    args = ap.parse_args()
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.reserve_gb > 0 and dev == 'cuda':
        _buf = torch.empty(int(args.reserve_gb * (1 << 30) // 4), dtype=torch.float32, device=dev)
        del _buf  # memory stays in the PyTorch caching allocator
    dls = {f: DataLoader(LMVDPair(args.data_dir, f), batch_size=args.batch_size,
                         shuffle=(f == 'train'), collate_fn=collate_pair, num_workers=4)
           for f in ['train', 'valid', 'test']}
    model = Fusion(args.ablate, args.hid).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    print(f"=== interp_noproxy {args.ablate} seed={args.seed} ===", flush=True)
    best_vf1, best_state, wait = -1, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for V, A, mv, ma, y in dls['train']:
            V, A, mv, ma, y = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev), y.to(dev)
            v_r, a_r = model.encode(V, A, mv, ma)
            mv0 = mv.float().sum(1, keepdim=True).clamp(min=1); ma0 = ma.float().sum(1, keepdim=True).clamp(min=1)
            Vd, Ad, mvd, mad = V, A, mv, ma
            r = random.random()
            if r < 0.33:
                Vd, Ad, mvd, mad = apply_drop(V, A, mv, ma, random.choice(['drop_v', 'drop_a']))
            elif r < 0.66 and args.ablate != 'no_frameaug':
                p = random.random() * 0.95
                if random.random() < 0.5: Vd, mvd = frame_drop(V, mv, p)
                else: Ad, mad = frame_drop(A, ma, p)
            out, _, _, wv, wa = model(Vd, Ad, mvd, mad)
            loss = ce(out, y)
            pv = model.proxy_v(a_r.detach()); pa = model.proxy_a(v_r.detach())
            loss = loss + args.lam_r * (F.mse_loss(pv, v_r.detach()) + F.mse_loss(pa, a_r.detach()))
            pres_v = (mvd.float().sum(1, keepdim=True) / mv0).clamp(0, 1)
            pres_a = (mad.float().sum(1, keepdim=True) / ma0).clamp(0, 1)
            loss = loss + args.lam_c * (F.binary_cross_entropy(wv, pres_v) + F.binary_cross_entropy(wa, pres_a))
            opt.zero_grad(); loss.backward(); opt.step()
        vf1 = eval_frame(model, dls['valid'], dev, [0.0])[0.0]
        if vf1 > best_vf1:
            best_vf1 = vf1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if wait >= args.patience: break
    model.load_state_dict(best_state)
    # deterministic check line
    print(f">>> [chk|{args.ablate}|seed{args.seed}] f0={eval_frame(model,dls['test'],dev,[0.0])[0.0]:.4f} "
          f"dropV={eval_mode(model,dls['test'],dev,'drop_v'):.4f} "
          f"dropA={eval_mode(model,dls['test'],dev,'drop_a'):.4f}", flush=True)
    # gate w under the three conditions
    for mode in ['full', 'drop_v', 'drop_a']:
        wv, wa = eval_w(model, dls['test'], dev, mode)
        print(f">>> [w|{args.ablate}|seed{args.seed}|{mode:<6}]  w_video={wv:.3f}  w_audio={wa:.3f}", flush=True)
    # w-r curve: visual frame-drop levels p, with r_v = 1-p
    for p in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
        wv_m, wv_s, wa_m = eval_w_frame(model, dls['test'], dev, p)
        print(f">>> [wr|{args.ablate}|seed{args.seed}] p={p:.2f} r={1-p:.2f} "
              f"w_v={wv_m:.3f}+-{wv_s:.3f} w_a={wa_m:.3f}", flush=True)

if __name__ == '__main__':
    main()
