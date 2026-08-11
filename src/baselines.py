#-*- coding: utf-8 -*-
"""Generic missing-handling baselines under the shared backbone (Table 4).

Same BiLSTM encoders and concatenation fusion; the methods differ only in how a
missing modality is handled:
  --method zero  : zero-fill the missing modality.
  --method token : replace it with a learned missing token.
  --method ae    : reconstruct it from the available modality with an autoencoder.
Evaluates intact / drop-video / drop-audio and the visual frame-level curve.
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

class BaselineFusion(nn.Module):
    def __init__(self, method, hid=128, nclass=2):
        super().__init__()
        self.method = method
        self.v_lstm = nn.LSTM(136, hid, batch_first=True, bidirectional=True)
        self.a_lstm = nn.LSTM(128, hid, batch_first=True, bidirectional=True)
        D = hid * 2
        self.miss_v = nn.Parameter(torch.randn(D) * 0.1)   # used by the missing-token method
        self.miss_a = nn.Parameter(torch.randn(D) * 0.1)
        if method == 'ae':
            self.ae_v = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
            self.ae_a = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.fc = nn.Sequential(nn.Linear(D * 2, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, nclass))
    def encode(self, V, A, mv, ma):
        return masked_mean(self.v_lstm(V)[0], mv), masked_mean(self.a_lstm(A)[0], ma)
    def forward(self, V, A, mv, ma, mode='full'):
        v_vec, a_vec = self.encode(V, A, mv, ma)
        B = v_vec.size(0)
        if mode == 'drop_v':
            if self.method == 'zero':  v_vec = torch.zeros_like(v_vec)
            elif self.method == 'token': v_vec = self.miss_v.unsqueeze(0).expand(B, -1)
            elif self.method == 'ae':   v_vec = self.ae_v(a_vec)
        elif mode == 'drop_a':
            if self.method == 'zero':  a_vec = torch.zeros_like(a_vec)
            elif self.method == 'token': a_vec = self.miss_a.unsqueeze(0).expand(B, -1)
            elif self.method == 'ae':   a_vec = self.ae_a(v_vec)
        return self.fc(torch.cat([v_vec, a_vec], -1)), v_vec, a_vec

@torch.no_grad()
def evaluate(model, loader, dev, mode='full', pv=None):
    """Returns (positive-class F1, macro-F1, acc). mode selects whole-modality drop; p is the visual frame-drop ratio.
    Under frame-level drop the compensation mechanism is not triggered, so the model pools over the remaining frames."""
    model.eval(); ys = []; ps = []
    for V, A, mv, ma, y in loader:
        V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
        V, A, mv, ma = apply_drop(V, A, mv, ma, mode)
        if pv is not None: V, mv = frame_drop(V, mv, pv)
        ps += model(V, A, mv, ma, mode)[0].argmax(1).cpu().tolist(); ys += y.tolist()
    return (f1_score(ys, ps, zero_division=0),
            f1_score(ys, ps, average='macro', zero_division=0),
            accuracy_score(ys, ps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/lmvd')
    ap.add_argument('--method', choices=['zero', 'token', 'ae'], required=True)
    ap.add_argument('--drop_p', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hid', type=int, default=128)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dump_probs', action='store_true')   # optional: export per-sample probabilities after training
    args = ap.parse_args()
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dls = {f: DataLoader(LMVDPair(args.data_dir, f), batch_size=args.batch_size,
                         shuffle=(f == 'train'), collate_fn=collate_pair, num_workers=4)
           for f in ['train', 'valid', 'test']}
    model = BaselineFusion(args.method, args.hid).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    print(f"=== baseline method={args.method} drop_p={args.drop_p} seed={args.seed} ===", flush=True)
    best_vf1, best_state, wait = -1, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for V, A, mv, ma, y in dls['train']:
            V, A, mv, ma, y = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev), y.to(dev)
            v_r, a_r = model.encode(V, A, mv, ma)
            mode = random.choice(['drop_v', 'drop_a']) if random.random() < args.drop_p else 'full'
            Vd, Ad, mvd, mad = apply_drop(V, A, mv, ma, mode)
            out, _, _ = model(Vd, Ad, mvd, mad, mode)
            loss = ce(out, y)
            if args.method == 'ae':   # AE reconstruction supervision
                loss = loss + F.mse_loss(model.ae_v(a_r.detach()), v_r.detach()) \
                            + F.mse_loss(model.ae_a(v_r.detach()), a_r.detach())
            opt.zero_grad(); loss.backward(); opt.step()
        vf1 = evaluate(model, dls['valid'], dev, 'full')[0]
        if vf1 > best_vf1:
            best_vf1 = vf1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if wait >= args.patience: break
    model.load_state_dict(best_state)
    ff = evaluate(model, dls['test'], dev, 'full')
    fdv = evaluate(model, dls['test'], dev, 'drop_v')
    fda = evaluate(model, dls['test'], dev, 'drop_a')
    fr = {p: evaluate(model, dls['test'], dev, 'full', pv=p)[0] for p in [0.25, 0.5, 0.75, 0.9]}
    print(f">>> [base|{args.method}|seed{args.seed}]  full={ff[0]:.4f}  dropVideo={fdv[0]:.4f}  dropAudio={fda[0]:.4f}  "
          f"frame-level f25={fr[0.25]:.4f} f50={fr[0.5]:.4f} f75={fr[0.75]:.4f} f90={fr[0.9]:.4f}", flush=True)
    print(f">>> [audit|base|{args.method}|seed{args.seed}] "
          f"full F1/mac/acc={ff[0]:.4f}/{ff[1]:.4f}/{ff[2]:.4f}  "
          f"dropV={fdv[0]:.4f}/{fdv[1]:.4f}/{fdv[2]:.4f}  "
          f"dropA={fda[0]:.4f}/{fda[1]:.4f}/{fda[2]:.4f}", flush=True)
    if args.dump_probs:
        outp = os.path.join("./outputs", f"base_{args.method}_s{args.seed}.npz")
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
                        ps += torch.softmax(model(Vd, Ad, mvd, mad, mode)[0], -1)[:, 1].cpu().tolist()
                        if mi == 0: ys += y.tolist()
                    dump[f'{split}_{mode}'] = np.array(ps, dtype='float32')
                dump[f'y_{split}'] = np.array(ys)
        np.savez(outp, **dump)
        print(f">>> probs -> {outp}", flush=True)

if __name__ == '__main__':
    main()
