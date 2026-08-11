#-*- coding: utf-8 -*-
"""Naive attention-fusion baseline (Tables 1 and 4).

Two-stream BiLSTM + vision-led cross-modal attention fusion.
--drop_p 0    : train on intact input only (the "naive fusion" rows).
--drop_p >0   : train with random whole-modality dropout at rate drop_p.
Evaluates intact / drop-video / drop-audio and the visual frame-level curve.
Paper protocol: --epochs 45 --patience 12, seeds {0, 1, 2, 42, 123}.
"""
import os, argparse, random, csv
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class LMVDPair(Dataset):
    """Returns video / audio separately (no concatenation)."""
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
    mv = (V.abs().sum(-1) != 0); ma = (A.abs().sum(-1) != 0)      # True = valid frame
    return V, A, mv, ma, torch.tensor(ys)

def masked_mean(seq, mask):
    m = mask.unsqueeze(-1).float()
    return (seq * m).sum(1) / m.sum(1).clamp(min=1)

class AdaFuse(nn.Module):
    """Vision-led: v_vec queries the audio sequence via cross-attention, with a sigmoid fusion gate."""
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
        a_ctx = torch.nan_to_num(a_ctx.squeeze(1))                # NaN -> 0 when audio is fully absent
        g = self.gate(torch.cat([v_vec, a_ctx], -1))
        return self.fc(torch.cat([v_vec, g * a_ctx], -1))

def apply_drop(V, A, mv, ma, mode):
    """mode: full / drop_a / drop_v. Returns the zeroed tensors."""
    if mode == 'drop_a':
        A = torch.zeros_like(A); ma = torch.zeros_like(ma)
    elif mode == 'drop_v':
        V = torch.zeros_like(V); mv = torch.zeros_like(mv)
    return V, A, mv, ma

def frame_drop(X, m, p):
    drop = (torch.rand(m.shape, device=X.device) < p) & m
    X = X.clone(); m = m.clone(); X[drop] = 0; m[drop.bool()] = 0
    return X, m

@torch.no_grad()
def evaluate(model, loader, dev, mode='full', pv=None):
    model.eval(); ys = []; ps = []
    for V, A, mv, ma, y in loader:
        V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
        V, A, mv, ma = apply_drop(V, A, mv, ma, mode)
        if pv is not None: V, mv = frame_drop(V, mv, pv)
        ps += model(V, A, mv, ma).argmax(1).cpu().tolist(); ys += y.tolist()
    return (f1_score(ys, ps, zero_division=0),
            f1_score(ys, ps, average='macro', zero_division=0),
            accuracy_score(ys, ps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/lmvd')
    ap.add_argument('--drop_p', type=float, default=0.0)      # 0 = intact-only training; >0 = random modality dropout
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
    model = AdaFuse(args.hid).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    mode_name = 'intact-only' if args.drop_p == 0 else 'modality-dropout'
    print(f"=== adafuse {mode_name} drop_p={args.drop_p} seed={args.seed} ===", flush=True)
    best_vf1, best_state, wait = -1, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for V, A, mv, ma, y in dls['train']:
            V, A, mv, ma, y = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev), y.to(dev)
            if args.drop_p > 0 and random.random() < args.drop_p:        # randomly drop one modality
                V, A, mv, ma = apply_drop(V, A, mv, ma, random.choice(['drop_a', 'drop_v']))
            opt.zero_grad(); loss = crit(model(V, A, mv, ma), y); loss.backward(); opt.step()
        vf1, _, _ = evaluate(model, dls['valid'], dev, 'full')
        if vf1 > best_vf1:
            best_vf1 = vf1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if wait >= args.patience: break
    model.load_state_dict(best_state)
    f_full, m_full, a_full = evaluate(model, dls['test'], dev, 'full')
    f_da, m_da, a_da = evaluate(model, dls['test'], dev, 'drop_a')
    f_dv, m_dv, a_dv = evaluate(model, dls['test'], dev, 'drop_v')
    # prints F1 / macro-F1 / acc plus the three missing-robustness conditions
    print(f">>> [adafuse_p{args.drop_p}] TEST  acc={a_full:.4f}  F1={f_full:.4f}  macroF1={m_full:.4f}  "
          f"| dropAudio_F1={f_da:.4f}  dropVideo_F1={f_dv:.4f}", flush=True)
    print(f">>> [audit|naive|seed{args.seed}] "
          f"full F1/mac/acc={f_full:.4f}/{m_full:.4f}/{a_full:.4f}  "
          f"dropV={f_dv:.4f}/{m_dv:.4f}/{a_dv:.4f}  "
          f"dropA={f_da:.4f}/{m_da:.4f}/{a_da:.4f}", flush=True)
    fr = {p: evaluate(model, dls['test'], dev, 'full', pv=p)[0] for p in [0.25, 0.5, 0.75, 0.9]}
    print(f">>> [naive_frame|seed{args.seed}] "
          f"frame-level f25={fr[0.25]:.4f} f50={fr[0.5]:.4f} f75={fr[0.75]:.4f} f90={fr[0.9]:.4f}", flush=True)
    if args.dump_probs:
        outp = os.path.join("./outputs", f"naive_s{args.seed}.npz")
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
