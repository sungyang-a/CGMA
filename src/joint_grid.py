"""Historical Figure 4 joint-missingness grid and representation dump.

The script is retained exactly at the training-protocol level used to produce the
released Figure 4 source logs: CGMA uses 45 epochs / patience 12, whereas the naive
AdaFuse comparison uses 40 epochs / patience 10. This difference is documented in
KNOWN_LIMITATIONS.md and must not be mistaken for the unified 45/12 main-table protocol.
Grid perturbations are evaluation-only and independently sampled for every cell.
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

# ---------------- CGMA(与 lmvd_ablation.py 逐字一致, 保 RNG) ----------------
class Fusion(nn.Module):
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
    def fused(self, V, A, mv, ma):
        v_vec, a_vec = self.encode(V, A, mv, ma)
        wv = self.comp_v(v_vec); wa = self.comp_a(a_vec)
        h_v = wv * v_vec; h_a = wa * a_vec
        g = self.gate(torch.cat([h_v, h_a], -1))
        return torch.cat([h_v, g * h_a], -1)
    def forward(self, V, A, mv, ma):
        v_vec, a_vec = self.encode(V, A, mv, ma)
        pv = self.proxy_v(a_vec); pa = self.proxy_a(v_vec)
        wv = self.comp_v(v_vec); wa = self.comp_a(a_vec)
        if self.ablate == 'no_proxy':
            h_v = wv * v_vec; h_a = wa * a_vec
        else:
            h_v = wv * v_vec + (1 - wv) * pv; h_a = wa * a_vec + (1 - wa) * pa
        g = self.gate(torch.cat([h_v, h_a], -1))
        return self.fc(torch.cat([h_v, g * h_a], -1)), v_vec, a_vec, wv, wa

# ---------------- AdaFuse(与 lmvd_adafuse.py 逐字一致) ----------------
class AdaFuse(nn.Module):
    def __init__(self, hid=128, nhead=4, nclass=2):
        super().__init__()
        self.v_lstm = nn.LSTM(136, hid, batch_first=True, bidirectional=True)
        self.a_lstm = nn.LSTM(128, hid, batch_first=True, bidirectional=True)
        D = hid * 2
        self.attn = nn.MultiheadAttention(D, nhead, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(D * 2, D), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(D * 2, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, nclass))
    def fused(self, V, A, mv, ma):
        v_seq, _ = self.v_lstm(V); a_seq, _ = self.a_lstm(A)
        v_vec = masked_mean(v_seq, mv)
        a_ctx, _ = self.attn(v_vec.unsqueeze(1), a_seq, a_seq, key_padding_mask=~ma)
        a_ctx = torch.nan_to_num(a_ctx.squeeze(1))
        g = self.gate(torch.cat([v_vec, a_ctx], -1))
        return torch.cat([v_vec, g * a_ctx], -1)
    def forward(self, V, A, mv, ma):
        v_seq, _ = self.v_lstm(V); a_seq, _ = self.a_lstm(A)
        v_vec = masked_mean(v_seq, mv)
        a_ctx, _ = self.attn(v_vec.unsqueeze(1), a_seq, a_seq, key_padding_mask=~ma)
        a_ctx = torch.nan_to_num(a_ctx.squeeze(1))
        g = self.gate(torch.cat([v_vec, a_ctx], -1))
        return self.fc(torch.cat([v_vec, g * a_ctx], -1))

def logits_of(model, method, V, A, mv, ma):
    out = model(V, A, mv, ma)
    return out[0] if method == 'ours' else out

@torch.no_grad()
def eval_cond(model, method, loader, dev, mode=None, pv=None, pa=None):
    model.eval(); ys = []; ps = []
    for V, A, mv, ma, y in loader:
        V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
        if mode is not None:
            V, A, mv, ma = apply_drop(V, A, mv, ma, mode)
        if pv: V, mv = frame_drop(V, mv, pv)
        if pa: A, ma = frame_drop(A, ma, pa)
        ps += logits_of(model, method, V, A, mv, ma).argmax(1).cpu().tolist()
        ys += y.tolist()
    return f1_score(ys, ps, zero_division=0)

@torch.no_grad()
def eval_valid_f0_ours(model, loader, dev):
    """验证段: 逐字复现 lmvd_ablation.eval_frame([0.0]) —— p=0 也调 frame_drop 消耗RNG,
    否则训练扰动序列从第2个epoch起偏离, 无法逐位复现论文模型(2026-07-02 确定性核对教训)。"""
    model.eval(); ys = []; preds = []
    for V, A, mv, ma, y in loader:
        V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
        V, mv = frame_drop(V, mv, 0.0)
        preds += model(V, A, mv, ma)[0].argmax(1).cpu().tolist(); ys += y.tolist()
    return f1_score(ys, preds, zero_division=0)

@torch.no_grad()
def dump_reps(model, method, loader, dev, out_path):
    model.eval()
    feats = {'full': [], 'drop_v': []}; preds = {'full': [], 'drop_v': []}; labels = []
    for V, A, mv, ma, y in loader:
        V, A, mv, ma = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev)
        feats['full'].append(model.fused(V, A, mv, ma).cpu().numpy())
        preds['full'] += logits_of(model, method, V, A, mv, ma).argmax(1).cpu().tolist()
        Vd, Ad, mvd, mad = apply_drop(V, A, mv, ma, 'drop_v')
        feats['drop_v'].append(model.fused(Vd, Ad, mvd, mad).cpu().numpy())
        preds['drop_v'] += logits_of(model, method, Vd, Ad, mvd, mad).argmax(1).cpu().tolist()
        labels += y.tolist()
    np.savez(out_path,
             full=np.concatenate(feats['full']), drop_v=np.concatenate(feats['drop_v']),
             pred_full=np.array(preds['full']), pred_drop_v=np.array(preds['drop_v']),
             labels=np.array(labels))
    print(f">>> reps → {out_path}", flush=True)

def train_ours(args, dls, dev):
    """逐字复现 lmvd_ablation no_proxy 训练(RNG消耗一致)。"""
    model = Fusion('no_proxy', args.hid).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best_vf1, best_state, wait = -1, None, 0
    for ep in range(1, 45 + 1):
        model.train()
        for V, A, mv, ma, y in dls['train']:
            V, A, mv, ma, y = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev), y.to(dev)
            v_r, a_r = model.encode(V, A, mv, ma)
            mv0 = mv.float().sum(1, keepdim=True).clamp(min=1); ma0 = ma.float().sum(1, keepdim=True).clamp(min=1)
            Vd, Ad, mvd, mad = V, A, mv, ma
            r = random.random()
            if r < 0.33:
                Vd, Ad, mvd, mad = apply_drop(V, A, mv, ma, random.choice(['drop_v', 'drop_a']))
            elif r < 0.66:
                p = random.random() * 0.95
                if random.random() < 0.5: Vd, mvd = frame_drop(V, mv, p)
                else: Ad, mad = frame_drop(A, ma, p)
            out, _, _, wv, wa = model(Vd, Ad, mvd, mad)
            loss = ce(out, y)
            pv = model.proxy_v(a_r.detach()); pa = model.proxy_a(v_r.detach())
            loss = loss + 1.0 * (F.mse_loss(pv, v_r.detach()) + F.mse_loss(pa, a_r.detach()))
            pres_v = (mvd.float().sum(1, keepdim=True) / mv0).clamp(0, 1)
            pres_a = (mad.float().sum(1, keepdim=True) / ma0).clamp(0, 1)
            loss = loss + 0.5 * (F.binary_cross_entropy(wv, pres_v) + F.binary_cross_entropy(wa, pres_a))
            opt.zero_grad(); loss.backward(); opt.step()
        # 与 ablation 相同: valid 帧级 f0 早停(含 RNG 消耗)
        vf1 = eval_valid_f0_ours(model, dls['valid'], dev)
        if vf1 > best_vf1:
            best_vf1 = vf1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if wait >= 12: break
    model.load_state_dict(best_state)
    return model

def train_adafuse(args, dls, dev):
    """逐字复现 lmvd_adafuse --drop_p 0 训练。"""
    model = AdaFuse(args.hid).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_vf1, best_state, wait = -1, None, 0
    for ep in range(1, 40 + 1):
        model.train()
        for V, A, mv, ma, y in dls['train']:
            V, A, mv, ma, y = V.to(dev), A.to(dev), mv.to(dev), ma.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(V, A, mv, ma), y); loss.backward(); opt.step()
        vf1 = eval_cond(model, 'adafuse', dls['valid'], dev)
        if vf1 > best_vf1:
            best_vf1 = vf1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if wait >= 10: break
    model.load_state_dict(best_state)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/lmvd')
    ap.add_argument('--method', choices=['ours', 'adafuse'], required=True)
    ap.add_argument('--hid', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dump_reps', action='store_true')
    ap.add_argument('--skip_grid', action='store_true')
    ap.add_argument('--output_dir', default='./outputs/joint_grid')
    args = ap.parse_args()
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dls = {f: DataLoader(LMVDPair(args.data_dir, f), batch_size=args.batch_size,
                         shuffle=(f == 'train'), collate_fn=collate_pair, num_workers=4)
           for f in ['train', 'valid', 'test']}
    print(f"=== grid2d {args.method} seed={args.seed} ===", flush=True)
    model = (train_ours if args.method == 'ours' else train_adafuse)(args, dls, dev)

    # 确定性核对
    chk_f0 = eval_cond(model, args.method, dls['test'], dev)
    chk_dv = eval_cond(model, args.method, dls['test'], dev, mode='drop_v')
    chk_da = eval_cond(model, args.method, dls['test'], dev, mode='drop_a')
    print(f">>> [chk|{args.method}|seed{args.seed}] f0={chk_f0:.4f} dropV={chk_dv:.4f} dropA={chk_da:.4f}", flush=True)

    # 6×6 网格
    if not args.skip_grid:
        grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for pv in grid:
            for pa in grid:
                f1 = eval_cond(model, args.method, dls['test'], dev, pv=pv, pa=pa)
                print(f">>> [grid|{args.method}|seed{args.seed}] pv={pv:.1f} pa={pa:.1f} f1={f1:.4f}", flush=True)

    if args.dump_reps:
        out = os.path.join(args.output_dir, f"rep_{args.method}_s{args.seed}.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        dump_reps(model, args.method, dls['test'], dev, out)

if __name__ == '__main__':
    main()
