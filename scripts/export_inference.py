#!/usr/bin/env python3
"""Export a training checkpoint to the proxy-free CGMA inference graph."""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cgma import Fusion as TrainingFusion
from cgma_inference import CGMAInference, parameter_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = torch.load(args.checkpoint, map_location="cpu")
    state = source.get("model_state", source)
    hid = int(source.get("hidden_size", 128)) if isinstance(source, dict) else 128
    source_ablation = source.get("ablate", "unknown") if isinstance(source, dict) else "unknown"
    if source_ablation not in {"no_proxy", "unknown"}:
        raise ValueError(
            f"Expected a final no_proxy checkpoint, received ablate={source_ablation!r}. "
            "Exporting another ablation would change its prediction graph."
        )

    inference = CGMAInference(hid=hid)
    filtered = {
        key: value for key, value in state.items()
        if not key.startswith("proxy_v.") and not key.startswith("proxy_a.")
    }
    inference.load_state_dict(filtered, strict=True)
    inference.eval()

    training = TrainingFusion("no_proxy", hid=hid)
    training.load_state_dict(state, strict=True)
    training.eval()

    generator = torch.Generator().manual_seed(0)
    video = torch.randn(3, 7, 136, generator=generator)
    audio = torch.randn(3, 5, 128, generator=generator)
    video_mask = torch.ones(3, 7, dtype=torch.bool)
    audio_mask = torch.ones(3, 5, dtype=torch.bool)
    video[1, 3] = 0; video_mask[1, 3] = False
    audio[2, 1] = 0; audio_mask[2, 1] = False
    with torch.no_grad():
        reference = training(video, audio, video_mask, audio_mask)[0]
        exported = inference(video, audio, video_mask, audio_mask)
    max_error = (reference - exported).abs().max().item()
    if max_error > 1e-6:
        raise RuntimeError(f"Inference export mismatch: max_abs_error={max_error:.3e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        "model_state": inference.state_dict(),
        "hidden_size": hid,
        "parameter_count": parameter_count(inference),
        "source_ablation": source_ablation,
    }, args.output)
    print(f"exported: {args.output}")
    print(f"parameters: {parameter_count(inference):,}")
    print(f"max_abs_logit_error: {max_error:.3e}")


if __name__ == "__main__":
    main()
