"""Smoke-test the LeDXA encoder on synthetic input — no real data required.

The UK Biobank / Human Phenotype Project DXA scans are access-controlled and are not
distributed here (see the top-level README). This script builds the encoder and runs a
forward pass on a random DXA-shaped batch, so you can confirm the model loads and produces
embeddings before wiring up your own data.

Requires `pip install -e .` (run from the repo root). Usage:
    python sample_data/demo.py
"""
import torch

from model.model import LeJEPA_Encoder

INPUT_SHAPE = (2, 3, 384, 128)  # (batch, channels, height, width) — whole-body DXA aspect ratio


def main():
    encoder = LeJEPA_Encoder(model_name="vit_small_patch16_384", img_size=(384, 128), proj_out_dim=128)
    encoder.eval()
    x = torch.randn(*INPUT_SHAPE)  # synthetic stand-in for a batch of DXA scans
    with torch.no_grad():
        features, projections = encoder(x)
    print(
        f"input {tuple(x.shape)} -> features {tuple(features.shape)} "
        f"-> projections {tuple(projections.shape)}"
    )


if __name__ == "__main__":
    main()
