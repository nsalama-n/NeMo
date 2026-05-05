"""
Convert cohere-transcribe-03-2026 from HuggingFace format to NeMo .nemo file.

MUST run inside the NeMo container on the cluster:
    nvcr.io/nvidia/nemo:25.02

Usage:
    python stc_asr/scripts/convert_cohere_to_nemo.py \
        --hf-path  /shared/models/cohere-transcribe-03-2026 \
        --config   stc_asr/configs/finetune_cohere_saudi.yaml \
        --output   /shared/models/cohere-transcribe-03-2026/cohere_transcribe.nemo

Why this works:
    Cohere confirmed that tensor names in their HuggingFace model already match
    NeMo's EncDecMultiTaskModel — so weights copy directly with no renaming.
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_hf_state_dict(hf_path: Path) -> dict:
    """Load weights from HuggingFace model folder (safetensors or pytorch_model.bin)."""
    import torch

    safetensors_path = hf_path / "model.safetensors"
    bin_path         = hf_path / "pytorch_model.bin"

    if safetensors_path.exists():
        logger.info(f"Loading weights from {safetensors_path}")
        from safetensors.torch import load_file
        return load_file(str(safetensors_path))

    if bin_path.exists():
        logger.info(f"Loading weights from {bin_path}")
        return torch.load(str(bin_path), map_location="cpu")

    raise FileNotFoundError(
        f"No model weights found in {hf_path}. "
        f"Expected model.safetensors or pytorch_model.bin."
    )


def print_config_summary(hf_path: Path):
    """Print key architecture values from config.json for verification."""
    config_path = hf_path / "config.json"
    if not config_path.exists():
        logger.warning("config.json not found — skipping summary")
        return

    with open(config_path) as f:
        cfg = json.load(f)

    print("\n── Cohere model architecture (from config.json) ────────────")
    print(f"  Encoder layers  : {cfg['encoder']['n_layers']}")
    print(f"  Encoder d_model : {cfg['encoder']['d_model']}")
    print(f"  Encoder n_heads : {cfg['encoder']['n_heads']}")
    print(f"  Decoder layers  : {cfg['transf_decoder']['config_dict']['num_layers']}")
    print(f"  Decoder hidden  : {cfg['transf_decoder']['config_dict']['hidden_size']}")
    print(f"  Vocab size      : {cfg['vocab_size']}")
    print(f"  Mel features    : {cfg['preprocessor']['features']}")
    print(f"  Sample rate     : {cfg['preprocessor']['sample_rate']} Hz")
    print()


def convert(hf_path: Path, config_path: Path, output_path: Path):
    import torch
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import EncDecMultiTaskModel

    print_config_summary(hf_path)

    # ── Step 1: Load HuggingFace weights ─────────────────────────────────────
    logger.info("Loading HuggingFace weights...")
    state_dict = load_hf_state_dict(hf_path)
    logger.info(f"Loaded {len(state_dict)} tensors")

    # ── Step 2: Load NeMo config ──────────────────────────────────────────────
    logger.info(f"Loading NeMo config from {config_path}")
    cfg = OmegaConf.load(str(config_path))

    # Remove init_from_nemo_model so NeMo doesn't try to load a checkpoint
    # during instantiation (we load weights manually below)
    cfg.pop("init_from_nemo_model", None)

    # ── Step 3: Instantiate NeMo model ───────────────────────────────────────
    logger.info("Instantiating NeMo EncDecMultiTaskModel...")
    nemo_model = EncDecMultiTaskModel(cfg=cfg.model)
    nemo_model.eval()

    # ── Step 4: Copy weights (tensor names match — no remapping needed) ───────
    logger.info("Loading weights into NeMo model...")
    missing, unexpected = nemo_model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"{len(missing)} missing keys (expected for prompt/tokenizer layers):")
        for k in missing[:10]:
            logger.warning(f"  MISSING: {k}")

    if unexpected:
        logger.warning(f"{len(unexpected)} unexpected keys:")
        for k in unexpected[:10]:
            logger.warning(f"  UNEXPECTED: {k}")

    if not missing and not unexpected:
        logger.info("All weights loaded successfully — perfect match.")

    # ── Step 5: Save as .nemo ────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving .nemo file to {output_path} ...")
    nemo_model.save_to(str(output_path))

    logger.info(f"Done. NeMo checkpoint saved to: {output_path}")
    print(f"\n── Next step ───────────────────────────────────────────────")
    print(f"  Update finetune_cohere_saudi.yaml:")
    print(f"  init_from_nemo_model: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert cohere-transcribe-03-2026 HF checkpoint to NeMo .nemo format."
    )
    parser.add_argument(
        "--hf-path", required=True,
        help="Path to downloaded HuggingFace model folder on cluster"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to finetune_cohere_saudi.yaml"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for .nemo file (e.g. /shared/models/cohere-transcribe-03-2026/cohere_transcribe.nemo)"
    )
    args = parser.parse_args()

    hf_path     = Path(args.hf_path)
    config_path = Path(args.config)
    output_path = Path(args.output)

    if not hf_path.exists():
        raise FileNotFoundError(f"HF model folder not found: {hf_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    convert(hf_path, config_path, output_path)


if __name__ == "__main__":
    main()
