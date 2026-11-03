"""Shared SDV setup that intentionally accepts training rows only."""
from __future__ import annotations

import json
import random
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

from phase2_data import ROOT, load_config, prepare, read_partition

PROCESSED = ROOT / "GENAI/data/processed"
MODEL_DIR = ROOT / "GENAI/models"
SYN_DIR = ROOT / "GENAI/data/synthetic"


def schema(train: pd.DataFrame, metadata_path: Path | None = None) -> Metadata:
    cfg = load_config()
    categorical = set(["PCOS (Y/N)", "Blood Group", "Cycle(R/I)", "Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)"])
    meta = Metadata.detect_from_dataframe(data=train, table_name="pcos", infer_sdtypes=False)
    for col in train.columns:
        meta.update_column(column_name=col, table_name="pcos", sdtype="categorical" if col in categorical else "numerical")
    meta.validate()
    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return meta


def train_and_sample(generator: str) -> tuple[pd.DataFrame, dict]:
    cfg = load_config()
    # prepare() writes deterministic partitions; only train.csv is loaded below.
    partition_info = prepare()
    train = read_partition("train")
    seed = int(cfg["random_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available() and cfg["sdv"].get("cuda", False):
        torch.cuda.manual_seed_all(seed)
    metadata_path = MODEL_DIR / "shared" / "pcos_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    meta = schema(train, metadata_path)
    cls = CTGANSynthesizer if generator == "ctgan" else TVAESynthesizer
    params = {"metadata": meta, "epochs": int(cfg["sdv"]["epochs"]), "batch_size": int(cfg["sdv"]["batch_size"]), "cuda": bool(cfg["sdv"].get("cuda", False)), "verbose": True}
    if generator == "ctgan": params["pac"] = int(cfg["sdv"].get("pac", 10))
    model = cls(**params)
    model.fit(train)
    n = int(cfg.get("synthetic_rows") or len(train))
    synthetic = model.sample(num_rows=n)
    out = SYN_DIR / generator
    out.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(out / f"synthetic_{generator}.csv", index=False)
    mdir = MODEL_DIR / generator
    mdir.mkdir(parents=True, exist_ok=True)
    model.save(filepath=str(mdir / f"{generator}_synthesizer.pkl"))
    model_config = {k: cfg["sdv"][k] for k in ("epochs", "batch_size", "pac", "cuda") if k in cfg["sdv"] and (generator == "ctgan" or k != "pac")}
    training_hash = hashlib.sha256((PROCESSED / "train.csv").read_bytes()).hexdigest()
    run = {
        "generator": generator.upper(), "seed": seed, "sdv_version": __import__("sdv").__version__,
        "torch_version": torch.__version__, "config": model_config, "training_rows": len(train),
        "synthetic_rows": len(synthetic), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_schema_version": meta.to_dict().get("METADATA_SPEC_VERSION"),
        "metadata_path": str(metadata_path.relative_to(ROOT)), "training_partition_sha256": training_hash,
        "raw_dataset_sha256": partition_info["source_sha256"],
        "fit_data_source": "GENAI/data/processed/train.csv only",
        "validation_or_test_passed_to_fit": False,
    }
    (out / "run_metadata.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return synthetic, run
