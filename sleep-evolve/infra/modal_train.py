"""Real LoRA fine-tune on a Modal GPU. This is an actual paid training job — NOT simulated —
and it is never imported by the test suite.

Usage:
    uv run modal run infra/modal_train.py --spec-path spec.json

spec.json is produced by sleep.trainer.build_training_spec() (adapter_id, base_model,
lora_rank, epochs, lr, max_len, examples_jsonl). The trained adapter lands on the
`sleep-evolve-adapters` volume at /adapters/<adapter_id>/ next to a manifest.json.

The same volume can be mounted by infra/modal_vllm.py, but serving a promoted adapter needs
vLLM's LoRA wiring (--enable-lora / LORA_MODULES / serve.lora_modules config) — that serving
path is a deliberate follow-up and is not implemented here. Promotion/serving decisions live
in sleep.trainer.CheckpointRegistry, gated on measured held-out improvement.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

APP_NAME = "sleep-evolve-train"
MINUTES = 60

if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness.config import load_config

    _cfg_path = os.environ.get(
        "TRAIN_CONFIG", str(Path(__file__).resolve().parents[1] / "configs" / "example.yaml"))
    _serve = load_config(_cfg_path).serve
    TRAIN = {
        "TRAIN_MODEL": os.environ.get("TRAIN_MODEL", _serve.model),
        "TRAIN_GPU": os.environ.get("TRAIN_GPU", "L40S:1"),
    }
else:  # inside the container: values were baked into the image env at deploy/run time
    TRAIN = {k: os.environ[k] for k in ("TRAIN_MODEL", "TRAIN_GPU")}

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "trl==0.12.1",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "bitsandbytes==0.44.1",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", **TRAIN})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
adapters = modal.Volume.from_name("sleep-evolve-adapters", create_if_missing=True)


@app.function(
    image=image,
    gpu=TRAIN["TRAIN_GPU"],
    timeout=3 * 60 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/adapters": adapters},
    secrets=[modal.Secret.from_name("sleep-evolve-vllm")],
)
def train(spec: dict) -> dict:
    """Train a 4-bit-quantized LoRA adapter on spec['examples_jsonl']; save it to the volume.

    Self-contained: only stdlib + image-installed packages may be imported here — the repo
    does not exist inside the container.
    """
    import json
    import tempfile
    from datetime import datetime, timezone

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    adapter_id = spec["adapter_id"]
    base_model = spec.get("base_model") or os.environ["TRAIN_MODEL"]

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(spec["examples_jsonl"])
        data_path = f.name
    ds = load_dataset("json", data_files=data_path, split="train")
    # SFTTrainer only needs the chat transcript; provenance columns must not leak into features.
    ds = ds.remove_columns([c for c in ds.column_names if c != "messages"])

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="auto", quantization_config=bnb)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    lora_rank = int(spec.get("lora_rank", 16))
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=int(spec.get("lora_alpha", 2 * lora_rank)),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    train_args = SFTConfig(
        output_dir="/tmp/sft-out",
        packing=False,
        max_seq_length=int(spec.get("max_len", 4096)),
        num_train_epochs=int(spec.get("epochs", 1)),
        learning_rate=float(spec.get("lr", 1e-4)),
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=train_args, peft_config=peft_config,
                         train_dataset=ds, tokenizer=tokenizer)
    result = trainer.train()
    train_loss = float(result.training_loss) if result.training_loss is not None else None

    dest = f"/adapters/{adapter_id}"
    trainer.save_model(dest)
    tokenizer.save_pretrained(dest)
    manifest = {
        "adapter_id": adapter_id,
        "base_model": base_model,
        "spec": spec,
        "train_loss": train_loss,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(f"{dest}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    adapters.commit()
    return {"adapter_id": adapter_id, "train_loss": train_loss, "adapter_path": dest}


@app.local_entrypoint()
def main(spec_path: str):
    """uv run modal run infra/modal_train.py --spec-path spec.json"""
    import json

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    result = train.remote(spec)
    print(json.dumps(result, indent=2))
