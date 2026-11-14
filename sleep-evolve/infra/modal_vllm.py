"""Optional: serve an open-weight model on Modal with vLLM's OpenAI-compatible server.

The model, GPU, vLLM version, and context length come from the `serve:` section of a
run config (default configs/example.yaml; override the path with SERVE_CONFIG). Set
SERVE_MODEL to override just the model.

One-time secret (the server's API key):
    uv run modal secret create sleep-evolve-vllm VLLM_API_KEY=<random string>
Deploy:
    uv run modal deploy infra/modal_vllm.py
Then point the harness at it:
    LLM_BASE_URL=https://<workspace>--sleep-evolve-vllm-serve.modal.run/v1
    LLM_API_KEY=<same random string>
    LLM_MODEL=<serve.model>
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "sleep-evolve-vllm"
PORT = 8000
MINUTES = 60

if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness.config import load_config

    _cfg_path = os.environ.get("SERVE_CONFIG", str(Path(__file__).resolve().parents[1] / "configs" / "example.yaml"))
    _serve = load_config(_cfg_path).serve
    SERVE = {
        "SERVE_MODEL": os.environ.get("SERVE_MODEL", _serve.model),
        "SERVE_GPU": _serve.gpu,
        "SERVE_VLLM_VERSION": _serve.vllm_version,
        "SERVE_MAX_MODEL_LEN": str(_serve.max_model_len),
    }
else:  # inside the container: values were baked into the image env at deploy time
    SERVE = {k: os.environ[k] for k in ("SERVE_MODEL", "SERVE_GPU", "SERVE_VLLM_VERSION", "SERVE_MAX_MODEL_LEN")}

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={SERVE['SERVE_VLLM_VERSION']}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", **SERVE})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=SERVE["SERVE_GPU"],
    scaledown_window=15 * MINUTES,
    timeout=24 * 60 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("sleep-evolve-vllm")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=20 * MINUTES)
def serve():
    model = os.environ["SERVE_MODEL"]
    cmd = [
        "vllm", "serve", model,
        "--served-model-name", model,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", os.environ["SERVE_MAX_MODEL_LEN"],
    ]
    subprocess.Popen(cmd)
