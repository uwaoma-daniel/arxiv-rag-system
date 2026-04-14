"""HuggingFace Spaces entrypoint — merges config-hf.yaml over config.yaml."""
import os, yaml, uvicorn
from pathlib import Path


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config() -> dict:
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    hf  = Path("config-hf.yaml")
    if hf.exists():
        cfg = _merge(cfg, yaml.safe_load(hf.read_text()))

    # HF Secrets → env vars → config
    for env_key, cfg_path in [
        ("QDRANT_URL",     ["vector_store", "qdrant_url"]),
        ("QDRANT_API_KEY", ["vector_store", "qdrant_api_key"]),
    ]:
        val = os.environ.get(env_key, "")
        if val:
            node = cfg
            for part in cfg_path[:-1]:
                node = node[part]
            node[cfg_path[-1]] = val

    return cfg


if __name__ == "__main__":
    config = load_config()
    os.environ["RAG_CONFIG_OVERRIDE"] = yaml.dump(config)
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
    )
