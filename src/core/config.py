"""Application configuration defaults."""

from dataclasses import dataclass
from pathlib import Path
import os


def _load_dotenv(dotenv_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file without overriding real env vars."""
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    app_name: str = "Perplexity Lite"
    default_top_k: int = 5
    retrieve_top_k: int = 30
    rerank_top_k: int = 5
    dataset_name: str = "hotpotqa"
    llm_provider: str = os.getenv("LLM_PROVIDER", "offline")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "120"))
    dense_vector_dim: int = int(os.getenv("DENSE_VECTOR_DIM", "128"))
    dense_encoder_backend: str = os.getenv("DENSE_ENCODER_BACKEND", "hash")
    dense_model_name: str = os.getenv(
        "DENSE_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    dense_encode_batch_size: int = int(os.getenv("DENSE_ENCODE_BATCH_SIZE", "64"))
    auto_build_indexes: bool = os.getenv("AUTO_BUILD_INDEXES", "true").lower() == "true"
    preload_retrievers: bool = os.getenv("PRELOAD_RETRIEVERS", "true").lower() == "true"
    prewarm_query_encoder: bool = os.getenv("PREWARM_QUERY_ENCODER", "true").lower() == "true"
    semantic_refiner_enabled: bool = os.getenv("SEMANTIC_REFINER_ENABLED", "false").lower() == "true"
    semantic_model_name: str = os.getenv(
        "SEMANTIC_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    semantic_refiner_top_n: int = int(os.getenv("SEMANTIC_REFINER_TOP_N", "16"))
    project_root: Path = PROJECT_ROOT
    raw_hotpot_dev_path: Path = Path(
        os.getenv(
            "HOTPOT_DEV_PATH",
            PROJECT_ROOT / "artifacts/raw/hotpotqa/hotpot_dev_distractor_v1.json",
        )
    )
    raw_hotpot_train_path: Path = Path(
        os.getenv(
            "HOTPOT_TRAIN_PATH",
            PROJECT_ROOT / "artifacts/raw/hotpotqa/hotpot_train_v1.1.json",
        )
    )
    chunk_output_dir: Path = Path(
        os.getenv(
            "CHUNK_OUTPUT_DIR",
            PROJECT_ROOT / "artifacts/chunks",
        )
    )
    index_output_dir: Path = Path(
        os.getenv(
            "INDEX_OUTPUT_DIR",
            PROJECT_ROOT / "artifacts/indexes",
        )
    )
    default_chunk_path: Path = Path(
        os.getenv(
            "DEFAULT_CHUNK_PATH",
            PROJECT_ROOT / "artifacts/chunks/train_chunks.jsonl",
        )
    )


settings = Settings()
