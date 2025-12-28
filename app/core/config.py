from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_name: str = Field(default="sse-chatbot", validation_alias="APP_NAME")
    app_env: str = Field(default="dev", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # OpenAI
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini",
                              validation_alias="OPENAI_MODEL")
    openai_embed_model: str = Field(
        default="text-embedding-3-small", validation_alias="OPENAI_EMBED_MODEL"
    )

    # Kafka
    kafka_bootstrap_servers: str = Field(
        default="kafka:9092", validation_alias="KAFKA_BOOTSTRAP_SERVERS"
    )
    rag_request_topic: str = Field(
        default="rag_requests", validation_alias="RAG_REQUEST_TOPIC"
    )
    rag_response_topic: str = Field(
        default="rag_responses", validation_alias="RAG_RESPONSE_TOPIC"
    )
    rag_worker_group_id: str = Field(
        default="rag-worker", validation_alias="RAG_WORKER_GROUP_ID"
    )

    # Dispatcher
    dispatcher_consumer_group: str = Field(
        default="api-dispatcher", validation_alias="DISPATCHER_CONSUMER_GROUP"
    )

    # DLQ / Retry
    rag_dlq_topic: str = Field(
        default="rag_requests_dlq", validation_alias="RAG_DLQ_TOPIC"
    )
    max_retries: int = Field(default=5, validation_alias="MAX_RETRIES")
    retry_base_seconds: float = Field(
        default=1.0, validation_alias="RETRY_BASE_SECONDS")
    retry_max_seconds: float = Field(
        default=30.0, validation_alias="RETRY_MAX_SECONDS")

    # Redis
    redis_url: str = Field(default="redis://redis:6379/0",
                           validation_alias="REDIS_URL")
    idempotency_ttl_seconds: int = Field(
        default=86400, validation_alias="IDEMPOTENCY_TTL_SECONDS"
    )

    # Mongo
    mongo_dsn: str = Field(default="mongodb://mongo:27017",
                           validation_alias="MONGO_DSN")
    mongo_db_name: str = Field(
        default="sse_chatbot", validation_alias="MONGO_DB_NAME")

    # Worker
    worker_max_concurrency: int = Field(
        default=32, validation_alias="WORKER_MAX_CONCURRENCY")

    # Qdrant
    qdrant_host: str = Field(default="qdrant", validation_alias="QDRANT_HOST")
    qdrant_grpc_port: int = Field(
        default=6334, validation_alias="QDRANT_GRPC_PORT")
    qdrant_http_port: int = Field(
        default=6333, validation_alias="QDRANT_HTTP_PORT")
    qdrant_collection: str = Field(
        default="rag_hybrid", validation_alias="QDRANT_COLLECTION")
    qdrant_dense_vector_name: str = Field(
        default="dense", validation_alias="QDRANT_DENSE_VECTOR_NAME"
    )
    qdrant_sparse_vector_name: str = Field(
        default="sparse", validation_alias="QDRANT_SPARSE_VECTOR_NAME"
    )

    # RAG knobs
    rag_top_k: int = Field(default=50, validation_alias="RAG_TOP_K")
    rag_rerank_top_k: int = Field(
        default=10, validation_alias="RAG_RERANK_TOP_K")
    rag_rerank_window_tokens: int = Field(
        default=512, validation_alias="RAG_RERANK_WINDOW_TOKENS")
    rag_rerank_stride_tokens: int = Field(
        default=128, validation_alias="RAG_RERANK_STRIDE_TOKENS")

    rag_chunk_tokens: int = Field(
        default=500, validation_alias="RAG_CHUNK_TOKENS")
    rag_overlap_tokens: int = Field(
        default=125, validation_alias="RAG_OVERLAP_TOKENS")
    rag_doc_profile_sample_chars: int = Field(
        default=8000, validation_alias="RAG_DOC_PROFILE_SAMPLE_CHARS"
    )
    rag_prf_limit: int = Field(default=3, validation_alias="RAG_PRF_LIMIT")
    rag_prf_snippets: int = Field(
        default=3, validation_alias="RAG_PRF_SNIPPETS")

    rag_rewrite_use_history: bool = Field(
        default=True, validation_alias="RAG_REWRITE_USE_HISTORY")
    rag_rewrite_history_max_messages: int = Field(
        default=5, validation_alias="RAG_REWRITE_HISTORY_MAX_MESSAGES"
    )

    rag_cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="RAG_CROSS_ENCODER_MODEL",
    )
    rag_cross_encoder_device: str = Field(
        default="cuda", validation_alias="RAG_CROSS_ENCODER_DEVICE"
    )

    rag_embed_batch: int = Field(
        default=64, validation_alias="RAG_EMBED_BATCH")
    rag_qdrant_upsert_batch: int = Field(
        default=128, validation_alias="RAG_QDRANT_UPSERT_BATCH")

    rag_sparse_tokenizer_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="RAG_SPARSE_TOKENIZER_MODEL",
    )
    rag_sparse_max_tokens: int = Field(
        default=1024, validation_alias="RAG_SPARSE_MAX_TOKENS")

    # SSE API timeouts
    sse_heartbeat_s: float = Field(
        default=15.0, validation_alias="SSE_HEARTBEAT_S")
    sse_max_idle_s: float = Field(
        default=60.0, validation_alias="SSE_MAX_IDLE_S")
    sse_max_total_s: float = Field(
        default=180.0, validation_alias="SSE_MAX_TOTAL_S")

    # LLM stream timeouts
    stream_first_token_timeout: float = Field(
        default=30.0, validation_alias="STREAM_FIRST_TOKEN_TIMEOUT"
    )
    stream_idle_timeout: float = Field(
        default=60.0, validation_alias="STREAM_IDLE_TIMEOUT")
    stream_hard_timeout: float = Field(
        default=120.0, validation_alias="STREAM_HARD_TIMEOUT")

    @property
    def dispatcher_topics(self) -> List[str]:
        return [self.rag_response_topic]


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
