from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import socket
import uuid

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError
from fastapi import FastAPI

from app.api.v1.api import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.infrastructure.kafka.dispatcher import ResponseDispatcher
from app.infrastructure.mongodb.client import close_mongo, init_mongo

logger = logging.getLogger(__name__)
settings = get_settings()


async def start_producer_with_retry(producer: AIOKafkaProducer, retries: int = 30, base_delay: float = 0.5) -> None:
    """Start Kafka producer with exponential backoff + jitter (behavior preserved)."""
    last: Exception | None = None
    for i in range(1, retries + 1):
        try:
            await producer.start()
            return
        except KafkaConnectionError as e:
            last = e
            delay = min(10.0, base_delay * (2 ** (i - 1)))
            delay *= (0.8 + random.random() * 0.4)
            logger.warning("Kafka not ready (%s/%s): %s; retry in %.2fs", i, retries, e, delay)
            await asyncio.sleep(delay)
    if last:
        raise last
    raise RuntimeError("Kafka producer start failed")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    configure_logging()
    app = FastAPI(title=settings.app_name)

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("Starting API service...")

        producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            key_serializer=lambda v: v.encode("utf-8") if isinstance(v, str) else v,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False, default=str).encode("utf-8"),
            acks="all",
            linger_ms=10,
        )
        await start_producer_with_retry(producer)
        app.state.producer = producer
        logger.info("Kafka producer started")

        instance_id = (
            os.getenv("APP_INSTANCE_ID")
            or os.getenv("HOSTNAME")
            or socket.gethostname()
            or uuid.uuid4().hex[:8]
        )
        dispatcher_group_id = f"{settings.dispatcher_consumer_group}-{instance_id}-pid{os.getpid()}"

        dispatcher = ResponseDispatcher(
            topics=settings.dispatcher_topics,
            group_id=dispatcher_group_id,
            max_queue_size=2000,
        )
        await dispatcher.start()
        app.state.dispatcher = dispatcher

        await init_mongo()
        logger.info("MongoDB initialized")

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("Shutting down API service...")

        dispatcher = getattr(app.state, "dispatcher", None)
        if dispatcher is not None:
            await dispatcher.stop()

        producer = getattr(app.state, "producer", None)
        if producer is not None:
            await producer.stop()

        await close_mongo()
        logger.info("Shutdown complete")

    app.include_router(api_router)
    return app
