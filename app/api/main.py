"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="autorace-exacta", version="0.1.0", lifespan=lifespan)
app.include_router(router)
