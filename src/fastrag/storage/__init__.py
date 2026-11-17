"""Raw file storage (D9): local volume by default, S3-compatible optional.

Keys are content-addressed, so re-uploading the same file is free."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Protocol

from fastrag.config import Settings


class Storage(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...
    async def get(self, key: str) -> bytes: ...
    async def delete(self, key: str) -> None: ...


def content_key(tenant: str, collection: str, data: bytes, suffix: str = "") -> str:
    digest = hashlib.sha256(data).hexdigest()
    return f"{tenant}/{collection}/{digest[:2]}/{digest}{suffix}"


class LocalStorage:
    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("storage key escapes root")
        return path

    async def put(self, key: str, data: bytes) -> None:
        path = self._path(key)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)

        await asyncio.to_thread(_write)

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._path(key).unlink, missing_ok=True)


class S3Storage:
    def __init__(self, s: Settings):
        import boto3

        if not s.s3_bucket:
            raise ValueError("FASTRAG_S3_BUCKET is required for the s3 storage backend")
        self.bucket = s.s3_bucket
        self.prefix = s.s3_prefix
        self.client = boto3.client("s3", endpoint_url=s.s3_endpoint_url)

    async def put(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self.client.put_object, Bucket=self.bucket, Key=self.prefix + key, Body=data)

    async def get(self, key: str) -> bytes:
        obj = await asyncio.to_thread(self.client.get_object, Bucket=self.bucket, Key=self.prefix + key)
        return await asyncio.to_thread(obj["Body"].read)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self.client.delete_object, Bucket=self.bucket, Key=self.prefix + key)


def build_storage(s: Settings) -> Storage:
    return S3Storage(s) if s.storage_backend == "s3" else LocalStorage(s.storage_path)
