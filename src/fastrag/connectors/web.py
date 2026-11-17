"""A list of web pages (the pilot's project pages and FAQs)."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import httpx

from fastrag.connectors.base import RemoteItem

MAX_BYTES = 10 * 1024 * 1024


def check_url(url: str, allow_private: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"only http(s) URLs are allowed: {url}")
    if allow_private:
        return
    for info in socket.getaddrinfo(parsed.hostname, None):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"URL resolves to a non-public address: {url}")


class WebConnector:
    def __init__(self, urls: list[str], allow_private: bool = False):
        self.urls = urls
        self.allow_private = allow_private

    async def list_items(self) -> list[RemoteItem]:
        items = []
        for url in self.urls:
            await asyncio.to_thread(check_url, url, self.allow_private)
            items.append(
                RemoteItem(
                    external_id=url,
                    name=urlparse(url).path.rsplit("/", 1)[-1] or urlparse(url).hostname or url,
                    path="",
                    mime="text/html",
                    modified_at=None,
                    version_tag=None,
                    uri=url,
                )
            )
        return items

    async def fetch(self, item: RemoteItem) -> tuple[bytes, str]:
        url = item.external_id
        async with httpx.AsyncClient(
            timeout=30, follow_redirects=False, headers={"User-Agent": "fastRag-sync/0.1"}
        ) as client:
            # Follow redirects by hand so every hop passes the address check.
            for _ in range(5):
                await asyncio.to_thread(check_url, url, self.allow_private)
                resp = await client.get(url)
                if resp.is_redirect and "location" in resp.headers:
                    url = str(resp.url.join(resp.headers["location"]))
                    continue
                break
            resp.raise_for_status()
            data = resp.content[:MAX_BYTES]
        lm = resp.headers.get("last-modified")
        if lm:
            try:
                item.modified_at = parsedate_to_datetime(lm).astimezone(UTC)
            except (TypeError, ValueError):
                pass
        item.modified_at = item.modified_at or datetime.now(UTC)
        ctype = resp.headers.get("content-type", "text/html").split(";")[0]
        name = hashlib.sha1(item.external_id.encode(), usedforsecurity=False).hexdigest()[:12]
        ext = ".pdf" if ctype == "application/pdf" else ".html"
        return data, name + ext
