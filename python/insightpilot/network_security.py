from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import httpx


class UnsafeURL(ValueError):
    pass


async def assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeURL("仅允许包含有效主机名的 http/https 公网 URL")
    if parsed.username or parsed.password:
        raise UnsafeURL("URL 不允许包含用户名或密码")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise UnsafeURL("禁止访问本机或局域网地址")

    def resolve() -> list[str]:
        return list({item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)})

    try:
        addresses = await asyncio.to_thread(resolve)
    except socket.gaierror as exc:
        raise UnsafeURL(f"域名解析失败：{hostname}") from exc
    if not addresses:
        raise UnsafeURL(f"域名没有可用地址：{hostname}")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise UnsafeURL(f"禁止访问非公网地址：{address}")


async def safe_get(client: httpx.AsyncClient, url: str, max_redirects: int = 5) -> httpx.Response:
    current = url
    for _ in range(max_redirects + 1):
        await assert_public_url(current)
        response = await client.get(current, follow_redirects=False)
        if response.status_code not in {301, 302, 303, 307, 308}:
            return response
        location = response.headers.get("location")
        if not location:
            return response
        current = urljoin(str(response.url), location)
    raise UnsafeURL("网页重定向次数过多")

