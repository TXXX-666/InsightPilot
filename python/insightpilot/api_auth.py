from __future__ import annotations

import hmac
import ipaddress
from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from .settings import settings


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_api_access(
    expected_token: str,
    authorization: str | None,
    client_host: str | None,
) -> None:
    """Require a bearer token when configured; otherwise permit loopback only."""
    if expected_token:
        scheme, _, supplied_token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not supplied_token or not hmac.compare_digest(
            supplied_token, expected_token
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效或缺失的 API Token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return

    if not _is_loopback(client_host):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="未配置 INSIGHTPILOT_API_TOKEN，仅允许本机访问",
        )


async def require_api_access(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    validate_api_access(
        settings.api_token,
        authorization,
        request.client.host if request.client else None,
    )
