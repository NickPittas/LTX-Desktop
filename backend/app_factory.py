"""FastAPI app factory decoupled from runtime bootstrap side effects."""

from __future__ import annotations

import base64
import hmac
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response as StarletteResponse

from _routes._errors import HTTPError, build_http_error_response
from _routes.generation import router as generation_router
from _routes.hf_auth import router as hf_auth_router
from _routes.health import router as health_router
from _routes.ic_lora import router as ic_lora_router
from _routes.image_gen import router as image_gen_router
from _routes.model_profiles import router as model_profiles_router
from _routes.models import router as models_router
from _routes.suggest_gap_prompt import router as suggest_gap_prompt_router
from _routes.retake import router as retake_router
from _routes.runtime_policy import router as runtime_policy_router
from _routes.settings import router as settings_router
from api_types import HTTPErrorResponse
from logging_policy import log_http_error, log_unhandled_exception
from services import memory_trace
from state import init_state_service

if TYPE_CHECKING:
    from app_handler import AppHandler

DEFAULT_ALLOWED_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

DEFAULT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    "4XX": {
        "model": HTTPErrorResponse,
        "description": "Client Error",
    },
    "5XX": {
        "model": HTTPErrorResponse,
        "description": "Server Error",
    },
}

#: Request-state attribute recording whether an exception handler already wrote
#: the single terminal ``http_error`` event. Lives on the shared request scope
#: state so it survives Starlette's ``BaseHTTPMiddleware`` task isolation (a bare
#: contextvar would not propagate back out of ``call_next``).
_TRACE_TERMINAL_ATTR = "ltx_memory_terminal_recorded"
_HTTP_TRACE_LABEL = "http"


def _trace_http_event(
    request: Request,
    event_type: str,
    *,
    status_code: int | None = None,
    code: str | None = None,
    message: str | None = None,
) -> None:
    """Write an HTTP trace event with the standard request fields."""
    fields: dict[str, object] = {
        "path": request.url.path,
        "method": request.method,
    }
    if status_code is not None:
        fields["status_code"] = status_code
    if code is not None:
        fields["code"] = code
    if message is not None:
        fields["message"] = message
    memory_trace.write_event(event_type, _HTTP_TRACE_LABEL, **fields)


def _record_http_terminal(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
) -> None:
    """Write the single terminal ``http_error`` and flag it on the request."""
    if not memory_trace.is_enabled():
        return
    try:
        _trace_http_event(
            request,
            "http_error",
            status_code=status_code,
            code=code,
            message=message,
        )
        setattr(request.state, _TRACE_TERMINAL_ATTR, True)
    except Exception:
        pass


def _http_terminal_recorded(request: Request) -> bool:
    return bool(getattr(request.state, _TRACE_TERMINAL_ATTR, False))


def create_app(
    *,
    handler: "AppHandler",
    allowed_origins: list[str] | None = None,
    title: str = "LTX-2 Video Generation Server",
    auth_token: str = "",
    admin_token: str = "",
) -> FastAPI:
    """Create a configured FastAPI app bound to the provided handler."""
    init_state_service(handler)

    app = FastAPI(title=title, responses=DEFAULT_ERROR_RESPONSES)
    app.state.admin_token = admin_token  # type: ignore[attr-defined]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or DEFAULT_ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _auth_middleware(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        call_next: Callable[[Request], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        if not auth_token:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path == "/api/auth/huggingface/callback":
            return await call_next(request)
        def _token_matches(candidate: str) -> bool:
            return hmac.compare_digest(candidate, auth_token)

        # WebSocket: check query param
        if request.headers.get("upgrade", "").lower() == "websocket":
            if _token_matches(request.query_params.get("token", "")):
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content=build_http_error_response(401, "Unauthorized").model_dump(),
            )
        # HTTP: Bearer or Basic auth
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer ") and _token_matches(auth_header[7:]):
            return await call_next(request)
        if auth_header.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth_header[6:]).decode()
                _, _, password = decoded.partition(":")
                if _token_matches(password):
                    return await call_next(request)
            except Exception:
                pass
        return JSONResponse(
            status_code=401,
            content=build_http_error_response(401, "Unauthorized").model_dump(),
        )

    @app.middleware("http")
    async def _memory_trace_middleware(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        call_next: Callable[[Request], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        if not memory_trace.is_enabled():
            return await call_next(request)
        run_id = (
            request.headers.get("x-ltx-memory-run-id")
            or os.environ.get("LTX_MEMORY_TRACE_RUN_ID")
            or "__process__"
        )
        case_id = request.headers.get("x-ltx-memory-case-id") or "__process__"
        ctx = memory_trace.MemoryTraceContext(run_id=run_id, case_id=case_id)
        setattr(request.state, _TRACE_TERMINAL_ATTR, False)
        with memory_trace.use_context(ctx):
            _trace_http_event(request, "http_start")
            try:
                response = await call_next(request)
            except Exception:
                if not _http_terminal_recorded(request):
                    _trace_http_event(
                        request,
                        "http_error",
                        status_code=500,
                        code="HTTP_500",
                        message="Internal Server Error",
                    )
                    setattr(request.state, _TRACE_TERMINAL_ATTR, True)
                raise
            if not _http_terminal_recorded(request):
                status = response.status_code
                if status < 400:
                    _trace_http_event(request, "http_end", status_code=status)
                else:
                    _trace_http_event(
                        request,
                        "http_error",
                        status_code=status,
                        code=f"HTTP_{status}",
                        message=f"HTTP {status}",
                    )
                    setattr(request.state, _TRACE_TERMINAL_ATTR, True)
        return response

    async def _route_http_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPError):
            log_http_error(request, exc)
            _record_http_terminal(
                request,
                status_code=exc.status_code,
                code=exc.code,
                message=exc.detail,
            )
            return JSONResponse(status_code=exc.status_code, content=exc.response.model_dump())
        resp = build_http_error_response(500, str(exc))
        _record_http_terminal(
            request,
            status_code=500,
            code=resp.code,
            message=resp.message,
        )
        return JSONResponse(status_code=500, content=resp.model_dump())

    async def _starlette_http_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, StarletteHTTPException):
            resp = build_http_error_response(exc.status_code, exc.detail)
            _record_http_terminal(
                request,
                status_code=exc.status_code,
                code=resp.code,
                message=resp.message,
            )
            return JSONResponse(status_code=exc.status_code, content=resp.model_dump())
        resp = build_http_error_response(500, str(exc))
        _record_http_terminal(
            request,
            status_code=500,
            code=resp.code,
            message=resp.message,
        )
        return JSONResponse(status_code=500, content=resp.model_dump())

    async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, RequestValidationError):
            resp = build_http_error_response(422, str(exc))
            _record_http_terminal(
                request,
                status_code=422,
                code=resp.code,
                message=resp.message,
            )
            return JSONResponse(status_code=422, content=resp.model_dump())
        resp = build_http_error_response(422, str(exc))
        _record_http_terminal(
            request,
            status_code=422,
            code=resp.code,
            message=resp.message,
        )
        return JSONResponse(status_code=422, content=resp.model_dump())

    async def _route_generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log_unhandled_exception(request, exc)
        resp = build_http_error_response(500, str(exc))
        _record_http_terminal(
            request,
            status_code=500,
            code=resp.code,
            message=resp.message,
        )
        return JSONResponse(status_code=500, content=resp.model_dump())

    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(HTTPError, _route_http_error_handler)
    app.add_exception_handler(StarletteHTTPException, _starlette_http_error_handler)
    app.add_exception_handler(Exception, _route_generic_error_handler)

    app.include_router(health_router)
    app.include_router(generation_router)
    app.include_router(models_router)
    app.include_router(model_profiles_router)
    app.include_router(settings_router)
    app.include_router(image_gen_router)
    app.include_router(suggest_gap_prompt_router)
    app.include_router(retake_router)
    app.include_router(ic_lora_router)
    app.include_router(runtime_policy_router)
    app.include_router(hf_auth_router)

    return app
