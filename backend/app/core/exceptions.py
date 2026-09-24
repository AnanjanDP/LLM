from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.core.logging import logger


class APIException(Exception):
    """Base API exception class with HTTP status code and message."""
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class LLMProviderError(APIException):
    """Exception raised when an LLM provider call fails."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=f"LLM Provider Error: {message}",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details
        )


class AuthenticationError(APIException):
    """Exception raised for authentication failures."""
    def __init__(self, message: str = "Invalid authentication credentials"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details={"headers": {"WWW-Authenticate": "Bearer"}}
        )


class NotFoundError(APIException):
    """Exception raised when a requested resource is missing."""
    def __init__(self, resource: str = "Resource"):
        super().__init__(
            message=f"{resource} not found",
            status_code=status.HTTP_404_NOT_FOUND
        )


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Global handler for custom API exceptions."""
    logger.error(f"API Error [{exc.status_code}] path={request.url.path}: {exc.message}")
    headers = exc.details.get("headers", {}) if exc.details else {}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.message,
            "details": {k: v for k, v in exc.details.items() if k != "headers"}
        },
        headers=headers
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Global handler for Pydantic validation errors."""
    logger.warning(f"Validation Error path={request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Request validation failed",
            "details": exc.errors()
        }
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global handler for unhandled server exceptions."""
    logger.exception(f"Unhandled Exception path={request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "An internal server error occurred.",
            "details": str(exc) if not request.app.debug else str(exc)
        }
    )
