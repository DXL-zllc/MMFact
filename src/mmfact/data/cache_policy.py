"""Execution-local policy for reusable input features and fitted resources."""

from contextlib import contextmanager
from contextvars import ContextVar

_MODE = ContextVar("mmfact_input_mode", default="raw")


class PreparedInputError(ValueError):
    """A required prepared input is missing or incompatible with this request."""


def input_mode(mode=None):
    mode = _MODE.get() if mode is None else mode
    if mode not in ("raw", "prepared"):
        raise ValueError("Input mode must be 'raw' or 'prepared'.")
    return mode


def prepared_only(mode=None):
    return input_mode(mode) == "prepared" or input_mode() == "prepared"


@contextmanager
def cache_policy(mode):
    """Scope raw/prepared behavior without changing other threads or async tasks."""
    token = _MODE.set(input_mode(mode))
    try:
        yield
    finally:
        _MODE.reset(token)


def require_prepared(path, kind="input", *, mode=None):
    if prepared_only(mode):
        raise PreparedInputError(
            f"Prepared {kind} is missing at {path}. Generate or import matching materials "
            "first, or select preprocessing.mode=raw to prepare them. Prepared mode never rebuilds."
        )
