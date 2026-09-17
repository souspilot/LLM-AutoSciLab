"""Crash-safe helpers shared by the paper benchmark runners.

The benchmark processes are intentionally disposable: every completed task and
every completed LLM response is persisted with an atomic rename.  A scheduler
preemption, Ctrl-C, or parent-process crash can therefore leave at most the
currently in-flight API request to repeat.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON durably without ever exposing a partially written target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, default=str, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def load_json(path: Path) -> Any | None:
    """Return decoded JSON, or ``None`` for a missing/corrupt checkpoint."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object, modulus: int = 2**32) -> int:
    """Stable integer seed (unlike Python's randomized built-in ``hash``)."""
    encoded = json.dumps(parts, separators=(",", ":"), default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % modulus


def safe_endpoint(endpoint: str | None) -> str | None:
    """Remove credentials and query parameters before recording an endpoint."""
    if not endpoint:
        return None
    split = urlsplit(endpoint)
    hostname = split.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = f":{split.port}" if split.port is not None else ""
    netloc = f"{hostname}{port}"
    return urlunsplit((split.scheme, netloc, split.path.rstrip("/"), "", ""))


def preflight_openai_endpoint(
    endpoint: str,
    model: str,
    api_key: str | None = None,
    timeout: float = 15.0,
) -> list[str]:
    """Check endpoint reachability and that its advertised model ID matches."""
    url = endpoint.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"OpenAI-compatible endpoint preflight failed for {safe_endpoint(endpoint)}: {exc}"
        ) from exc

    data = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]
    if model_ids and model not in model_ids:
        preview = ", ".join(model_ids[:8])
        raise RuntimeError(
            f"Model {model!r} is not advertised by {safe_endpoint(endpoint)}. "
            f"Available model IDs: {preview}"
        )
    return model_ids


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def ensure_run_config(directory: Path, config: dict[str, Any]) -> str:
    """Create or validate the immutable configuration for one result directory."""
    directory = Path(directory)
    fingerprint = config_fingerprint(config)
    target = directory / "run_config.json"
    existing = load_json(target)
    if existing is not None:
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Refusing to mix incompatible runs in {directory}. "
                "Use a different --out-dir, or restore the original arguments."
            )
        return fingerprint
    atomic_write_json(
        target,
        {"schema_version": 1, "fingerprint": fingerprint, "config": config},
    )
    return fingerprint


def completed_result(path: Path, fingerprint: str | None = None) -> dict[str, Any] | None:
    """Load a valid terminal result; error rows remain pending for a retry."""
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None
    if fingerprint is not None and payload.get("run_fingerprint") != fingerprint:
        return None
    if payload.get("status") in (None, "error"):
        return None
    return payload


class ResponseCache:
    """Task-local replay cache preserving repeated stochastic call order."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        raw = load_json(self.path) if self.path else None
        self._entries: dict[str, list[str | None]] = (
            raw.get("entries", {}) if isinstance(raw, dict) else {}
        )
        self._positions: dict[str, int] = {}

    @staticmethod
    def fingerprint(payload: dict[str, Any]) -> str:
        return config_fingerprint(payload)

    def lookup(self, key: str) -> tuple[int, str | None]:
        position = self._positions.get(key, 0)
        self._positions[key] = position + 1
        values = self._entries.get(key, [])
        value = values[position] if position < len(values) else None
        return position, value

    def store(self, key: str, position: int, value: str) -> None:
        if self.path is None:
            return
        values = self._entries.setdefault(key, [])
        while len(values) <= position:
            values.append(None)
        values[position] = value
        atomic_write_json(
            self.path,
            {"schema_version": 1, "entries": self._entries},
        )
