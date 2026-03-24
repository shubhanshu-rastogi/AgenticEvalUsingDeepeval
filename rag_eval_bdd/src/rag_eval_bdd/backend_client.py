from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rag_eval_bdd.models import BackendConfig


class BackendClient:
    def __init__(self, config: BackendConfig):
        self.config = config
        self.session = requests.Session()
        self._ask_cache: "OrderedDict[Tuple[str, str], Tuple[float, Dict[str, Any]]]" = OrderedDict()

        retries = Retry(
            total=config.retries,
            read=config.retries,
            connect=config.retries,
            backoff_factor=config.backoff_s,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"]),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        api_key = os.getenv("API_KEY")
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _url(self, endpoint: str) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        base = self.config.base_url.rstrip("/")
        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return f"{base}{endpoint}"

    def check_reachable(self) -> None:
        candidates = [self.config.base_url, self._url("/docs"), self._url("/health")]
        last_error: Exception | None = None

        for url in candidates:
            try:
                resp = self.session.get(url, timeout=self.config.timeout_s)
                if resp.status_code < 500:
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        if last_error:
            raise RuntimeError(f"Backend is not reachable: {last_error}")
        raise RuntimeError("Backend is not reachable")

    def upload_document(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        with file_path.open("rb") as fh:
            files = {"file": (file_path.name, fh)}
            response = self.session.post(
                self._url(self.config.upload_endpoint),
                files=files,
                timeout=self.config.timeout_s,
            )
        response.raise_for_status()
        payload = response.json()
        session_id = payload.get("session_id")
        if not session_id:
            raise RuntimeError("Upload response does not contain session_id")
        return str(session_id), payload

    def ask_question(
        self,
        session_id: str,
        question: str,
        use_cache: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        payload = {"session_id": session_id, "question": question}
        cache_key = (session_id, question.strip())
        if use_cache:
            cached = self._get_cached_ask(cache_key)
            if cached is not None:
                return payload, cached

        response = self.session.post(
            self._url(self.config.ask_endpoint),
            json=payload,
            timeout=self.config.timeout_s,
        )
        response.raise_for_status()
        response_payload = response.json()
        if use_cache:
            self._set_cached_ask(cache_key, response_payload)
        return payload, response_payload

    def _get_cached_ask(self, cache_key: Tuple[str, str]) -> Dict[str, Any] | None:
        now = time.monotonic()
        self._prune_ask_cache(now)
        cached = self._ask_cache.get(cache_key)
        if cached is None:
            return None
        inserted_at, payload = cached
        ttl = max(0, int(self.config.ask_cache_ttl_s))
        if ttl > 0 and (now - inserted_at) > ttl:
            self._ask_cache.pop(cache_key, None)
            return None
        self._ask_cache.move_to_end(cache_key)
        return deepcopy(payload)

    def _set_cached_ask(self, cache_key: Tuple[str, str], payload: Dict[str, Any]) -> None:
        now = time.monotonic()
        self._prune_ask_cache(now)
        self._ask_cache[cache_key] = (now, deepcopy(payload))
        self._ask_cache.move_to_end(cache_key)

        max_entries = max(1, int(self.config.ask_cache_max_entries))
        while len(self._ask_cache) > max_entries:
            self._ask_cache.popitem(last=False)

    def _prune_ask_cache(self, now: float) -> None:
        ttl = max(0, int(self.config.ask_cache_ttl_s))
        if ttl == 0:
            return
        expired_keys = [
            key
            for key, (inserted_at, _) in self._ask_cache.items()
            if (now - inserted_at) > ttl
        ]
        for key in expired_keys:
            self._ask_cache.pop(key, None)

    def get_current_session(self) -> Dict[str, Any]:
        response = self.session.get(
            self._url(self.config.current_session_endpoint),
            timeout=self.config.timeout_s,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code == 404:
                raise RuntimeError(
                    "Backend endpoint '/session/current' is unavailable or no active UI session exists. "
                    "Restart backend after pulling latest changes, upload a file in UI, then retry."
                ) from exc
            raise
        payload = response.json()
        session_id = payload.get("session_id")
        if not session_id:
            raise RuntimeError("Current-session response does not contain session_id")
        return payload

    def get_session_chunks(self, limit: int | None = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = int(limit)
        response = self.session.get(
            self._url(self.config.session_chunks_endpoint),
            params=params,
            timeout=self.config.timeout_s,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code == 404:
                raise RuntimeError(
                    "Backend endpoint '/session/chunks' is unavailable or no active UI session exists. "
                    "Restart backend after pulling latest changes, upload a file in UI, then retry."
                ) from exc
            raise
        payload = response.json()
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            raise RuntimeError("Session-chunks response does not contain a valid chunks list")
        return chunks
