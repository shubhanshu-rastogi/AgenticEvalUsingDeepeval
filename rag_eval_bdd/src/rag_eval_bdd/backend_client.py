from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rag_eval_bdd.models import BackendConfig


class BackendClient:
    def __init__(self, config: BackendConfig):
        self.config = config
        self.session = requests.Session()
        self._ask_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

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
        if use_cache and cache_key in self._ask_cache:
            return payload, deepcopy(self._ask_cache[cache_key])

        response = self.session.post(
            self._url(self.config.ask_endpoint),
            json=payload,
            timeout=self.config.timeout_s,
        )
        response.raise_for_status()
        response_payload = response.json()
        if use_cache:
            self._ask_cache[cache_key] = deepcopy(response_payload)
        return payload, response_payload
