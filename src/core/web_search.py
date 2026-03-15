from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from src.schemas.prompt_coach import WebSearchResult


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[WebSearchResult] = []
        self._capture_title = False
        self._capture_snippet = False
        self._current_url: str | None = None
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        class_name = attr_map.get("class", "") or ""
        if tag == "a" and "result__a" in class_name:
            self._capture_title = True
            self._current_url = _normalize_duckduckgo_url(attr_map.get("href"))
            self._current_title = []
            self._current_snippet = []
        elif tag == "a" and "result__snippet" in class_name:
            self._capture_snippet = True

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._current_title.append(data)
        elif self._capture_snippet:
            self._current_snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a":
            return
        if self._capture_snippet:
            self._capture_snippet = False
            return
        if self._capture_title:
            self._capture_title = False
            title = " ".join(part.strip() for part in self._current_title if part.strip()).strip()
            snippet = " ".join(part.strip() for part in self._current_snippet if part.strip()).strip() or None
            if title and self._current_url:
                self.results.append(WebSearchResult(title=title, url=self._current_url, snippet=snippet))


class WebSearchClient:
    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        response = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "evalpipeline-prompt-coach/1.0"},
            timeout=self.timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
        parser = DuckDuckGoHTMLParser()
        parser.feed(response.text)
        return parser.results[:max_results]


def _normalize_duckduckgo_url(url: str | None) -> str | None:
    if not url:
        return None
    if url.startswith("//"):
        return f"https:{url}"
    if "duckduckgo.com/l/?" not in url:
        return url
    parsed = urlparse(url)
    uddg = parse_qs(parsed.query).get("uddg")
    if not uddg:
        return url
    return unquote(uddg[0])
