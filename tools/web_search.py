from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool

_TRUSTED_DOMAINS = ["cdc.gov", "who.int", "nih.gov", "mayoclinic.org", "healthline.com"]

_ddg = DuckDuckGoSearchResults(max_results=5)

@tool
def medical_web_search(query: str) -> str:
    """
    Free web search using DuckDuckGo. We bias towards trusted medical domains
    by adding site: filters to the query.
    """
    domain_filter = " OR ".join([f"site:{d}" for d in _TRUSTED_DOMAINS])
    q = f"{query} ({domain_filter})"
    return _ddg.invoke({"query": q})