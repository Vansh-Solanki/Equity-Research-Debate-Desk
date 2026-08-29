"""SEC EDGAR filing lookup tool.

Resolves a company name or ticker to a CIK, finds its most recent filing of a
given type, downloads the primary document, and strips HTML down to plain text.
"""

import os
import re
from functools import lru_cache

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}"

MAX_RAW_TEXT_CHARS = 3_000_000


def _session() -> requests.Session:
    user_agent = os.getenv("SEC_EDGAR_USER_AGENT")
    if not user_agent or "your-name" in user_agent:
        raise RuntimeError(
            "SEC_EDGAR_USER_AGENT must be set to a real, descriptive contact "
            "string (SEC blocks generic/default user agents)."
        )
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


@lru_cache(maxsize=1)
def _ticker_index():
    """Returns (ticker_upper -> cik, title_lower -> cik)."""
    resp = _session().get(TICKERS_URL, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    by_ticker = {}
    by_title = {}
    for entry in raw.values():
        cik = int(entry["cik_str"])
        by_ticker[entry["ticker"].upper()] = cik
        by_title[entry["title"].lower()] = cik
    return by_ticker, by_title


def _resolve_cik(company: str) -> int | None:
    by_ticker, by_title = _ticker_index()

    exact_ticker = by_ticker.get(company.upper())
    if exact_ticker is not None:
        return exact_ticker

    exact_title = by_title.get(company.lower())
    if exact_title is not None:
        return exact_title

    needle = company.lower()
    for title, cik in by_title.items():
        if needle in title:
            return cik

    return None


def _strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:MAX_RAW_TEXT_CHARS]


def get_filing(company: str, filing_type: str = "10-K") -> dict:
    try:
        session = _session()

        cik = _resolve_cik(company)
        if cik is None:
            return {"success": False, "data": None, "error": f"company not found: {company}"}

        resp = session.get(SUBMISSIONS_URL.format(cik=cik), timeout=15)
        resp.raise_for_status()
        submissions = resp.json()

        recent = submissions["filings"]["recent"]
        forms = recent["form"]
        accession_numbers = recent["accessionNumber"]
        filing_dates = recent["filingDate"]
        primary_docs = recent["primaryDocument"]

        match_idx = next((i for i, form in enumerate(forms) if form == filing_type), None)
        if match_idx is None:
            return {
                "success": False,
                "data": None,
                "error": f"no recent {filing_type} filing found for {company}",
            }

        accession_nodash = accession_numbers[match_idx].replace("-", "")
        doc_url = ARCHIVES_URL.format(cik=cik, accession_nodash=accession_nodash, doc=primary_docs[match_idx])

        doc_resp = session.get(doc_url, timeout=30)
        doc_resp.raise_for_status()
        raw_text = _strip_html(doc_resp.text)

        return {
            "success": True,
            "data": {
                "company": submissions.get("name", company),
                "filing_type": filing_type,
                "filed_date": filing_dates[match_idx],
                "raw_text": raw_text,
                "source_url": doc_url,
            },
            "error": None,
        }
    except requests.RequestException as exc:
        return {"success": False, "data": None, "error": f"request failed: {exc}"}
    except Exception as exc:  # noqa: BLE001 - surface any unexpected failure through the tool contract
        return {"success": False, "data": None, "error": str(exc)}
