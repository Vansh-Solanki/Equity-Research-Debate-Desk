"""Phase 2 acceptance check: each MCP tool returns valid data for 3 test
companies, and error handling is verified for an invalid company.

Usage:
    python scripts/test_mcp_tools.py
"""

import sys

from dotenv import load_dotenv

from mcp_server.tools import get_filing, get_news, get_price

TEST_COMPANIES = ["AAPL", "MSFT", "TSLA"]
INVALID_COMPANY = "NOT_A_REAL_COMPANY_XYZ"


def check(label: str, result: dict, expect_success: bool) -> bool:
    ok = result.get("success") == expect_success
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: success={result.get('success')} error={result.get('error')}")
    if not ok:
        print(f"       full result: {result}")
    return ok


def main() -> int:
    load_dotenv()
    all_ok = True

    print("=== get_filing ===")
    for company in TEST_COMPANIES:
        result = get_filing(company)
        all_ok &= check(f"get_filing({company})", result, expect_success=True)
        if result["success"]:
            data = result["data"]
            print(f"       filed_date={data['filed_date']} chars={len(data['raw_text'])} url={data['source_url']}")
    all_ok &= check(f"get_filing({INVALID_COMPANY})", get_filing(INVALID_COMPANY), expect_success=False)

    print("\n=== get_price ===")
    for company in TEST_COMPANIES:
        result = get_price(company)
        all_ok &= check(f"get_price({company})", result, expect_success=True)
        if result["success"]:
            data = result["data"]
            print(f"       points={len(data['prices'])} stats={data['summary_stats']}")
    all_ok &= check(f"get_price({INVALID_COMPANY})", get_price(INVALID_COMPANY), expect_success=False)

    print("\n=== get_news ===")
    for company in TEST_COMPANIES:
        result = get_news(company)
        all_ok &= check(f"get_news({company})", result, expect_success=True)
        if result["success"]:
            articles = result["data"]["articles"]
            print(f"       articles={len(articles)}")
            if articles:
                print(f"       first: {articles[0]['title']!r} ({articles[0]['source']})")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
