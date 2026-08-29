"""Price history tool, backed by yfinance."""

import yfinance as yf


def get_price(company: str, period: str = "1y") -> dict:
    try:
        ticker_obj = yf.Ticker(company)
        history = ticker_obj.history(period=period)
        history = history[history["Close"].notna()]

        if history.empty:
            return {"success": False, "data": None, "error": f"no price data found for {company}"}

        prices = [
            {"date": index.strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 4)}
            for index, row in history.iterrows()
        ]

        first_close = float(history["Close"].iloc[0])
        last_close = float(history["Close"].iloc[-1])
        pct_change = ((last_close - first_close) / first_close) * 100 if first_close else 0.0
        avg_volume = float(history["Volume"].mean())

        return {
            "success": True,
            "data": {
                "ticker": company.upper(),
                "prices": prices,
                "summary_stats": {
                    "pct_change": round(pct_change, 4),
                    "avg_volume": round(avg_volume, 2),
                },
            },
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - surface any unexpected failure through the tool contract
        return {"success": False, "data": None, "error": str(exc)}
