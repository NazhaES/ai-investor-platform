from sqlalchemy import text

from database.postgres_sql import get_engine


def to_text(value) -> str | None:
    """Join a list into lines; keep strings and None as-is."""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return value


def save_metrics(
    company: str,
    year: int,
    metrics: dict
) -> None:
    """
    Save extracted financial metrics to PostgreSQL.

    Args:
        company: Company name.
        year: Fiscal year.
        metrics: Extracted KPI dictionary.
    """
    engine = get_engine()

    query = """
    INSERT INTO financial_metrics (
        company,
        year,
        revenue,
        net_income,
        operating_income,
        cash_flow,
        total_assets,
        total_liabilities,
        risk_factors,
        growth_drivers
    )
    VALUES (
        :company,
        :year,
        :revenue,
        :net_income,
        :operating_income,
        :cash_flow,
        :total_assets,
        :total_liabilities,
        :risk_factors,
        :growth_drivers
    )
    ON CONFLICT (company, year) DO UPDATE SET
        revenue = EXCLUDED.revenue,
        net_income = EXCLUDED.net_income,
        operating_income = EXCLUDED.operating_income,
        cash_flow = EXCLUDED.cash_flow,
        total_assets = EXCLUDED.total_assets,
        total_liabilities = EXCLUDED.total_liabilities,
        risk_factors = EXCLUDED.risk_factors,
        growth_drivers = EXCLUDED.growth_drivers,
        created_at = CURRENT_TIMESTAMP
    """

    params = {
        "company": company,
        "year": str(year),
        "revenue": metrics.get("Revenue") or metrics.get("revenue"),
        "net_income": metrics.get("Net Income") or metrics.get("net_income"),
        "operating_income": metrics.get("Operating Income") or metrics.get("operating_income"),
        "cash_flow": metrics.get("Cash Flow from Operating Activities") or metrics.get("cash_flow"),
        "total_assets": metrics.get("Total Assets") or metrics.get("total_assets"),
        "total_liabilities": metrics.get("Total Liabilities") or metrics.get("total_liabilities"),
        "risk_factors": to_text(metrics.get("Top Risk Factors") or metrics.get("risk_factors")),
        "growth_drivers": to_text(metrics.get("Top Growth Drivers") or metrics.get("growth_drivers"))
    }

    with engine.begin() as connection:
        connection.execute(text(query), params)

    print(
        f"Successfully saved metrics for {company} {year}"
    )


def get_metrics() -> list[dict]:
    """
    Load all saved financial metrics, one row per company and year.

    Returns:
        List of metric rows.
    """
    engine = get_engine()

    query = """
    SELECT *
    FROM financial_metrics
    ORDER BY company, year
    """

    with engine.connect() as connection:
        result = connection.execute(text(query))
        rows = [dict(row._mapping) for row in result]

    return rows


if __name__ == "__main__":
    for row in get_metrics():
        print(row)
