import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llm.llm import get_structured_completion
from vectorstore.pinecone_store import PineconeVectorStore, Retriever
import config

load_dotenv()


class ProfitabilityMetrics(BaseModel):
    """Income statement and cash flow figures."""
    revenue: str | int | None = Field(None, alias="Revenue")
    net_income: str | int | None = Field(None, alias="Net Income")
    operating_income: str | int | None = Field(None, alias="Operating Income")
    cash_flow: str | int | None = Field(None, alias="Cash Flow from Operating Activities")


class BalanceSheetQualitativeMetrics(BaseModel):
    """Balance sheet figures plus qualitative insights."""
    total_assets: str | int | None = Field(None, alias="Total Assets")
    total_liabilities: str | int | None = Field(None, alias="Total Liabilities")
    risk_factors: str | list | None = Field(None, alias="Top Risk Factors")
    growth_drivers: str | list | None = Field(None, alias="Top Growth Drivers")


PROFITABILITY_QUERY = """
Income statement, revenue, net income, operating income, earnings,
cash flow from operating activities, results of operations
for {company} fiscal year {year}
"""

BALANCE_SHEET_QUERY = """
Balance sheet, total assets, total liabilities, debt, material cash
requirements, risks, uncertainties, growth drivers, business outlook
for {company} fiscal year {year}
"""


def retrieve_context(
    retriever: Retriever,
    company: str,
    year: int,
    query: str
) -> str:
    """
    Retrieve context for a focused extraction query.
    """
    documents = retriever.invoke(
        query=query.format(company=company, year=year),
        company=company,
        year=year,
        top_k=config.KPI_TOP_K
    )
    return "\n\n".join(
        doc.page_content
        for doc in documents
    )


def build_profitability_prompt(
    company: str,
    year: int,
    context: str
) -> str:
    """
    Build prompt for income statement / cash flow extraction (pass 1).
    """
    return f"""
You are an expert financial analyst.

Company: {company}
Year: {year}

Context:
{context}

Extract the following information:

1. Revenue
2. Net Income
3. Operating Income
4. Cash Flow from Operating Activities

Instructions:

- Use only the provided context.
- Financial values must match the report exactly.
- Return null for a field ONLY if the context contains absolutely no
  information about it.
- Return valid JSON only.
"""


def build_balance_sheet_prompt(
    company: str,
    year: int,
    context: str
) -> str:
    """
    Build prompt for balance sheet + risks/growth extraction (pass 2).
    """
    return f"""
You are an expert financial analyst.

Company: {company}
Year: {year}

Context:
{context}

Extract the following information:

1. Total Assets
2. Total Liabilities
3. Top Risk Factors
4. Top Growth Drivers

Instructions:

- Use only the provided context.
- Financial values must match the report exactly.
- For Risk Factors and Growth Drivers: return 4 to 6 items each, as a
  list of specific, detailed statements (1-2 sentences) that include
  concrete evidence from the report: figures, percentages, segment
  names, product names, or events.
- Do NOT write generic items (e.g. "expansion of manufacturing
  capacity" or "market competition"). Every item must be grounded in
  the context.
- Return null for a field ONLY if the context contains absolutely no
  information about it. If the report discusses the topic at all,
  extract the most relevant points.
- Return valid JSON only.
"""


def extract_financial_metrics(
    retriever: Retriever,
    company: str,
    year: int
) -> dict:
    """
    Extract KPIs using two focused RAG passes:
      1. Income statement figures (revenue, income, cash flow)
      2. Balance sheet figures + risks/growth drivers

    Splitting into focused passes keeps each value near the top of a
    short, relevant context instead of buried at the end of one large
    generic context, which improves extraction reliability.
    """
    print(f"Extracting KPIs for {company} {year} (pass 1/2: income statement)...")
    context = retrieve_context(
        retriever=retriever,
        company=company,
        year=year,
        query=PROFITABILITY_QUERY
    )
    prompt = build_profitability_prompt(company, year, context)
    profitability = get_structured_completion(
        prompt=prompt,
        response_model=ProfitabilityMetrics
    )

    print(f"Extracting KPIs for {company} {year} (pass 2/2: balance sheet + risks/growth)...")
    context = retrieve_context(
        retriever=retriever,
        company=company,
        year=year,
        query=BALANCE_SHEET_QUERY
    )
    prompt = build_balance_sheet_prompt(company, year, context)
    balance = get_structured_completion(
        prompt=prompt,
        response_model=BalanceSheetQualitativeMetrics
    )

    results = {
        **profitability.model_dump(),
        **balance.model_dump()
    }

    print(f"KPI extraction complete for {company} {year}.")

    return results


def main(input_dir: str | None = None) -> None:
    """
    Extract and save KPIs for every ingested PDF.
    """
    from pathlib import Path

    from database.metrics import save_metrics
    from ingestion.embeddings import get_embeddings
    from utils import parse_company_year

    embeddings = get_embeddings()

    vector_store = PineconeVectorStore(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=config.PINECONE_INDEX_NAME,
        namespace=config.PINECONE_NAMESPACE
    )

    retriever = Retriever(
        vector_store.client,
        embeddings,
        vector_store.namespace
    )

    input_dir = input_dir or config.RAW_PDFS_DIR

    for pdf_file in Path(input_dir).glob("*.pdf"):
        company, year = parse_company_year(pdf_file)
        year = int(year) if year.isdigit() else None

        results = extract_financial_metrics(
            retriever=retriever,
            company=company,
            year=year
        )

        print(f"\nExtracted KPIs for {company} {year}\n")

        for key, value in results.items():
            print(f"{key}:")
            print(value)
            print("-" * 80)

        save_metrics(
            company=company,
            year=year,
            metrics=results
        )

if __name__ == "__main__":
    main()
