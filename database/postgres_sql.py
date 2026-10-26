import os
from urllib.parse import quote

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_engine(database: str | None = None):
    """
    Create PostgreSQL connection.

    Args:
        database: Database name. Defaults to POSTGRES_DATABASE from .env.
    """
    if database is None:
        database = os.getenv("POSTGRES_DATABASE")

    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    encoded_user = quote(user, safe="")
    encoded_password = quote(password, safe="")

    connection_string = (
        f"postgresql+psycopg2://"
        f"{encoded_user}:{encoded_password}@{host}:{port}/{database}"
    )

    return create_engine(connection_string)
