from psycopg import connect
from audio_reprompt.config import get_db_connection_string


def get_conn():
    """Establishes a connection to the PostgreSQL database."""
    return connect(get_db_connection_string(), autocommit=True)
