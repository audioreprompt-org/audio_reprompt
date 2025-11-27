import os
from psycopg import connect


def get_conn():
    db_user = os.environ["DB_USER"]
    db_password = os.environ["DB_PASSWORD"]
    db_host = os.environ["DB_HOST"]
    db_name = os.environ["DB_NAME"]

    return connect(
        f"dbname={db_name} user={db_user} password={db_password} host={db_host}",
        autocommit=True,
    )
