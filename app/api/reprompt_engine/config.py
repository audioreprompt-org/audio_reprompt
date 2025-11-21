import os

def get_db_connection_string():
    return (
        f"dbname={os.environ.get('DB_NAME', 'postgres')} "
        f"user={os.environ.get('DB_USER', 'postgres')} "
        f"password={os.environ.get('DB_PASSWORD', 'postgres')} "
        f"host={os.environ.get('DB_HOST', 'localhost')}"
    )

def get_moonshot_api_key():
    return os.getenv("MOONSHOT_API_KEY")
