"""Infrastructure secrets and credentials manager.

Provides reliable, reusable helpers for retrieving and setting up database
(RDS/PostgreSQL) and Kaggle credentials from AWS Secrets Manager or environment
variables without fragile shell parsing or inline python snippets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[secrets %(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def fetch_secret(secret_name: str, region: str = "us-east-1") -> dict[str, Any]:
    """Retrieve and parse a JSON secret from AWS Secrets Manager."""
    # Try boto3 first
    try:
        import boto3

        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=secret_name)
        secret_str = resp.get("SecretString")
        if secret_str:
            return json.loads(secret_str)
    except Exception as e:
        logger.debug(f"boto3 retrieval failed for {secret_name} ({e}); trying aws cli")

    # Fallback to aws CLI
    try:
        cmd = [
            "aws",
            "secretsmanager",
            "get-secret-value",
            "--secret-id",
            secret_name,
            "--region",
            region,
            "--query",
            "SecretString",
            "--output",
            "text",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(res.stdout.strip())
    except Exception as e:
        logger.warning(f"Failed to fetch secret '{secret_name}' from Secrets Manager: {e}")
        return {}


def resolve_db_credentials(
    secret_name: str | None = None,
    region: str = "us-east-1",
) -> dict[str, str]:
    """Resolve database environment variables from Secrets Manager or existing environment."""
    env_vars: dict[str, str] = {}

    if secret_name:
        data = fetch_secret(secret_name, region)
        if data:
            host = data.get("host") or data.get("address")
            port = str(data.get("port", "5432"))
            user = data.get("username") or data.get("user")
            pwd = data.get("password")
            dbname = data.get("dbname") or data.get("db_name")

            if host:
                env_vars["DB_HOST"] = str(host)
            if port:
                env_vars["DB_PORT"] = str(port)
            if user:
                env_vars["DB_USER"] = str(user)
            if pwd:
                env_vars["DB_PASSWORD"] = str(pwd)
            if dbname:
                env_vars["DB_NAME"] = str(dbname)

    return env_vars


def resolve_kaggle_credentials(
    secret_name: str | None = None,
    username: str | None = None,
    key: str | None = None,
    region: str = "us-east-1",
    sync_config_file: bool = True,
) -> dict[str, str]:
    """Resolve Kaggle credentials, optionally syncing ~/.kaggle/kaggle.json with 0600 permissions."""
    env_vars: dict[str, str] = {}

    # 1. From Secrets Manager if provided
    if secret_name:
        data = fetch_secret(secret_name, region)
        if data:
            s_user = data.get("username") or data.get("user") or data.get("KAGGLE_USERNAME")
            s_key = data.get("key") or data.get("token") or data.get("KAGGLE_KEY")
            if s_user:
                username = s_user
            if s_key:
                key = s_key

    # 2. Direct overrides
    if username:
        env_vars["KAGGLE_USERNAME"] = username
    if key:
        env_vars["KAGGLE_KEY"] = key

    eff_user = env_vars.get("KAGGLE_USERNAME") or os.getenv("KAGGLE_USERNAME")
    eff_key = env_vars.get("KAGGLE_KEY") or os.getenv("KAGGLE_KEY")

    # 3. Write/update ~/.kaggle/kaggle.json if both user and key are available
    if eff_user and eff_key and sync_config_file:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        conf_file = kaggle_dir / "kaggle.json"
        conf_file.write_text(json.dumps({"username": eff_user, "key": eff_key}, indent=2))
        conf_file.chmod(0o600)

    # 4. If credentials not set in env, check existing config file for username
    if not eff_user:
        conf_file = Path.home() / ".kaggle" / "kaggle.json"
        if conf_file.exists():
            try:
                conf = json.loads(conf_file.read_text())
                if conf.get("username"):
                    env_vars["KAGGLE_USERNAME"] = conf["username"]
            except Exception:
                pass

    return env_vars


def to_shell_exports(vars_dict: dict[str, str]) -> str:
    """Format dictionary into safely quoted shell export statements."""
    lines = []
    for k, v in sorted(vars_dict.items()):
        lines.append(f"export {k}={shlex.quote(str(v))}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve and format infrastructure secrets")
    parser.add_argument("--db-secret", help="AWS Secrets Manager secret name for database credentials")
    parser.add_argument("--kaggle-secret", help="AWS Secrets Manager secret name for Kaggle credentials")
    parser.add_argument("--kaggle-user", help="Explicit Kaggle username override")
    parser.add_argument("--kaggle-key", help="Explicit Kaggle API key override")
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
        help="AWS Region (default: us-east-1)",
    )
    parser.add_argument(
        "--format",
        choices=["shell", "json"],
        default="shell",
        help="Output format: shell (export statements for eval) or json",
    )

    args = parser.parse_args()

    resolved_vars: dict[str, str] = {}

    if args.db_secret:
        db_vars = resolve_db_credentials(args.db_secret, region=args.region)
        resolved_vars.update(db_vars)

    if args.kaggle_secret or args.kaggle_user or args.kaggle_key:
        kaggle_vars = resolve_kaggle_credentials(
            secret_name=args.kaggle_secret,
            username=args.kaggle_user,
            key=args.kaggle_key,
            region=args.region,
        )
        resolved_vars.update(kaggle_vars)

    if args.format == "shell":
        out = to_shell_exports(resolved_vars)
        if out:
            print(out)
    else:
        print(json.dumps(resolved_vars, indent=2))


if __name__ == "__main__":
    main()
