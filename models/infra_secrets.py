"""Infrastructure secrets and credentials manager.

Provides reliable, reusable helpers for retrieving and setting up database
(RDS/PostgreSQL), Kaggle, LLM and git (deploy key) credentials from AWS Secrets
Manager or environment variables without fragile shell parsing or inline
python snippets.

Any secret that is explicitly requested but cannot be resolved raises
``SecretResolutionError`` (CLI exit code 1) so unattended jobs fail fast
instead of running for hours without credentials.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[secrets %(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

GIT_DEPLOY_KEY_PATH = Path.home() / ".ssh" / "ablation_deploy_key"


class SecretResolutionError(RuntimeError):
    """A requested secret could not be retrieved or is missing required fields."""


def fetch_secret_string(secret_name: str, region: str = "us-east-1") -> str:
    """Retrieve the raw SecretString from AWS Secrets Manager.

    Raises SecretResolutionError if neither boto3 nor the aws CLI can read it.
    """
    errors: list[str] = []

    # Try boto3 first
    try:
        import boto3

        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=secret_name)
        secret_str = resp.get("SecretString")
        if secret_str:
            return secret_str
        errors.append("boto3: secret has no SecretString")
    except Exception as e:
        errors.append(f"boto3: {e}")
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
        if res.stdout.strip():
            return res.stdout
        errors.append("aws cli: empty SecretString")
    except Exception as e:
        errors.append(f"aws cli: {e}")

    raise SecretResolutionError(
        f"Failed to fetch secret '{secret_name}' ({region}): " + "; ".join(errors)
    )


def fetch_secret(secret_name: str, region: str = "us-east-1") -> dict[str, Any]:
    """Retrieve and parse a JSON secret from AWS Secrets Manager."""
    raw = fetch_secret_string(secret_name, region)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SecretResolutionError(f"Secret '{secret_name}' is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise SecretResolutionError(f"Secret '{secret_name}' must be a JSON object")
    return data


def resolve_db_credentials(
    secret_name: str | None = None,
    region: str = "us-east-1",
) -> dict[str, str]:
    """Resolve database environment variables from Secrets Manager or existing environment."""
    env_vars: dict[str, str] = {}

    if secret_name:
        data = fetch_secret(secret_name, region)
        host = data.get("host") or data.get("address")
        port = str(data.get("port") or "5432")
        user = data.get("username") or data.get("user")
        pwd = data.get("password")
        dbname = data.get("dbname") or data.get("db_name")

        missing = [k for k, v in (("host", host), ("username", user), ("password", pwd), ("dbname", dbname)) if not v]
        if missing:
            raise SecretResolutionError(f"DB secret '{secret_name}' is missing fields: {missing}")

        env_vars["DB_HOST"] = str(host)
        env_vars["DB_PORT"] = port
        env_vars["DB_USER"] = str(user)
        env_vars["DB_PASSWORD"] = str(pwd)
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
        s_user = data.get("username") or data.get("user") or data.get("KAGGLE_USERNAME")
        s_key = data.get("key") or data.get("token") or data.get("KAGGLE_KEY")
        if not (s_user and s_key):
            raise SecretResolutionError(
                f"Kaggle secret '{secret_name}' must contain 'username' and 'key'"
            )
        username = username or s_user  # explicit --kaggle-user still wins
        key = key or s_key

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


def resolve_llm_credentials(
    secret_name: str | None = None,
    region: str = "us-east-1",
) -> dict[str, str]:
    """Resolve LLM provider API keys from Secrets Manager."""
    env_vars: dict[str, str] = {}
    if not secret_name:
        return env_vars

    data = fetch_secret(secret_name, region)
    openai_key = data.get("OPENAI_API_KEY") or data.get("openai_api_key") or ""
    openai_base = data.get("OPENAI_BASE_URL") or data.get("openai_base_url") or ""

    if not openai_key:
        raise SecretResolutionError(
            f"LLM secret '{secret_name}' must contain 'OPENAI_API_KEY'"
        )

    if openai_key:
        env_vars["OPENAI_API_KEY"] = str(openai_key)
    if openai_base:
        env_vars["OPENAI_BASE_URL"] = str(openai_base)

    return env_vars


def resolve_git_ssh_key(
    secret_name: str,
    region: str = "us-east-1",
    key_path: Path = GIT_DEPLOY_KEY_PATH,
) -> dict[str, str]:
    """Install a GitHub deploy key from Secrets Manager and return GIT_SSH_COMMAND.

    The secret may be the raw OpenSSH private key or a JSON object with a
    ``private_key`` (or ``key``) field.
    """
    raw = fetch_secret_string(secret_name, region).strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SecretResolutionError(f"Git SSH secret '{secret_name}' is invalid JSON: {e}") from e
        raw = str(data.get("private_key") or data.get("key") or "").strip()
    if "PRIVATE KEY" not in raw:
        raise SecretResolutionError(f"Git SSH secret '{secret_name}' does not contain a private key")

    key_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    key_path.touch(mode=0o600, exist_ok=True)
    key_path.chmod(0o600)
    key_path.write_text(raw + "\n")  # OpenSSH requires a trailing newline

    ssh_cmd = (
        f"ssh -i {shlex.quote(str(key_path))} -o IdentitiesOnly=yes "
        "-o StrictHostKeyChecking=accept-new"
    )
    return {"GIT_SSH_COMMAND": ssh_cmd}


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
        "--llm-secret",
        help="AWS Secrets Manager secret name for LLM API keys (OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--git-ssh-secret", help="AWS Secrets Manager secret name holding a GitHub deploy key"
    )
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

    try:
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

        if args.llm_secret:
            resolved_vars.update(resolve_llm_credentials(args.llm_secret, region=args.region))

        if args.git_ssh_secret:
            resolved_vars.update(resolve_git_ssh_key(args.git_ssh_secret, region=args.region))
    except SecretResolutionError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.format == "shell":
        out = to_shell_exports(resolved_vars)
        if out:
            print(out)
    else:
        print(json.dumps(resolved_vars, indent=2))


if __name__ == "__main__":
    main()
