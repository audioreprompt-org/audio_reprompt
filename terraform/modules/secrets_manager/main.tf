resource "random_id" "secret_suffix" {
  byte_length = 4

  keepers = {
    name_prefix = var.secret_name
  }
}

# Agregamos un sufijo aleatorio al nombre del secreto para evitar colisiones
resource "aws_secretsmanager_secret" "db_credentials" {
  name        = "${var.secret_name}${random_id.secret_suffix.hex}"
  description = "Secret for RDS database credentials including connection details"
}

resource "aws_secretsmanager_secret_version" "db_credentials_version" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = var.db_username,
    password = var.db_password,
    engine   = var.db_engine,
    host     = var.db_host,
    port     = var.db_port,
    dbname   = var.db_name
  })
}

locals {
  create_kaggle_secret  = nonsensitive(var.kaggle_username != "" && var.kaggle_key != "")
  create_git_ssh_secret = nonsensitive(var.git_ssh_private_key != "")
  create_llm_secret     = nonsensitive(var.openai_api_key != "")
}

# Kaggle credentials secret
resource "aws_secretsmanager_secret" "kaggle_credentials" {
  count       = local.create_kaggle_secret ? 1 : 0
  name        = "${var.kaggle_secret_name}-${random_id.secret_suffix.hex}"
  description = "Secret for Kaggle API credentials (username and key)"
}

resource "aws_secretsmanager_secret_version" "kaggle_credentials_version" {
  count     = local.create_kaggle_secret ? 1 : 0
  secret_id = aws_secretsmanager_secret.kaggle_credentials[0].id
  secret_string = jsonencode({
    username = var.kaggle_username
    key      = var.kaggle_key
  })
}

# Git SSH deploy key secret
resource "aws_secretsmanager_secret" "git_ssh_key" {
  count       = local.create_git_ssh_secret ? 1 : 0
  name        = "${var.git_ssh_secret_name}-${random_id.secret_suffix.hex}"
  description = "Secret holding GitHub SSH deploy key for publishing ablation results"
}

resource "aws_secretsmanager_secret_version" "git_ssh_key_version" {
  count     = local.create_git_ssh_secret ? 1 : 0
  secret_id = aws_secretsmanager_secret.git_ssh_key[0].id
  secret_string = jsonencode({
    private_key = var.git_ssh_private_key
  })
}

# LLM provider API keys secret
resource "aws_secretsmanager_secret" "llm_credentials" {
  count       = local.create_llm_secret ? 1 : 0
  name        = "${var.llm_secret_name}-${random_id.secret_suffix.hex}"
  description = "Secret for LLM provider API keys (OpenAI)"
}

resource "aws_secretsmanager_secret_version" "llm_credentials_version" {
  count     = local.create_llm_secret ? 1 : 0
  secret_id = aws_secretsmanager_secret.llm_credentials[0].id
  secret_string = jsonencode({
    OPENAI_API_KEY   = var.openai_api_key
    OPENAI_BASE_URL  = var.openai_base_url
  })
}
