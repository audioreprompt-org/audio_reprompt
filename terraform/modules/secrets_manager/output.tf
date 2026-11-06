output "secret_arn" {
  description = "ARN del secreto de AWS Secrets Manager."
  value       = aws_secretsmanager_secret.db_credentials.arn
}

output "secret_name" {
  description = "Nombre del secreto de AWS Secrets Manager."
  value       = aws_secretsmanager_secret.db_credentials.name
}

output "kaggle_secret_arn" {
  description = "ARN del secreto de credenciales de Kaggle."
  value       = local.create_kaggle_secret ? aws_secretsmanager_secret.kaggle_credentials[0].arn : ""
}

output "kaggle_secret_name" {
  description = "Nombre del secreto de credenciales de Kaggle."
  value       = local.create_kaggle_secret ? aws_secretsmanager_secret.kaggle_credentials[0].name : ""
}

output "git_ssh_secret_arn" {
  description = "ARN del secreto de deploy key SSH de Git."
  value       = local.create_git_ssh_secret ? aws_secretsmanager_secret.git_ssh_key[0].arn : ""
}

output "git_ssh_secret_name" {
  description = "Nombre del secreto de deploy key SSH de Git."
  value       = local.create_git_ssh_secret ? aws_secretsmanager_secret.git_ssh_key[0].name : ""
}

output "llm_secret_arn" {
  description = "ARN del secreto de API keys de LLM."
  value       = local.create_llm_secret ? aws_secretsmanager_secret.llm_credentials[0].arn : ""
}

output "llm_secret_name" {
  description = "Nombre del secreto de API keys de LLM."
  value       = local.create_llm_secret ? aws_secretsmanager_secret.llm_credentials[0].name : ""
}
