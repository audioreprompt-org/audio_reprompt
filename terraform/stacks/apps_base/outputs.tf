output "rds_address" {
  description = "Endpoint de la instancia RDS."
  value       = module.project_rds_db.address
}

output "rds_port" {
  description = "Puerto de la instancia RDS."
  value       = module.project_rds_db.port
}

output "rds_engine" {
  description = "Motor de la base de datos RDS."
  value       = module.project_rds_db.engine
}

output "rds_db_name" {
  description = "Nombre de la base de datos RDS."
  value       = module.project_rds_db.db_name
}

output "secrets_manager_secret_arn" {
  description = "ARN del secreto de AWS Secrets Manager."
  value       = module.project_secrets_manager.secret_arn
}

output "ec2_runner_instance_id" {
  description = "ID de la instancia EC2 runner (si está habilitada)"
  value       = var.enable_ec2_runner ? module.ec2_runner[0].instance_id : null
}

output "ec2_runner_start_command" {
  description = "Comando AWS CLI para encender el runner on-demand"
  value       = var.enable_ec2_runner ? module.ec2_runner[0].start_command : null
}

output "ec2_runner_ssm_trigger_command" {
  description = "Comando AWS SSM para ejecutar comandos en el runner sin claves SSH"
  value       = var.enable_ec2_runner ? module.ec2_runner[0].ssm_trigger_command : null
}

