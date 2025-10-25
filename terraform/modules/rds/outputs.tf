output "address" {
  description = "Endpoint de la instancia RDS."
  value       = aws_db_instance.mi_rds_postgres.address
}

output "port" {
  description = "Puerto de la instancia RDS."
  value       = aws_db_instance.mi_rds_postgres.port
}

output "engine" {
  description = "Motor de la base de datos RDS."
  value       = aws_db_instance.mi_rds_postgres.engine
}

output "db_name" { # Esto es el nombre de la BD interna, un output válido
  description = "Nombre de la instancia de base de datos RDS."
  value       = aws_db_instance.mi_rds_postgres.db_name
}
