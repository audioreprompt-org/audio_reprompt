# modules/rds/variables.tf
variable "db_allocated_storage_gib" {
  description = "Almacenamiento inicial en GB para la BD en gibibytes"
  type        = number
}

variable "db_name" {
  description = "Nombre de la base de datos en RDS."
  type        = string
}

variable "db_username" {
  description = "Usuario administrador para la base de datos RDS."
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Contraseña para el usuario administrador de la base de datos RDS."
  type        = string
  sensitive   = true
}

variable "db_publicly_accessible" {
  description = "Indica si la BD debe ser accesible públicamente."
  type        = bool
  default     = false
}

variable "sg_ingress_cidr_blocks" {
  description = "Lista de bloques CIDR blocks que pueden acceder a la instancia RDS PostgreSQL."
  type        = list(string)
}
