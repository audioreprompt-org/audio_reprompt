variable "region" {
  description = "La región de AWS donde se desplegarán los recursos."
  type        = string
  nullable    = false
}

variable "owner" {
  description = "Dueño de los recursos. Para propósito acadmémico."
  type        = string
  nullable    = false
}

variable "db_publicly_accessible" {
  description = "Indica si la BD debe ser accesible públicamente."
  type        = bool
}

variable "sg_ingress_cidr_blocks" {
  description = "Lista de bloques CIDR permitidos para acceder a la instancia RDS PostgreSQL."
  type        = list(string)
}

variable "secret_name" {
  description = "Nombre del secreto en AWS Secrets Manager."
  type        = string
}

variable "db_allocated_storage_gib" {
  description = "Almacenamiento inicial en GB para la BD en gibibytes."
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

variable "enable_ec2_runner" {
  description = "Desplegar VM EC2 para ejecuciones desatendidas (on-demand)"
  type        = bool
  default     = false
}

variable "runner_instance_type" {
  description = "Tipo de instancia EC2 para el runner"
  type        = string
  default     = "t3.xlarge"
}

variable "runner_experiment_tag" {
  description = "Experimento inicial (tag AblationExperiment, p. ej. B2); vacío = sin auto-ejecución"
  type        = string
  default     = ""
}

variable "runner_git_branch" {
  description = "Rama que el runner ejecuta y donde publica resultados"
  type        = string
  default     = "ablations-experiments"
}

variable "runner_kaggle_secret_name" {
  description = "Secreto de Secrets Manager con credenciales de Kaggle ({\"username\",\"key\"})"
  type        = string
  default     = ""
}

variable "runner_git_ssh_secret_name" {
  description = "Secreto de Secrets Manager con la deploy key de GitHub (escritura) para publicar resultados"
  type        = string
  default     = ""
}

