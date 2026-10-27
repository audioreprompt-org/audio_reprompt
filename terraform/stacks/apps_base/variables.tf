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

  validation {
    condition     = contains(["", "all", "generation", "B1", "B2", "B3", "B1a", "B1b", "B1c", "B1d", "B2a", "B2b", "B3a", "B3b", "B3c", "B3d", "B3e"], var.runner_experiment_tag)
    error_message = "runner_experiment_tag debe ser vacío o un ID de experimento válido (B1, B2, B3, B1a-B1d, B2a-B2b, B3a-B3e, all, generation)."
  }
}

variable "runner_sample_size" {
  description = "Tamaño de muestra estratificada por defecto para el runner (tag AblationSampleSize)"
  type        = number
  default     = 30
}

variable "runner_seed" {
  description = "Semilla aleatoria por defecto para el runner (tag AblationSeed)"
  type        = number
  default     = 42
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

variable "runner_llm_secret_name" {
  description = "Secreto de Secrets Manager con las API keys de LLM (OPENAI_API_KEY, OPENAI_BASE_URL)"
  type        = string
  default     = ""
}

variable "kaggle_username" {
  description = "Usuario de Kaggle para almacenar en Secrets Manager"
  type        = string
  sensitive   = true
  default     = ""
}

variable "kaggle_key" {
  description = "API Key de Kaggle para almacenar en Secrets Manager"
  type        = string
  sensitive   = true
  default     = ""
}

variable "git_ssh_private_key" {
  description = "Clave privada SSH (deploy key) para publicar resultados en GitHub"
  type        = string
  sensitive   = true
  default     = ""
}

variable "openai_api_key" {
  description = "API Key de OpenAI / compatible endpoint para almacenar en Secrets Manager"
  type        = string
  sensitive   = true
  default     = ""
}

variable "openai_base_url" {
  description = "URL base opcional para el cliente OpenAI"
  type        = string
  default     = ""
}
