variable "secret_name" {
  description = "Nombre del secreto en AWS Secrets Manager."
  type        = string
}

variable "db_username" {
  description = "Usuario de la base de datos."
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Contraseña de la base de datos."
  type        = string
  sensitive   = true
}

variable "db_engine" {
  description = "Motor de la base de datos."
  type        = string
}

variable "db_host" {
  description = "Host (endpoint) de la base de datos."
  type        = string
}

variable "db_port" {
  description = "Puerto de la base de datos."
  type        = number
}

variable "db_name" {
  description = "Nombre de la base de datos."
  type        = string
}

variable "kaggle_secret_name" {
  description = "Nombre base del secreto en Secrets Manager para credenciales de Kaggle."
  type        = string
  default     = "kaggle-ablation-credentials"
}

variable "kaggle_username" {
  description = "Usuario de Kaggle."
  type        = string
  sensitive   = true
  default     = ""
}

variable "kaggle_key" {
  description = "API Key de Kaggle."
  type        = string
  sensitive   = true
  default     = ""
}

variable "git_ssh_secret_name" {
  description = "Nombre base del secreto en Secrets Manager para la deploy key SSH de GitHub."
  type        = string
  default     = "git-ssh-ablation-deploy-key"
}

variable "git_ssh_private_key" {
  description = "Clave privada SSH (OpenSSH) con permisos de push en el repositorio."
  type        = string
  sensitive   = true
  default     = ""
}

variable "llm_secret_name" {
  description = "Nombre base del secreto en Secrets Manager para API keys de proveedores LLM."
  type        = string
  default     = "llm-ablation-api-keys"
}

variable "openai_api_key" {
  description = "API Key para OpenAI / compatible endpoint."
  type        = string
  sensitive   = true
  default     = ""
}

variable "openai_base_url" {
  description = "Base URL opcional para OpenAI compatible endpoint."
  type        = string
  default     = ""
}
