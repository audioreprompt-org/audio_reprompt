variable "instance_type" {
  description = "EC2 instance type for the runner VM"
  type        = string
  default     = "t3.xlarge"
}

variable "volume_size_gb" {
  description = "Root EBS disk size in GiB"
  type        = number
  default     = 50
}

variable "secret_arn" {
  description = "ARN of the AWS Secrets Manager secret for RDS credentials"
  type        = string
}

variable "dvc_s3_bucket_arn" {
  description = "ARN of the S3 bucket used as DVC remote"
  type        = string
  default     = "arn:aws:s3:::maia-music-reprompt"
}

variable "subnet_id" {
  description = "Subnet ID where the runner instance will be deployed"
  type        = string
  default     = ""
}

variable "vpc_id" {
  description = "VPC ID where the runner instance and RDS reside"
  type        = string
  default     = ""
}

variable "experiment_tag" {
  description = "Default experiment to run on instance start (e.g. B2, or empty to require manual trigger)"
  type        = string
  default     = ""
}

variable "owner" {
  description = "Resource owner tag"
  type        = string
  default     = "student"
}
