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

variable "kaggle_secret_name" {
  description = "Secrets Manager secret with Kaggle credentials ({\"username\",\"key\"}); empty = rely on ~/.kaggle/kaggle.json"
  type        = string
  default     = ""
}

variable "git_ssh_secret_name" {
  description = "Secrets Manager secret holding a GitHub deploy key (write access) used to push results; empty disables git push"
  type        = string
  default     = ""
}

variable "llm_secret_name" {
  description = "Secrets Manager secret holding LLM API keys ({\"OPENAI_API_KEY\",\"OPENAI_BASE_URL\"})"
  type        = string
  default     = ""
}

variable "additional_secret_arns" {
  description = "Additional Secrets Manager secret ARNs the runner is allowed to read"
  type        = list(string)
  default     = []
}

variable "git_repo_url" {
  description = "Repository URL used to clone/fetch (public repo: HTTPS, no credentials needed)"
  type        = string
  default     = "https://github.com/audioreprompt-org/audio_reprompt.git"
}

variable "git_push_url" {
  description = "Push URL used together with the deploy key"
  type        = string
  default     = "git@github.com:audioreprompt-org/audio_reprompt.git"
}

variable "git_branch" {
  description = "Branch the runner checks out, runs and publishes to"
  type        = string
  default     = "ablations-experiments"
}

variable "git_author_name" {
  description = "Git author/committer name for automated publish commits"
  type        = string
  default     = "ablation-bot"
}

variable "git_author_email" {
  description = "Git author/committer email for automated publish commits"
  type        = string
  default     = "ablation-bot@users.noreply.github.com"
}

variable "owner" {
  description = "Resource owner tag"
  type        = string
  default     = "student"
}
