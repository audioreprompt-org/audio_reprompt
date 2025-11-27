variable "region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "owner" {
  description = "Resource owner"
  type        = string
  default     = "student"
}

variable "cluster_name" {
  description = "EKS Cluster Name"
  type        = string
  default     = "audio-reprompt-cluster"
}

variable "k8s_cluster_version" {
  description = "Kubernetes Version"
  type        = string
  default     = "1.30"
}
