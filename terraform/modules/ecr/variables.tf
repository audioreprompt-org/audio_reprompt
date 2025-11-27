variable "repository_name" {
  description = "The name of the repository in ECR."
  type        = string
}

variable "keep_tags_number" {
  description = "Number of images to retain."
  type        = number
  default     = 10
}
