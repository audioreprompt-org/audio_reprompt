output "repository_url" {
  value = aws_ecr_repository.ecr.repository_url
  description = "The URL of the repository."
}

output "repository_arn" {
  value = aws_ecr_repository.ecr.arn
}
