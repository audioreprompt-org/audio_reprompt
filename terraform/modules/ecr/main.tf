resource "aws_ecr_repository" "ecr" {
  name                 = var.repository_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

resource "aws_ecr_lifecycle_policy" "lifecycle_policy" {
  repository = aws_ecr_repository.ecr.name
  policy     = local.policy_document
}
