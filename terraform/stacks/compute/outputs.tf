output "backend_repo_url" {
  description = "URL for the backend ECR repository"
  value       = module.ecr_backend.repository_url
}

output "frontend_repo_url" {
  description = "URL for the frontend ECR repository"
  value       = module.ecr_frontend.repository_url
}

output "eks_update_kubeconfig_command" {
  description = "Command to update your local kubeconfig"
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks_cluster.cluster_name}"
}
