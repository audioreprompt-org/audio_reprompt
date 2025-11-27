module "ecr_backend" {
  source           = "../../modules/ecr"
  repository_name  = "uniandes/tasty-backend"
  keep_tags_number = 10
}

module "ecr_frontend" {
  source           = "../../modules/ecr"
  repository_name  = "uniandes/tasty-frontend"
  keep_tags_number = 10
}

module "eks_cluster" {
  source = "../../modules/eks"

  cluster_name                   = var.cluster_name
  k8s_cluster_version            = var.k8s_cluster_version
  cluster_endpoint_public_access = true
}
