module "project_rds_db" {
  source                   = "../../modules/rds"
  db_allocated_storage_gib = var.db_allocated_storage_gib
  db_name                  = var.db_name
  db_username              = var.db_username
  db_password              = var.db_password
  db_publicly_accessible   = var.db_publicly_accessible
  sg_ingress_cidr_blocks   = var.sg_ingress_cidr_blocks
}

module "project_secrets_manager" {
  source      = "../../modules/secrets_manager"
  secret_name = var.secret_name
  db_username = var.db_username               # Mismo usuario que se usa para RDS
  db_password = var.db_password               # Misma contraseña que se usa para RDS
  db_engine   = module.project_rds_db.engine  # RDS engine type, e.g., "postgres"
  db_host     = module.project_rds_db.address # RDS host address
  db_port     = module.project_rds_db.port    # RDS host port, e.g., "5432"
  db_name     = module.project_rds_db.db_name # RDS database name

  kaggle_secret_name  = var.runner_kaggle_secret_name != "" ? var.runner_kaggle_secret_name : "kaggle-ablation-credentials"
  kaggle_username     = var.kaggle_username
  kaggle_key          = var.kaggle_key
  git_ssh_secret_name = var.runner_git_ssh_secret_name != "" ? var.runner_git_ssh_secret_name : "git-ssh-ablation-deploy-key"
  git_ssh_private_key = var.git_ssh_private_key
  llm_secret_name     = var.runner_llm_secret_name != "" ? var.runner_llm_secret_name : "llm-ablation-api-keys"
  openai_api_key      = var.openai_api_key
  openai_base_url     = var.openai_base_url

  # Explicit dependency to ensure RDS is fully created before Secrets Manager tries to read its outputs
  depends_on = [
    module.project_rds_db
  ]
}

module "ec2_runner" {
  count         = var.enable_ec2_runner ? 1 : 0
  source        = "../../modules/ec2_runner"
  secret_arn    = module.project_secrets_manager.secret_arn
  instance_type = var.runner_instance_type
  owner         = var.owner

  experiment_tag      = var.runner_experiment_tag
  git_branch          = var.runner_git_branch
  kaggle_secret_name  = module.project_secrets_manager.kaggle_secret_name != "" ? module.project_secrets_manager.kaggle_secret_name : var.runner_kaggle_secret_name
  git_ssh_secret_name = module.project_secrets_manager.git_ssh_secret_name != "" ? module.project_secrets_manager.git_ssh_secret_name : var.runner_git_ssh_secret_name
  llm_secret_name     = module.project_secrets_manager.llm_secret_name != "" ? module.project_secrets_manager.llm_secret_name : var.runner_llm_secret_name
  additional_secret_arns = compact([
    module.project_secrets_manager.kaggle_secret_arn,
    module.project_secrets_manager.git_ssh_secret_arn,
    module.project_secrets_manager.llm_secret_arn,
  ])

  depends_on = [
    module.project_secrets_manager
  ]
}

