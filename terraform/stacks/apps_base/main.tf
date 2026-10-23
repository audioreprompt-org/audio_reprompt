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
  kaggle_secret_name  = var.runner_kaggle_secret_name
  git_ssh_secret_name = var.runner_git_ssh_secret_name

  depends_on = [
    module.project_secrets_manager
  ]
}

