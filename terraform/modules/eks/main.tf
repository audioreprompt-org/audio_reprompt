module "eks_cluster" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 18.31"

  cluster_name    = var.cluster_name
  cluster_version = var.k8s_cluster_version
  vpc_id          = data.aws_vpc.default.id
  subnet_ids      = data.aws_subnets.all.ids

  cluster_endpoint_public_access = var.cluster_endpoint_public_access

  create_iam_role = false
  iam_role_arn    = local.cluster_role_arn

  eks_managed_node_groups = {
    default = {
      create_iam_role = false
      iam_role_arn    = local.node_role_arn
      ami_type        = "AL2023_x86_64_STANDARD"
      instance_types  = ["t3.medium"]
      desired_size    = 1
      min_size        = 1
      max_size        = 2
    }
  }

  aws_auth_roles = [
    {
      rolearn  = local.lab_role_arn
      username = "lab-admin"
      groups   = ["system:masters"]
    },
    {
      rolearn  = local.lab_role_arn
      username = "system:node:{{EC2PrivateDNSName}}"
      groups   = ["system:bootstrappers", "system:nodes"]
    }
  ]

  node_security_group_additional_rules = {
    ingress_control_plane_to_nodes_webhook = {
      description                   = "Allow Control Plane to Ingress webhook"
      protocol                      = "tcp"
      from_port                     = 8443
      to_port                       = 8443
      type                          = "ingress"
      source_cluster_security_group = true
    }
    ingress_nodes_to_nodes_all_traffic = {
      description = "Allow nodes to communicate with each other"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
    egress_nodes_to_internet = {
      description = "Allow nodes to connect to internet (ECR, external APIs)"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "egress"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  enable_irsa               = false
  create_aws_auth_configmap = false
  manage_aws_auth_configmap = true
}
