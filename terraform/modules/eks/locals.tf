locals {
  # Safely taking the first found ARN for the Lab roles
  cluster_role_arn = tolist(data.aws_iam_roles.eks_cluster.arns)[0]
  node_role_arn    = tolist(data.aws_iam_roles.eks_node.arns)[0]
  lab_role_arn     = tolist(data.aws_iam_roles.lab_role.arns)[0]
}
