data "aws_region" "current" {}

locals {
  # Secrets the runner may read: RDS credentials + optional Kaggle / git deploy key / LLM keys.
  # Named secrets get a "*" or "-*" suffix because Secrets Manager appends a random id to ARNs.
  runner_secret_arns = distinct(compact(concat([
    var.secret_arn,
    "arn:aws:secretsmanager:*:*:secret:kaggle-*",
    var.kaggle_secret_name != "" ? "arn:aws:secretsmanager:*:*:secret:${var.kaggle_secret_name}*" : "",
    var.git_ssh_secret_name != "" ? "arn:aws:secretsmanager:*:*:secret:${var.git_ssh_secret_name}*" : "",
    var.llm_secret_name != "" ? "arn:aws:secretsmanager:*:*:secret:${var.llm_secret_name}*" : "",
  ], var.additional_secret_arns)))
}

data "aws_vpc" "default" {
  default = var.vpc_id == "" ? true : null
  id      = var.vpc_id != "" ? var.vpc_id : null
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ── IAM Role & Profile ────────────────────────────────────────────────────────
resource "aws_iam_role" "runner_role" {
  name = "audio-reprompt-ec2-runner-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# Attach AWS SSM Core policy for passwordless/keyless management via Systems Manager
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.runner_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Policy for Secrets Manager and DVC S3
resource "aws_iam_policy" "runner_policy" {
  name        = "audio-reprompt-ec2-runner-policy"
  description = "Allows access to RDS database secrets and DVC S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecretsManagerAccess"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = local.runner_secret_arns
      },
      {
        Sid    = "DvcS3Access"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          var.dvc_s3_bucket_arn,
          "${var.dvc_s3_bucket_arn}/*"
        ]
      }
      # Instance tags are read through IMDS (instance_metadata_tags), so
      # ec2:DescribeTags is no longer required.
    ]
  })
}

resource "aws_iam_role_policy_attachment" "runner_attach" {
  role       = aws_iam_role.runner_role.name
  policy_arn = aws_iam_policy.runner_policy.arn
}

resource "aws_iam_instance_profile" "runner_profile" {
  name = "audio-reprompt-ec2-runner-profile"
  role = aws_iam_role.runner_role.name
}

# ── Security Group ────────────────────────────────────────────────────────────
resource "aws_security_group" "runner_sg" {
  name        = "audio-reprompt-ec2-runner-sg"
  description = "Security group for runner instance"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name  = "audio-reprompt-ec2-runner-sg"
    owner = var.owner
  }
}

# ── EC2 Runner Instance ───────────────────────────────────────────────────────
resource "aws_instance" "runner" {
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.instance_type
  subnet_id            = var.subnet_id != "" ? var.subnet_id : data.aws_subnets.default.ids[0]
  iam_instance_profile = aws_iam_instance_profile.runner_profile.name

  # Crucial: when `sudo shutdown -h now` is called by the job script,
  # the instance transitions to `stopped` rather than `terminated`.
  instance_initiated_shutdown_behavior = "stop"

  vpc_security_group_ids = [aws_security_group.runner_sg.id]

  root_block_device {
    volume_size           = var.volume_size_gb
    volume_type           = "gp3"
    delete_on_termination = true
  }

  # IMDSv2 only; expose instance tags to IMDS so the boot-time trigger can read
  # `AblationExperiment` without the AWS CLI or ec2:DescribeTags.
  metadata_options {
    http_endpoint          = "enabled"
    http_tokens            = "required"
    instance_metadata_tags = "enabled"
  }

  user_data = templatefile("${path.module}/../../../deploy/aws/cloud-init.yaml.tftpl", {
    aws_region          = data.aws_region.current.name
    db_secret_name      = var.secret_arn
    kaggle_secret_name  = var.kaggle_secret_name
    git_ssh_secret_name = var.git_ssh_secret_name
    llm_secret_name     = var.llm_secret_name
    git_repo_url        = var.git_repo_url
    git_push_url        = var.git_ssh_secret_name != "" ? var.git_push_url : ""
    git_branch          = var.git_branch
    git_author_name     = var.git_author_name
    git_author_email    = var.git_author_email
  })
  # cloud-init only runs on first boot: recreate the VM when provisioning changes
  user_data_replace_on_change = true

  tags = merge(
    {
      Name  = "audio-reprompt-ec2-runner"
      owner = var.owner
    },
    var.experiment_tag != "" ? { "AblationExperiment" = var.experiment_tag } : {}
  )
}
