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
        Resource = [
          var.secret_arn,
          "arn:aws:secretsmanager:*:*:secret:kaggle-*"
        ]
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
      },
      {
        Sid    = "EC2TagRead"
        Effect = "Allow"
        Action = [
          "ec2:DescribeTags"
        ]
        Resource = "*"
      }
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

  user_data = file("${path.module}/../../../deploy/aws/cloud-init.yaml")

  tags = merge(
    {
      Name  = "audio-reprompt-ec2-runner"
      owner = var.owner
    },
    var.experiment_tag != "" ? { "AblationExperiment" = var.experiment_tag } : {}
  )
}
