output "instance_id" {
  description = "EC2 Instance ID of the ablation runner"
  value       = aws_instance.runner.id
}

output "instance_private_ip" {
  description = "Private IP address of the runner"
  value       = aws_instance.runner.private_ip
}

output "start_command" {
  description = "AWS CLI command to power on the VM on demand"
  value       = "aws ec2 start-instances --instance-ids ${aws_instance.runner.id}"
}

output "ssm_trigger_command" {
  description = "AWS SSM command to trigger an ablation run without SSH keys"
  # Goes through systemd (env file, isolation, journald) and returns at once,
  # so SSM's 1h executionTimeout never kills the job. Follow logs with:
  #   journalctl -u ablation@B2 -f
  value = "aws ssm send-command --instance-ids ${aws_instance.runner.id} --document-name 'AWS-RunShellScript' --parameters 'commands=[\"mkdir -p /run/ablation && printf \\\"SAMPLE_SIZE=30\\\\nSEED=42\\\\n\\\" > /run/ablation/override.env && systemctl start --no-block ablation@B2.service\"]'"
}
