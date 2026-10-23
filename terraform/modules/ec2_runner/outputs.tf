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
  value       = "aws ssm send-command --instance-ids ${aws_instance.runner.id} --document-name 'AWS-RunShellScript' --parameters 'commands=[\"sudo -u ubuntu /opt/audio_reprompt/scripts/run_ablation_job.sh --experiment B2 --shutdown\"]'"
}
