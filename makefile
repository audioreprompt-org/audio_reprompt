# terraform initialize
init:
	terraform -chdir=$(shell pwd)/terraform/stacks/${STACK} init -backend-config=$(shell pwd)/terraform/environments/${ENV}/${STACK}/backend.tfvars

# terraform plan and create a plan file
plan:init
	terraform -chdir=$(shell pwd)/terraform/stacks/${STACK} plan -var-file=$(shell pwd)/terraform/environments/${ENV}/${STACK}/terraform.tfvars -out=.tfplan

# apply the created plan file
apply:plan
	terraform -chdir=$(shell pwd)/terraform/stacks/${STACK} apply .tfplan

# terraform destroy all resources
destroy:
	terraform -chdir=$(shell pwd)/terraform/stacks/${STACK} destroy -var-file=$(shell pwd)/terraform/environments/${ENV}/${STACK}/terraform.tfvars
