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

# Build the python distribution wheel for the models
build-reprompt:
	@rm -rf ./models/build ./models/dist ./models/*.egg-info
	@rm -f ./app/api/wheels/*.whl
	@cd ./models && uv build --wheel && cd ..
	@mkdir -p ./app/api/wheels/ && cp ./models/dist/*.whl ./app/api/wheels/
	@rm -rf ./models/build ./models/dist ./*.egg-info
