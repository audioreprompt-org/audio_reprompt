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

# ── Ablation experiments ──────────────────────────────────────
ABLATION_CMD = uv run python -m models.ablation_runner
LIMIT ?= 100000

# Listar experimentos
ablation-list:
	$(ABLATION_CMD) --list --experiment all

# ── Retrieval ─────────────────────────────────────────────────
ablation-A1a:
	$(ABLATION_CMD) --experiment A1a --limit $(LIMIT)

ablation-A1b:
	$(ABLATION_CMD) --experiment A1b --limit $(LIMIT)

ablation-A1: ablation-A1a ablation-A1b

ablation-A2a:
	$(ABLATION_CMD) --experiment A2a --limit $(LIMIT)

ablation-A2b:
	$(ABLATION_CMD) --experiment A2b --limit $(LIMIT)

ablation-A2c:
	$(ABLATION_CMD) --experiment A2c --limit $(LIMIT)

ablation-A2: ablation-A2a ablation-A2b ablation-A2c

ablation-A3a:
	$(ABLATION_CMD) --experiment A3a --limit $(LIMIT)

ablation-A3b:
	$(ABLATION_CMD) --experiment A3b --limit $(LIMIT)

ablation-A3: ablation-A3a ablation-A3b

ablation-retrieval: ablation-A1 ablation-A3 ablation-A2

# ── Generation ────────────────────────────────────────────────
ablation-B1a:
	$(ABLATION_CMD) --experiment B1a --limit $(LIMIT)

ablation-B1b:
	$(ABLATION_CMD) --experiment B1b --limit $(LIMIT)

ablation-B1c:
	$(ABLATION_CMD) --experiment B1c --limit $(LIMIT)

ablation-B1d:
	$(ABLATION_CMD) --experiment B1d --limit $(LIMIT)

ablation-B1: ablation-B1a ablation-B1b ablation-B1c ablation-B1d

ablation-B2a:
	$(ABLATION_CMD) --experiment B2a --limit $(LIMIT)

ablation-B2b:
	$(ABLATION_CMD) --experiment B2b --limit $(LIMIT)

ablation-B2: ablation-B2a ablation-B2b

ablation-B3a:
	$(ABLATION_CMD) --experiment B3a --limit $(LIMIT)

ablation-B3b:
	$(ABLATION_CMD) --experiment B3b --limit $(LIMIT)

ablation-B3c:
	$(ABLATION_CMD) --experiment B3c --limit $(LIMIT)

ablation-B3d:
	$(ABLATION_CMD) --experiment B3d --limit $(LIMIT)

ablation-B3e:
	$(ABLATION_CMD) --experiment B3e --limit $(LIMIT)

ablation-B3: ablation-B3a ablation-B3b ablation-B3c ablation-B3d ablation-B3e

ablation-generation: ablation-B1 ablation-B2 ablation-B3

# ── All ablations (ordered per Ablations.md) ──────────────────
ablation-all: ablation-retrieval ablation-generation
