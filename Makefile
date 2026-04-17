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
ABLATION_CMD = uv run --env-file .env-example python -m models.ablation_runner
SAMPLE_SIZE ?= 30
SEED ?= 42
PHASE ?= all
RETRIEVAL_ARGS = --sample-size $(SAMPLE_SIZE) --seed $(SEED)
GENERATION_ARGS = --sample-size $(SAMPLE_SIZE) --seed $(SEED) --phase $(PHASE)

# Listar experimentos
ablation-list:
	$(ABLATION_CMD) --list --experiment all

# Calcular CLAP scores para CSVs existentes (uso: make ablation-score-csv CSV="path1.csv path2.csv")
ablation-score-csv:
	$(ABLATION_CMD) --phase score --reprompt-csv $(CSV)

# ── Retrieval (solo reprompts — métricas textuales, ambos modelos) ─
ablation-A1a:
	$(ABLATION_CMD) --experiment A1a $(RETRIEVAL_ARGS)

ablation-A1b:
	$(ABLATION_CMD) --experiment A1b $(RETRIEVAL_ARGS)

ablation-A1: ablation-A1a ablation-A1b

ablation-A2a:
	$(ABLATION_CMD) --experiment A2a $(RETRIEVAL_ARGS)

ablation-A2b:
	$(ABLATION_CMD) --experiment A2b $(RETRIEVAL_ARGS)

ablation-A2c:
	$(ABLATION_CMD) --experiment A2c $(RETRIEVAL_ARGS)

ablation-A2: ablation-A2a ablation-A2b ablation-A2c

ablation-A3a:
	$(ABLATION_CMD) --experiment A3a $(RETRIEVAL_ARGS)

ablation-A3b:
	$(ABLATION_CMD) --experiment A3b $(RETRIEVAL_ARGS)

ablation-A3: ablation-A3a ablation-A3b

ablation-retrieval: ablation-A1 ablation-A3 ablation-A2

# ── Generation (requieren audio/Kaggle + CLAP, usar PHASE) ────
ablation-B1a:
	$(ABLATION_CMD) --experiment B1a $(GENERATION_ARGS)

ablation-B1b:
	$(ABLATION_CMD) --experiment B1b $(GENERATION_ARGS)

ablation-B1c:
	$(ABLATION_CMD) --experiment B1c $(GENERATION_ARGS)

ablation-B1d:
	$(ABLATION_CMD) --experiment B1d $(GENERATION_ARGS)

ablation-B1: ablation-B1a ablation-B1b ablation-B1c ablation-B1d

ablation-B2a:
	$(ABLATION_CMD) --experiment B2a $(GENERATION_ARGS)

ablation-B2b:
	$(ABLATION_CMD) --experiment B2b $(GENERATION_ARGS)

ablation-B2: ablation-B2a ablation-B2b

ablation-B3a:
	$(ABLATION_CMD) --experiment B3a $(GENERATION_ARGS)

ablation-B3b:
	$(ABLATION_CMD) --experiment B3b $(GENERATION_ARGS)

ablation-B3c:
	$(ABLATION_CMD) --experiment B3c $(GENERATION_ARGS)

ablation-B3d:
	$(ABLATION_CMD) --experiment B3d $(GENERATION_ARGS)

ablation-B3e:
	$(ABLATION_CMD) --experiment B3e $(GENERATION_ARGS)

ablation-B3: ablation-B3a ablation-B3b ablation-B3c ablation-B3d ablation-B3e

ablation-generation: ablation-B1 ablation-B2 ablation-B3

# ── Full pipeline: reprompt → Kaggle → score → analysis ───────
FULL_ARGS = --sample-size $(SAMPLE_SIZE) --seed $(SEED) --phase full

ablation-B1-full:
	$(ABLATION_CMD) --experiment B1 $(FULL_ARGS)

ablation-B2-full:
	$(ABLATION_CMD) --experiment B2 $(FULL_ARGS)

ablation-B3-full:
	$(ABLATION_CMD) --experiment B3 $(FULL_ARGS)

# ── Kaggle remote audio generation ────────────────────────────
KAGGLE_CMD = uv run python -m models.kaggle_runner
RUN_ID ?=

# Upload CSVs + push notebook to Kaggle (usage: make kaggle-upload CSVS="data/ablations/reprompts/*.csv")
kaggle-upload:
	$(KAGGLE_CMD) upload --csvs $(CSVS)

# Check Kaggle kernel execution status
kaggle-status:
	$(KAGGLE_CMD) status

# Download generated audio from Kaggle
kaggle-download:
	$(KAGGLE_CMD) download $(if $(RUN_ID),--run-id $(RUN_ID))

# Full remote pipeline: upload → poll → download (usage: make kaggle-run CSVS="data/ablations/reprompts/*.csv")
kaggle-run:
	$(KAGGLE_CMD) run --csvs $(CSVS) $(if $(RUN_ID),--run-id $(RUN_ID))

# Kaggle + score for existing CSVs (usage: make kaggle-score CSVS="..." EXPERIMENT=B2 RUN_ID=R20260416_220924)
kaggle-score:
	$(KAGGLE_CMD) run --csvs $(CSVS) $(if $(RUN_ID),--run-id $(RUN_ID))
	$(ABLATION_CMD) --experiment $(EXPERIMENT) --phase score $(if $(RUN_ID),--run-id $(RUN_ID))

# ── Statistical analysis ──────────────────────────────────────
ANALYSIS_CMD = uv run python -m models.ablation_analysis
RUN =

# Run statistical analysis (usage: make ablation-analysis EXPERIMENT=B2 RUN=R20260416_194531)
ablation-analysis:
	$(ANALYSIS_CMD) --experiment $(EXPERIMENT) $(if $(RUN),--run $(RUN))

# List experiment groups with available score data
ablation-analysis-list:
	$(ANALYSIS_CMD) --list

# ── All ablations (ordered per Ablations.md) ──────────────────
ablation-all: ablation-retrieval ablation-generation
