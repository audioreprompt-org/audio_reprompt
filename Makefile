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
ABLATION_CMD = uv run --env-file .env python -m models.ablation_runner
SAMPLE_SIZE ?= 30
SEED ?= 42
PHASE ?= all
GENERATION_ARGS = --sample-size $(SAMPLE_SIZE) --seed $(SEED) --phase $(PHASE)

# Listar experimentos
ablation-list:
	$(ABLATION_CMD) --list --experiment all

# Calcular CLAP scores para CSVs existentes (uso: make ablation-score-csv CSV="path1.csv path2.csv")
ablation-score-csv:
	$(ABLATION_CMD) --phase score --reprompt-csv $(CSV)

# Calcular métricas NLP para CSVs existentes (uso: make ablation-nlp-csv CSV="path1.csv path2.csv")
ablation-nlp-csv:
	$(ABLATION_CMD) --phase nlp --reprompt-csv $(CSV)

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
FULL_ARGS = --sample-size $(SAMPLE_SIZE) --seed $(SEED) --phase full $(if $(RUN_ID),--run-id $(RUN_ID))

ablation-B1-full:
	$(ABLATION_CMD) --experiment B1 $(FULL_ARGS)

ablation-B2-full:
	$(ABLATION_CMD) --experiment B2 $(FULL_ARGS)

ablation-B3-full:
	$(ABLATION_CMD) --experiment B3 $(FULL_ARGS)

# Concurrency-safe publish to DVC and git
ablation-publish:
	./scripts/ablation_publish.sh $(if $(RUN_ID),--run-id $(RUN_ID)) $(if $(BRANCH),--branch $(BRANCH))

# On-demand unattended runner (runs pipeline, publishes DVC hash, optional shutdown)
ablation-job:
	./scripts/run_ablation_job.sh --experiment $(or $(EXPERIMENT),B2) $(if $(RUN_ID),--run-id $(RUN_ID)) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE)) $(if $(SEED),--seed $(SEED)) $(if $(BRANCH),--branch $(BRANCH)) $(if $(SHUTDOWN),--shutdown) $(if $(NO_SYNC),--no-sync) $(if $(KAGGLE_USER),--kaggle-user $(KAGGLE_USER)) $(if $(KAGGLE_SECRET),--kaggle-secret $(KAGGLE_SECRET)) $(if $(GIT_SSH_SECRET),--git-ssh-secret $(GIT_SSH_SECRET))


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
ablation-all: ablation-generation

# ── RAG IR Quality Evaluation ─────────────────────────────────
IR_EVAL_CMD = uv run --env-file .env python -m models.ir_eval.cli
IR_OUTPUT_DIR ?= $(OUTPUT_DIR)
IR_QUERIES ?= $(QUERIES)
EXTRA_ARGS ?=

# Run RAG IR evaluation using ir_eval CLI (usage: make rag-eval [SEED=42] [IR_OUTPUT_DIR=path] [IR_QUERIES=path] [EXTRA_ARGS="..."])
rag-eval:
	$(IR_EVAL_CMD) $(if $(SEED),--seed $(SEED)) $(if $(IR_OUTPUT_DIR),--output-dir $(IR_OUTPUT_DIR)) $(if $(IR_QUERIES),--queries $(IR_QUERIES)) $(EXTRA_ARGS)

# Run only RAG diagnostics (skips retrieval loops)
rag-eval-diagnostics:
	$(IR_EVAL_CMD) $(if $(SEED),--seed $(SEED)) $(if $(IR_OUTPUT_DIR),--output-dir $(IR_OUTPUT_DIR)) $(if $(IR_QUERIES),--queries $(IR_QUERIES)) --skip-retrieval $(EXTRA_ARGS)
