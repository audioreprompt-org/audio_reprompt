# ── Environment & Terraform Configuration ─────────────────────
-include .env

STACK ?= apps_base
ENV ?= student1
ENABLE_EC2_RUNNER ?= true

# Pass GIT_SSH_PRIVATE_KEY via shell env inside recipes (avoid GNU Make $(shell cat ...),
# which converts all newlines in the SSH private key into spaces and breaks libcrypto).
ifneq ($(strip $(GIT_SSH_KEY_PATH)),)
GIT_SSH_ENV = TF_VAR_git_ssh_private_key="$$(cat $(GIT_SSH_KEY_PATH))"
else ifneq ($(strip $(GIT_SSH_PRIVATE_KEY)),)
export TF_VAR_git_ssh_private_key = $(GIT_SSH_PRIVATE_KEY)
GIT_SSH_ENV =
else
GIT_SSH_ENV =
endif

# Map environment variables (.env) or CLI overrides to Terraform -var flags
TF_VARS = \
	-var="enable_ec2_runner=$(ENABLE_EC2_RUNNER)" \
	$(if $(KAGGLE_SECRET_NAME),-var="runner_kaggle_secret_name=$(KAGGLE_SECRET_NAME)") \
	$(if $(KAGGLE_USERNAME),-var="kaggle_username=$(KAGGLE_USERNAME)") \
	$(if $(KAGGLE_KEY),-var="kaggle_key=$(KAGGLE_KEY)") \
	$(if $(GIT_SSH_SECRET_NAME),-var="runner_git_ssh_secret_name=$(GIT_SSH_SECRET_NAME)") \
	$(if $(LLM_SECRET_NAME),-var="runner_llm_secret_name=$(LLM_SECRET_NAME)") \
	$(if $(OPENAI_API_KEY),-var="openai_api_key=$(OPENAI_API_KEY)") \
	$(if $(OPENAI_BASE_URL),-var="openai_base_url=$(OPENAI_BASE_URL)") \
	$(if $(BRANCH),-var="runner_git_branch=$(BRANCH)") \
	$(EXTRA_TF_VARS)

# Terraform core targets
init:
	terraform -chdir=$(CURDIR)/terraform/stacks/$(STACK) init -backend-config=$(CURDIR)/terraform/environments/$(ENV)/$(STACK)/backend.tfvars

plan: init
	$(GIT_SSH_ENV) terraform -chdir=$(CURDIR)/terraform/stacks/$(STACK) plan \
		-var-file=$(CURDIR)/terraform/environments/$(ENV)/$(STACK)/terraform.tfvars \
		$(TF_VARS) \
		-out=.tfplan

apply: plan
	terraform -chdir=$(CURDIR)/terraform/stacks/$(STACK) apply .tfplan

destroy:
	$(GIT_SSH_ENV) terraform -chdir=$(CURDIR)/terraform/stacks/$(STACK) destroy \
		-var-file=$(CURDIR)/terraform/environments/$(ENV)/$(STACK)/terraform.tfvars \
		$(TF_VARS)

# ── Ablation EC2 Runner Deployment ───────────────────────────
# Despliega RDS + Secrets Manager + EC2 Runner en apps_base
ablation-deploy: apply

# Muestra outputs (ID de la instancia EC2, comando SSM, comando de inicio)
ablation-deploy-output:
	terraform -chdir=$(CURDIR)/terraform/stacks/apps_base output

# Destruye la infraestructura desplegada
ablation-destroy: destroy

EXPERIMENT ?= B2
LOG_FILE ?= logs_ablations_latest.txt

# Descarga los logs de consola más recientes de la instancia EC2
ablation-ec2-logs:
	@INSTANCE_ID=$$(aws ec2 describe-instances \
		--filters "Name=instance-state-name,Values=running,stopped" \
		--query "Reservations[].Instances[].InstanceId" \
		--output text | awk '{print $$1}'); \
	if [ -z "$$INSTANCE_ID" ] || [ "$$INSTANCE_ID" = "None" ] || [ "$$INSTANCE_ID" = "null" ]; then \
		echo "Error: No EC2 instance found via aws ec2 describe-instances." >&2; exit 1; \
	fi; \
	echo "==> Fetching latest console logs for instance ($$INSTANCE_ID) to $(LOG_FILE)..."; \
	aws ec2 get-console-output \
		--instance-id "$$INSTANCE_ID" \
		--latest \
		--output text > $(LOG_FILE); \
	echo "==> Logs saved to $(LOG_FILE)"

# Ejecuta un experimento de ablación en la instancia EC2 remota vía AWS SSM
ablation-remote: ablation-ec2-run

ablation-ec2-run:
	@INSTANCE_ID=$$(aws ec2 describe-instances \
		--filters "Name=instance-state-name,Values=running,stopped" \
		--query "Reservations[].Instances[].InstanceId" \
		--output text | awk '{print $$1}'); \
	if [ -z "$$INSTANCE_ID" ] || [ "$$INSTANCE_ID" = "None" ] || [ "$$INSTANCE_ID" = "null" ]; then \
		echo "Error: No EC2 instance found via aws ec2 describe-instances." >&2; exit 1; \
	fi; \
	echo "==> Using EC2 instance ($$INSTANCE_ID) for experiment $(EXPERIMENT)..."; \
	aws ec2 start-instances --instance-ids "$$INSTANCE_ID" >/dev/null; \
	aws ec2 wait instance-running --instance-ids "$$INSTANCE_ID"; \
	while [ "$$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$$INSTANCE_ID" --query "InstanceInformationList[0].PingStatus" --output text 2>/dev/null)" != "Online" ]; do \
		sleep 5; \
	done; \
	CMD_ID=$$(aws ssm send-command \
		--instance-ids "$$INSTANCE_ID" \
		--document-name "AWS-RunShellScript" \
		--comment "Run ablation experiment $(EXPERIMENT)" \
		--parameters 'commands=["if [ ! -f /etc/systemd/system/ablation@.service ]; then if command -v cloud-init >/dev/null 2>&1; then cloud-init status --wait || true; fi; if [ ! -f /etc/systemd/system/ablation@.service ] && [ -x /usr/local/sbin/ablation-bootstrap ]; then /usr/local/sbin/ablation-bootstrap; fi; while [ ! -f /etc/systemd/system/ablation@.service ]; do sleep 2; done; fi","mkdir -p /run/ablation","printf '\''SAMPLE_SIZE=%s\nSEED=%s\n'\'' $(SAMPLE_SIZE) $(SEED) > /run/ablation/override.env","systemctl daemon-reload","systemctl start --no-block ablation@$(EXPERIMENT).service"]' \
		--query "Command.CommandId" \
		--output text); \
	echo "==> Triggered ablation@$(EXPERIMENT) (Command: $$CMD_ID)"; \
	echo "    Logs (console output): make ablation-ec2-logs";

# ── One-time Local Postgres Export & RDS Hydration ───────────
DB_DUMP_FILE ?= local_pg.dump
RDS_USER ?= temp
RDS_PASSWORD ?= temp
RDS_DB ?= temp

# Export local Docker Postgres (app/docker-compose.yml) to a snapshot file
db-export:
	docker compose -f app/docker-compose.yml exec -T db \
		pg_dump -Fc --no-owner --no-acl -U $(or $(DB_USER),postgres) -d $(or $(DB_NAME),postgres) > $(DB_DUMP_FILE)
	@ls -lh $(DB_DUMP_FILE)

# Hydrate RDS from the snapshot file and verify table row counts
db-hydrate:
	@RDS_HOST=$${RDS_HOST:-$$(terraform -chdir=$(CURDIR)/terraform/stacks/$(STACK) output -raw rds_address)}; \
	echo "Hydrating RDS ($$RDS_HOST) from $(DB_DUMP_FILE)..."; \
	docker compose -f app/docker-compose.yml exec -T -e PGPASSWORD="$(RDS_PASSWORD)" db \
		pg_restore --no-owner --no-acl --clean --if-exists \
		-h "$$RDS_HOST" -p 5432 -U "$(RDS_USER)" -d "$(RDS_DB)" < $(DB_DUMP_FILE) || [ $$? -eq 1 ]; \
	echo "Verifying RDS table row counts:"; \
	docker compose -f app/docker-compose.yml exec -T -e PGPASSWORD="$(RDS_PASSWORD)" db \
		psql -h "$$RDS_HOST" -p 5432 -U "$(RDS_USER)" -d "$(RDS_DB)" \
		-c "SELECT relname AS table, n_live_tup AS rows FROM pg_stat_user_tables ORDER BY relname;"



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

ablation-B3f:
	$(ABLATION_CMD) --experiment B3f $(GENERATION_ARGS)

ablation-B3: ablation-B3a ablation-B3b ablation-B3c ablation-B3d ablation-B3e ablation-B3f

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
	./scripts/run_ablation_job.sh --experiment $(or $(EXPERIMENT),B2) $(if $(RUN_ID),--run-id $(RUN_ID)) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE)) $(if $(SEED),--seed $(SEED)) $(if $(BRANCH),--branch $(BRANCH)) $(if $(SHUTDOWN),--shutdown) $(if $(NO_SYNC),--no-sync) $(if $(NO_PUBLISH),--no-publish) $(if $(DB_SECRET_NAME),--secret-name $(DB_SECRET_NAME)) $(if $(KAGGLE_USER),--kaggle-user $(KAGGLE_USER)) $(if $(or $(KAGGLE_SECRET_NAME),$(KAGGLE_SECRET)),--kaggle-secret $(or $(KAGGLE_SECRET_NAME),$(KAGGLE_SECRET))) $(if $(or $(GIT_SSH_SECRET_NAME),$(GIT_SSH_SECRET)),--git-ssh-secret $(or $(GIT_SSH_SECRET_NAME),$(GIT_SSH_SECRET))) $(if $(LLM_SECRET_NAME),--llm-secret $(LLM_SECRET_NAME))


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
