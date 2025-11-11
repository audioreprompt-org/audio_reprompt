# Makefile para Audio Reprompt.

# Variables.
COMPOSE=docker compose

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

# Levantar servicios.
up:
	$(COMPOSE) up --build -d

# Detener servicios.
stop:
	$(COMPOSE) stop

# Detener y eliminar contenedores, redes y volúmenes.
down:
	$(COMPOSE) down -v

# Ver logs del frontend.
logs-frontend:
	$(COMPOSE) logs -f frontend

# Ver logs del backend.
logs-backend:
	$(COMPOSE) logs -f backend

# Ver imágenes y contenedores.
ps:
	$(COMPOSE) ps

images:
	$(COMPOSE) images

# Limpiar imágenes por ID.
rmi:
	docker rmi $(image_id)

# Reiniciar todo (detener + levantar).
restart: down up
