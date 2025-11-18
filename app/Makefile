# Makefile para Popularity Prediction Dashboard.

# Variables.
COMPOSE=docker compose

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
