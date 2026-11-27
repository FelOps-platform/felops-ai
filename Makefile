COMPOSE_PROFILES ?= cpu
export COMPOSE_PROFILES

up:
	docker compose up -d --build --wait

up-gpu: 
	COMPOSE_PROFILES=gpu docker compose up -d --build --wait
# host needs nvidia-container-toolkit to run the gpu profile

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

reset:
	docker compose down -v
