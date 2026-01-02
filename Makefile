# =============================================================================
# SOVEREIGN TRADING SYSTEM - MAKEFILE
# =============================================================================
# Common operations for Docker deployment

.PHONY: help build up down restart logs status clean test

# Default target
help:
	@echo "Sovereign Trading System - Docker Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build      Build Docker images"
	@echo "  up         Start all services"
	@echo "  down       Stop all services"
	@echo "  restart    Restart all services"
	@echo "  logs       View logs (all services)"
	@echo "  logs-gft   View GFT bot logs"
	@echo "  logs-5ers  View The5ers bot logs"
	@echo "  status     Show service status"
	@echo "  clean      Remove containers and volumes"
	@echo "  test       Run tests"
	@echo "  shell-gft  Shell into GFT container"
	@echo "  shell-5ers Shell into The5ers container"

# Build images
build:
	docker-compose -f docker/docker-compose.yml build

# Start services
up:
	docker-compose -f docker/docker-compose.yml up -d

# Start with full stack (including Postgres)
up-full:
	docker-compose -f docker/docker-compose.yml --profile full up -d

# Stop services
down:
	docker-compose -f docker/docker-compose.yml down

# Restart services
restart: down up

# View all logs
logs:
	docker-compose -f docker/docker-compose.yml logs -f

# View specific service logs
logs-gft:
	docker-compose -f docker/docker-compose.yml logs -f gft-bot

logs-5ers:
	docker-compose -f docker/docker-compose.yml logs -f the5ers-bot

logs-prometheus:
	docker-compose -f docker/docker-compose.yml logs -f prometheus

logs-grafana:
	docker-compose -f docker/docker-compose.yml logs -f grafana

# Show status
status:
	docker-compose -f docker/docker-compose.yml ps

# Clean up
clean:
	docker-compose -f docker/docker-compose.yml down -v --remove-orphans
	docker image prune -f

# Run tests
test:
	docker-compose -f docker/docker-compose.yml run --rm gft-bot python -m pytest -v

# Shell access
shell-gft:
	docker-compose -f docker/docker-compose.yml exec gft-bot /bin/bash

shell-5ers:
	docker-compose -f docker/docker-compose.yml exec the5ers-bot /bin/bash

# Backup volumes
backup:
	@echo "Backing up data volumes..."
	docker run --rm \
		-v sovereign-gft-data:/data \
		-v $(PWD)/backups:/backup \
		alpine tar czf /backup/gft-data-$(shell date +%Y%m%d).tar.gz /data
	docker run --rm \
		-v sovereign-the5ers-data:/data \
		-v $(PWD)/backups:/backup \
		alpine tar czf /backup/the5ers-data-$(shell date +%Y%m%d).tar.gz /data
	@echo "Backup complete!"

# Pull latest images
pull:
	docker-compose -f docker/docker-compose.yml pull

# Show resource usage
stats:
	docker stats --no-stream $(shell docker-compose -f docker/docker-compose.yml ps -q)

# Prometheus reload config
prometheus-reload:
	curl -X POST http://localhost:9090/-/reload

# Check config files
validate:
	docker-compose -f docker/docker-compose.yml config --quiet && echo "docker-compose.yml: OK"
	@echo "Validating Prometheus config..."
	docker run --rm -v $(PWD)/docker/prometheus:/etc/prometheus prom/prometheus promtool check config /etc/prometheus/prometheus.yml
