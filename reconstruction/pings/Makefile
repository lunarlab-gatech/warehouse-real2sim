.PHONY: build run

SHELL = /bin/sh

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

GPUS ?= 0
MACHINE ?= default

build:
	COMPOSE_DOCKER_CLI_BUILD=1 docker --context $(MACHINE) compose build pings --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

run:
	docker --context $(MACHINE) compose run -e CUDA_VISIBLE_DEVICES=$(GPUS) pings bash

# type exit to exit the bash shell

