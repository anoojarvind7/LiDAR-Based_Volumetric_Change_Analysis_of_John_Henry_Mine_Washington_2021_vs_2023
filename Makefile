# Full reproduction: `make all` (needs the conda env from environment.yml).
CONFIG ?= config/john_henry.yaml
MV = minevol --config $(CONFIG)

.PHONY: all test lint fetch process volumes change report smrf-check

all: fetch process volumes change report

test:
	pytest -q

lint:
	ruff check src tests

fetch:
	$(MV) fetch

process:
	$(MV) process

volumes:
	$(MV) volumes
	$(MV) resolution

change:
	$(MV) change

report:
	$(MV) report

# Independent ground classification (SMRF) to test sensitivity to the vendor's classes.
smrf-check:
	$(MV) process --ground smrf
	$(MV) volumes --ground smrf
