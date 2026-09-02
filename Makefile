# Out-of-tree Triton Ascend extensions (triton-ext layout).
# Each extension under extensions/ builds independently.

BUILD_DIR ?= build

default: build

.PHONY: list
list:
	ci/list_extensions.py

.PHONY: build
build:
	@for EXT_DIR in $(shell ci/list_extensions.py path); do \
		$(MAKE) -C $$EXT_DIR build; \
	done

.PHONY: install
install:
	@for EXT_DIR in $(shell ci/list_extensions.py path); do \
		$(MAKE) -C $$EXT_DIR install; \
	done

.PHONY: test
test:
	pytest

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: clean-all
clean-all: clean
	rm -rf triton-* llvm-*
