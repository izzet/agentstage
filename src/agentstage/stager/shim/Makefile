# LD_PRELOAD shim build.
#
# Produces libagentstage_shim.so. Load via:
#   LD_PRELOAD=$DFTRACER_SO:$(realpath libagentstage_shim.so) ...
#
# DFTracer (if used) MUST come first so it logs the agent's intent
# (cold path) before our shim redirects.

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -fPIC -D_GNU_SOURCE
LDFLAGS ?= -shared -ldl -lpthread

SHIM_SRC = agentstage_shim.c
SHIM_LIB = libagentstage_shim.so

.PHONY: all clean test

all: $(SHIM_LIB)

$(SHIM_LIB): $(SHIM_SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(SHIM_LIB)

# Build C tests too. Tests live under tests/shim/ at the repo root.
TEST_DIR = ../../../../tests/shim
TEST_SRCS = $(wildcard $(TEST_DIR)/test_*.c)
TEST_BINS = $(TEST_SRCS:.c=.bin)

test: $(SHIM_LIB) $(TEST_BINS)
	@echo "running shim C tests..."
	@for t in $(TEST_BINS); do \
		echo "--- $$t ---"; \
		$$t || exit 1; \
	done
	@echo "all shim tests passed"

$(TEST_DIR)/%.bin: $(TEST_DIR)/%.c $(SHIM_LIB)
	$(CC) -O2 -Wall -D_GNU_SOURCE -o $@ $< -lpthread

clean-tests:
	rm -f $(TEST_BINS)
