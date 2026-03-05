CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -g
LDFLAGS = -lnuma -lpthread

TARGET  = mig_bench

all: $(TARGET)

$(TARGET): mig_bench.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ── Check prerequisites before running ──
check:
	@echo "=== Pre-flight checks ==="
	@test -f /sys/kernel/debug/mig_timing \
		|| (echo "ERROR: /sys/kernel/debug/mig_timing missing." \
		    && echo "  → Instrumented kernel not running, or debugfs not mounted." \
		    && echo "  → Mount debugfs: sudo mount -t debugfs none /sys/kernel/debug" \
		    && exit 1)
	@grep -q "numa=fake" /proc/cmdline \
		|| (echo "WARNING: numa=fake not detected in /proc/cmdline" \
		    && echo "  → Add 'numa=fake=2' to boot params and reboot")
	@echo "Timer frequency (arch_timer):" \
		&& grep -m1 "mult\|freq" /proc/timer_list 2>/dev/null || true
	@echo "NUMA topology:"
	@numactl --hardware 2>/dev/null || (echo "numactl not found, install: apt install numactl" && exit 1)
	@echo "THP status:"
	@cat /sys/kernel/mm/transparent_hugepage/enabled
	@echo "All checks passed."

run: $(TARGET) check
	@echo "=== Running benchmarks (requires root) ==="
	sudo ./$(TARGET)
	@echo "=== Running analysis ==="
	python3 analyze_timing.py timing_*.csv

# Enable THP in madvise mode (recommended for Benchmark B)
thp-madvise:
	echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# Disable NUMA balancing during benchmarks (reduces noise from auto-migration)
disable-numa-balancing:
	echo 0 | sudo tee /proc/sys/kernel/numa_balancing

# Reset debugfs buffer manually
reset-buffer:
	echo 1 | sudo tee /sys/kernel/debug/mig_timing

clean:
	rm -f $(TARGET) timing_*.csv *.png

.PHONY: all check run thp-madvise disable-numa-balancing reset-buffer clean