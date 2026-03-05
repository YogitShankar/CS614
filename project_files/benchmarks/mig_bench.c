/*
 * mig_bench.c — Userspace Migration Benchmark Driver
 * Target: ARM64, Linux 6.1.4, emulated NUMA (numa=fake=2), 4KB pages
 *
 * Drives three benchmark scenarios:
 *   A: Base 4KB page migration    — measures per-stage costs at small granularity
 *   B: 2MB THP migration          — copy-dominated, reveals DMA offload benefit
 *   C: Shared page migration      — rmap walk scaling vs sharing degree
 *
 * Build:
 *   gcc -O2 -Wall -o mig_bench mig_bench.c -lnuma -lpthread
 *
 * Run (requires instrumented kernel + debugfs mounted):
 *   sudo mount -t debugfs none /sys/kernel/debug   # if not already mounted
 *   sudo ./mig_bench
 *
 * Notes for numa=fake=2:
 *   Both NUMA nodes share the same physical DIMM(s). Migration is a
 *   metadata + copy operation with no real cross-DIMM latency penalty.
 *   Copy times will be ~2-3x faster than real cross-node NUMA.c
 *   The stage *ratios* and structural breakdown are still valid for
 *   identifying which stages dominate.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <numaif.h>
#include <numa.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>

/* ─────────────────────────────────────────────────────────────────── */
/* Configuration                                                       */
/* ─────────────────────────────────────────────────────────────────── */

#define BASE_PAGE_SIZE      4096UL
#define HUGE_PAGE_SIZE      (2UL * 1024 * 1024)   /* 2MB THP, valid for 4KB base */
#define DEBUGFS_PATH        "/sys/kernel/debug/mig_timing"
#define THP_ENABLED_PATH    "/sys/kernel/mm/transparent_hugepage/enabled"

/* Number of pages per scenario — tune based on available RAM */
#define N_BASE_PAGES_SMALL  512
#define N_BASE_PAGES_LARGE  2048
#define N_THP_SMALL         32
#define N_THP_LARGE         128
#define N_SHARED_PAGES      64

/* ─────────────────────────────────────────────────────────────────── */
/* Utility                                                             */
/* ─────────────────────────────────────────────────────────────────── */

static void die(const char *msg)
{
    fprintf(stderr, "FATAL: %s: %s\n", msg, strerror(errno));
    exit(EXIT_FAILURE);
}

static void warn(const char *msg)
{
    fprintf(stderr, "WARN: %s: %s\n", msg, strerror(errno));
}

/*
 * pin_to_cpu — bind this thread to a specific CPU core.
 * On ARM64 emulated NUMA, all CPUs may be on "node 0" from the
 * scheduler's view, but numa_fake splits memory. Pinning ensures
 * consistent instruction timing and avoids migration noise.
 */
static void pin_to_cpu(int cpu)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0)
        die("sched_setaffinity");
}

/* Reset the kernel timing buffer via a write to debugfs */
static void timing_reset(void)
{
    int fd = open(DEBUGFS_PATH, O_WRONLY);
    if (fd < 0)
        die("open debugfs for reset — is the instrumented kernel running?");
    if (write(fd, "1", 1) < 0)
        die("write to debugfs reset");
    close(fd);
    /* Brief pause to ensure the kernel's reset completes before
     * we trigger new migrations. */
    usleep(20000);
}

/*
 * timing_read — copy the kernel's CSV data to an output file.
 * Uses a read loop to handle the seq_file interface correctly:
 * the kernel may return data in multiple chunks.
 */
static void timing_read(const char *output_filename)
{
    char buf[8192];
    ssize_t n;
    int rfd, wfd;

    rfd = open(DEBUGFS_PATH, O_RDONLY);
    if (rfd < 0)
        die("open debugfs for read");

    wfd = open(output_filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (wfd < 0)
        die("open output CSV for write");

    while ((n = read(rfd, buf, sizeof(buf))) > 0) {
        if (write(wfd, buf, n) != n)
            die("write CSV data");
    }

    close(rfd);
    close(wfd);
}

/*
 * timing_read_filtered — like timing_read but writes only rows where
 * page_order == target_order.  Extracts real THP records (order=9)
 * from a mixed buffer that also contains split 4KB records (order=0).
 *
 * page_order is the 6th column (index 5, 0-based) in the CSV:
 *   seq,cpu,pid,src_pfn,dst_pfn,page_order,...
 */
static void timing_read_filtered(const char *output_filename, int target_order)
{
    FILE *rfd, *wfd;
    char  line[1024];
    int   header_done = 0;
    int   kept = 0, skipped = 0;

    rfd = fopen(DEBUGFS_PATH, "r");
    if (!rfd) { warn("open debugfs for filtered read"); return; }

    wfd = fopen(output_filename, "w");
    if (!wfd) { fclose(rfd); warn("open filtered CSV for write"); return; }

    while (fgets(line, sizeof(line), rfd)) {
        if (!header_done) {
            fputs(line, wfd);
            header_done = 1;
            continue;
        }
        /* page_order is the 6th comma-separated field (index 5) */
        char  tmp[1024];
        int   field = 0, order = -1;
        char *tok;
        strncpy(tmp, line, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';
        tok = strtok(tmp, ",");
        while (tok) {
            if (field == 5) { order = atoi(tok); break; }
            field++;
            tok = strtok(NULL, ",");
        }
        if (order == target_order) { fputs(line, wfd); kept++; }
        else skipped++;
    }
    fclose(rfd);
    fclose(wfd);
    printf("  THP filter: %d order-%d records kept, %d split/other skipped\n",
           kept, target_order, skipped);
}

/*
 * read_vmstat_thp — print thp_migration_* counters from /proc/vmstat.
 * Counters are cumulative since boot — useful for seeing the split rate.
 */
static void read_vmstat_thp(void)
{
    FILE *f = fopen("/proc/vmstat", "r");
    char  line[256];
    if (!f) { warn("open /proc/vmstat"); return; }
    printf("  THP vmstat counters (cumulative since boot):\n");
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "thp_migration", 13) == 0)
            printf("    %s", line);
    }
    fclose(f);
}

/*
 * get_page_node — query which NUMA node a virtual address is on.
 * Uses move_pages(2) with no destination (status query mode).
 * Returns node ID, or -1 on error.
 */
static int get_page_node(void *addr)
{
    void *pages[1] = { (void *)((uintptr_t)addr & ~(BASE_PAGE_SIZE - 1)) };
    int status[1]  = { -1 };

    if (move_pages(0, 1, pages, NULL, status, 0) < 0)
        return -1;
    return status[0];
}

/*
 * verify_placement — check that all pages are on the expected node.
 * Returns number of misplaced pages (0 = all correct).
 */
static int verify_placement(void **pages, int count, int expected_node)
{
    int misplaced = 0, i;
    for (i = 0; i < count; i++) {
        int node = get_page_node(pages[i]);
        if (node != expected_node) {
            misplaced++;
            if (misplaced <= 3)
                fprintf(stderr, "  [verify] page %d on node %d, expected %d\n",
                        i, node, expected_node);
        }
    }
    return misplaced;
}

/*
 * elapsed_ms — compute milliseconds between two CLOCK_MONOTONIC timestamps.
 */
static double elapsed_ms(struct timespec *start, struct timespec *end)
{
    return (end->tv_sec  - start->tv_sec)  * 1000.0
         + (end->tv_nsec - start->tv_nsec) / 1.0e6;
}

/* Print a section header */
static void print_header(const char *title, int count,
                          int src_node, int dst_node)
{
    printf("\n");
    printf("┌──────────────────────────────────────────────────────┐\n");
    printf("│  %-52s│\n", title);
    printf("│  Count: %-5d   Node %d → Node %d                      │\n",
           count, src_node, dst_node);
    printf("└──────────────────────────────────────────────────────┘\n");
}

/* ─────────────────────────────────────────────────────────────────── */
/* Benchmark A: 4KB base page migration                               */
/* ─────────────────────────────────────────────────────────────────── */

/*
 * benchmark_base_pages — allocate 'count' individual 4KB pages on
 * src_node, write to each to force physical allocation and rmap setup,
 * then migrate them all to dst_node in a single move_pages(2) call.
 *
 * Each page produces one timing record in the kernel buffer.
 * For N pages we get N independent samples of the full 5-stage cost.
 *
 * Why write to each page?
 *   On ARM64 (and all Linux), a page is not physically allocated until
 *   first write (demand paging). An unwritten page has no physical frame,
 *   no PFN, no rmap entry — it would migrate trivially. We must write
 *   to ensure the kernel sets up the PTE and the anon_vma rmap entry,
 *   giving try_to_unmap() a real mapping to walk.
 */
static void benchmark_base_pages(int count, int src_node, int dst_node,
                                  const char *output_file)
{
    void   **pages  = NULL;
    int    *nodes   = NULL;
    int    *status  = NULL;
    int     i, misplaced, failed;
    long    ret;
    struct  timespec t0, t1;
    double  wall_ms, throughput_mbs;

    print_header("Benchmark A: 4KB Base Page Migration", count, src_node, dst_node);

    pages  = calloc(count, sizeof(void *));
    nodes  = calloc(count, sizeof(int));
    status = calloc(count, sizeof(int));
    if (!pages || !nodes || !status)
        die("calloc benchmark arrays");

    /* Allocate on source node using libnuma */
    printf("  Allocating %d × 4KB pages on node %d ...\n", count, src_node);
    for (i = 0; i < count; i++) {
        pages[i] = numa_alloc_onnode(BASE_PAGE_SIZE, src_node);
        if (!pages[i])
            die("numa_alloc_onnode 4KB");

        /*
         * Write a distinct pattern per page. Using (0xAA | (i & 0xF))
         * gives 16 distinct patterns — helps identify data integrity
         * issues if the copy stage produces wrong results.
         */
        memset(pages[i], (int)(0xAA | (i & 0xF)), BASE_PAGE_SIZE);
    }

    misplaced = verify_placement(pages, count, src_node);
    if (misplaced)
        printf("  WARNING: %d/%d pages not on node %d after allocation\n",
               misplaced, count, src_node);
    else
        printf("  Placement verified: all %d pages on node %d\n", count, src_node);

    for (i = 0; i < count; i++)
        nodes[i] = dst_node;

    /* Reset kernel timing buffer immediately before migration */
    timing_reset();

    printf("  Migrating to node %d ...\n", dst_node);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    ret = move_pages(0, count, pages, nodes, status, MPOL_MF_MOVE);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    wall_ms = elapsed_ms(&t0, &t1);

    if (ret < 0)
        warn("move_pages returned error");

    failed = 0;
    for (i = 0; i < count; i++) {
        if (status[i] < 0) {
            if (++failed <= 5)
                fprintf(stderr, "  page[%d] status=%d (%s)\n",
                        i, status[i], strerror(-status[i]));
        }
    }

    int succeeded = count - failed;
    throughput_mbs = (double)succeeded * BASE_PAGE_SIZE
                     / (wall_ms / 1000.0) / (1024.0 * 1024.0);

    printf("  Succeeded: %d/%d   Failed: %d\n", succeeded, count, failed);
    printf("  Wall time: %.1f ms   Throughput: %.0f MB/s\n",
           wall_ms, throughput_mbs);

    timing_read(output_file);
    printf("  Timing data → %s\n", output_file);

    for (i = 0; i < count; i++)
        numa_free(pages[i], BASE_PAGE_SIZE);
    free(pages);
    free(nodes);
    free(status);
}

/* ─────────────────────────────────────────────────────────────────── */
/* Benchmark B: 2MB THP migration                                     */
/* ─────────────────────────────────────────────────────────────────── */

/*
 * check_thp_enabled — print THP status and warn if not available.
 * With numa=fake=2, THPs should work normally since all memory is
 * from the same physical DIMM.
 */
static void check_thp_enabled(void)
{
    int fd;
    char buf[256];
    ssize_t n;

    fd = open(THP_ENABLED_PATH, O_RDONLY);
    if (fd < 0) {
        warn("cannot read THP status");
        return;
    }
    n = read(fd, buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        /* Remove trailing newline */
        char *nl = strchr(buf, '\n');
        if (nl) *nl = '\0';
        printf("  THP status: %s\n", buf);
        if (!strstr(buf, "[always]") && !strstr(buf, "[madvise]"))
            printf("  NOTE: THP may not be active. Run:\n"
                   "    echo madvise > %s\n", THP_ENABLED_PATH);
    }
    close(fd);
}

/*
 * verify_thp — check /proc/self/smaps to count AnonHugePages for addr.
 * Returns 1 if the mapping is backed by a THP, 0 otherwise.
 */
static int verify_thp(void *addr, size_t size)
{
    FILE *f;
    char line[256];
    unsigned long map_start, map_end;
    int in_mapping = 0;
    unsigned long anon_huge = 0;
    int found = 0;

    f = fopen("/proc/self/smaps", "r");
    if (!f)
        return 0;

    while (fgets(line, sizeof(line), f)) {
        /* Parse VMA start lines: "aabbccdd-eeffgghh ..." */
        if (sscanf(line, "%lx-%lx", &map_start, &map_end) == 2) {
            in_mapping = (map_start == (unsigned long)addr);
            if (in_mapping)
                found = 1;
        }
        if (in_mapping && sscanf(line, "AnonHugePages: %lu", &anon_huge) == 1) {
            break;
        }
    }
    fclose(f);

    return found && (anon_huge >= (size / 1024));
}

/*
 * benchmark_huge_pages — allocate 'count' 2MB-aligned anonymous regions,
 * advise the kernel to use THP backing (MADV_HUGEPAGE), bind to src_node,
 * touch all 2MB to force allocation as a single PMD-level THP, then migrate.
 *
 * Each THP produces one timing record with page_order=9.
 * The copy stage will dominate (~260µs per THP with intra-DIMM bandwidth).
 *
 * ARM64 THP note: with 4KB base pages, THPs are order-9 (512 × 4KB = 2MB).
 * folio_copy() iterates copy_highpage() 512 times, each calling __pi_memcpy.
 */
static void benchmark_huge_pages(int count, int src_node, int dst_node,
                                  const char *output_file)
{
    void   **pages  = NULL;
    int    *nodes   = NULL;
    int    *status  = NULL;
    int     i, thp_confirmed, failed;
    long    ret;
    struct  timespec t0, t1;
    double  wall_ms, throughput_mbs;

    print_header("Benchmark B: 2MB THP Migration", count, src_node, dst_node);

    check_thp_enabled();

    pages  = calloc(count, sizeof(void *));
    nodes  = calloc(count, sizeof(int));
    status = calloc(count, sizeof(int));
    if (!pages || !nodes || !status)
        die("calloc benchmark arrays");

    printf("  Allocating %d × 2MB THPs on node %d ...\n", count, src_node);

    for (i = 0; i < count; i++) {
        unsigned long nodemask;
        
        /* 1. Allocate double the size to ensure we can find a 2MB aligned block */
        void *raw_mem = mmap(NULL, HUGE_PAGE_SIZE * 2, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw_mem == MAP_FAILED)
            die("mmap for THP region");

        /* 2. Calculate the perfectly aligned 2MB boundary */
        uintptr_t raw_addr = (uintptr_t)raw_mem;
        uintptr_t aligned_addr = (raw_addr + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);
        pages[i] = (void *)aligned_addr;

        /* 3. Bind the virtual memory area to the source node FIRST */
        nodemask = 1UL << src_node;
        if (mbind(pages[i], HUGE_PAGE_SIZE, MPOL_BIND,
                  &nodemask, sizeof(nodemask) * 8,
                  MPOL_MF_MOVE | MPOL_MF_STRICT) < 0)
            warn("mbind THP to src_node");

        /* 4. Advise the kernel to back this aligned region with THPs */
        if (madvise(pages[i], HUGE_PAGE_SIZE, MADV_HUGEPAGE) < 0)
            warn("madvise MADV_HUGEPAGE");

        /* 5. FAULT IT IN: This physically allocates the 2MB frame on the source node */
        memset(pages[i], (int)(0xBB | (i & 0xF)), HUGE_PAGE_SIZE);
        
        /* Lock it in RAM so it doesn't swap out before migration */
        mlock(pages[i], HUGE_PAGE_SIZE);
    }

    /* Verify THP backing — at least check a few */
    thp_confirmed = 0;
    for (i = 0; i < count && i < 4; i++) {
        if (verify_thp(pages[i], HUGE_PAGE_SIZE))
            thp_confirmed++;
    }
    if (thp_confirmed < (count < 4 ? count : 4))
        printf("  WARNING: Not all mappings confirmed as THP — "
               "some may have fallen back to 4KB pages.\n"
               "  Check: grep -A5 'VmFlags.*ht' /proc/self/smaps\n");
    else
        printf("  THP backing confirmed for sampled pages\n");

    for (i = 0; i < count; i++)
        nodes[i] = dst_node;

    timing_reset();

    printf("  Migrating to node %d ...\n", dst_node);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    ret = move_pages(0, count, pages, nodes, status, MPOL_MF_MOVE);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    wall_ms = elapsed_ms(&t0, &t1);

    if (ret < 0)
        warn("move_pages THP returned error");

    failed = 0;
    for (i = 0; i < count; i++) {
        if (status[i] < 0) {
            if (++failed <= 5)
                fprintf(stderr, "  THP[%d] status=%d (%s)\n",
                        i, status[i], strerror(-status[i]));
        }
    }

    int succeeded = count - failed;
    throughput_mbs = (double)succeeded * HUGE_PAGE_SIZE
                     / (wall_ms / 1000.0) / (1024.0 * 1024.0);

    printf("  Succeeded: %d/%d   Failed: %d\n", succeeded, count, failed);
    printf("  Wall time: %.1f ms   Throughput: %.0f MB/s\n",
           wall_ms, throughput_mbs);

    timing_read(output_file);
    printf("  Timing data → %s\n", output_file);

    /* Extract order-9 (whole THP) records into a separate clean CSV.
     * Renames "timing_2mb_32.csv" → "timing_thp_real_32.csv" etc. */
    {
        char thp_file[256];
        const char *pfx = "timing_2mb_";
        if (strncmp(output_file, pfx, strlen(pfx)) == 0)
            snprintf(thp_file, sizeof(thp_file), "timing_thp_real_%s",
                     output_file + strlen(pfx));
        else
            snprintf(thp_file, sizeof(thp_file), "timing_thp_real_%s", output_file);
        timing_read_filtered(thp_file, 9);
        printf("  Real THP records → %s\n", thp_file);
    }
    read_vmstat_thp();

    for (i = 0; i < count; i++)
        munmap(pages[i], HUGE_PAGE_SIZE);
    free(pages);
    free(nodes);
    free(status);
}

/* ─────────────────────────────────────────────────────────────────── */
/* Benchmark C: Shared page migration (rmap walk scaling)             */
/* ─────────────────────────────────────────────────────────────────── */

/*
 * benchmark_shared_pages — measures how unmap cost scales with the
 * number of processes mapping each page.
 *
 * Setup: parent creates MAP_SHARED|MAP_ANONYMOUS pages on src_node,
 * writes to each. Then forks (sharing_degree - 1) children. Each child
 * reads every page (establishing a PTE in its page table), then sleeps.
 * Parent migrates all pages. Each page now has 'sharing_degree' PTEs
 * in 'sharing_degree' different mm_structs.
 *
 * try_to_unmap() must walk all sharing_degree PTEs, issuing one TLBI
 * per PTE per page on ARM64. Cost should scale linearly with sharing_degree.
 * This is the workload that batch TLBI (Optimization 2) would accelerate.
 *
 * Data interpretation:
 *   At sharing_degree=1: baseline unmap cost
 *   At sharing_degree=N: N × baseline + TLBI broadcast overhead
 *   The difference tells us the TLBI broadcast overhead per shootdown.
 */
static void benchmark_shared_pages(int num_pages, int sharing_degree,
                                    int src_node, int dst_node,
                                    const char *output_file)
{
    void   **pages   = NULL;
    int    *nodes    = NULL;
    int    *status   = NULL;
    pid_t  *children = NULL;
    int     i, c, failed;
    long    ret;
    struct  timespec t0, t1;
    double  wall_ms;

    printf("\n  [Shared] degree=%-3d  pages=%-3d  node %d→%d\n",
           sharing_degree, num_pages, src_node, dst_node);

    pages    = calloc(num_pages, sizeof(void *));
    nodes    = calloc(num_pages, sizeof(int));
    status   = calloc(num_pages, sizeof(int));
    children = calloc(sharing_degree, sizeof(pid_t));
    if (!pages || !nodes || !status || !children)
        die("calloc shared benchmark arrays");

    /* Parent creates shared anonymous mappings on src_node */
    for (i = 0; i < num_pages; i++) {
        unsigned long nodemask;

        pages[i] = mmap(NULL, BASE_PAGE_SIZE,
                        PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (pages[i] == MAP_FAILED)
            die("mmap shared anonymous");

        nodemask = 1UL << src_node;
        if (mbind(pages[i], BASE_PAGE_SIZE, MPOL_BIND,
                  &nodemask, sizeof(nodemask) * 8,
                  MPOL_MF_MOVE | MPOL_MF_STRICT) < 0)
            warn("mbind shared page");

        /* Write to force physical allocation */
        ((volatile char *)pages[i])[0] = (char)('A' + (i % 26));
    }

    /*
     * Fork children. Each child:
     *   1. Reads all pages (establishes PTE in its own page table)
     *   2. Calls pause() — sleeps until parent sends SIGTERM
     *
     * The read in each child is essential: without it, the child's
     * PTE is not yet faulted in, and try_to_unmap won't find it.
     */
    for (c = 0; c < sharing_degree - 1; c++) {
        children[c] = fork();
        if (children[c] < 0)
            die("fork");

        if (children[c] == 0) {
            /* Child process */
            volatile char sink = 0;
            for (i = 0; i < num_pages; i++)
                sink ^= ((volatile char *)pages[i])[0];
            (void)sink;
            pause();  /* Sleep until SIGTERM */
            _exit(0);
        }
    }

    /* Give children time to fault in all their pages */
    usleep(150000);

    for (i = 0; i < num_pages; i++)
        nodes[i] = dst_node;

    timing_reset();

    clock_gettime(CLOCK_MONOTONIC, &t0);

    /*
     * MPOL_MF_MOVE_ALL: migrate pages even if shared across processes.
     * Without this flag, move_pages(2) refuses to migrate pages with
     * mapcount > 1. We need this for the sharing degree > 1 cases.
     */
    ret = move_pages(0, num_pages, pages, nodes, status, MPOL_MF_MOVE_ALL);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    wall_ms = elapsed_ms(&t0, &t1);

    if (ret < 0)
        warn("move_pages shared returned error");

    failed = 0;
    for (i = 0; i < num_pages; i++)
        if (status[i] < 0) failed++;

    printf("  Succeeded: %d/%d   Wall: %.1f ms\n",
           num_pages - failed, num_pages, wall_ms);

    timing_read(output_file);
    printf("  Timing data → %s\n", output_file);

    /* Terminate children */
    for (c = 0; c < sharing_degree - 1; c++) {
        kill(children[c], SIGTERM);
        waitpid(children[c], NULL, 0);
    }

    for (i = 0; i < num_pages; i++)
        munmap(pages[i], BASE_PAGE_SIZE);
    free(pages);
    free(nodes);
    free(status);
    free(children);
}

/* ─────────────────────────────────────────────────────────────────── */
/* Benchmark D: Migration Downtime                                     */
/* ─────────────────────────────────────────────────────────────────── */

/*
 * ARM64 Generic Timer helpers.
 *
 * CNTVCT_EL0 increments at CNTFRQ_EL0 Hz (typically 24 MHz on most
 * ARM64 SoCs → 41.67 ns/tick). We use ISB to serialise the read
 * against surrounding memory instructions — equivalent to x86 LFENCE
 * before RDTSC. Without ISB, the OoO pipeline can advance the timer
 * read past the memory access we're timing.
 */
static double dt_ns_per_tick;

static inline uint64_t arm64_read_timer(void)
{
    uint64_t v;
    __asm__ volatile ("isb" ::: "memory");
    __asm__ volatile ("mrs %0, cntvct_el0" : "=r" (v));
    return v;
}

static void dt_calibrate_timer(void)
{
    uint64_t t0, t1;
    struct timespec ts0, ts1;
    unsigned long freq;

    __asm__ volatile ("mrs %0, cntfrq_el0" : "=r" (freq));

    clock_gettime(CLOCK_MONOTONIC, &ts0);
    t0 = arm64_read_timer();
    usleep(100000);
    t1 = arm64_read_timer();
    clock_gettime(CLOCK_MONOTONIC, &ts1);

    uint64_t ticks   = t1 - t0;
    uint64_t real_ns = (uint64_t)(ts1.tv_sec  - ts0.tv_sec)  * 1000000000ULL
                     + (uint64_t)(ts1.tv_nsec - ts0.tv_nsec);

    dt_ns_per_tick = (double)real_ns / (double)ticks;
    printf("  [downtime] ARM64 timer: %lu Hz, %.2f ns/tick\n",
           freq, dt_ns_per_tick);
}

/*
 * Shared state between access thread and migration thread.
 *
 * phase:
 *   0 = setup (access thread spinning at barrier)
 *   1 = baseline recording (no migration yet)
 *   2 = migration in progress
 *   3 = post-migration recording
 *   4 = done — both threads exit
 */
#define DT_MAX_SAMPLES 400000

struct dt_sample {
    uint64_t t_before;
    uint64_t t_after;
};

struct dt_shared {
    _Atomic int          phase;
    volatile char       *target_page;
    struct dt_sample    *samples;
    _Atomic int          sample_count;
    int                  access_cpu;
    int                  migrate_cpu;
    int                  src_node;
    int                  dst_node;
    const char          *csv_out;
};

/*
 * Access thread: continuously reads one byte from target_page,
 * timing each access with ISB-serialised CNTVCT_EL0 reads.
 *
 * A migration stall appears as a large latency spike: the LDR triggers
 * a Data Abort → kernel fault path → migration_entry_wait() → thread
 * sleeps until the migrator calls folio_unlock() after set_pte_at().
 *
 * The volatile qualifier prevents the compiler from:
 *   - Hoisting the load out of the loop into a register
 *   - Eliminating it as dead code
 *   - Reordering it relative to the ISB-bracketed timer reads
 */
static void *dt_access_thread(void *arg)
{
    struct dt_shared *sh = arg;
    volatile char sink;
    int n = 0;

    pin_to_cpu(sh->access_cpu);

    /* Spin-wait for migration thread to signal phase 1 */
    while (atomic_load(&sh->phase) == 0)
        __asm__ volatile ("yield");

    while (atomic_load(&sh->phase) < 4 && n < DT_MAX_SAMPLES) {
        uint64_t tb, ta;

        /* ISB before: ensure timer read doesn't execute before prior stores */
        __asm__ volatile ("isb" ::: "memory");
        tb = arm64_read_timer();

        sink = sh->target_page[0];   /* The timed access */

        /* ISB after: ensure LDR retires before the timer read executes */
        __asm__ volatile ("isb" ::: "memory");
        ta = arm64_read_timer();

        sh->samples[n].t_before = tb;
        sh->samples[n].t_after  = ta;
        n++;
    }

    (void)sink;
    atomic_store(&sh->sample_count, n);
    return NULL;
}

/*
 * Migration thread: waits for access thread to warm up, then
 * calls move_pages() to migrate the target page. Repeats round-trip.
 * Uses /sys/kernel/debug/mig_downtime to reset + read kernel records.
 */
static void *dt_migrate_thread(void *arg)
{
    struct dt_shared *sh = arg;
    void *page_ptr = (void *)sh->target_page;
    int dest, status;

    pin_to_cpu(sh->migrate_cpu);

    /* Reset kernel downtime buffer */
    {
        int fd = open("/sys/kernel/debug/mig_downtime", O_WRONLY);
        if (fd >= 0) { (void)write(fd, "r", 1); close(fd); }
    }

    /* Phase 1: give access thread 50 ms of baseline samples */
    atomic_store(&sh->phase, 1);
    usleep(50000);

    /* ── Migration 1: src → dst ── */
    dest = sh->dst_node;
    atomic_store(&sh->phase, 2);
    if (move_pages(0, 1, &page_ptr, &dest, &status, MPOL_MF_MOVE) < 0)
        warn("move_pages (fwd)");

    /* Phase 3: 50 ms post-migration baseline */
    atomic_store(&sh->phase, 3);
    usleep(50000);

    /* ── Migration 2: dst → src (measures a second stall event) ── */
    dest = sh->src_node;
    atomic_store(&sh->phase, 2);
    if (move_pages(0, 1, &page_ptr, &dest, &status, MPOL_MF_MOVE) < 0)
        warn("move_pages (rev)");

    atomic_store(&sh->phase, 3);
    usleep(50000);
    atomic_store(&sh->phase, 4);

    return NULL;
}

static void dt_analyze_and_save(struct dt_shared *sh)
{
    int total     = atomic_load(&sh->sample_count);
    const double  STALL_THRESH = 5000.0;   /* 5 µs — well above any TLB miss */
    int    n_normal = 0, n_stall = 0;
    double sum_normal = 0.0, max_stall = 0.0;
    int    max_idx = -1;
    int    i;

    for (i = 0; i < total; i++) {
        double lat = (double)(sh->samples[i].t_after - sh->samples[i].t_before)
                   * dt_ns_per_tick;
        if (lat > STALL_THRESH) {
            n_stall++;
            if (lat > max_stall) { max_stall = lat; max_idx = i; }
            (void)max_idx;  /* recorded for future stall-profile use */
        } else {
            n_normal++;
            sum_normal += lat;
        }
    }

    double mean_normal = (n_normal > 0) ? sum_normal / n_normal : 0.0;

    printf("\n  ── Downtime Results ──────────────────────────\n");
    printf("  Total samples  : %d\n",   total);
    printf("  Normal accesses: %d  (mean %.1f ns)\n", n_normal, mean_normal);
    printf("  Stall events   : %d  (threshold %.0f ns)\n", n_stall, STALL_THRESH);
    printf("  Max stall      : %.1f ns  =  %.2f µs\n",
           max_stall, max_stall / 1000.0);
    printf("\n  ► Downtime ≈ %.2f µs\n", max_stall / 1000.0);

    if (n_stall > 0) {
        /*
         * Estimated decomposition on ARM64:
         *   ~2.5 µs = fixed fault-handling overhead
         *             (Data Abort entry + kernel path + context switch)
         *   remainder = migration_entry_wait() time
         *             = copy_remaining + remap
         *
         * DMA offload implication: DMA frees the CPU during the copy,
         * but the faulting thread stalls for the same total time.
         * Downtime is not reduced; only migrator CPU utilisation improves.
         */
        double overhead_ns = 2500.0;
        double wait_ns     = max_stall - overhead_ns;

        printf("\n  Estimated breakdown:\n");
        printf("    Fixed fault overhead : ~%.1f µs\n", overhead_ns / 1000.0);
        printf("    Migration wait       : ~%.1f µs\n", wait_ns / 1000.0);
        printf("\n  DMA implication: faulting thread still stalls %.2f µs;\n"
               "  DMA benefits are off the application's critical path.\n",
               max_stall / 1000.0);
    }

    /* Write userspace CSV */
    FILE *f = fopen(sh->csv_out, "w");
    if (f) {
        fprintf(f, "sample_idx,t_before,t_after,ticks,latency_ns,is_stall\n");
        for (i = 0; i < total; i++) {
            uint64_t ticks = sh->samples[i].t_after - sh->samples[i].t_before;
            double   lat   = (double)ticks * dt_ns_per_tick;
            fprintf(f, "%d,%lu,%lu,%lu,%.1f,%d\n",
                    i,
                    (unsigned long)sh->samples[i].t_before,
                    (unsigned long)sh->samples[i].t_after,
                    (unsigned long)ticks,
                    lat,
                    (lat > STALL_THRESH) ? 1 : 0);
        }
        fclose(f);
        printf("  Userspace CSV   → %s\n", sh->csv_out);
    }

    /* Read + save kernel-side downtime records */
    {
        int fd = open("/sys/kernel/debug/mig_downtime", O_RDONLY);
        if (fd >= 0) {
            static char kbuf[65536];
            ssize_t n = read(fd, kbuf, sizeof(kbuf) - 1);
            close(fd);
            if (n > 0) {
                kbuf[n] = '\0';
                char kfile[256];
                snprintf(kfile, sizeof(kfile), "kernel_%s", sh->csv_out);
                FILE *kf = fopen(kfile, "w");
                if (kf) { fwrite(kbuf, 1, n, kf); fclose(kf); }
                printf("  Kernel CSV      → %s\n", kfile);

                /* Print first few kernel records */
                char *line = strtok(kbuf, "\n");
                int lc = 0;
                printf("  Kernel records (first 4):\n");
                while (line && lc < 5) {
                    printf("    %s\n", line);
                    line = strtok(NULL, "\n");
                    lc++;
                }
            } else {
                printf("  NOTE: mig_downtime debugfs empty — "
                       "kernel patch applied?\n");
            }
        } else {
            printf("  NOTE: cannot open /sys/kernel/debug/mig_downtime\n");
        }
    }
}

/*
 * benchmark_downtime — allocate one 4KB page, spawn two threads:
 *   access thread: continuously reads the page, timing each access
 *   migrate thread: migrates the page back and forth
 *
 * The access thread will hit migration PTEs during the migrations,
 * causing measurable stall spikes in the latency series.
 *
 * access_cpu and migrate_cpu should be on different physical cores to
 * avoid SMT scheduling artefacts. For same-node vs cross-node
 * comparisons, vary which NUMA node each CPU belongs to.
 */
static void benchmark_downtime(int access_cpu, int migrate_cpu,
                                int src_node, int dst_node,
                                const char *label, const char *csv_out)
{
    struct dt_shared sh = {};
    struct dt_sample *samples;
    pthread_t atid, mtid;
    unsigned long nodemask;

    printf("\n┌──────────────────────────────────────────────────────┐\n");
    printf("│  Benchmark D: %s\n", label);
    printf("│  access=CPU%-3d  migrate=CPU%-3d  Node %d → Node %d    │\n",
           access_cpu, migrate_cpu, src_node, dst_node);
    printf("└──────────────────────────────────────────────────────┘\n");

    samples = calloc(DT_MAX_SAMPLES, sizeof(*samples));
    if (!samples) die("calloc dt samples");

    /* Allocate and fault in the target page on src_node */
    void *pg = mmap(NULL, BASE_PAGE_SIZE, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pg == MAP_FAILED) die("mmap target page");

    nodemask = 1UL << src_node;
    if (mbind(pg, BASE_PAGE_SIZE, MPOL_BIND,
              &nodemask, 64, MPOL_MF_MOVE | MPOL_MF_STRICT) < 0)
        warn("mbind target page (non-fatal)");

    memset(pg, 0xBB, BASE_PAGE_SIZE);   /* fault page in physically */

    int node = get_page_node(pg);
    printf("  Page on Node %d (expected %d)\n", node, src_node);

    sh.phase        = 0;
    sh.sample_count = 0;
    sh.target_page  = (volatile char *)pg;
    sh.samples      = samples;
    sh.access_cpu   = access_cpu;
    sh.migrate_cpu  = migrate_cpu;
    sh.src_node     = src_node;
    sh.dst_node     = dst_node;
    sh.csv_out      = csv_out;

    pthread_create(&atid, NULL, dt_access_thread,  &sh);
    usleep(5000);   /* let access thread reach its spin-wait */
    pthread_create(&mtid, NULL, dt_migrate_thread, &sh);

    pthread_join(mtid, NULL);
    pthread_join(atid, NULL);

    dt_analyze_and_save(&sh);

    munmap(pg, BASE_PAGE_SIZE);
    free(samples);
}

/* ─────────────────────────────────────────────────────────────────── */
/* Pre-flight checks                                                   */
/* ─────────────────────────────────────────────────────────────────── */

static void preflight_checks(int *src_node, int *dst_node)
{
    int max_node;

    /* Check NUMA availability */
    if (numa_available() < 0) {
        fprintf(stderr,
            "ERROR: NUMA not available.\n"
            "  Did you boot with numa=fake=2 ?\n"
            "  Add 'numa=fake=2' to GRUB_CMDLINE_LINUX in /etc/default/grub\n"
            "  then run: sudo update-grub && sudo reboot\n");
        exit(EXIT_FAILURE);
    }

    max_node = numa_max_node();
    if (max_node < 1) {
        fprintf(stderr,
            "ERROR: Need at least 2 NUMA nodes, found %d.\n"
            "  Boot parameter: numa=fake=2\n", max_node + 1);
        exit(EXIT_FAILURE);
    }

    *src_node = 0;
    *dst_node = 1;

    /* Check debugfs interface */
    if (access(DEBUGFS_PATH, R_OK | W_OK) < 0) {
        fprintf(stderr,
            "ERROR: Cannot access %s\n"
            "  Is the instrumented kernel running?\n"
            "  Is debugfs mounted? Try:\n"
            "    sudo mount -t debugfs none /sys/kernel/debug\n",
            DEBUGFS_PATH);
        exit(EXIT_FAILURE);
    }

    /* Check we're running as root (needed for move_pages on other pids,
     * and for debugfs write access) */
    if (getuid() != 0)
        fprintf(stderr,
            "WARNING: Not running as root. Some operations may fail.\n"
            "  Run: sudo ./mig_bench\n");

    printf("  NUMA nodes: %d (nodes 0..%d)\n", max_node + 1, max_node);
    printf("  Source node: %d   Destination node: %d\n", *src_node, *dst_node);
    printf("  Debugfs interface: %s ✓\n", DEBUGFS_PATH);

    /* Print timer frequency — relevant for ARM64 clock resolution */
    {
        FILE *f = fopen("/proc/timer_list", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strstr(line, "mult") || strstr(line, "freq") ||
                    strstr(line, "arch_sys_counter")) {
                    printf("  %s", line);
                    break;
                }
            }
            fclose(f);
        }
    }
}

/* ─────────────────────────────────────────────────────────────────── */
/* Main                                                                */
/* ─────────────────────────────────────────────────────────────────── */

int main(void)
{
    int src_node, dst_node;

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   Migration Function Decomposition Benchmark         ║\n");
    printf("║   ARM64 / Linux 6.1.4 / 4KB pages / numa=fake=2     ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    preflight_checks(&src_node, &dst_node);

    /*
     * Pin to CPU 0. This ensures:
     *   - Consistent ktime_get_ns() readings (same arch_timer instance)
     *   - move_pages() kernel path runs on this CPU, so raw_smp_processor_id()
     *     in mig_timing_store() is deterministic
     */
    pin_to_cpu(0);
    printf("  Pinned to CPU 0\n\n");

    /* ── Benchmark A: 4KB pages ── */
    benchmark_base_pages(N_BASE_PAGES_SMALL, src_node, dst_node,
                         "timing_4kb_512.csv");
    benchmark_base_pages(N_BASE_PAGES_LARGE, src_node, dst_node,
                         "timing_4kb_2048.csv");

    /* ── Benchmark B: 2MB THPs ── */
    benchmark_huge_pages(N_THP_SMALL, src_node, dst_node,
                         "timing_2mb_32.csv");
    benchmark_huge_pages(N_THP_LARGE, src_node, dst_node,
                         "timing_2mb_128.csv");

    /* ── Benchmark C: Shared pages — sweep sharing degree ── */
    printf("\n┌──────────────────────────────────────────────────────┐\n");
    printf("│  Benchmark C: Shared Page Migration (rmap scaling)  │\n");
    printf("└──────────────────────────────────────────────────────┘\n");

    {
        int degrees[] = {1, 2, 4, 8, 16, 32, 64};
        int nd = (int)(sizeof(degrees) / sizeof(degrees[0]));
        int d;

        for (d = 0; d < nd; d++) {
            char filename[64];
            snprintf(filename, sizeof(filename),
                     "timing_shared_deg%03d.csv", degrees[d]);
            benchmark_shared_pages(N_SHARED_PAGES, degrees[d],
                                   src_node, dst_node, filename);
        }
    }

    /* ── Benchmark D: Downtime ── */
    dt_calibrate_timer();

    /*
     * Find a CPU on Node 1 for the cross-node test.
     * We read /sys/devices/system/node/node1/cpulist directly —
     * avoids the libnuma cpumask API which varies across versions.
     */
    int node1_cpu = -1;
    {
        FILE *f = fopen("/sys/devices/system/node/node1/cpulist", "r");
        if (f) {
            int c;
            if (fscanf(f, "%d", &c) == 1)
                node1_cpu = c;
            fclose(f);
        }
    }

    printf("\n┌──────────────────────────────────────────────────────┐\n");
    printf("│  Benchmark D: Migration Downtime                    │\n");
    printf("└──────────────────────────────────────────────────────┘\n");

    /*
     * Test 1: Both CPUs on Node 0 (same-node).
     * Access CPU=1, Migrate CPU=0. Measures baseline stall when
     * both threads are near each other in the NUMA topology.
     */
    benchmark_downtime(1, 0, src_node, dst_node,
                       "Same-node, 4KB page",
                       "downtime_samenode.csv");

    /*
     * Test 2: Access thread on Node 1 CPU, migrator on Node 0.
     * Cross-node wakeup latency adds ~1-2 µs to stall overhead.
     * Skipped if no Node 1 CPU is found.
     */
    if (node1_cpu > 1) {
        benchmark_downtime(node1_cpu, 0, src_node, dst_node,
                           "Cross-node, 4KB page",
                           "downtime_crossnode.csv");
    } else {
        printf("  [skip] cross-node test (node1_cpu=%d)\n", node1_cpu);
    }

    printf("\n══════════════════════════════════════════════════════\n");
    printf("  All benchmarks complete.\n");
    printf("  Run: python3 analyze_timing.py timing_*.csv\n");
    printf("       python3 analyze_downtime.py downtime_*.csv\n");
    printf("══════════════════════════════════════════════════════\n\n");

    return 0;
}