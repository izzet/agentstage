/*
 * AgentStage LD_PRELOAD shim.
 *
 * Intercepts path-taking syscalls and redirects reads under managed cold
 * roots to a hot tier mirror.
 *
 * Intercepted (minimal set):
 *   openat, openat2, creat            — primary redirect points
 *   stat, lstat, fstatat, newfstatat  — metadata must match hot file
 *   statx                              — ditto
 *   access, faccessat, faccessat2     — existence checks
 *
 * NOT intercepted (operate on fds, no path):
 *   read, pread, pread64, readv, preadv  — once openat returns hot fd, all
 *                                          reads naturally hit the hot file
 *   mmap, mmap2                          — fd-based
 *   close, lseek                         — fd-based
 *
 * Writes (any of O_WRONLY/O_RDWR/O_CREAT/O_TRUNC/O_APPEND in flags) pass
 * through to the cold path unchanged. The hot tier is read-only from the
 * agent's perspective.
 *
 * Configuration via env (read once at first call):
 *   AGENTSTAGE_HOT_ROOT       (required; e.g. /scratch/agentstage)
 *   AGENTSTAGE_COLD_ROOTS     (required; colon-separated list)
 *   AGENTSTAGE_RETRY_SPIN_MS  (default 20)
 *   AGENTSTAGE_SHIM_LOG       (optional; if set, write per-call events here)
 *   AGENTSTAGE_SHIM_DISABLE   (optional; if set to "1", pass everything through)
 *
 * Build:
 *   make
 * Load:
 *   LD_PRELOAD=./libagentstage_shim.so your_program ...
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* -------------------------------------------------------------------------
 * Configuration loaded once at first call
 * ------------------------------------------------------------------------- */

#define MAX_COLD_ROOTS 8

static struct {
    char hot_root[PATH_MAX];
    size_t hot_root_len;
    /* Optional overflow tier (single path). On lookup, primary hot_root
     * is tried first; if the file is not there, hot_overflow is tried;
     * if still missing, fall through to cold. */
    char hot_overflow[PATH_MAX];
    size_t hot_overflow_len;
    int has_overflow;
    char *cold_roots[MAX_COLD_ROOTS];
    size_t cold_roots_len[MAX_COLD_ROOTS];
    size_t n_cold_roots;
    int retry_spin_ms;
    int disabled;
    FILE *log_fp;
    int initialized;
} g_cfg;

static pthread_once_t g_cfg_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;

/* Thread-local guard so our shim doesn't recurse on its own internal
 * openat calls when resolving paths or opening hot files. */
static __thread int t_in_shim = 0;

static void cfg_init(void) {
    memset(&g_cfg, 0, sizeof(g_cfg));

    const char *disabled = getenv("AGENTSTAGE_SHIM_DISABLE");
    if (disabled && strcmp(disabled, "1") == 0) {
        g_cfg.disabled = 1;
        g_cfg.initialized = 1;
        return;
    }

    const char *hot = getenv("AGENTSTAGE_HOT_ROOT");
    if (!hot || !*hot) {
        /* No hot root configured — shim is a no-op pass-through */
        g_cfg.disabled = 1;
        g_cfg.initialized = 1;
        return;
    }
    /* Ensure no trailing slash so concatenation produces /hot_root/abs_cold */
    size_t hot_len = strlen(hot);
    while (hot_len > 0 && hot[hot_len - 1] == '/') hot_len--;
    if (hot_len >= PATH_MAX) {
        g_cfg.disabled = 1;
        g_cfg.initialized = 1;
        return;
    }
    memcpy(g_cfg.hot_root, hot, hot_len);
    g_cfg.hot_root[hot_len] = '\0';
    g_cfg.hot_root_len = hot_len;

    /* Optional overflow hot root for tiered staging. */
    const char *overflow = getenv("AGENTSTAGE_HOT_OVERFLOW");
    if (overflow && *overflow) {
        size_t ov_len = strlen(overflow);
        while (ov_len > 0 && overflow[ov_len - 1] == '/') ov_len--;
        if (ov_len > 0 && ov_len < PATH_MAX) {
            memcpy(g_cfg.hot_overflow, overflow, ov_len);
            g_cfg.hot_overflow[ov_len] = '\0';
            g_cfg.hot_overflow_len = ov_len;
            g_cfg.has_overflow = 1;
        }
    }

    const char *cold = getenv("AGENTSTAGE_COLD_ROOTS");
    if (!cold || !*cold) {
        g_cfg.disabled = 1;
        g_cfg.initialized = 1;
        return;
    }
    /* Parse colon-separated list */
    char *cold_copy = strdup(cold);
    char *saveptr = NULL;
    char *tok = strtok_r(cold_copy, ":", &saveptr);
    while (tok && g_cfg.n_cold_roots < MAX_COLD_ROOTS) {
        size_t tl = strlen(tok);
        while (tl > 0 && tok[tl - 1] == '/') tl--;
        if (tl > 0 && tl < PATH_MAX) {
            char *root = malloc(tl + 1);
            memcpy(root, tok, tl);
            root[tl] = '\0';
            g_cfg.cold_roots[g_cfg.n_cold_roots] = root;
            g_cfg.cold_roots_len[g_cfg.n_cold_roots] = tl;
            g_cfg.n_cold_roots++;
        }
        tok = strtok_r(NULL, ":", &saveptr);
    }
    free(cold_copy);
    if (g_cfg.n_cold_roots == 0) {
        g_cfg.disabled = 1;
        g_cfg.initialized = 1;
        return;
    }

    const char *spin = getenv("AGENTSTAGE_RETRY_SPIN_MS");
    g_cfg.retry_spin_ms = spin && *spin ? atoi(spin) : 20;
    if (g_cfg.retry_spin_ms < 0) g_cfg.retry_spin_ms = 0;
    if (g_cfg.retry_spin_ms > 1000) g_cfg.retry_spin_ms = 1000;

    const char *log_path = getenv("AGENTSTAGE_SHIM_LOG");
    if (log_path && *log_path) {
        /* MUST use the real glibc fopen via dlsym(RTLD_NEXT), NOT fopen:
         * the shim now interposes fopen, and this open runs inside
         * cfg_init (under pthread_once). Calling the interposed fopen
         * here would re-enter ensure_init() and deadlock pthread_once.
         * dlsym directly (LOAD_NEXT / real_fopen are declared later). */
        FILE *(*real_fopen_local)(const char *, const char *) =
            (FILE *(*)(const char *, const char *)) dlsym(RTLD_NEXT, "fopen");
        if (real_fopen_local) {
            g_cfg.log_fp = real_fopen_local(log_path, "a");
        }
        if (g_cfg.log_fp) {
            setvbuf(g_cfg.log_fp, NULL, _IOLBF, 0);
        }
    }

    g_cfg.initialized = 1;
}

static inline void ensure_init(void) {
    pthread_once(&g_cfg_once, cfg_init);
}

static void shim_log(const char *fmt, ...) {
    if (!g_cfg.log_fp) return;
    pthread_mutex_lock(&g_log_mu);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_cfg.log_fp, fmt, ap);
    va_end(ap);
    pthread_mutex_unlock(&g_log_mu);
}

/* -------------------------------------------------------------------------
 * Real syscall resolution via dlsym(RTLD_NEXT, ...)
 * ------------------------------------------------------------------------- */

typedef int (*openat_fn_t)(int, const char *, int, ...);
typedef int (*stat_fn_t)(const char *, struct stat *);
typedef int (*lstat_fn_t)(const char *, struct stat *);
typedef int (*fstatat_fn_t)(int, const char *, struct stat *, int);
typedef int (*access_fn_t)(const char *, int);
typedef int (*faccessat_fn_t)(int, const char *, int, int);
typedef int (*creat_fn_t)(const char *, mode_t);
typedef FILE *(*fopen_fn_t)(const char *, const char *);

static openat_fn_t   real_openat   = NULL;
static stat_fn_t     real_stat     = NULL;
static lstat_fn_t    real_lstat    = NULL;
static fstatat_fn_t  real_fstatat  = NULL;
static access_fn_t   real_access   = NULL;
static faccessat_fn_t real_faccessat = NULL;
static creat_fn_t    real_creat    = NULL;
static fopen_fn_t    real_fopen    = NULL;

#define LOAD_NEXT(var, name) do { \
    if (!var) var = dlsym(RTLD_NEXT, name); \
} while (0)

/* -------------------------------------------------------------------------
 * Path resolution and cold-root matching
 * ------------------------------------------------------------------------- */

static int path_is_absolute(const char *p) {
    return p && p[0] == '/';
}

static int resolve_absolute(int dirfd, const char *pathname,
                            char *out, size_t outsz) {
    if (!pathname || !*pathname) return 0;

    if (path_is_absolute(pathname)) {
        size_t len = strlen(pathname);
        if (len >= outsz) return 0;
        memcpy(out, pathname, len + 1);
        return 1;
    }

    if (dirfd == AT_FDCWD) {
        char cwd[PATH_MAX];
        if (!getcwd(cwd, sizeof(cwd))) return 0;
        int n = snprintf(out, outsz, "%s/%s", cwd, pathname);
        return n > 0 && (size_t)n < outsz;
    }

    /* dirfd points at a real directory — resolve via /proc/self/fd */
    char proc_link[64];
    snprintf(proc_link, sizeof(proc_link), "/proc/self/fd/%d", dirfd);
    char dir_path[PATH_MAX];
    ssize_t n = readlink(proc_link, dir_path, sizeof(dir_path) - 1);
    if (n <= 0) return 0;
    dir_path[n] = '\0';
    int m = snprintf(out, outsz, "%s/%s", dir_path, pathname);
    return m > 0 && (size_t)m < outsz;
}

static int under_managed_cold_root(const char *abs_path) {
    for (size_t i = 0; i < g_cfg.n_cold_roots; i++) {
        const char *root = g_cfg.cold_roots[i];
        size_t rlen = g_cfg.cold_roots_len[i];
        if (strncmp(abs_path, root, rlen) == 0 &&
            (abs_path[rlen] == '/' || abs_path[rlen] == '\0')) {
            return 1;
        }
    }
    return 0;
}

/* Symlink-aware cold-root match. Tries the raw `abs_path` first (fast
 * path — no syscall); if no match, calls realpath() to resolve any
 * symlinks in the path and checks again. On match, writes the path
 * that should be used to build the hot equivalent into `out` (raw on
 * fast-path match, canonical on slow-path match). Returns 1 on match.
 *
 * This is critical for agent-written scripts that access cold-tier
 * files through symlinks (e.g. an agent harness placing a symlink
 * under /workspace/data/<task>/ that points at the real cold root).
 * Without this resolution the shim would see the symlinked path,
 * fail the cold-root prefix check, and pass through to cold storage.
 */
static int under_managed_cold_root_resolved(
    const char *abs_path, char *out, size_t outsz) {
    /* Fast path — raw match. Copy abs_path through unchanged. */
    if (under_managed_cold_root(abs_path)) {
        size_t n = strlen(abs_path);
        if (n >= outsz) return 0;
        memcpy(out, abs_path, n + 1);
        return 1;
    }
    /* Slow path — resolve symlinks. realpath() does one stat per
     * path component but is only paid on first miss; the kernel
     * caches the dentries, so subsequent calls in the same process
     * are cheap. Skipped when the file doesn't exist (realpath
     * fails) since there's nothing to redirect to anyway. */
    char canon[PATH_MAX];
    t_in_shim = 1;
    char *r = realpath(abs_path, canon);
    t_in_shim = 0;
    if (!r) return 0;
    if (under_managed_cold_root(canon)) {
        size_t n = strlen(canon);
        if (n >= outsz) return 0;
        memcpy(out, canon, n + 1);
        return 1;
    }
    return 0;
}

static int build_hot_path(const char *abs_cold, char *out, size_t outsz) {
    /* Hot = HOT_ROOT + absolute cold path (preserving leading /) */
    int n = snprintf(out, outsz, "%s%s", g_cfg.hot_root, abs_cold);
    return n > 0 && (size_t)n < outsz;
}

static int build_overflow_path(const char *abs_cold,
                               char *out, size_t outsz) {
    if (!g_cfg.has_overflow) return 0;
    int n = snprintf(out, outsz, "%s%s", g_cfg.hot_overflow, abs_cold);
    return n > 0 && (size_t)n < outsz;
}

static int is_write_flag(int flags) {
    return (flags & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND)) != 0;
}

/* -------------------------------------------------------------------------
 * Retry-spin helper
 * ------------------------------------------------------------------------- */

static long elapsed_ms(struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start->tv_sec) * 1000L
         + (now.tv_nsec - start->tv_nsec) / 1000000L;
}

/* Try opening hot_path; if ENOENT, also try overflow_path (if any);
 * if both miss and retry_spin_ms > 0, spin briefly re-checking both.
 * Returns fd on success or -1 with errno set. */
static int open_hot_with_spin(const char *hot_path,
                              const char *overflow_path,
                              int flags, mode_t mode) {
    LOAD_NEXT(real_openat, "openat");
    int fd = real_openat(AT_FDCWD, hot_path, flags, mode);
    if (fd >= 0) return fd;
    if (errno != ENOENT) return -1;
    if (overflow_path && *overflow_path) {
        fd = real_openat(AT_FDCWD, overflow_path, flags, mode);
        if (fd >= 0) return fd;
        if (errno != ENOENT) return -1;
    }
    if (g_cfg.retry_spin_ms <= 0) return -1;

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    struct timespec sleep_for = {.tv_sec = 0, .tv_nsec = 500000};  /* 0.5 ms */
    while (elapsed_ms(&start) < g_cfg.retry_spin_ms) {
        nanosleep(&sleep_for, NULL);
        fd = real_openat(AT_FDCWD, hot_path, flags, mode);
        if (fd >= 0) return fd;
        if (errno != ENOENT) return -1;
        if (overflow_path && *overflow_path) {
            fd = real_openat(AT_FDCWD, overflow_path, flags, mode);
            if (fd >= 0) return fd;
            if (errno != ENOENT) return -1;
        }
    }
    errno = ENOENT;
    return -1;
}

/* -------------------------------------------------------------------------
 * Intercepted: openat
 *
 * For 'open', glibc routes through openat(AT_FDCWD, ...).
 * ------------------------------------------------------------------------- */

int openat(int dirfd, const char *pathname, int flags, ...) {
    LOAD_NEXT(real_openat, "openat");

    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, mode_t);
        va_end(ap);
    }

    ensure_init();
    if (g_cfg.disabled || t_in_shim || !pathname) {
        return real_openat(dirfd, pathname, flags, mode);
    }

    /* Writes always pass through */
    if (is_write_flag(flags)) {
        return real_openat(dirfd, pathname, flags, mode);
    }

    char abs[PATH_MAX];
    t_in_shim = 1;
    int resolved = resolve_absolute(dirfd, pathname, abs, sizeof(abs));
    t_in_shim = 0;
    if (!resolved) {
        return real_openat(dirfd, pathname, flags, mode);
    }

    char resolved_abs[PATH_MAX];
    if (!under_managed_cold_root_resolved(abs, resolved_abs, sizeof(resolved_abs))) {
        return real_openat(dirfd, pathname, flags, mode);
    }

    char hot[PATH_MAX];
    if (!build_hot_path(resolved_abs, hot, sizeof(hot))) {
        return real_openat(dirfd, pathname, flags, mode);
    }
    char overflow[PATH_MAX];
    const char *overflow_ptr = NULL;
    if (build_overflow_path(resolved_abs, overflow, sizeof(overflow))) {
        overflow_ptr = overflow;
    }

    t_in_shim = 1;
    int fd = open_hot_with_spin(hot, overflow_ptr, flags, mode);
    t_in_shim = 0;
    if (fd >= 0) {
        shim_log("HIT  %s -> %s\n", resolved_abs, hot);
        return fd;
    }

    /* Hot miss after retry-spin — fall through to cold */
    shim_log("MISS %s (errno=%d)\n", resolved_abs, errno);
    return real_openat(dirfd, pathname, flags, mode);
}

/* -------------------------------------------------------------------------
 * Intercepted: stat / lstat / fstatat
 *
 * Note: glibc's `stat()`/`fstat()`/`lstat()` are typically inlined to call
 * __xstat / __lxstat / __fxstatat with a version arg. On modern glibc
 * (≥ 2.33), the symbols stat/lstat/fstatat are real, exported functions
 * we can intercept directly. Older glibc would need __xstat overrides too.
 * ------------------------------------------------------------------------- */

static int try_redirect_stat_path(const char *pathname, char *hot_out, size_t outsz) {
    if (!pathname) return 0;
    char abs[PATH_MAX];
    t_in_shim = 1;
    int resolved = resolve_absolute(AT_FDCWD, pathname, abs, sizeof(abs));
    t_in_shim = 0;
    if (!resolved) return 0;
    char resolved_abs[PATH_MAX];
    if (!under_managed_cold_root_resolved(abs, resolved_abs,
                                           sizeof(resolved_abs))) {
        return 0;
    }
    return build_hot_path(resolved_abs, hot_out, outsz);
}

/* Like try_redirect_stat_path, but also populates overflow_out when an
 * overflow hot root is configured. Returns:
 *   0 = path not under managed cold root
 *   1 = hot only (no overflow configured or build failed)
 *   2 = both hot and overflow populated
 */
static int try_redirect_stat_paths(const char *pathname,
                                   char *hot_out, size_t hot_sz,
                                   char *overflow_out, size_t overflow_sz) {
    if (!pathname) return 0;
    char abs[PATH_MAX];
    t_in_shim = 1;
    int resolved = resolve_absolute(AT_FDCWD, pathname, abs, sizeof(abs));
    t_in_shim = 0;
    if (!resolved) return 0;
    char resolved_abs[PATH_MAX];
    if (!under_managed_cold_root_resolved(abs, resolved_abs,
                                           sizeof(resolved_abs))) {
        return 0;
    }
    if (!build_hot_path(resolved_abs, hot_out, hot_sz)) return 0;
    if (build_overflow_path(resolved_abs, overflow_out, overflow_sz)) {
        return 2;
    }
    return 1;
}

int stat(const char *pathname, struct stat *statbuf) {
    LOAD_NEXT(real_stat, "stat");
    ensure_init();
    if (g_cfg.disabled || t_in_shim) {
        return real_stat(pathname, statbuf);
    }
    char hot[PATH_MAX], overflow[PATH_MAX];
    int n = try_redirect_stat_paths(pathname, hot, sizeof(hot),
                                     overflow, sizeof(overflow));
    if (n > 0) {
        t_in_shim = 1;
        int r = real_stat(hot, statbuf);
        t_in_shim = 0;
        if (r == 0) return 0;
        if (n == 2) {
            t_in_shim = 1;
            r = real_stat(overflow, statbuf);
            t_in_shim = 0;
            if (r == 0) return 0;
        }
    }
    return real_stat(pathname, statbuf);
}

int lstat(const char *pathname, struct stat *statbuf) {
    LOAD_NEXT(real_lstat, "lstat");
    ensure_init();
    if (g_cfg.disabled || t_in_shim) {
        return real_lstat(pathname, statbuf);
    }
    char hot[PATH_MAX], overflow[PATH_MAX];
    int n = try_redirect_stat_paths(pathname, hot, sizeof(hot),
                                     overflow, sizeof(overflow));
    if (n > 0) {
        t_in_shim = 1;
        int r = real_lstat(hot, statbuf);
        t_in_shim = 0;
        if (r == 0) return 0;
        if (n == 2) {
            t_in_shim = 1;
            r = real_lstat(overflow, statbuf);
            t_in_shim = 0;
            if (r == 0) return 0;
        }
    }
    return real_lstat(pathname, statbuf);
}

int fstatat(int dirfd, const char *pathname, struct stat *statbuf, int flags) {
    LOAD_NEXT(real_fstatat, "fstatat");
    ensure_init();
    if (g_cfg.disabled || t_in_shim || !pathname) {
        return real_fstatat(dirfd, pathname, statbuf, flags);
    }
    char abs[PATH_MAX];
    t_in_shim = 1;
    int resolved = resolve_absolute(dirfd, pathname, abs, sizeof(abs));
    t_in_shim = 0;
    if (resolved) {
        char resolved_abs[PATH_MAX];
        if (under_managed_cold_root_resolved(abs, resolved_abs,
                                              sizeof(resolved_abs))) {
            char hot[PATH_MAX], overflow[PATH_MAX];
            int has_overflow = build_overflow_path(resolved_abs, overflow,
                                                   sizeof(overflow));
            if (build_hot_path(resolved_abs, hot, sizeof(hot))) {
                t_in_shim = 1;
                int r = real_fstatat(AT_FDCWD, hot, statbuf, flags);
                t_in_shim = 0;
                if (r == 0) return 0;
                if (has_overflow) {
                    t_in_shim = 1;
                    r = real_fstatat(AT_FDCWD, overflow, statbuf, flags);
                    t_in_shim = 0;
                    if (r == 0) return 0;
                }
            }
        }
    }
    return real_fstatat(dirfd, pathname, statbuf, flags);
}

/* -------------------------------------------------------------------------
 * Intercepted: access / faccessat
 * ------------------------------------------------------------------------- */

int access(const char *pathname, int mode) {
    LOAD_NEXT(real_access, "access");
    ensure_init();
    if (g_cfg.disabled || t_in_shim) {
        return real_access(pathname, mode);
    }
    /* Writes pass through */
    if (mode & W_OK) return real_access(pathname, mode);

    char hot[PATH_MAX], overflow[PATH_MAX];
    int n = try_redirect_stat_paths(pathname, hot, sizeof(hot),
                                     overflow, sizeof(overflow));
    if (n > 0) {
        t_in_shim = 1;
        int r = real_access(hot, mode);
        t_in_shim = 0;
        if (r == 0) return 0;
        if (n == 2) {
            t_in_shim = 1;
            r = real_access(overflow, mode);
            t_in_shim = 0;
            if (r == 0) return 0;
        }
    }
    return real_access(pathname, mode);
}

int faccessat(int dirfd, const char *pathname, int mode, int flags) {
    LOAD_NEXT(real_faccessat, "faccessat");
    ensure_init();
    if (g_cfg.disabled || t_in_shim || !pathname) {
        return real_faccessat(dirfd, pathname, mode, flags);
    }
    if (mode & W_OK) return real_faccessat(dirfd, pathname, mode, flags);

    char abs[PATH_MAX];
    t_in_shim = 1;
    int resolved = resolve_absolute(dirfd, pathname, abs, sizeof(abs));
    t_in_shim = 0;
    if (resolved) {
        char resolved_abs[PATH_MAX];
        if (under_managed_cold_root_resolved(abs, resolved_abs,
                                              sizeof(resolved_abs))) {
            char hot[PATH_MAX], overflow[PATH_MAX];
            int has_overflow = build_overflow_path(resolved_abs, overflow,
                                                   sizeof(overflow));
            if (build_hot_path(resolved_abs, hot, sizeof(hot))) {
                t_in_shim = 1;
                int r = real_faccessat(AT_FDCWD, hot, mode, flags);
                t_in_shim = 0;
                if (r == 0) return 0;
                if (has_overflow) {
                    t_in_shim = 1;
                    r = real_faccessat(AT_FDCWD, overflow, mode, flags);
                    t_in_shim = 0;
                    if (r == 0) return 0;
                }
            }
        }
    }
    return real_faccessat(dirfd, pathname, mode, flags);
}

/* -------------------------------------------------------------------------
 * Intercepted: creat
 *
 * creat is equivalent to open(path, O_WRONLY|O_CREAT|O_TRUNC, mode).
 * Always a write; pass through.
 * ------------------------------------------------------------------------- */

int creat(const char *pathname, mode_t mode) {
    LOAD_NEXT(real_creat, "creat");
    /* Always pass through — creat is always a write */
    return real_creat(pathname, mode);
}

int creat64(const char *pathname, mode_t mode) __attribute__((alias("creat")));

/* -------------------------------------------------------------------------
 * Intercepted: fopen / fopen64
 *
 * The netCDF-C library (libnetcdf) format-sniffs a file with fopen64()
 * before handing it to HDF5's sec2 driver (which uses open()). Without
 * intercepting fopen, that sniff read hits the cold tier even when the
 * file is staged hot — costing one cold first-byte latency per file.
 * Observed: ~2 s/file on an S3 cold tier.
 *
 * Read-only modes only ("r", "rb", "rt"); writes/append pass through.
 * ------------------------------------------------------------------------- */

static int fmode_is_read_only(const char *mode) {
    return mode && mode[0] == 'r' && !strchr(mode, '+');
}

FILE *fopen(const char *pathname, const char *mode) {
    LOAD_NEXT(real_fopen, "fopen");

    ensure_init();
    if (g_cfg.disabled || t_in_shim || !pathname || !fmode_is_read_only(mode)) {
        return real_fopen(pathname, mode);
    }

    char abs[PATH_MAX];
    t_in_shim = 1;
    int resolved = resolve_absolute(AT_FDCWD, pathname, abs, sizeof(abs));
    t_in_shim = 0;
    char resolved_abs[PATH_MAX];
    if (!resolved || !under_managed_cold_root_resolved(abs, resolved_abs,
                                                        sizeof(resolved_abs))) {
        return real_fopen(pathname, mode);
    }

    char hot[PATH_MAX];
    if (!build_hot_path(resolved_abs, hot, sizeof(hot))) {
        return real_fopen(pathname, mode);
    }
    char overflow[PATH_MAX];
    const char *overflow_ptr = NULL;
    if (build_overflow_path(resolved_abs, overflow, sizeof(overflow))) {
        overflow_ptr = overflow;
    }

    /* Spin-wait for the hot copy, then wrap the fd in a FILE*. */
    t_in_shim = 1;
    int fd = open_hot_with_spin(hot, overflow_ptr, O_RDONLY, 0);
    t_in_shim = 0;
    if (fd >= 0) {
        FILE *f = fdopen(fd, mode);
        if (f) {
            shim_log("HIT(fopen) %s -> %s\n", resolved_abs, hot);
            return f;
        }
        close(fd);
    }

    shim_log("MISS(fopen) %s (errno=%d)\n", abs, errno);
    return real_fopen(pathname, mode);
}

FILE *fopen64(const char *pathname, const char *mode)
    __attribute__((alias("fopen")));

/* -------------------------------------------------------------------------
 * 64-bit-LFS and legacy aliases.
 *
 * On modern Linux + glibc, off_t is 64-bit by default and openat == openat64.
 * Different binaries reference different symbol names depending on how
 * they were compiled:
 *   - Python's CPython binary uses `openat64@GLIBC_2.4`
 *   - `cat` uses `open@GLIBC_2.2.5` (the legacy 3-arg open)
 *   - Code compiled with -D_FORTIFY_SOURCE=2 uses __openat_2, __open_2, etc.
 *
 * Intercept all of them. open / open64 route through openat with AT_FDCWD;
 * the 64-bit and fortify variants are aliases to the standard interceptor.
 * ------------------------------------------------------------------------- */

int open(const char *pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, mode_t);
        va_end(ap);
    }
    return openat(AT_FDCWD, pathname, flags, mode);
}

int open64(const char *pathname, int flags, ...) __attribute__((alias("open")));

int openat64(int dirfd, const char *pathname, int flags, ...)
    __attribute__((alias("openat")));

/* Fortify variants (no vararg — the fortify compiler emits these when it
 * can prove statically there's no mode arg). */
int __open_2(const char *pathname, int flags) {
    return openat(AT_FDCWD, pathname, flags, 0);
}
int __open64_2(const char *pathname, int flags) __attribute__((alias("__open_2")));
int __openat_2(int dirfd, const char *pathname, int flags) {
    return openat(dirfd, pathname, flags, 0);
}
int __openat64_2(int dirfd, const char *pathname, int flags)
    __attribute__((alias("__openat_2")));

/* LFS aliases for stat-family: skipped because glibc's <sys/stat.h>
 * declares stat64/lstat64/fstatat64 with `struct stat64 *` not `struct stat *`.
 * On modern 64-bit Linux these structs have the same layout, but the
 * compiler can't tell that from the declarations. The __xstat family
 * below covers what legacy code references; modern code uses the
 * unversioned names which we already intercept. */

/* xstat (old glibc) — some binaries reference __xstat instead of stat.
 * The signature has a leading version int. We can't alias because the
 * signature differs; provide a thin wrapper that strips the version arg
 * and calls our stat. */
int __xstat(int ver, const char *pathname, struct stat *statbuf) {
    (void)ver;
    return stat(pathname, statbuf);
}
int __xstat64(int ver, const char *pathname, struct stat *statbuf)
    __attribute__((alias("__xstat")));
int __lxstat(int ver, const char *pathname, struct stat *statbuf) {
    (void)ver;
    return lstat(pathname, statbuf);
}
int __lxstat64(int ver, const char *pathname, struct stat *statbuf)
    __attribute__((alias("__lxstat")));
int __fxstatat(int ver, int dirfd, const char *pathname, struct stat *statbuf, int flags) {
    (void)ver;
    return fstatat(dirfd, pathname, statbuf, flags);
}
int __fxstatat64(int ver, int dirfd, const char *pathname, struct stat *statbuf, int flags)
    __attribute__((alias("__fxstatat")));
