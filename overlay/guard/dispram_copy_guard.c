// SPDX-License-Identifier: Apache-2.0
// LD_PRELOAD trip-wire: attributes CUDA runtime memcpy/memset calls to the carveout KV range.
// Written from the public CUDA runtime API signatures. Interposes the cudart entry points that
// torch/vLLM call through the dynamic linker; calls resolved with cuGetProcAddress or a statically
// linked cudart are NOT seen (the smoke's positive control proves the torch path is seen).
//   dispram_guard_set(lo, hi)  range [lo, hi) to watch (the hook calls it after building the pool)
//   dispram_guard_hits()       calls whose src or dst range overlaps the watched range
//   dispram_guard_seen()       all interposed calls (proves interposition is live)
//   DISPRAM_GUARD_ABORT=1      abort() on the first hit (default: log once per call site kind, count)
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int cerr;  // cudaError_t
static _Atomic uintptr_t g_lo, g_hi;
static _Atomic unsigned long long g_hits, g_seen;

void dispram_guard_set(uintptr_t lo, uintptr_t hi) { atomic_store(&g_lo, lo); atomic_store(&g_hi, hi); }
unsigned long long dispram_guard_hits(void) { return atomic_load(&g_hits); }
unsigned long long dispram_guard_seen(void) { return atomic_load(&g_seen); }
void dispram_guard_reset(void) { atomic_store(&g_hits, 0); atomic_store(&g_seen, 0); }

int dispram_guard_overlaps(uintptr_t p, size_t n) {
  uintptr_t lo = atomic_load(&g_lo), hi = atomic_load(&g_hi);
  return n && lo < hi && p < hi && p + n > lo;
}

static void check(const char *fn, const void *dst, size_t dn, const void *src, size_t sn) {
  atomic_fetch_add(&g_seen, 1);
  if (!(dispram_guard_overlaps((uintptr_t)dst, dn) || dispram_guard_overlaps((uintptr_t)src, sn))) return;
  unsigned long long h = atomic_fetch_add(&g_hits, 1) + 1;
  fprintf(stderr, "glm-dispram-guard: HIT %s dst=%p src=%p bytes=%zu hits=%llu\n", fn, dst, src, dn ? dn : sn, h);
  const char *a = getenv("DISPRAM_GUARD_ABORT");
  if (a && a[0] == '1') abort();
}

#define NEXT(fn) static __typeof__(&fn) real; \
  if (!real) real = (__typeof__(&fn))dlsym(RTLD_NEXT, #fn); \
  if (!real) { fprintf(stderr, "glm-dispram-guard: no next %s\n", #fn); abort(); }

cerr cudaMemcpy(void *d, const void *s, size_t n, int k) {
  NEXT(cudaMemcpy);
  check("cudaMemcpy", d, n, s, n); return real(d, s, n, k);
}
cerr cudaMemcpyAsync(void *d, const void *s, size_t n, int k, void *st) {
  NEXT(cudaMemcpyAsync);
  check("cudaMemcpyAsync", d, n, s, n); return real(d, s, n, k, st);
}
cerr cudaMemcpy2DAsync(void *d, size_t dp, const void *s, size_t sp, size_t w, size_t h, int k, void *st) {
  NEXT(cudaMemcpy2DAsync);
  check("cudaMemcpy2DAsync", d, h ? dp * (h - 1) + w : 0, s, h ? sp * (h - 1) + w : 0);
  return real(d, dp, s, sp, w, h, k, st);
}
cerr cudaMemcpyPeerAsync(void *d, int dd, const void *s, int sd, size_t n, void *st) {
  NEXT(cudaMemcpyPeerAsync);
  check("cudaMemcpyPeerAsync", d, n, s, n); return real(d, dd, s, sd, n, st);
}
cerr cudaMemset(void *d, int v, size_t n) {
  NEXT(cudaMemset);
  check("cudaMemset", d, n, 0, 0); return real(d, v, n);
}
cerr cudaMemsetAsync(void *d, int v, size_t n, void *st) {
  NEXT(cudaMemsetAsync);
  check("cudaMemsetAsync", d, n, 0, 0); return real(d, v, n, st);
}
cerr cudaMemset2DAsync(void *d, size_t p, int v, size_t w, size_t h, void *st) {
  NEXT(cudaMemset2DAsync);
  check("cudaMemset2DAsync", d, h ? p * (h - 1) + w : 0, 0, 0); return real(d, p, v, w, h, st);
}
