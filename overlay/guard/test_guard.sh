#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# CPU test of the LD_PRELOAD copy guard against a fake cudart (colima, gcc). Prints GUARD-TEST PASS/FAIL.
set -euo pipefail
G=$(cd "$(dirname "$0")" && pwd); T=$(mktemp -d)
cat > "$T/fake.c" <<'EOF'
#include <stddef.h>
int calls;
int cudaMemcpyAsync(void *d, const void *s, size_t n, int k, void *st) { calls++; return 0; }
int cudaMemsetAsync(void *d, int v, size_t n, void *st) { calls++; return 0; }
int cudaMemcpy2DAsync(void *d, size_t dp, const void *s, size_t sp, size_t w, size_t h, int k, void *st) { calls++; return 0; }
EOF
cat > "$T/main.c" <<'EOF'
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
extern int calls;
int cudaMemcpyAsync(void *, const void *, size_t, int, void *);
int cudaMemsetAsync(void *, int, size_t, void *);
int cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, int, void *);
int main(void) {
  void (*set)(uintptr_t, uintptr_t) = dlsym(RTLD_DEFAULT, "dispram_guard_set");
  unsigned long long (*hits)(void) = dlsym(RTLD_DEFAULT, "dispram_guard_hits");
  unsigned long long (*seen)(void) = dlsym(RTLD_DEFAULT, "dispram_guard_seen");
  if (!set || !hits || !seen) { puts("GUARD-TEST FAIL not preloaded"); return 1; }
  set(0x10000, 0x20000);
  cudaMemcpyAsync((void *)0x1000, (void *)0x2000, 0x100, 3, 0);      /* outside */
  cudaMemcpyAsync((void *)0x1000, (void *)0x1ff00, 0x200, 3, 0);     /* src straddles lo..hi end */
  cudaMemcpyAsync((void *)0xff00, (void *)0x2000, 0x101, 3, 0);      /* dst ends 1 byte inside */
  cudaMemcpyAsync((void *)0xff00, (void *)0x2000, 0x100, 3, 0);      /* dst ends exactly at lo */
  cudaMemsetAsync((void *)0x18000, 0, 16, 0);                        /* inside */
  cudaMemcpy2DAsync((void *)0x8000, 0x1000, (void *)0x3000, 0x10, 0x10, 9, 3, 0); /* rows reach 0x10010 */
  int ok = hits() == 4 && seen() == 6 && calls == 6;
  printf("GUARD-TEST %s hits=%llu seen=%llu forwarded=%d\n", ok ? "PASS" : "FAIL", hits(), seen(), calls);
  return !ok;
}
EOF
gcc -O2 -Wall -Werror -shared -fPIC "$G/dispram_copy_guard.c" -o "$T/libguard.so" -ldl
gcc -O2 -shared -fPIC "$T/fake.c" -o "$T/libfakecudart.so"
gcc -O2 "$T/main.c" -o "$T/main" -L"$T" -lfakecudart -ldl -Wl,-rpath,"$T"
LD_PRELOAD="$T/libguard.so" "$T/main" 2>"$T/err.txt"
grep -c 'glm-dispram-guard: HIT' "$T/err.txt" | sed 's/^/stderr HIT lines: /'
