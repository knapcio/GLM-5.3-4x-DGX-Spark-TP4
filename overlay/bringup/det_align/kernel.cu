// SPDX-License-Identifier: Apache-2.0
// Original deterministic counting-sort implementation; no imported vLLM kernel.
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <stdint.h>

namespace glm_det {
constexpr int EMAX = 256, THREADS = 256, WARPS = 8, TILE = 1024;
using Scan = cub::BlockScan<int, THREADS>;

template<class Id>
__device__ __forceinline__ int route(const Id* ids, const int* map,
                                    int i, int n, int e, bool premap) {
  if (i >= n) return -1;
  // Stock get_local_expert_id converts even int64 inputs to signed int32.
  const int expert = int(ids[i]);
  if (expert < 0 || expert >= e) return -1;
  return premap && map ? map[expert] : expert;
}
// One writer per (warp, expert); match_any gives each token's ascending lane rank.
__device__ __forceinline__ int peers(int expert, int* hist) {
  const int t = threadIdx.x, lane = t & 31;
  const unsigned same = __match_any_sync(0xffffffffu, expert);
  if (expert >= 0 && lane == __ffs(same) - 1)
    hist[(t / 32) * EMAX + expert] = __popc(same);
  return __popc(same & ((1u << lane) - 1u));
}

// Write labels, padding, and unused tail only. Valid slots are scatter-owned.
__device__ void layout(int* ids, int* experts, int* post, const int* map,
                       bool premap, int n, int e, int b, int capacity,
                       int count, int start, int rounded, int total) {
  const int t = threadIdx.x;
  if (t < e) {
    const int mapped = map && !premap ? map[t] : t;
    for (int i = start / b; i < (start + rounded) / b; ++i) experts[i] = mapped;
    for (int i = start + count; i < start + rounded; ++i) ids[i] = n;
  }
  for (int i = total + t; i < capacity; i += THREADS) ids[i] = n;
  const int tail = map && !premap ? map[e - 1] : -1;
  for (int i = total / b + t; i < (capacity + b - 1) / b; i += THREADS)
    experts[i] = tail;
  if (t == 0) *post = total;
}

template<class Id>
__global__ void decode(const Id* input, const int* map, int* ids, int* experts,
                       int* post, int n, int e, int b, int capacity, bool premap) {
  __shared__ int hist[WARPS * EMAX];
  __shared__ int starts[EMAX];
  __shared__ Scan::TempStorage scan;
  const int t = threadIdx.x;
  for (int i = t; i < WARPS * EMAX; i += THREADS) hist[i] = 0;
  __syncthreads();
  const int expert = route(input, map, t, n, e, premap);
  int rank = peers(expert, hist);
  __syncthreads();
  int count = 0;
  if (t < e) for (int w = 0; w < WARPS; ++w) count += hist[w * EMAX + t];
  const int rounded = (count + b - 1) / b * b;
  int start, total;
  Scan(scan).ExclusiveSum(rounded, start, total);
  starts[t] = start;
  layout(ids, experts, post, map, premap, n, e, b, capacity, count, start, rounded, total);
  __syncthreads();
  if (expert >= 0) {
    for (int w = 0; w < t / 32; ++w) rank += hist[w * EMAX + expert];
    ids[starts[expert] + rank] = t;
  }
}

template<class Id>
__device__ int rank_tile(const Id* input, const int* map, int* ranks,
                         int n, int e, int begin, bool premap) {
  __shared__ int hist[WARPS * EMAX];
  __shared__ int prior[EMAX];
  const int t = threadIdx.x;
  prior[t] = 0;
  // Contiguous rounds preserve flat-index order and reduce tile-scan traffic.
  for (int part = 0; part < TILE && begin + part < n; part += THREADS) {
    for (int j = t; j < WARPS * EMAX; j += THREADS) hist[j] = 0;
    __syncthreads();
    const int i = begin + part + t;
    const int expert = route(input, map, i, n, e, premap);
    int rank = peers(expert, hist);
    __syncthreads();
    int count = 0;
    if (t < e) for (int w = 0; w < WARPS; ++w) count += hist[w * EMAX + t];
    if (i < n) {
      if (expert >= 0) {
        rank += prior[expert];
        for (int w = 0; w < t / 32; ++w) rank += hist[w * EMAX + expert];
      }
      ranks[i] = rank;
    }
    __syncthreads(); // all prior reads complete before expert lanes update it
    prior[t] += count;
    __syncthreads(); // all hist reads complete before the next round clears it
  }
  return prior[t];
}

template<class Id>
__global__ void histogram(const Id* input, const int* map, int* counts, int* ranks,
                          int n, int e, bool premap) {
  const int count = rank_tile(input, map, ranks, n, e, blockIdx.x * TILE, premap);
  if (threadIdx.x < e) counts[blockIdx.x * e + threadIdx.x] = count;
}

// M64 (512 entries) and other compact batches avoid two extra kernel launches.
template<class Id>
__global__ void compact(const Id* input, const int* map, int* ids, int* experts,
                        int* post, int* ranks, int n, int e, int b, int capacity, bool premap) {
  __shared__ int starts[EMAX];
  __shared__ Scan::TempStorage scan;
  const int t = threadIdx.x;
  const int count = rank_tile(input, map, ranks, n, e, 0, premap);
  const int rounded = (count + b - 1) / b * b;
  int start, total;
  Scan(scan).ExclusiveSum(rounded, start, total);
  starts[t] = start;
  layout(ids, experts, post, map, premap, n, e, b, capacity, count, start, rounded, total);
  __syncthreads();
  for (int i = t; i < n; i += THREADS) {
    const int expert = route(input, map, i, n, e, premap);
    if (expert >= 0) ids[starts[expert] + ranks[i]] = i;
  }
}

__global__ void prefix(int* counts, int* ids, int* experts, int* post, const int* map,
                       int n, int e, int b, int capacity, int tiles, bool premap) {
  __shared__ Scan::TempStorage scan;
  const int t = threadIdx.x;
  int count = 0;
  if (t < e) {
    // In-place exclusive tile scan; coalesced across expert lanes.
    for (int tile = 0; tile < tiles; ++tile) {
      const int v = counts[tile * e + t];
      counts[tile * e + t] = count;
      count += v;
    }
  }
  const int rounded = (count + b - 1) / b * b;
  int start, total;
  Scan(scan).ExclusiveSum(rounded, start, total);
  if (t < e) for (int tile = 0; tile < tiles; ++tile) counts[tile * e + t] += start;
  layout(ids, experts, post, map, premap, n, e, b, capacity, count, start, rounded, total);
}

template<class Id>
__global__ void scatter(const Id* input, const int* map, const int* offsets,
                        const int* ranks, int* ids, int n, int e, bool premap) {
  const int i = blockIdx.x * THREADS + threadIdx.x;
  const int expert = route(input, map, i, n, e, premap);
  if (expert >= 0) ids[offsets[(i / TILE) * e + expert] + ranks[i]] = i;
}

template<class Id>
cudaError_t launch(const void* input, const int* map, int* ids, int* experts, int* post,
            int* counts, int* ranks, int n, int e, int b, int capacity,
            bool premap, cudaStream_t stream) {
  const auto* in = static_cast<const Id*>(input);
  if (n <= 128) {
    decode<<<1, THREADS, 0, stream>>>(in, map, ids, experts, post, n, e, b, capacity, premap);
  } else if (n <= TILE) {
    compact<<<1, THREADS, 0, stream>>>(in, map, ids, experts, post, ranks, n, e, b, capacity, premap);
  } else {
    const int tiles = (n + TILE - 1) / TILE;
    histogram<<<tiles, THREADS, 0, stream>>>(in, map, counts, ranks, n, e, premap);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    prefix<<<1, THREADS, 0, stream>>>(counts, ids, experts, post, map, n, e, b, capacity, tiles, premap);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    scatter<<<(n + THREADS - 1) / THREADS, THREADS, 0, stream>>>(in, map, counts, ranks, ids, n, e, premap);
  }
  return cudaGetLastError(); // launch status only; never synchronize
}
} // namespace glm_det

// Benchmark-only boundary polling, on the same host thread/runtime as launches.
// Production align remains asynchronous and uses only the launch status below.
extern "C" int glm_det_last_error() { return int(cudaGetLastError()); }

extern "C" int glm_det_align(const void* input, const int* map, int* ids,
    int* experts, int* post, int* counts, int* ranks, int n, int e, int b,
    int capacity, int id_bytes, int premap, void* stream) {
  if (id_bytes == 8)
    return int(glm_det::launch<int64_t>(input, map, ids, experts, post, counts, ranks,
        n, e, b, capacity, premap, static_cast<cudaStream_t>(stream)));
  else
    return int(glm_det::launch<int32_t>(input, map, ids, experts, post, counts, ranks,
        n, e, b, capacity, premap, static_cast<cudaStream_t>(stream)));
}
