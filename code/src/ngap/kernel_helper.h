#ifndef NGAP_KERNEL_HELPER_H_
#define NGAP_KERNEL_HELPER_H_

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace ngap_nb {

__device__ __forceinline__ static void namedBarrierSync(int name,
                                                        int numThreads) {
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}

__device__ static void addToBuffer(volatile int *data, uint length, int iter,
                                   int *buffer, int *buffer_idx,
                                   uint buffer_start_pos, uint *buffer_end_pos,
                                   int buffer_size) {
  if (length <= 0)
    return;
  uint idx = atomicAdd((uint *)buffer_end_pos, length);
  assert((idx - buffer_start_pos + buffer_size) % buffer_size + length <
         buffer_size - 100);
  // while (!((idx - *d_filter_buffer_start + data_buffer_size_per_block) %
  //          data_buffer_size_per_block) <
  //        (data_buffer_size_per_block - 100))
  //   ;
  idx = idx % buffer_size;
  // printf("idx=%d length=%d buffer_size=%d iter=%d\n", idx, length,
  // buffer_size,
  //        iter);
  if (idx + length <= buffer_size) {
    memcpy((int *)(buffer + idx), (int *)data, sizeof(int) * length);
    thrust::device_ptr<int> dev_ptr =
        thrust::device_pointer_cast((int *)(buffer_idx + idx));
    thrust::fill(dev_ptr, (dev_ptr + length), iter);
  } else {
    int cutoff = idx + length - buffer_size;
    memcpy((int *)(buffer + idx), (int *)data, sizeof(int) * (length - cutoff));
    memcpy((int *)buffer, (int *)(data + (length - cutoff)),
           sizeof(int) * cutoff);
    thrust::device_ptr<int> dev_ptr =
        thrust::device_pointer_cast((int *)buffer_idx);
    thrust::fill(dev_ptr + idx, (dev_ptr + idx + (length - cutoff)), iter);
    thrust::fill(dev_ptr, (dev_ptr + cutoff), iter);
  }
}

__device__ static __forceinline__ void
addToBufferSimple(int data, int iter, int *buffer, int *buffer_idx,
                  uint buffer_start_pos, uint *buffer_end_pos_tmp,
                  int buffer_size) {
  int old = atomicAdd(buffer_end_pos_tmp, 1) % buffer_size;
  assert((old - buffer_start_pos + buffer_size) % buffer_size <
         buffer_size - 128);
  buffer[old] = data;
  buffer_idx[old] = iter;
}

__device__ __host__ static __forceinline__ uint64_t getResult(uint32_t node,
                                                              uint32_t index) {
  node = (node | (blockIdx.x << 22));
  return ((uint64_t)(node) << 32) | index;
};

__device__ __forceinline__ static void addResult(int vertex, int iteration,
                                                 uint64_t *results,
                                                 int *results_size,
                                                 bool report_off) {
  int old = atomicAdd(results_size, 1);
  if (!report_off) {
    // assert((old - buffer_start_pos + buffer_size) % buffer_size <
    //        buffer_size - 128);
    results[old] = getResult(vertex, iteration);
  }
}

__device__ __forceinline__ static void
addResult2(int vertex, int iteration, uint32_t *d_results_v,
           uint32_t *d_results_i, int *results_size,
           unsigned long long int result_capacity, bool report_off) {
  int old = atomicAdd(results_size, 1);
  if (!report_off) {
    assert(old < result_capacity - 128);
    d_results_i[old] = iteration;
    // d_results_i[old] = blockIdx.x;
    vertex = (vertex | (blockIdx.x << 22));
    d_results_v[old] = vertex;
  }
}

__device__ __forceinline__ static void
addResult2(int vertex, int iteration, uint32_t *d_results_v,
           uint32_t *d_results_i, unsigned long long int *results_size,
           unsigned long long int result_capacity, bool report_off) {
  unsigned long long int old = atomicAdd(results_size, 1);
  if (!report_off) {
    assert(old < result_capacity - 128);
    d_results_i[old] = iteration;
    // d_results_i[old] = blockIdx.x;
    vertex = (vertex | (blockIdx.x << 22));
    d_results_v[old] = vertex;
  }
}

// __device__ __forceinline__ static void
// addBufferandResult(int vertex, int iteration, int *buffer, int *buffer_idx,
//                    uint buffer_start_pos, uint *buffer_end_pos_tmp,
//                    int buffer_size, uint32_t *d_results_v,
//                    uint32_t *d_results_i, int *results_size,
//                    uint8_t *node_attrs) {
//   addToBufferSimple(vertex, iteration, buffer, buffer_idx, buffer_start_pos,
//                     buffer_end_pos_tmp, buffer_size);
//   if (node_attrs[vertex] & 0b10)
//     addResult2(vertex, iteration, d_results_v, d_results_i, results_size);
// }

__device__ static __forceinline__ void
printIndexQueue(int *queueIndexStart, uint length, int &iter_id) {
  printf("iter %d(%u): ", iter_id, length);
  int start, last_index;
  if (length <= 0)
    return;
  for (int i = 0; i < length; i++) {
    if (i == 0) {
      last_index = queueIndexStart[i];
      start = i;
    } else {
      if (last_index != queueIndexStart[i]) {
        printf("%d(%d), ", last_index, i - start);
        start = i;
        last_index = queueIndexStart[i];
      }
    }
  }
  printf("%d(%d), ", queueIndexStart[length - 1], length - start);
  printf("\n");
  iter_id++;
}

} // namespace ngap_nb
#endif