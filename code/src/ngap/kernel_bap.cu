#include "group_graph.h"
#include "kernel_helper.h"
#include "ngap_buffer.h"

using namespace ngap_nb;

template <bool unique, bool record_fs>
__global__ void
// __launch_bounds__(256, 6)
advanceAndFilterNonBlockingNAPGroups(NonBlockingBuffer nblb,
                                     uint8_t *arr_input_streams,
                                     int arr_input_streams_size,
                                     GroupMatchset gms, GroupNodeAttrs gna,
                                     GroupAAS gaas, GroupCsr gcsr) {

  Matchset symbol_set = gms.groups_ms[blockIdx.x];
  uint8_t *node_attrs = gna.groups_node_attrs[blockIdx.x];
  int *always_active_nodes = gaas.groups_always_active_states[blockIdx.x];
  Csr csr = gcsr.groups_csr[blockIdx.x];
  int input_index = blockIdx.y;

  uint blockId = blockIdx.y * gridDim.x + blockIdx.x;

  const int buffer_capacity_per_block = nblb.buffer_capacity_per_block;
  const int data_buffer_fetch_size = nblb.data_buffer_fetch_size;
  // const int add_aas_start = nblb.add_aas_start;
  // const int add_aas_interval = nblb.add_aas_interval;
  const int input_bound = (input_index + 1) * arr_input_streams_size;
  int *d_buffer;
  int *d_buffer_idx;
  if (blockId < gridDim.x * gridDim.y / 2) {
    d_buffer = nblb.d_buffer + blockId * buffer_capacity_per_block;
    d_buffer_idx = nblb.d_buffer_idx + blockId * buffer_capacity_per_block;
  } else {
    d_buffer = nblb.d_buffer2 + (blockId - gridDim.x * gridDim.y / 2) *
                                    buffer_capacity_per_block;
    d_buffer_idx = nblb.d_buffer_idx2 + (blockId - gridDim.x * gridDim.y / 2) *
                                            buffer_capacity_per_block;
  }
  uint *d_buffer_start = nblb.d_buffer_start + blockId;
  uint *d_buffer_end = nblb.d_buffer_end + blockId;
  uint *d_buffer_end_tmp = nblb.d_buffer_end_tmp + blockId;
  // uint *length = nblb.length + blockIdx.y;
  // uint64_t *results = nblb.d_results;
  uint32_t *d_results_i = nblb.d_results_i;
  uint32_t *d_results_v = nblb.d_results_v;
  unsigned long long int *results_size = nblb.d_results_size;
  // todo(tge): reduce table memory size
  int *d_symbol_table = nblb.d_symbol_table +
                        blockIdx.x * (nblb.num_seg * arr_input_streams_size);
  int *newest_idx = nblb.d_newest_idx + blockId;
  int *fakeiter2 = nblb.d_fakeiter2 + blockId * nblb.d_fakeiter_capacity;
  int *fakeiter_size2 = nblb.d_fakeiter_size2 + blockId;

  if (csr.alwaysActiveNum == 0) {
    *newest_idx = input_bound;
  }

  // uint threadIdInGlobal = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef PRINT_INDEX_QUEUE
  int iteration_id = 0;
#endif

  while (*d_buffer_start != *d_buffer_end || *newest_idx < input_bound) {

    if (threadIdx.x == 0)
      if (*d_buffer_start == *d_buffer_end && *newest_idx < input_bound) {
        int oldf2 = atomicAdd(fakeiter_size2, 1);
        assert(oldf2 < 1);
        fakeiter2[oldf2] = *newest_idx;
        // atomicAdd(newest_idx, 1)
        *newest_idx += 1;
      }

    uint length =
        (*d_buffer_end - *d_buffer_start + buffer_capacity_per_block) %
        buffer_capacity_per_block;

#ifdef PRINT_INDEX_QUEUE
    if (threadIdx.x == 0)
      printIndexQueue(d_buffer_idx +
                          (*d_buffer_start) % buffer_capacity_per_block,
                      length, iteration_id);
#endif

    if (length > data_buffer_fetch_size)
      length = data_buffer_fetch_size;
    if (length > 0) {
      uint threadId = threadIdx.x;
      while (threadId < length) {
        uint offset = (*d_buffer_start + threadId) % buffer_capacity_per_block;
        int vertex = d_buffer[offset];
        int iter = d_buffer_idx[offset];
        if (iter >= input_index * arr_input_streams_size &&
            iter < input_bound) {
          // Add fake vertices in batch.
          if (iter == *newest_idx - 1 &&
              !atomicCAS((int *)(d_symbol_table + *newest_idx), 0, 1)) {
            int oldf2 = atomicAdd(fakeiter_size2, 1);
            assert(oldf2 < 1);
            fakeiter2[oldf2] = iter + 1;
            // atomicAdd(newest_idx, 1);
            *newest_idx += 1;
          }
          // If vertex < 0, add always active nodes, and fiter them.
          // If vertex > 0, do advance and filter.
          if (vertex < 0) {
          } else {
            // advance + filter
            if (iter < input_bound - 1) {
              uint8_t next_symbol = arr_input_streams[iter + 1];
              int n_start = csr.GetNeighborListOffset(vertex);
              int n_end = n_start + csr.GetNeighborListLength(vertex);
              bool isUnique = ((nblb.unique_frequency <= 0) ||
                               ((offset % nblb.unique_frequency) == 0));
              while (n_start < n_end) {
                int neighbor = csr.d_column_indices[n_start++];
                if (symbol_set.test(neighbor, next_symbol)) {
                  if (unique && isUnique) {
                    int mask1 = __match_any_sync(__activemask(),
                                                 getResult(neighbor, iter));
                    int leader = __ffs(mask1) - 1;
                    if (threadIdx.x % 32 == leader) {
                      addToBufferSimple(neighbor, iter + 1, d_buffer,
                                        d_buffer_idx, *d_buffer_start,
                                        d_buffer_end_tmp,
                                        buffer_capacity_per_block);
                      if (node_attrs[neighbor] & 0b10)
                        addResult2(neighbor, iter + 1, d_results_v, d_results_i,
                                   results_size, nblb.results_capacity,
                                   nblb.report_off);
                    }
                  } else {
                    addToBufferSimple(neighbor, iter + 1, d_buffer,
                                      d_buffer_idx, *d_buffer_start,
                                      d_buffer_end_tmp,
                                      buffer_capacity_per_block);
                    if (node_attrs[neighbor] & 0b10)
                      addResult2(neighbor, iter + 1, d_results_v, d_results_i,
                                 results_size, nblb.results_capacity,
                                 nblb.report_off);
                  }
                }
              }
            }
          }
        }
        threadId += blockDim.x;
      }
    }
    __syncthreads();
    if (*fakeiter_size2 > 0) {
      int f_iter = *fakeiter2;
      uint8_t f_symbol = arr_input_streams[f_iter];
      uint threadId = threadIdx.x;
      while (threadId < csr.alwaysActiveNum) {
        int aan = always_active_nodes[threadId];
        if (symbol_set.test(aan, f_symbol)) {
          addToBufferSimple(aan, f_iter, d_buffer, d_buffer_idx,
                            *d_buffer_start, d_buffer_end_tmp,
                            buffer_capacity_per_block);
          if (node_attrs[aan] & 0b10)
            addResult2(aan, f_iter, d_results_v, d_results_i, results_size,
                       nblb.results_capacity, nblb.report_off);
        }
        threadId += blockDim.x;
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      *d_buffer_end = *d_buffer_end_tmp % buffer_capacity_per_block;
      *d_buffer_end_tmp = *d_buffer_end_tmp % buffer_capacity_per_block;
      *d_buffer_start = (*d_buffer_start + length) % buffer_capacity_per_block;
      *fakeiter_size2 = 0;
      if (record_fs) {
        int old = atomicAdd(nblb.d_froniter_end, 1);
        nblb.d_froniter_length[old] = length;
      }
    }
    // __threadfence();
    __syncthreads();
  }
}

#define __advanceAndFilterNonBlockingNAPGroups(T1, T2)                         \
  template __global__ void advanceAndFilterNonBlockingNAPGroups<T1, T2>(       \
      NonBlockingBuffer nblb, uint8_t * arr_input_streams,                     \
      int arr_input_streams_size, GroupMatchset gms, GroupNodeAttrs gna,       \
      GroupAAS gaas, GroupCsr gcsr);

__advanceAndFilterNonBlockingNAPGroups(false, false);
__advanceAndFilterNonBlockingNAPGroups(true, false);
__advanceAndFilterNonBlockingNAPGroups(false, true);
__advanceAndFilterNonBlockingNAPGroups(true, true);