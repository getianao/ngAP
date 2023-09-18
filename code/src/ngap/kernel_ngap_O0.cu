#include "group_graph.h"
#include "kernel_helper.h"
#include "ngap_buffer.h"

using namespace ngap_nb;

__global__ void advanceAndFilterBlockingGroups(
    BlockingBuffer blb, uint8_t *arr_input_streams, int arr_input_streams_size,
    GroupMatchset gms, GroupNodeAttrs gna, GroupAAS gaas, GroupCsr gcsr) {

  Matchset symbol_set = gms.groups_ms[blockIdx.x];
  uint8_t *node_attrs = gna.groups_node_attrs[blockIdx.x];
  int *always_active_nodes = gaas.groups_always_active_states[blockIdx.x];
  Csr csr = gcsr.groups_csr[blockIdx.x];
  int input_index = blockIdx.y;

  uint blockId = blockIdx.y * gridDim.x + blockIdx.x;

  const int buffer_capacity_per_block = blb.buffer_capacity_per_block;
  const bool unique = blb.unique;
  int *d_buffer = blb.d_buffer + blockId * buffer_capacity_per_block;
  int *d_buffer_size = blb.d_buffer_size + blockId;
  uint64_t *results = blb.d_results;
  unsigned long long int *results_size = blb.d_results_size;

  for (int iter = input_index * arr_input_streams_size + 1;
       iter < (input_index + 1) * arr_input_streams_size; iter++) {
    int *d_buffer_cur =
        iter % 2 ? d_buffer : d_buffer + buffer_capacity_per_block / 2;
    int *d_buffer_next =
        iter % 2 ? d_buffer + buffer_capacity_per_block / 2 : d_buffer;
    int symbol = arr_input_streams[iter];
    int length = *d_buffer_size;

#ifdef DEBUG_FRONTIER_SIZE
    if (threadIdx.x == 0) {
      int old = atomicAdd(blb.d_froniter_end, 1);
      blb.d_froniter_length[old] = length;
    }
#endif
    // __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
      *d_buffer_size = 0;
    // __threadfence();
    __syncthreads();
    // advance + filter
    uint threadId = threadIdx.x;
    while (threadId < length) {
      int vertex = d_buffer_cur[threadId];
      int n_start = csr.GetNeighborListOffset(vertex);
      uint n_length = csr.GetNeighborListLength(vertex);
      for (int i = 0; i < n_length; i++) {
        int neighbor = csr.d_column_indices[n_start + i];
        if (symbol_set.test(neighbor, symbol)) {
          if (unique) {
            int mask1 = __match_any_sync(__activemask(), neighbor);
            int leader = __ffs(mask1) - 1;
            if (threadIdx.x % 32 == leader) {
              int old = atomicAdd(d_buffer_size, 1);
              assert(old < buffer_capacity_per_block / 2);
              d_buffer_next[old] = neighbor;
              if (node_attrs[neighbor] & 0b10) {
                unsigned long long int rs_idx = atomicAdd(results_size, 1);
                if (!blb.report_off) {
                  assert(rs_idx < blb.results_capacity);
                  results[rs_idx] = getResult(neighbor, iter);
                }
              }
            }
          } else {
            int old = atomicAdd(d_buffer_size, 1);
            assert(old < buffer_capacity_per_block / 2);
            d_buffer_next[old] = neighbor;
            if (node_attrs[neighbor] & 0b10) {
              unsigned long long int rs_idx = atomicAdd(results_size, 1);
              if (!blb.report_off) {
                assert(rs_idx < blb.results_capacity);
                results[rs_idx] = getResult(neighbor, iter);
              }
            }
          }
        }
      }
      threadId += blockDim.x;
    }

    // filter always active nodes
    threadId = threadIdx.x;
    while (threadId < csr.alwaysActiveNum) {
      int vertex = always_active_nodes[threadId];
      if (symbol_set.test(vertex, symbol)) {
        int old = atomicAdd(d_buffer_size, 1);
        assert(old < buffer_capacity_per_block / 2);
        d_buffer_next[old] = vertex;
        if (node_attrs[vertex] & 0b10) {
          unsigned long long int rs_idx = atomicAdd(results_size, 1);
          if (!blb.report_off) {
            assert(rs_idx < blb.results_capacity);
            results[rs_idx] = getResult(vertex, iter);
          }
        }
      }
      threadId += blockDim.x;
    }

    // if (threadIdx.x == 0) {
    //   for (int i = 0; i < csr.alwaysActiveNum; i++) {
    //     int vertex = always_active_nodes[i];
    //     if (symbol_set.test(vertex, symbol)) {
    //       int old = atomicAdd(d_buffer_size, 1);
    //       assert(old < buffer_capacity_per_block / 2);
    //       d_buffer_next[old] = vertex;
    //       if (node_attrs[vertex] & 0b10) {
    //         unsigned long long int rs_idx = atomicAdd(results_size, 1);
    //         if (!blb.report_off) {
    //           assert(rs_idx < blb.results_capacity);
    //           results[rs_idx] = getResult(vertex, iter);
    //         }
    //       }
    //     }
    //   }
    // }
    // __threadfence();
    __syncthreads();
  }
}