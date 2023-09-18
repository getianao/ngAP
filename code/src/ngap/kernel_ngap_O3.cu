#include "group_graph.h"
#include "kernel_helper.h"
#include "ngap_buffer.h"

using namespace ngap_nb;

template <bool unique, int precompute_depth, bool record_fs>
__global__ void
// __launch_bounds__(256, 6)
advanceAndFilterNonBlockingPrecGroups(NonBlockingBuffer nblb,
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
  const int add_aas_start = nblb.add_aas_start;
  const int add_aas_interval = nblb.add_aas_interval;
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
  // uint *length = nblb.length + blockId;
  // uint64_t *results = nblb.d_results;
  uint32_t *d_results_i = nblb.d_results_i;
  uint32_t *d_results_v = nblb.d_results_v;
  unsigned long long int *results_size = nblb.d_results_size;
  // todo(tge): reduce table memory size
  int *d_symbol_table = nblb.d_symbol_table +
                        blockIdx.x * (nblb.num_seg * arr_input_streams_size);
  int *newest_idx = nblb.d_newest_idx + blockId;
  if (csr.alwaysActiveNum == 0) {
    *newest_idx = input_bound;
  }
  // int *fakeiter = nblb.d_fakeiter + blockId * arr_input_streams_size;
  // int *fakeiter_size = nblb.d_fakeiter_size + blockId;
  int *fakeiter2 = nblb.d_fakeiter2 + blockId * nblb.d_fakeiter_capacity;
  int *fakeiter_size2 = nblb.d_fakeiter_size2 + blockId;
  // int *cutoffnum = nblb.cutoffnum + blockId;

  // uint threadIdInGlobal = blockIdx.x * blockDim.x + threadIdx.x;

  while (*d_buffer_start != *d_buffer_end || *newest_idx < input_bound) {

    if (threadIdx.x == 0)
      if (*d_buffer_start == *d_buffer_end && *newest_idx < input_bound) {
        addToBufferSimple(-1, *newest_idx, d_buffer, d_buffer_idx,
                          *d_buffer_start, d_buffer_end_tmp,
                          buffer_capacity_per_block);
      }

    uint length2 =
        (*d_buffer_end - *d_buffer_start + buffer_capacity_per_block) %
        buffer_capacity_per_block;
    uint length = length2;
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
          if ((iter == *newest_idx) &&
              !atomicCAS((int *)(d_symbol_table + *newest_idx), 0, 1)) {
            // int old = atomicAdd(d_buffer_end_tmp, 1);
            // assert((old - *d_buffer_start + buffer_capacity_per_block) %
            //              buffer_capacity_per_block <
            //          buffer_capacity_per_block - 128);
            // d_buffer[old % buffer_capacity_per_block] = -1;
            // d_buffer_idx[old % buffer_capacity_per_block] = iter;

            int iter_rank = *newest_idx - input_index * arr_input_streams_size;
            if (iter_rank >= add_aas_start &&
                ((iter_rank - add_aas_start) % add_aas_interval) == 0) {
              int end_idx = *newest_idx + add_aas_interval;
              if (end_idx > input_bound)
                end_idx = input_bound;
              int start_idx = *newest_idx;
              int number_idx = end_idx - start_idx;
              *newest_idx = end_idx;
              uint old = atomicAdd(d_buffer_end_tmp, number_idx);
              assert((old - *d_buffer_start + buffer_capacity_per_block) %
                         buffer_capacity_per_block <
                     buffer_capacity_per_block - 128);
              for (int i = 0; i < number_idx; i++) {
                d_buffer[(old + i) % buffer_capacity_per_block] = -1;
                d_buffer_idx[(old + i) % buffer_capacity_per_block] =
                    start_idx + i;
              }
            }
          }
          // If vertex < 0, add always active nodes, and fiter them.
          // If vertex > 0, do advance and filter.
          if (vertex < 0) {
            if (precompute_depth == 0) {
              uint8_t symbol = arr_input_streams[iter];
              for (int i = 0; i < csr.alwaysActiveNum; i++) {
                int aan = always_active_nodes[i];
                if (symbol_set.test(aan, symbol)) {
                  addToBufferSimple(aan, iter, d_buffer, d_buffer_idx,
                                    *d_buffer_start, d_buffer_end_tmp,
                                    buffer_capacity_per_block);
                  if (node_attrs[aan] & 0b10)
                    addResult2(aan, iter, d_results_v, d_results_i,
                               results_size, nblb.results_capacity,
                               nblb.report_off);
                }
              }
            } else {
              for (int pcd = precompute_depth; pcd > 0; pcd--) { // 3 2 1
                if (iter <= input_bound - pcd) {
                  // Add vertices.
                  uint32_t symbol_idx = 0;
                  for (int loop = 0; loop < pcd; loop++) {
                    symbol_idx =
                        256 * symbol_idx + arr_input_streams[iter + loop];
                  }
                  int symbol_real_idx =
                      csr.d_pts[pcd - 1].getVertexSymbolIndex(symbol_idx);
                  if (symbol_real_idx >= 0) {
                    int poffset =
                        csr.d_pts[pcd - 1].d_vertices_offsets[symbol_real_idx];
                    uint plength =
                        csr.d_pts[pcd - 1]
                            .d_vertices_offsets[symbol_real_idx + 1] -
                        csr.d_pts[pcd - 1].d_vertices_offsets[symbol_real_idx];
                    if (pcd == precompute_depth) {
                      if (plength > csr.d_pts[pcd - 1].cutoff) {
                        int oldf2 = atomicAdd(fakeiter_size2, 1);
                        assert(oldf2 < nblb.d_fakeiter_capacity);
                        fakeiter2[oldf2] = iter;
                      }
                      int min_cut = min(csr.d_pts[pcd - 1].cutoff, plength);
                      for (int j = 0; j < min_cut; j++) {
                        int pov = csr.d_pts[pcd - 1].d_vertices[poffset + j];
                        addToBufferSimple(pov, iter + pcd - 1, d_buffer,
                                          d_buffer_idx, *d_buffer_start,
                                          d_buffer_end_tmp,
                                          buffer_capacity_per_block);
                      }
                    } else {
                      for (int j = 0; j < plength; j++) {
                        addToBufferSimple(
                            csr.d_pts[pcd - 1].d_vertices[poffset + j],
                            iter + pcd - 1, d_buffer, d_buffer_idx,
                            *d_buffer_start, d_buffer_end_tmp,
                            buffer_capacity_per_block);
                      }
                    }
                  }
                  // Add result during precomputing.
                  for (int loop = 0; loop < pcd; loop++) {
                    uint32_t symbol_idx2 = 0;
                    for (int loop2 = 0; loop2 < loop + 1; loop2++) {
                      symbol_idx2 =
                          256 * symbol_idx2 + arr_input_streams[iter + loop2];
                    }
                    int symbol_real_idx2 =
                        csr.d_pts[loop].getResultSymbolIndex(symbol_idx2);
                    if (symbol_real_idx2 >= 0) {
                      int roffset =
                          csr.d_pts[loop].d_results_offsets[symbol_real_idx2];
                      uint rlength =
                          csr.d_pts[loop]
                              .d_results_offsets[symbol_real_idx2 + 1] -
                          roffset;
                      for (int r = 0; r < rlength; r++) {
                        addResult2(csr.d_pts[loop].d_results[roffset + r],
                                   iter + loop, d_results_v, d_results_i,
                                   results_size, nblb.results_capacity,
                                   nblb.report_off);
                      }
                    }
                  }
                  break;
                }
              }
            }
          } else {
            // advance + filter
            if (iter < input_bound - 1) {
              bool isUnique = ((nblb.unique_frequency <= 0) ||
                               ((offset % nblb.unique_frequency) == 0));
              uint8_t next_symbol = arr_input_streams[iter + 1];
              int n_start = csr.GetNeighborListOffset(vertex);
              int n_end = n_start + csr.GetNeighborListLength(vertex);
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

    if (precompute_depth != 0) {
      int ssize2 = *fakeiter_size2;
      if (ssize2 > 0) {
        uint threadId = threadIdx.x;
        int pcd = precompute_depth;
        while (threadId < ssize2) {
          int fiter = fakeiter2[threadId];
          uint32_t symbol_idx = 0;
          for (int loop = 0; loop < pcd; loop++)
            symbol_idx = 256 * symbol_idx + arr_input_streams[fiter + loop];
          int symbol_real_idx =
              csr.d_pts[pcd - 1].getVertexSymbolIndex(symbol_idx);
          if (symbol_real_idx >= 0) {
            int fpoffset =
                csr.d_pts[pcd - 1].d_vertices_offsets[symbol_real_idx] +
                csr.d_pts[pcd - 1].cutoff;
            uint fplength =
                csr.d_pts[pcd - 1].d_vertices_offsets[symbol_real_idx + 1] -
                fpoffset;
            for (int i = 0; i < fplength; i++) {
              int pov = csr.d_pts[pcd - 1].d_vertices[fpoffset + i];
              addToBufferSimple(pov, fiter + pcd - 1, d_buffer, d_buffer_idx,
                                *d_buffer_start, d_buffer_end_tmp,
                                buffer_capacity_per_block);
            }
          }
          threadId += blockDim.x;
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      *d_buffer_end = *d_buffer_end_tmp % buffer_capacity_per_block;
      *d_buffer_end_tmp = *d_buffer_end_tmp % buffer_capacity_per_block;
      *d_buffer_start = (*d_buffer_start + length) % buffer_capacity_per_block;
      if (precompute_depth > 0) {
        *fakeiter_size2 = 0;
      }
      if (record_fs) {
        int old = atomicAdd(nblb.d_froniter_end, 1);
        nblb.d_froniter_length[old] = length;
      }
    }
    // __threadfence();
    __syncthreads();
  }
}

#define __advanceAndFilterNonBlockingPrecGroups(T1, T2, T3)                    \
  template __global__ void advanceAndFilterNonBlockingPrecGroups<T1, T2, T3>(  \
      NonBlockingBuffer nblb, uint8_t * arr_input_streams,                     \
      int arr_input_streams_size, GroupMatchset gms, GroupNodeAttrs gna,       \
      GroupAAS gaas, GroupCsr gcsr);

__advanceAndFilterNonBlockingPrecGroups(false, 0, false);
__advanceAndFilterNonBlockingPrecGroups(true, 0, false);
__advanceAndFilterNonBlockingPrecGroups(false, 0, true);
__advanceAndFilterNonBlockingPrecGroups(true, 0, true);
__advanceAndFilterNonBlockingPrecGroups(false, 1, false);
__advanceAndFilterNonBlockingPrecGroups(true, 1, false);
__advanceAndFilterNonBlockingPrecGroups(false, 1, true);
__advanceAndFilterNonBlockingPrecGroups(true, 1, true);
__advanceAndFilterNonBlockingPrecGroups(false, 2, false);
__advanceAndFilterNonBlockingPrecGroups(true, 2, false);
__advanceAndFilterNonBlockingPrecGroups(false, 2, true);
__advanceAndFilterNonBlockingPrecGroups(true, 2, true);
__advanceAndFilterNonBlockingPrecGroups(false, 3, false);
__advanceAndFilterNonBlockingPrecGroups(true, 3, false);
__advanceAndFilterNonBlockingPrecGroups(false, 3, true);
__advanceAndFilterNonBlockingPrecGroups(true, 3, true);
