#include "group_graph.h"
#include "kernel_helper.h"
#include "ngap_buffer.h"

using namespace ngap_nb;
template <bool unique>
__global__ void
// __launch_bounds__(256, 6)
advanceAndFilterNonBlockingR1Groups(NonBlockingBuffer nblb,
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
  // uint *length = nblb.length + blockIdx.y;
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

  // int max_depth = 1;
  auto processRealVertexR0 = [&](int rvertex, int riter, int depth) {
    // advance + filter
    if (riter >= input_bound - 1)
      return;
    uint8_t rsymbol = arr_input_streams[riter + 1];
    int rn_start = csr.GetNeighborListOffset(rvertex);
    int rn_end = rn_start + csr.GetNeighborListLength(rvertex);
    // if (csr.GetNeighborListLength(rvertex) > 3)
    //   printf("%d, ", csr.GetNeighborListLength(rvertex));
    // #pragma unroll 4
    while (rn_start < rn_end) {
      int rneighbor = csr.d_column_indices[rn_start++];
      if (symbol_set.test(rneighbor, rsymbol)) {
        if (false) {
          int mask1 =
              __match_any_sync(__activemask(), getResult(rneighbor, riter));
          int leader = __ffs(mask1) - 1;
          if (threadIdx.x % 32 == leader) {
            addToBufferSimple(rneighbor, riter + 1, d_buffer, d_buffer_idx,
                              *d_buffer_start, d_buffer_end_tmp,
                              buffer_capacity_per_block);
            if (node_attrs[rneighbor] & 0b10)
              addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
                         results_size, nblb.results_capacity, nblb.report_off);
          }
        } else {
          addToBufferSimple(rneighbor, riter + 1, d_buffer, d_buffer_idx,
                            *d_buffer_start, d_buffer_end_tmp,
                            buffer_capacity_per_block);
          if (node_attrs[rneighbor] & 0b10)
            addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
                       results_size, nblb.results_capacity, nblb.report_off);
        }
      }
    }
  };

  auto processRealVertexR1 = [&](int rvertex, int riter, int depth,
                                 bool isUnique) {
    // advance + filter
    if (riter >= input_bound - 1)
      return;
    uint8_t rsymbol = arr_input_streams[riter + 1];
    int rn_start = csr.GetNeighborListOffset(rvertex);
    int rn_end = rn_start + csr.GetNeighborListLength(rvertex);
    // if (csr.GetNeighborListLength(rvertex) > 3)
    //   printf("%d, ", csr.GetNeighborListLength(rvertex));
#pragma unroll 2
    while (rn_start < rn_end) {
      int rneighbor = csr.d_column_indices[rn_start++];
      if (symbol_set.test(rneighbor, rsymbol)) {
        if (unique && isUnique) {
          int mask1 =
              __match_any_sync(__activemask(), getResult(rneighbor, riter));
          int leader = __ffs(mask1) - 1;
          if (threadIdx.x % 32 == leader) {
            if (node_attrs[rneighbor] & 0b10)
              addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
                         results_size, nblb.results_capacity, nblb.report_off);
            if (__popc(__activemask()) <= nblb.active_threshold) {
              addToBufferSimple(rneighbor, riter + 1, d_buffer, d_buffer_idx,
                                *d_buffer_start, d_buffer_end_tmp,
                                buffer_capacity_per_block);
            } else {
              processRealVertexR0(rneighbor, riter + 1, depth + 1);
            }
          }
        } else {
          if (node_attrs[rneighbor] & 0b10)
            addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
                       results_size, nblb.results_capacity, nblb.report_off);
          if (__popc(__activemask()) <= nblb.active_threshold) {
            addToBufferSimple(rneighbor, riter + 1, d_buffer, d_buffer_idx,
                              *d_buffer_start, d_buffer_end_tmp,
                              buffer_capacity_per_block);
          } else {
            processRealVertexR0(rneighbor, riter + 1, depth + 1);
          }
        }
      }
    }
  };

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
            uint8_t symbol = arr_input_streams[iter];
            for (int i = 0; i < csr.alwaysActiveNum; i++) {
              int aan = always_active_nodes[i];
              if (symbol_set.test(aan, symbol)) {
                addToBufferSimple(aan, iter, d_buffer, d_buffer_idx,
                                  *d_buffer_start, d_buffer_end_tmp,
                                  buffer_capacity_per_block);
                if (node_attrs[aan] & 0b10)
                  addResult2(aan, iter, d_results_v, d_results_i, results_size,
                             nblb.results_capacity, nblb.report_off);
              }
            }
          } else {
            bool isUnique = ((nblb.unique_frequency <= 0) ||
                             ((offset % nblb.unique_frequency) == 0));
            processRealVertexR1(vertex, iter, 0, isUnique);
          }
        }
        threadId += blockDim.x;
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      *d_buffer_end = *d_buffer_end_tmp % buffer_capacity_per_block;
      *d_buffer_start = (*d_buffer_start + length) % buffer_capacity_per_block;
    }
    // __threadfence();
    __syncthreads();
  }
}

template __global__ void advanceAndFilterNonBlockingR1Groups<false>(
    NonBlockingBuffer nblb, uint8_t *arr_input_streams,
    int arr_input_streams_size, GroupMatchset gms, GroupNodeAttrs gna,
    GroupAAS gaas, GroupCsr gcsr);

template __global__ void advanceAndFilterNonBlockingR1Groups<true>(
    NonBlockingBuffer nblb, uint8_t *arr_input_streams,
    int arr_input_streams_size, GroupMatchset gms, GroupNodeAttrs gna,
    GroupAAS gaas, GroupCsr gcsr);

template <bool unique>
__global__ void
// __launch_bounds__(256, 6)
advanceAndFilterNonBlockingR2Groups(NonBlockingBuffer nblb,
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
  // uint *length = nblb.length + blockIdx.y;
  // uint64_t *results = nblb.d_results;
  uint32_t *d_results_i = nblb.d_results_i;
  uint32_t *d_results_v = nblb.d_results_v;
  unsigned long long int *results_size = nblb.d_results_size;
  // todo(tge): reduce table memory size
  int *d_symbol_table = nblb.d_symbol_table +
                        blockIdx.x * (nblb.num_seg * arr_input_streams_size);
  int *newest_idx = nblb.d_newest_idx + blockId;

  // int max_depth = 2;
  auto processRealVertexR0 = [&](int rvertex, int riter, int depth) {
    // advance + filter
    if (riter >= input_bound - 1)
      return;
    uint8_t rsymbol = arr_input_streams[riter + 1];
    int rn_start = csr.GetNeighborListOffset(rvertex);
    int rn_end = rn_start + csr.GetNeighborListLength(rvertex);
    // #pragma unroll 2
    while (rn_start < rn_end) {
      int rneighbor = csr.d_column_indices[rn_start++];
      if (symbol_set.test(rneighbor, rsymbol)) {
        if (false) {
          int mask1 =
              __match_any_sync(__activemask(), getResult(rneighbor, riter));
          int leader = __ffs(mask1) - 1;
          if (threadIdx.x % 32 == leader) {
            addToBufferSimple(rneighbor, riter + 1, d_buffer, d_buffer_idx,
                              *d_buffer_start, d_buffer_end_tmp,
                              buffer_capacity_per_block);
            if (node_attrs[rneighbor] & 0b10)
              addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
                         results_size, nblb.results_capacity, nblb.report_off);
          }
        } else {
          addToBufferSimple(rneighbor, riter + 1, d_buffer, d_buffer_idx,
                            *d_buffer_start, d_buffer_end_tmp,
                            buffer_capacity_per_block);
          if (node_attrs[rneighbor] & 0b10)
            addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
                       results_size, nblb.results_capacity, nblb.report_off);
        }
      }
    }
  };

  auto processRealVertexR1 = [&](int rvertex, int riter, int depth) {
    // advance + filter
    if (riter >= input_bound - 1)
      return;
    uint8_t rsymbol = arr_input_streams[riter + 1];
    int rn_start = csr.GetNeighborListOffset(rvertex);
    int rn_end = rn_start + csr.GetNeighborListLength(rvertex);
#pragma unroll 2
    while (rn_start < rn_end) {
      int rneighbor = csr.d_column_indices[rn_start++];
      if (symbol_set.test(rneighbor, rsymbol)) {
        if (__popc(__activemask()) <= nblb.active_threshold) {
          if (false) {
            int mask1 =
                __match_any_sync(__activemask(), getResult(rneighbor, riter));
            int leader = __ffs(mask1) - 1;
            if (threadIdx.x % 32 == leader) {
              addToBufferSimple(rneighbor, riter + 1, d_buffer, d_buffer_idx,
                                *d_buffer_start, d_buffer_end_tmp,
                                buffer_capacity_per_block);
              if (node_attrs[rneighbor] & 0b10)
                addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
                           results_size, nblb.results_capacity,
                           nblb.report_off);
            }
          } else {
            addToBufferSimple(rneighbor, riter + 1, d_buffer, d_buffer_idx,
                              *d_buffer_start, d_buffer_end_tmp,
                              buffer_capacity_per_block);
            if (node_attrs[rneighbor] & 0b10)
              addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
                         results_size, nblb.results_capacity, nblb.report_off);
          }
        } else {
          if (node_attrs[rneighbor] & 0b10)
            addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
                       results_size, nblb.results_capacity, nblb.report_off);
          processRealVertexR0(rneighbor, riter + 1, depth + 1);
        }
      }
    }
  };

  auto processRealVertexR2 = [&](int rvertex, int riter, int depth,
                                 bool isUnique) {
    // advance + filter
    if (riter >= input_bound - 1)
      return;
    uint8_t rsymbol = arr_input_streams[riter + 1];
    int rn_start = csr.GetNeighborListOffset(rvertex);
    int rn_end = rn_start + csr.GetNeighborListLength(rvertex);
#pragma unroll 4
    while (rn_start < rn_end) {
      int rneighbor = csr.d_column_indices[rn_start++];
      if (symbol_set.test(rneighbor, rsymbol)) {
        if (unique && isUnique) {
          int mask1 =
              __match_any_sync(__activemask(), getResult(rneighbor, riter));
          int leader = __ffs(mask1) - 1;
          if (threadIdx.x % 32 == leader) {
            if (node_attrs[rneighbor] & 0b10)
              addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
                         results_size, nblb.results_capacity, nblb.report_off);
            if (__popc(__activemask()) <= nblb.active_threshold) {
              addToBufferSimple(rneighbor, riter + 1, d_buffer, d_buffer_idx,
                                *d_buffer_start, d_buffer_end_tmp,
                                buffer_capacity_per_block);
            } else {
              processRealVertexR1(rneighbor, riter + 1, depth + 1);
            }
          }
        } else {
          if (node_attrs[rneighbor] & 0b10)
            addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
                       results_size, nblb.results_capacity, nblb.report_off);
          if (__popc(__activemask()) <= nblb.active_threshold) {
            addToBufferSimple(rneighbor, riter + 1, d_buffer, d_buffer_idx,
                              *d_buffer_start, d_buffer_end_tmp,
                              buffer_capacity_per_block);

          } else {
            processRealVertexR1(rneighbor, riter + 1, depth + 1);
          }
        }
      }
    }
  };

  //   auto processRealVertex = [&](auto &&processRealVertex, int rvertex,
  //                                int riter, int depth) {
  //     // advance + filter
  //     if (riter >= input_bound - 1)
  //       return;
  //     uint8_t rsymbol = arr_input_streams[riter + 1];
  //     int rn_start = csr.GetNeighborListOffset(rvertex);
  //     int rn_end = rn_start + csr.GetNeighborListLength(rvertex);
  // #pragma unroll 32
  //     while (rn_start < rn_end) {
  //       int rneighbor = csr.d_column_indices[rn_start++];
  //       if (symbol_set.test(rneighbor, rsymbol)) {
  //         if (depth >= max_depth ||
  //             __popc(__activemask()) <= nblb.active_threshold) {
  //           if (unique) {
  //             int mask1 =
  //                 __match_any_sync(__activemask(), getResult(rneighbor,
  //                 riter));
  //             int leader = __ffs(mask1) - 1;
  //             if (threadIdx.x % 32 == leader) {
  //               addToBufferSimple(rneighbor, riter + 1, d_buffer,
  //               d_buffer_idx,
  //                                 *d_buffer_start, d_buffer_end_tmp,
  //                                 buffer_capacity_per_block);
  //               if (node_attrs[rneighbor] & 0b10)
  //                 addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
  //                            results_size, nblb.results_capacity);
  //             }
  //           } else {
  //             addToBufferSimple(rneighbor, riter + 1, d_buffer, d_buffer_idx,
  //                                 *d_buffer_start, d_buffer_end_tmp,
  //                                 buffer_capacity_per_block);
  //               if (node_attrs[rneighbor] & 0b10)
  //                 addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
  //                            results_size, nblb.results_capacity);
  //           }
  //         } else {
  //           if (node_attrs[rneighbor] & 0b10)
  //               addResult2(rneighbor, riter + 1, d_results_v, d_results_i,
  //                          results_size, nblb.results_capacity);
  //           processRealVertex(processRealVertex, rneighbor, riter + 1, depth
  //           + 1);
  //         }
  //       }
  //     }
  //   };

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
          // If vertex > 0 and we didn't met it before, add a fake
          // vertex(-1) for its iteration.
          if (vertex < 0) {
            // d_symbol_table[iter] = 2;
            uint8_t symbol = arr_input_streams[iter];
            for (int i = 0; i < csr.alwaysActiveNum; i++) {
              int aan = always_active_nodes[i];
              if (symbol_set.test(aan, symbol)) {
                addToBufferSimple(aan, iter, d_buffer, d_buffer_idx,
                                  *d_buffer_start, d_buffer_end_tmp,
                                  buffer_capacity_per_block);
                if (node_attrs[aan] & 0b10)
                  addResult2(aan, iter, d_results_v, d_results_i, results_size,
                             nblb.results_capacity, nblb.report_off);
              }
            }
          } else {
            bool isUnique = ((nblb.unique_frequency <= 0) ||
                             ((offset % nblb.unique_frequency) == 0));
            processRealVertexR2(vertex, iter, 0, isUnique);
            // processRealVertex(processRealVertex, vertex, iter, 0);
          }
        }
        threadId += blockDim.x;
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      *d_buffer_end = *d_buffer_end_tmp % buffer_capacity_per_block;
      *d_buffer_end_tmp = *d_buffer_end_tmp % buffer_capacity_per_block;
      *d_buffer_start = (*d_buffer_start + length) % buffer_capacity_per_block;
    }
    // __threadfence();
    __syncthreads();
  }
}

template __global__ void advanceAndFilterNonBlockingR2Groups<false>(
    NonBlockingBuffer nblb, uint8_t *arr_input_streams,
    int arr_input_streams_size, GroupMatchset gms, GroupNodeAttrs gna,
    GroupAAS gaas, GroupCsr gcsr);

template __global__ void advanceAndFilterNonBlockingR2Groups<true>(
    NonBlockingBuffer nblb, uint8_t *arr_input_streams,
    int arr_input_streams_size, GroupMatchset gms, GroupNodeAttrs gna,
    GroupAAS gaas, GroupCsr gcsr);