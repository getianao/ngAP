#ifndef RUN_AHEAD_KERNELS_H_
#define RUN_AHEAD_KERNELS_H_

#include "common.h"
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>
#include <cuda.h>

#include "scan_kernels.h"

using namespace cub;

static const int ACTIVE_STATE_ARRAY_SIZE_BIG = 200;

// static const int TEMP_XID_LOOK = 0;

template <typename Key, int BLOCK_THREADS, int ITEMS_PER_THREAD,
          typename ValueT>
__launch_bounds__(BLOCK_THREADS) __global__
    void ir_handle_stage1(Key *intermediate_report_offset_array,
                          ValueT *intermediate_report_sid_array, int *ir_len1,
                          // intermediate report.

                          match_pair *real_output_array,
                          unsigned long long int *tail_of_real_output_array,
                          // real report

                          const OutEdges *transition_table,
                          const char *is_report, const uint8_t *input_stream,
                          const int input_stream_length, const int num_of_state
                          // const bool block_sort
                          //  handling part.
    ) {

  if (blockIdx.x * blockDim.x + threadIdx.x >= *ir_len1) {
    intermediate_report_offset_array[blockIdx.x * blockDim.x + threadIdx.x] =
        0x7fffffff;
  }

  __syncthreads();

  enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

  typedef BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD,
                    BLOCK_LOAD_WARP_TRANSPOSE>
      BlockLoadT;

  typedef BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD, ValueT>
      BlockRadixSortT;

  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockRadixSortT::TempStorage sort;
  } temp_storage;

  Key items[ITEMS_PER_THREAD];
  ValueT thread_values[ITEMS_PER_THREAD];

  // Our current block's offset
  int block_offset = blockIdx.x * TILE_SIZE;
  // Load items into a blocked arrangement

  BlockLoadT(temp_storage.load)
      .Load(intermediate_report_offset_array + block_offset, items);
  BlockLoadT(temp_storage.load)
      .Load(intermediate_report_sid_array + block_offset, thread_values);

  __syncthreads();

  BlockRadixSortT(temp_storage.sort).Sort(items, thread_values);

  __syncthreads();

  //
  StoreDirectStriped<BLOCK_THREADS>(
      threadIdx.x, intermediate_report_offset_array + block_offset, items);
  StoreDirectStriped<BLOCK_THREADS>(
      threadIdx.x, intermediate_report_sid_array + block_offset, thread_values);
};

__global__ void
ir_handle_stage2(int *intermediate_report_offset_array,
                 int *intermediate_report_sid_array, int *ir_len1,
                 // intermediate report.

                 match_pair *real_output_array,
                 unsigned long long int *tail_of_real_output_array,
                 // real report

                 const OutEdges *transition_table, const char *is_report,
                 const uint8_t *input_stream, const int input_stream_length,
                 const int num_of_state) {
  int ir_handle_start = blockIdx.x * blockDim.x;
  int ir_handle_end = blockIdx.x * blockDim.x + blockDim.x;

  if (ir_handle_end > *ir_len1) {
    ir_handle_end = *ir_len1;
  }

  extern __shared__ int is_active_state[]; // size: 2 * num_of_state

  int offset = threadIdx.x;
  while (offset < 2 * num_of_state) {
    is_active_state[offset] = false;
    offset += blockDim.x;
  }

  __shared__ bool jump;
  jump = false;

  __syncthreads();

  int next_ir_pos = ir_handle_start;

  for (int symbol_pos = 0; symbol_pos < input_stream_length; symbol_pos++) {
    jump = true;

    __syncthreads();

    offset = threadIdx.x;
    while (offset < num_of_state) {
      if (is_active_state[offset]) {
        jump = false;
      }
      offset += blockDim.x;
    }

    __syncthreads();

    if (jump) {
      if (next_ir_pos < ir_handle_end) {
        // if (!(intermediate_report_offset_array[next_ir_pos] >= symbol_pos)) {
        //	printf("jump from %d to %d\n", symbol_pos,
        // intermediate_report_offset_array[next_ir_pos]);

        //}
        symbol_pos = intermediate_report_offset_array[next_ir_pos];
      } else {
        break;
      }
    }

    //__syncthreads();

    // the below loop could be parallelized. let's do it later.
    // offset = threadIdx.x;
    // while (offset + next_ir_pos < ir_len &&
    // intermediate_output_array_offset[offset + next_ir_pos] == symbol_pos) {
    //	is_active_state[intermediate_output_array_sid[offset + next_ir_pos]] =
    // true;
    //}

    while (next_ir_pos < ir_handle_end &&
           intermediate_report_offset_array[next_ir_pos] == symbol_pos) {
      // printf("enable %d\n", intermediate_report_sid_array[next_ir_pos]);
      is_active_state[intermediate_report_sid_array[next_ir_pos]] = true;
      next_ir_pos++;
    }

    __syncthreads();

    uint8_t symbol = input_stream[symbol_pos];

    // printf("symbol %c\n", symbol);

    offset = threadIdx.x;
    while (offset < num_of_state) {
      if (is_active_state[offset]) {
        int idx_transtable = offset * 256 + (int)symbol;
        OutEdges out_nodes = transition_table[idx_transtable];

        if (out_nodes.x != -1) {
          is_active_state[out_nodes.x + num_of_state] = true;
        }

        if (out_nodes.y != -1) {
          is_active_state[out_nodes.y + num_of_state] = true;
        }

        if (out_nodes.z != -1) {
          is_active_state[out_nodes.z + num_of_state] = true;
        }

        if (out_nodes.w != -1) {
          is_active_state[out_nodes.w + num_of_state] = true;
        }

        if (is_report[offset]) {
          unsigned long long int write_to =
              atomicAdd(tail_of_real_output_array, 1);
          real_output_array[write_to].symbol_offset = symbol_pos - 1;
          real_output_array[write_to].state_id = offset;
        }
      }

      offset += blockDim.x;
    }

    __syncthreads();

    offset = threadIdx.x;
    while (offset < num_of_state) {
      is_active_state[offset] = is_active_state[offset + num_of_state];

      offset += blockDim.x;
    }

    __syncthreads();

    offset = threadIdx.x;
    while (offset < num_of_state) {
      is_active_state[offset + num_of_state] = false;
      offset += blockDim.x;
    }

    __syncthreads();

  } // loop on symbols.
};

#endif