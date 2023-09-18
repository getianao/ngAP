#ifndef SCAN_KERNELS_H_
#define SCAN_KERNELS_H_

#include "common.h"
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>
#include <cuda.h>

__device__ inline bool insert_val_to_array(int *arr, int &len, int val) {
  if (val == -1) {
    return false;
  }
  assert(len < 8192);

  for (int i = 0; i < len; i++) {
    if (arr[i] == val) {
      return true;
    }
  }

  arr[len++] = val;

  return true;
};

template <int NUM_STATE_WL_SHR_MEM, int BLOCK_SIZE = 128>
__device__ __forceinline__ int
insert_val_to_array_cached(int *arr, int &len, int val, int *arr_cached) {
  if (val == -1) {
    return 1;
  }

  for (int i = 0; i < min(len, NUM_STATE_WL_SHR_MEM); i++) {
    if (arr_cached[i * BLOCK_SIZE + threadIdx.x] == val) {
      return 0;
    }
  }

  for (int i = NUM_STATE_WL_SHR_MEM; i < len; i++) {
    if (arr[i] == val) {
      return 0;
    }
  }

  if (len < NUM_STATE_WL_SHR_MEM) {
    arr_cached[len * BLOCK_SIZE + threadIdx.x] = val;
  } else {
    // arr[len - NUM_STATE_WL_SHR_MEM] = val;
    arr[len] = val;
  }

  len++;
  return 2;
};

/**
active + active
default
**/
template <
    int ACTIVE_STATE_ARRAY_SIZE, bool record_intermediate_report = true,
    bool report_on =
        true> // Each thread has a small buffer to store the activated states.
__global__ void scanning_input_kernel2(
    const OutEdges *__restrict__ transition_table,
    const uint8_t *__restrict__ input_stream, const int len_stream,
    const int *__restrict__ start_state_id, const int n_start_node,
    const char *__restrict__ is_report,
    int *__restrict__ intermediate_output_array_offset,
    int *__restrict__ intermediate_output_array_sid,
    int *__restrict__ tail_of_intermediate_output_array,
    match_pair *__restrict__ real_output_array,
    unsigned long long int *__restrict__ tail_of_real_output_array,
    const int R) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int pos = xid;

  if (xid >= len_stream) {
    return;
  }

  while (xid < len_stream) {
    int cur_active2[ACTIVE_STATE_ARRAY_SIZE];
    int cur_array_tail = 0;
    int next_active2[ACTIVE_STATE_ARRAY_SIZE];
    int next_array_tail = 0;

    int *cur_active = cur_active2;
    int *next_active = next_active2;

    for (int i = 0; i < n_start_node; i++) {
      next_active[i] = start_state_id[i];
    }

    next_array_tail = n_start_node;

    for (pos = xid; pos < xid + R && pos < len_stream; pos++) {
      uint8_t symbol = input_stream[pos];

      int *tmp = cur_active;
      cur_active = next_active;
      next_active = tmp;

      // for (int i = 0; i < next_array_tail; i++) {
      // 	cur_active[i] = next_active[i];
      // }

      cur_array_tail = next_array_tail;

      if (cur_array_tail == 0) {
        break;
      }

      next_array_tail = 0;
      for (int i = 0; i < cur_array_tail; i++) {
        int idx_transtable = cur_active[i] * 256 + (int)symbol;
        OutEdges out_nodes = transition_table[idx_transtable];

        bool b = insert_val_to_array(next_active, next_array_tail, out_nodes.x);
        if (!b)
          continue;
        b = insert_val_to_array(next_active, next_array_tail, out_nodes.y);
        if (!b)
          continue;
        b = insert_val_to_array(next_active, next_array_tail, out_nodes.z);
        if (!b)
          continue;
        b = insert_val_to_array(next_active, next_array_tail, out_nodes.w);
        if (!b)
          continue;
      }

      for (int i = 0; i < next_array_tail; i++) {
        if (is_report[next_active[i]]) {
          unsigned long long int write_to =
              atomicAdd(tail_of_real_output_array, 1);

          // match_pair m;
          // m.symbol_offset = pos;
          // m.state_id = next_active[i];
          if (report_on) {
            real_output_array[write_to].state_id = next_active[i];
            real_output_array[write_to].symbol_offset = pos;
          }

          // printf("offset = %d sid = %d\n", pos, next_active[i]);
        }
      }
    }

    if (record_intermediate_report) {
      // write intermediate reports.
      for (int i = 0; i < next_array_tail; i++) {
        if (!is_report[next_active[i]] && pos < len_stream) {
          int write_to = atomicAdd(tail_of_intermediate_output_array, 1);
          intermediate_output_array_offset[write_to] =
              pos; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          intermediate_output_array_sid[write_to] = next_active[i];
        }
      }
    }

    xid += gridDim.x * blockDim.x;
  }
};

/**
active + active
**/
template <
    int ACTIVE_STATE_ARRAY_SIZE, int NUM_STATE_WL_SHR_MEM,
    bool record_intermediate_report = true,
    bool report_on =
        true> // Each thread has a small buffer to store the activated states.
__global__ void scanning_input_kernel3(
    const OutEdges *__restrict__ transition_table,
    const uint8_t *__restrict__ input_stream, const int len_stream,
    const int *__restrict__ start_state_id, const int n_start_node,
    const char *__restrict__ is_report,
    int *__restrict__ intermediate_output_array_offset,
    int *__restrict__ intermediate_output_array_sid,
    int *__restrict__ tail_of_intermediate_output_array,
    match_pair *__restrict__ real_output_array,
    unsigned long long int *__restrict__ tail_of_real_output_array,
    const int R) {

  const int BLOCK_SIZE = 128;
  __shared__ int cur_shr_worklist[NUM_STATE_WL_SHR_MEM * BLOCK_SIZE];
  __shared__ int next_shr_worklist[NUM_STATE_WL_SHR_MEM * BLOCK_SIZE];

  int cur_active2[ACTIVE_STATE_ARRAY_SIZE];
  int next_active2[ACTIVE_STATE_ARRAY_SIZE];

  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int pos = xid;

  int *__restrict__ cur_active = cur_active2 - NUM_STATE_WL_SHR_MEM;
  int *__restrict__ next_active = next_active2 - NUM_STATE_WL_SHR_MEM;

  if (xid >= len_stream) {
    return;
  }

  // int *cur_shr_worklist_this_thread = cur_shr_worklist + threadIdx.x *
  // NUM_STATE_WL_SHR_MEM; int *next_shr_worklist_this_thread =
  // next_shr_worklist + threadIdx.x * NUM_STATE_WL_SHR_MEM;

  int *__restrict__ cur_shr_worklist_this_thread =
      cur_shr_worklist; // + threadIdx.x;
  int *__restrict__ next_shr_worklist_this_thread =
      next_shr_worklist; // + threadIdx.x;

  while (xid < len_stream) {
    int cur_array_tail = 0;
    int next_array_tail = 0;
    for (int i = 0; i < n_start_node && i < NUM_STATE_WL_SHR_MEM; i++) {
      // next_shr_worklist[threadIdx.x * NUM_STATE_WL_SHR_MEM + i] =
      // start_state_id[i]; next_shr_worklist_this_thread[i] =
      // start_state_id[i];
      // next_shr_worklist_this_thread[i * NUM_STATE_WL_SHR_MEM] =
      // start_state_id[i];

      next_shr_worklist_this_thread[i * BLOCK_SIZE + threadIdx.x] =
          start_state_id[i];
    }

    for (int i = NUM_STATE_WL_SHR_MEM; i < n_start_node; i++) {
      next_active[i] = start_state_id[i];
    }

    next_array_tail = n_start_node;
    for (pos = xid; pos < xid + R && pos < len_stream; pos++) {
      uint8_t symbol = input_stream[pos];

      int *tmp = next_shr_worklist_this_thread;
      next_shr_worklist_this_thread = cur_shr_worklist_this_thread;
      cur_shr_worklist_this_thread = tmp;

      // for (int i = 0; i < min(next_array_tail, NUM_STATE_WL_SHR_MEM); i++) {
      // 	// cur_shr_worklist[threadIdx.x * NUM_STATE_WL_SHR_MEM + i] =
      // next_shr_worklist[threadIdx.x * NUM_STATE_WL_SHR_MEM + i];
      // 	cur_shr_worklist_this_thread[i * NUM_STATE_WL_SHR_MEM] =
      // next_shr_worklist_this_thread[i * NUM_STATE_WL_SHR_MEM];
      // }

      tmp = next_active;
      next_active = cur_active;
      cur_active = tmp;

      // for (int i = NUM_STATE_WL_SHR_MEM; i < next_array_tail; i++) {
      // 	cur_active[i] = next_active[i];
      // }

      cur_array_tail = next_array_tail;

      if (cur_array_tail == 0) {
        break;
      }

      next_array_tail = 0;
      for (int i = 0; i < min(NUM_STATE_WL_SHR_MEM, cur_array_tail); i++) {
        int idx_transtable =
            cur_shr_worklist_this_thread[i * BLOCK_SIZE + threadIdx.x] * 256 +
            (int)symbol;
        // cur_shr_worklist[threadIdx.x * NUM_STATE_WL_SHR_MEM + i] * 256 +
        // (int) symbol;
        OutEdges out_nodes = transition_table[idx_transtable];

        int t = insert_val_to_array_cached<NUM_STATE_WL_SHR_MEM>(
            next_active, next_array_tail, out_nodes.x,
            next_shr_worklist_this_thread);
        if (t == 1) {
          continue;
        }

        t = insert_val_to_array_cached<NUM_STATE_WL_SHR_MEM>(
            next_active, next_array_tail, out_nodes.y,
            next_shr_worklist_this_thread);
        if (t == 1) {
          continue;
        }

        t = insert_val_to_array_cached<NUM_STATE_WL_SHR_MEM>(
            next_active, next_array_tail, out_nodes.z,
            next_shr_worklist_this_thread);
        if (t == 1) {
          continue;
        }

        insert_val_to_array_cached<NUM_STATE_WL_SHR_MEM>(
            next_active, next_array_tail, out_nodes.w,
            next_shr_worklist_this_thread);
      }

      for (int i = NUM_STATE_WL_SHR_MEM; i < cur_array_tail; i++) {
        // int t;
        int idx_transtable = cur_active[i] * 256 + (int)symbol;
        OutEdges out_nodes = transition_table[idx_transtable];
        int t;
        t = insert_val_to_array_cached<NUM_STATE_WL_SHR_MEM>(
            next_active, next_array_tail, out_nodes.x,
            next_shr_worklist_this_thread);
        if (t == 1)
          continue;
        t = insert_val_to_array_cached<NUM_STATE_WL_SHR_MEM>(
            next_active, next_array_tail, out_nodes.y,
            next_shr_worklist_this_thread);
        if (t == 1)
          continue;
        t = insert_val_to_array_cached<NUM_STATE_WL_SHR_MEM>(
            next_active, next_array_tail, out_nodes.z,
            next_shr_worklist_this_thread);
        if (t == 1)
          continue;
        insert_val_to_array_cached<NUM_STATE_WL_SHR_MEM>(
            next_active, next_array_tail, out_nodes.w,
            next_shr_worklist_this_thread);
      }

      for (int i = 0; i < min(next_array_tail, NUM_STATE_WL_SHR_MEM); i++) {
        if (is_report[next_shr_worklist_this_thread[i * BLOCK_SIZE +
                                                    threadIdx.x]]) {
          unsigned long long int write_to =
              atomicAdd(tail_of_real_output_array, 1);
          if (report_on) {
            real_output_array[write_to].state_id =
                next_shr_worklist_this_thread[i * BLOCK_SIZE + threadIdx.x];
            real_output_array[write_to].symbol_offset = pos;
          }
        }
      }

      for (int i = NUM_STATE_WL_SHR_MEM; i < next_array_tail; i++) {
        if (is_report[next_active[i]]) {
          unsigned long long int write_to =
              atomicAdd(tail_of_real_output_array, 1);
          if (report_on) {
            real_output_array[write_to].state_id = next_active[i];
            real_output_array[write_to].symbol_offset = pos;
          }
        }
      }
    }

    // if (record_intermediate_report) {
    // 	// write intermediate reports.
    // 	if (xid == 0) {
    // 		printf("not implemented\n");
    // 	}

    // 	assert(0);

    // 	// for (int i = 0; i < next_array_tail; i++) {
    // 	// 	if (!is_report[next_active[i]] && pos < len_stream) {
    // 	// 		int write_to =
    // atomicAdd(tail_of_intermediate_output_array, 1);
    // 	// 		intermediate_output_array_offset[write_to] = pos; //
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // 	// 		intermediate_output_array_sid[write_to] =
    // next_active[i];
    // 	// 	}
    // 	// }
    // }

    xid += gridDim.x * blockDim.x;
  }
};

#endif
