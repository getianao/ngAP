
#include "ngap_buffer.h"

__host__ void BlockingBuffer::init(Array2<uint8_t> *input_stream,
                                   int input_total_size, int input_num,
                                   int multi_ss_size, Graph &graph,
                                   ngap_option *plo) {
  buffer_capacity = DATA_BUFFER_SIZE_FRONTIER;
  results_capacity = RESULTS_SIZE;
  buffer_capacity_per_block = buffer_capacity / input_num;
  unique = plo->unique;
  report_off = plo->report_off;
  CHECK_ERROR(cudaMalloc((void **)&d_buffer, sizeof(int) * buffer_capacity));
  CHECK_ERROR(cudaMalloc((void **)&d_buffer_size, sizeof(int) * input_num));
  CHECK_ERROR(
      cudaMalloc((void **)&d_buffer_idx, sizeof(int) * buffer_capacity));
  motivate_worklist_length = plo->motivate_worklist_length;

  if (plo->motivate_worklist_length) {
    CHECK_ERROR(cudaMalloc((void **)&d_froniter_length, sizeof(int) * 2000000));
    CHECK_ERROR(
        cudaMemset((void *)d_froniter_length, 0, sizeof(int) * 2000000));
    CHECK_ERROR(cudaMalloc((void **)&d_froniter_end, sizeof(int)));
    CHECK_ERROR(cudaMemset((void *)d_froniter_end, 0, sizeof(int)));
  }

  // characterize divergence
  // CHECK_ERROR(cudaMalloc((void **)&d_froniter_divergence_end, sizeof(int)));
  // CHECK_ERROR(cudaMalloc((void **)&d_froniter_divergence_advance, sizeof(int)
  // * 2000000)); CHECK_ERROR(cudaMalloc((void **)&d_froniter_divergence_filter,
  // sizeof(int) * 2000000)); CHECK_ERROR(cudaMalloc((void
  // **)&d_froniter_workload_end, sizeof(int))); CHECK_ERROR(cudaMalloc((void
  // **)&d_froniter_workload, sizeof(int) * 200000000));

  // CHECK_ERROR(cudaMemset((void *)d_froniter_divergence_end, 0, sizeof(int)));
  // CHECK_ERROR(cudaMemset((void *)d_froniter_workload_end, 0, sizeof(int)));
  // CHECK_ERROR(cudaMemset((void *)d_froniter_divergence_advance, 0,
  // sizeof(int)* 2000000)); CHECK_ERROR(cudaMemset((void
  // *)d_froniter_divergence_filter, 0, sizeof(int)* 2000000));
  // CHECK_ERROR(cudaMemset((void *)d_froniter_workload, 0, sizeof(int)*
  // 200000000));

  auto getResult = [](uint32_t node, uint32_t index) -> uint64_t {
    uint64_t r = 0;
    r = (uint32_t)node;
    r = r << 32;
    r = r | (uint32_t)index;
    return r;
  };

  int *h_buffer = new int[buffer_capacity_per_block];
  unsigned long long int h_results_size = 0;
  uint64_t *h_results = new uint64_t[results_capacity];
  memset(h_results, 0, sizeof(uint64_t) * results_capacity);
  for (int i = 0; i < input_num; i++) {
    uint8_t first_symbol = input_stream->get_host()[i * multi_ss_size];
    int h_buffer_size = 0;

    memset(h_buffer, 0, sizeof(int) * buffer_capacity_per_block);
    for (int j = 0; j < graph.alwaysActiveNum; j++) {
      int vertex = graph.always_active_nodes->get_host()[j];
      if (graph.symbol_sets->get_host()[vertex].test(first_symbol)) {
        if (graph.node_attrs->get_host()[vertex] & 0b10) {
          h_results[h_results_size] = getResult(vertex, i * multi_ss_size);
          h_results_size++;
        }
        h_buffer[h_buffer_size] = vertex;
        h_buffer_size++;
      }
    }
    if (i == 0) {
      for (int j = 0; j < graph.startActiveNum; j++) {
        int vertex = graph.start_active_nodes->get_host()[j];
        if (graph.symbol_sets->get_host()[vertex].test(first_symbol)) {
          if (graph.node_attrs->get_host()[vertex] & 0b10) {
            h_results[h_results_size] = getResult(vertex, i * multi_ss_size);
            h_results_size++;
          }
          h_buffer[h_buffer_size] = vertex;
          h_buffer_size++;
        }
      }
    }

    CHECK_ERROR(cudaMemcpy((void *)(d_buffer + i * buffer_capacity_per_block),
                           h_buffer, sizeof(int) * buffer_capacity_per_block,
                           cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy((void *)(d_buffer_size + i), &h_buffer_size,
                           sizeof(int), cudaMemcpyHostToDevice));
  }

  CHECK_ERROR(
      cudaMalloc((void **)&d_results, sizeof(uint64_t) * results_capacity));
  CHECK_ERROR(
      cudaMalloc((void **)&d_results_size, sizeof(unsigned long long int)));
  CHECK_ERROR(
      cudaMemset((void *)d_results, 0, sizeof(uint64_t) * results_capacity));
  CHECK_ERROR(
      cudaMemset((void *)d_results_size, 0, sizeof(unsigned long long int)));

  CHECK_ERROR(cudaMemcpy((void *)d_results, h_results,
                         sizeof(uint64_t) * results_capacity,
                         cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy((void *)d_results_size, &h_results_size,
                         sizeof(unsigned long long int),
                         cudaMemcpyHostToDevice));
  delete[] h_buffer;
  delete[] h_results;
}

__host__ void BlockingBuffer::init_nfagroups(Array2<uint8_t> *input_stream,
                                             int input_total_size,
                                             int input_num, int multi_ss_size,
                                             std::vector<Graph *> gs,
                                             ngap_option *plo) {

  bool isDup = false;
  if (plo->duplicate_input_stream > 1 &&
      (plo->split_chunk_size == -1 ||
       plo->split_chunk_size == plo->input_len)) {
    isDup = true;
  }

  report_off = plo->report_off;

  int block_num = plo->group_num * input_num;
  buffer_capacity = DATA_BUFFER_SIZE_FRONTIER;
  results_capacity = plo->output_capacity;
  printf("results_capacity = %llu \n", results_capacity);
  buffer_capacity_per_block = buffer_capacity / block_num;
  unique = plo->unique;
  CHECK_ERROR(cudaMalloc((void **)&d_buffer, sizeof(int) * buffer_capacity));
  CHECK_ERROR(cudaMalloc((void **)&d_buffer_size, sizeof(int) * block_num));
  CHECK_ERROR(
      cudaMalloc((void **)&d_buffer_idx, sizeof(int) * buffer_capacity));
  motivate_worklist_length = plo->motivate_worklist_length;

  if (plo->motivate_worklist_length) {
    CHECK_ERROR(cudaMalloc((void **)&d_froniter_length, sizeof(int) * 2000000));
    CHECK_ERROR(
        cudaMemset((void *)d_froniter_length, 0, sizeof(int) * 2000000));
    CHECK_ERROR(cudaMalloc((void **)&d_froniter_end, sizeof(int)));
    CHECK_ERROR(cudaMemset((void *)d_froniter_end, 0, sizeof(int)));
  }
  num_seg = input_num;
  group_num = plo->group_num;

  // characterize divergence
  // CHECK_ERROR(cudaMalloc((void **)&d_froniter_divergence_end, sizeof(int)));
  // CHECK_ERROR(cudaMalloc((void **)&d_froniter_divergence_advance, sizeof(int)
  // * 2000000)); CHECK_ERROR(cudaMalloc((void **)&d_froniter_divergence_filter,
  // sizeof(int) * 2000000)); CHECK_ERROR(cudaMalloc((void
  // **)&d_froniter_workload_end, sizeof(int))); CHECK_ERROR(cudaMalloc((void
  // **)&d_froniter_workload, sizeof(int) * 200000000));

  // CHECK_ERROR(cudaMemset((void *)d_froniter_divergence_end, 0, sizeof(int)));
  // CHECK_ERROR(cudaMemset((void *)d_froniter_workload_end, 0, sizeof(int)));
  // CHECK_ERROR(cudaMemset((void *)d_froniter_divergence_advance, 0,
  // sizeof(int)* 2000000)); CHECK_ERROR(cudaMemset((void
  // *)d_froniter_divergence_filter, 0, sizeof(int)* 2000000));
  // CHECK_ERROR(cudaMemset((void *)d_froniter_workload, 0, sizeof(int)*
  // 200000000));

  auto getResult = [](uint32_t node, uint32_t index) -> uint64_t {
    uint64_t r = 0;
    r = (uint32_t)node;
    r = r << 32;
    r = r | (uint32_t)index;
    return r;
  };

  int *h_buffer = new int[buffer_capacity_per_block];
  unsigned long long int h_results_size = 0;
  uint64_t *h_results = new uint64_t[results_capacity];
  memset(h_results, 0, sizeof(uint64_t) * results_capacity);
  for (int i = 0; i < block_num; i++) {
    int nfa_index = i % group_num;
    int input_index = i / group_num;

    uint8_t first_symbol =
        input_stream->get_host()[input_index * multi_ss_size];
    int h_buffer_size = 0;

    memset(h_buffer, 0, sizeof(int) * buffer_capacity_per_block);
    for (int j = 0; j < gs[nfa_index]->alwaysActiveNum; j++) {
      int vertex = gs[nfa_index]->always_active_nodes->get_host()[j];
      if (gs[nfa_index]->symbol_sets->get_host()[vertex].test(first_symbol)) {
        if (gs[nfa_index]->node_attrs->get_host()[vertex] & 0b10) {
          h_results[h_results_size] =
              getResult(vertex, input_index * multi_ss_size);
          h_results_size++;
        }
        h_buffer[h_buffer_size] = vertex;
        h_buffer_size++;
      }
    }
    if (input_index == 0 || isDup) {
      for (int j = 0; j < gs[nfa_index]->startActiveNum; j++) {
        int vertex = gs[nfa_index]->start_active_nodes->get_host()[j];
        if (gs[nfa_index]->symbol_sets->get_host()[vertex].test(first_symbol)) {
          if (gs[nfa_index]->node_attrs->get_host()[vertex] & 0b10) {
            h_results[h_results_size] =
                getResult(vertex, input_index * multi_ss_size);
            h_results_size++;
          }
          h_buffer[h_buffer_size] = vertex;
          h_buffer_size++;
        }
      }
    }

    CHECK_ERROR(cudaMemcpy((void *)(d_buffer + i * buffer_capacity_per_block),
                           h_buffer, sizeof(int) * buffer_capacity_per_block,
                           cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy((void *)(d_buffer_size + i), &h_buffer_size,
                           sizeof(int), cudaMemcpyHostToDevice));
  }

  CHECK_ERROR(
      cudaMalloc((void **)&d_results_size, sizeof(unsigned long long int)));
  CHECK_ERROR(
      cudaMemset((void *)d_results_size, 0, sizeof(unsigned long long int)));
  CHECK_ERROR(cudaMemcpy((void *)d_results_size, &h_results_size,
                         sizeof(unsigned long long int),
                         cudaMemcpyHostToDevice));
  if (plo->use_uvm) {
    CHECK_ERROR(cudaMallocManaged((void **)&d_results,
                                  sizeof(uint64_t) * results_capacity));
    memcpy(d_results, h_results, h_results_size * sizeof(int));
  } else {
    CHECK_ERROR(
        cudaMalloc((void **)&d_results, sizeof(uint64_t) * results_capacity));
    CHECK_ERROR(
        cudaMemset((void *)d_results, 0, sizeof(uint64_t) * results_capacity));
    CHECK_ERROR(cudaMemcpy((void *)d_results, h_results,
                           sizeof(uint64_t) * h_results_size,
                           cudaMemcpyHostToDevice));
  }

  delete[] h_buffer;
  delete[] h_results;
}

__host__ void BlockingBuffer::release() {
  cudaFree((void *)d_buffer);
  cudaFree((void *)d_buffer_size);
  cudaFree((void *)d_results);
  cudaFree((void *)d_results_size);

  if (motivate_worklist_length) {
    cudaFree((void *)d_froniter_length);
    cudaFree((void *)d_froniter_end);
  }
}

__host__ void NonBlockingBuffer::init(Array2<uint8_t> *input_stream,
                                      int input_total_size, int input_num,
                                      int multi_ss_size, Graph &graph,
                                      ngap_option *plo) {
  unique_frequency = plo->unique_frequency;
  buffer_capacity = DATA_BUFFER_SIZE;
  results_capacity = plo->output_capacity;
  report_off = plo->report_off;
  printf("results_capacity = %llu \n", results_capacity);
  buffer_capacity_per_block = buffer_capacity / ((input_num + 1) / 2);
  printf("buffer_capacity_per_block = %lld\n", buffer_capacity_per_block);
  if (plo->data_buffer_fetch_size > buffer_capacity_per_block) {
    printf("data_buffer_fetch_size is larger than buffer_capacity_per_block. "
           "Set it to %lld\n",
           buffer_capacity_per_block);
    data_buffer_fetch_size = buffer_capacity_per_block;
  } else {
    data_buffer_fetch_size = plo->data_buffer_fetch_size;
  }
  add_aas_start = plo->add_aas_start;
  active_threshold = plo->active_threshold;
  unique = plo->unique;
  if (add_aas_start > multi_ss_size)
    add_aas_start = multi_ss_size; // add all aas
  add_aas_interval = plo->add_aas_interval;

  // CHECK_ERROR(cudaMalloc((void **)&d_fakeiter,
  //                        sizeof(int) * add_aas_interval * input_num));
  // CHECK_ERROR(cudaMalloc((void **)&d_fakeiter_size, sizeof(int) *
  // input_num)); CHECK_ERROR(cudaMemset((void *)d_fakeiter, 0,
  //                        sizeof(int) * add_aas_interval * input_num));
  // CHECK_ERROR(cudaMemset((void *)d_fakeiter_size, 0, sizeof(int) *
  // input_num));

  CHECK_ERROR(cudaMalloc((void **)&d_fakeiter2,
                         sizeof(int) * add_aas_interval * input_num));
  CHECK_ERROR(cudaMalloc((void **)&d_fakeiter_size2, sizeof(int) * input_num));
  CHECK_ERROR(cudaMemset((void *)d_fakeiter2, 0,
                         sizeof(int) * add_aas_interval * input_num));
  CHECK_ERROR(cudaMemset((void *)d_fakeiter_size2, 0, sizeof(int) * input_num));
  CHECK_ERROR(cudaMalloc((void **)&cutoffnum, sizeof(int) * input_num));
  CHECK_ERROR(cudaMemset((void *)cutoffnum, 0, sizeof(int) * input_num));

  CHECK_ERROR(cudaMalloc((void **)&d_buffer, sizeof(int) * buffer_capacity));
  CHECK_ERROR(
      cudaMalloc((void **)&d_buffer_idx, sizeof(int) * buffer_capacity));
  CHECK_ERROR(cudaMalloc((void **)&d_buffer2, sizeof(int) * buffer_capacity));
  CHECK_ERROR(
      cudaMalloc((void **)&d_buffer_idx2, sizeof(int) * buffer_capacity));
  //   CHECK_ERROR(cudaMalloc((void **)&d_buffer_size, sizeof(int) *
  //   input_num));
  // CHECK_ERROR(cudaMalloc((void **)&d_buffer_test, sizeof(int) *
  // buffer_capacity)); CHECK_ERROR(
  //     cudaMalloc((void **)&d_buffer_idx_test, sizeof(int) *
  //     buffer_capacity));
  // CHECK_ERROR(cudaMalloc((void **)&d_buffer_end_tmp_test, sizeof(uint) *
  // input_num));

  CHECK_ERROR(cudaMalloc((void **)&d_buffer_start, sizeof(uint) * input_num));
  CHECK_ERROR(cudaMalloc((void **)&d_buffer_end, sizeof(uint) * input_num));
  CHECK_ERROR(cudaMalloc((void **)&d_buffer_end_tmp, sizeof(uint) * input_num));
  CHECK_ERROR(cudaMemset((void *)d_buffer_start, 0, sizeof(uint) * input_num));
  CHECK_ERROR(cudaMemset((void *)d_buffer_end, 0, sizeof(uint) * input_num));
  CHECK_ERROR(
      cudaMemset((void *)d_buffer_end_tmp, 0, sizeof(uint) * input_num));
  CHECK_ERROR(
      cudaMalloc((void **)&d_newest_idx, sizeof(int) * (input_num + 1)));
  CHECK_ERROR(
      cudaMemset((void *)d_newest_idx, 0, sizeof(int) * (input_num + 1)));

  auto getResult = [](uint32_t node, uint32_t index) -> uint64_t {
    uint64_t r = 0;
    r = (uint32_t)node;
    r = r << 32;
    r = r | (uint32_t)index;
    return r;
  };

  int *h_buffer = new int[buffer_capacity_per_block];
  int *h_buffer_idx = new int[buffer_capacity_per_block];
  unsigned long long int h_results_size = 0;
  uint64_t *h_results = new uint64_t[results_capacity];
  uint32_t *h_results_v = new uint32_t[results_capacity];
  uint32_t *h_results_i = new uint32_t[results_capacity];
  int pc_cnt = 0, pc_total = 0;
  for (int i = 0; i < input_num; i++) {
    int h_buffer_size = 0;
    memset(h_buffer, 0, sizeof(int) * buffer_capacity_per_block);
    memset(h_buffer_idx, 0, sizeof(int) * buffer_capacity_per_block);
    // Fake nodes are use for activate the always active node for one iter
    if (graph.alwaysActiveNum > 0) {
      for (int fake = 0; fake < add_aas_start; fake++) {
        // TODO(tge): if length is 0, don't add it to buffer.
        if (precompute_depth > 0 &&
            (i * multi_ss_size + fake) < input_total_size) {
          int pc_length =
              h_pts[precompute_depth - 1].vertices_length
                  [input_stream->get_host()[i * multi_ss_size + fake]];
          if (pc_length == 0)
            pc_cnt++;
          // else
          //   printf("%d, ", pc_length);
          pc_total++;
        }
        h_buffer[h_buffer_size] = -1;
        h_buffer_idx[h_buffer_size] = i * multi_ss_size + fake;
        h_buffer_size++;
      }
    }

    // start nodes
    if (i == 0) {
      uint8_t first_symbol = input_stream->get_host()[i * multi_ss_size];
      for (int j = 0; j < graph.startActiveNum; j++) {
        int vertex = graph.start_active_nodes->get_host()[j];
        if (graph.symbol_sets->get_host()[vertex].test(first_symbol)) {
          if (graph.node_attrs->get_host()[vertex] & 0b10) {
            h_results_v[h_results_size] = vertex;
            h_results_i[h_results_size] = i * multi_ss_size;
            h_results[h_results_size] = getResult(vertex, i * multi_ss_size);
            h_results_size++;
          }
          h_buffer[h_buffer_size] = vertex;
          h_buffer_idx[h_buffer_size] = i * multi_ss_size;
          h_buffer_size++;
        }
      }
    }

    // if (i == 0) {
    //   for (int j = 0; j < graph.startActiveNum; j++) {
    //     int vertex = graph.start_active_nodes->get_host()[j];
    //     h_buffer[h_buffer_size] = vertex;
    //     h_buffer_idx[h_buffer_size] = i * multi_ss_size;
    //     h_buffer_size++;
    //   }
    // }
    if (i < input_num / 2) {
      CHECK_ERROR(cudaMemcpy((void *)(d_buffer + i * buffer_capacity_per_block),
                             h_buffer, sizeof(int) * buffer_capacity_per_block,
                             cudaMemcpyHostToDevice));
      CHECK_ERROR(cudaMemcpy(
          (void *)(d_buffer_idx + i * buffer_capacity_per_block), h_buffer_idx,
          sizeof(int) * buffer_capacity_per_block, cudaMemcpyHostToDevice));
    } else {
      CHECK_ERROR(cudaMemcpy(
          (void *)(d_buffer2 + (i - input_num / 2) * buffer_capacity_per_block),
          h_buffer, sizeof(int) * buffer_capacity_per_block,
          cudaMemcpyHostToDevice));
      CHECK_ERROR(
          cudaMemcpy((void *)(d_buffer_idx2 +
                              (i - input_num / 2) * buffer_capacity_per_block),
                     h_buffer_idx, sizeof(int) * buffer_capacity_per_block,
                     cudaMemcpyHostToDevice));
    }
    CHECK_ERROR(cudaMemcpy((void *)(d_buffer_end + i), &h_buffer_size,
                           sizeof(int), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy((void *)(d_buffer_end_tmp + i), &h_buffer_size,
                           sizeof(int), cudaMemcpyHostToDevice));
    int tmp_newest_idx = i * multi_ss_size + add_aas_start;
    CHECK_ERROR(cudaMemcpy((void *)(d_newest_idx + i), &tmp_newest_idx,
                           sizeof(int), cudaMemcpyHostToDevice));
  }
  if (precompute_depth > 0) {
    printf("Precompute preview: zero = %d, total = %d\n", pc_cnt, pc_total);
  }

  CHECK_ERROR(
      cudaMalloc((void **)&d_results, sizeof(uint64_t) * results_capacity));
  CHECK_ERROR(
      cudaMalloc((void **)&d_results_size, sizeof(unsigned long long int)));
  CHECK_ERROR(
      cudaMemset((void *)d_results, 0, sizeof(uint64_t) * results_capacity));
  CHECK_ERROR(
      cudaMalloc((void **)&d_results_v, sizeof(uint32_t) * results_capacity));
  CHECK_ERROR(
      cudaMalloc((void **)&d_results_i, sizeof(uint32_t) * results_capacity));
  CHECK_ERROR(
      cudaMemset((void *)d_results_v, 0, sizeof(uint32_t) * results_capacity));
  CHECK_ERROR(
      cudaMemset((void *)d_results_i, 0, sizeof(uint32_t) * results_capacity));

  CHECK_ERROR(
      cudaMemset((void *)d_results_size, 0, sizeof(unsigned long long int)));
  CHECK_ERROR(
      cudaMalloc((void **)&d_symbol_table, sizeof(int) * input_total_size));
  CHECK_ERROR(
      cudaMemset((void *)d_symbol_table, 0, sizeof(int) * input_total_size));
  CHECK_ERROR(cudaMemcpy((void *)d_results, h_results,
                         sizeof(uint64_t) * results_capacity,
                         cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy((void *)d_results_v, h_results_v,
                         sizeof(uint32_t) * results_capacity,
                         cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy((void *)d_results_i, h_results_i,
                         sizeof(uint32_t) * results_capacity,
                         cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy((void *)d_results_size, &h_results_size,
                         sizeof(unsigned long long int),
                         cudaMemcpyHostToDevice));

  // CHECK_ERROR(cudaMalloc((void **)&preresult, sizeof(int) *
  // buffer_capacity)); CHECK_ERROR(
  //     cudaMalloc((void **)&preresult_iter, sizeof(int) * buffer_capacity));
  // CHECK_ERROR(cudaMalloc((void **)&preresult_size, sizeof(int)));
  // CHECK_ERROR(cudaMalloc((void **)&preresult_end, sizeof(int)));
  // CHECK_ERROR(cudaMemset((void *)preresult, 0, sizeof(int) *
  // buffer_capacity)); CHECK_ERROR(
  //     cudaMemset((void *)preresult_iter, 0, sizeof(int) * buffer_capacity));
  // CHECK_ERROR(cudaMemset((void *)preresult_size, 0, sizeof(int)));
  // CHECK_ERROR(cudaMemset((void *)preresult_end, 0, sizeof(int)));

  delete[] h_buffer;
  delete[] h_results;
  delete[] h_results_v;
  delete[] h_results_i;
  delete[] h_buffer_idx;
}

__host__ void NonBlockingBuffer::init_nfagroups(
    Array2<uint8_t> *input_stream, int input_total_size, int input_num,
    int multi_ss_size, std::vector<Graph *> gs, ngap_option *plo) {

  bool isDup = false;
  if (plo->duplicate_input_stream > 1 &&
      (plo->split_chunk_size == -1 ||
       plo->split_chunk_size == plo->input_len)) {
    isDup = true;
  }

  report_off = plo->report_off;

  if (plo->motivate_worklist_length) {
    CHECK_ERROR(
        cudaMalloc((void **)&d_froniter_length, sizeof(int) * 1000000000));
    CHECK_ERROR(
        cudaMemset((void *)d_froniter_length, 0, sizeof(int) * 1000000000));
    CHECK_ERROR(cudaMalloc((void **)&d_froniter_end, sizeof(int)));
    CHECK_ERROR(cudaMemset((void *)d_froniter_end, 0, sizeof(int)));
  }

  unique_frequency = (plo->unique_frequency / 32) * 32;
  int block_num = plo->group_num * input_num;
  if (plo->use_uvm)
    buffer_capacity = DATA_BUFFER_SIZE * 2.1;
  else
    buffer_capacity = DATA_BUFFER_SIZE;
  results_capacity = plo->output_capacity;
  printf("results_capacity = %llu \n", results_capacity);
  buffer_capacity_per_block = buffer_capacity / ((block_num + 1) / 2);
  printf("buffer_capacity_per_block = %lld %d\n", buffer_capacity_per_block,
         block_num);
  if (plo->data_buffer_fetch_size > buffer_capacity_per_block) {
    printf("data_buffer_fetch_size is larger than buffer_capacity_per_block. "
           "Set it to %lld\n",
           buffer_capacity_per_block);
    data_buffer_fetch_size = buffer_capacity_per_block;
  } else {
    data_buffer_fetch_size = plo->data_buffer_fetch_size;
  }
  active_threshold = plo->active_threshold;
  unique = plo->unique;
  add_aas_start = plo->add_aas_start;
  if (add_aas_start > multi_ss_size)
    add_aas_start = multi_ss_size; // add all aas
  add_aas_interval = plo->add_aas_interval;
  if (add_aas_interval > multi_ss_size)
    add_aas_interval = multi_ss_size; // add all aas

  num_seg = input_num;
  group_num = plo->group_num;

  // CHECK_ERROR(cudaMalloc((void **)&d_fakeiter,
  //                        sizeof(int) * add_aas_interval * block_num));
  // CHECK_ERROR(cudaMalloc((void **)&d_fakeiter_size, sizeof(int) *
  // block_num)); CHECK_ERROR(cudaMemset((void *)d_fakeiter, 0,
  //                        sizeof(int) * add_aas_interval * block_num));
  // CHECK_ERROR(cudaMemset((void *)d_fakeiter_size, 0, sizeof(int) *
  // block_num));

  d_fakeiter_capacity = std::max(add_aas_interval, add_aas_start) + 1;
  d_fakeiter_capacity = std::min(d_fakeiter_capacity, data_buffer_fetch_size);
  CHECK_ERROR(cudaMalloc((void **)&d_fakeiter2,
                         sizeof(int) * d_fakeiter_capacity * block_num));
  CHECK_ERROR(cudaMalloc((void **)&d_fakeiter_size2, sizeof(int) * block_num));
  CHECK_ERROR(cudaMemset((void *)d_fakeiter2, 0,
                         sizeof(int) * d_fakeiter_capacity * block_num));
  CHECK_ERROR(cudaMemset((void *)d_fakeiter_size2, 0, sizeof(int) * block_num));
  CHECK_ERROR(cudaMalloc((void **)&cutoffnum, sizeof(int) * block_num));
  CHECK_ERROR(cudaMemset((void *)cutoffnum, 0, sizeof(int) * block_num));

  if (plo->use_uvm) {
    printf("Use UVM\n");
    CHECK_ERROR(
        cudaMallocManaged((void **)&d_buffer, sizeof(int) * buffer_capacity));
    CHECK_ERROR(cudaMallocManaged((void **)&d_buffer_idx,
                                  sizeof(int) * buffer_capacity));
    CHECK_ERROR(
        cudaMallocManaged((void **)&d_buffer2, sizeof(int) * buffer_capacity));
    CHECK_ERROR(cudaMallocManaged((void **)&d_buffer_idx2,
                                  sizeof(int) * buffer_capacity));
  } else {
    CHECK_ERROR(cudaMalloc((void **)&d_buffer, sizeof(int) * buffer_capacity));
    CHECK_ERROR(
        cudaMalloc((void **)&d_buffer_idx, sizeof(int) * buffer_capacity));
    CHECK_ERROR(cudaMalloc((void **)&d_buffer2, sizeof(int) * buffer_capacity));
    CHECK_ERROR(
        cudaMalloc((void **)&d_buffer_idx2, sizeof(int) * buffer_capacity));
  }

  //   CHECK_ERROR(cudaMalloc((void **)&d_buffer_size, sizeof(int) *
  //   block_num));
  // CHECK_ERROR(cudaMalloc((void **)&d_buffer_test, sizeof(int) *
  // buffer_capacity)); CHECK_ERROR(
  //     cudaMalloc((void **)&d_buffer_idx_test, sizeof(int) *
  //     buffer_capacity));
  // CHECK_ERROR(cudaMalloc((void **)&d_buffer_end_tmp_test, sizeof(uint) *
  // block_num));

  CHECK_ERROR(cudaMalloc((void **)&d_buffer_start, sizeof(uint) * block_num));
  CHECK_ERROR(cudaMalloc((void **)&d_buffer_end, sizeof(uint) * block_num));
  CHECK_ERROR(cudaMalloc((void **)&d_buffer_end_tmp, sizeof(uint) * block_num));
  CHECK_ERROR(cudaMemset((void *)d_buffer_start, 0, sizeof(uint) * block_num));
  CHECK_ERROR(cudaMemset((void *)d_buffer_end, 0, sizeof(uint) * block_num));
  CHECK_ERROR(
      cudaMemset((void *)d_buffer_end_tmp, 0, sizeof(uint) * block_num));
  CHECK_ERROR(
      cudaMalloc((void **)&d_newest_idx, sizeof(int) * (block_num + 1)));
  CHECK_ERROR(
      cudaMemset((void *)d_newest_idx, 0, sizeof(int) * (block_num + 1)));

  // auto getResult = [](uint32_t node, uint32_t index) -> uint64_t {
  //   uint64_t r = 0;
  //   r = (uint32_t)node;
  //   r = r << 32;
  //   r = r | (uint32_t)index;
  //   return r;
  // };

  int *h_buffer = new int[buffer_capacity_per_block];
  int *h_buffer_idx = new int[buffer_capacity_per_block];
  unsigned long long int h_results_size = 0;
  // uint64_t *h_results = new uint64_t[results_capacity];
  uint32_t *h_results_v = new uint32_t[results_capacity];
  uint32_t *h_results_i = new uint32_t[results_capacity];
  for (int i = 0; i < block_num; i++) {
    int nfa_index = i % group_num;
    int input_index = i / group_num;

    uint h_buffer_size = 0;
    memset(h_buffer, 0, sizeof(int) * buffer_capacity_per_block);
    memset(h_buffer_idx, 0, sizeof(int) * buffer_capacity_per_block);
    // Always active nodes
    // Fake nodes are use for activate the always active node for one iter
    if (gs[nfa_index]->alwaysActiveNum > 0) {
      for (int fake = 0; fake < add_aas_start; fake++) {
        // TODO(tge): if length is 0, don't add it to buffer.
        h_buffer[h_buffer_size] = -1;
        h_buffer_idx[h_buffer_size] = input_index * multi_ss_size + fake;
        h_buffer_size++;
      }
    }

    // Start nodes
    if (input_index == 0 || isDup) {
      uint8_t first_symbol =
          input_stream->get_host()[input_index * multi_ss_size];
      for (int j = 0; j < gs[nfa_index]->startActiveNum; j++) {
        int vertex = gs[nfa_index]->start_active_nodes->get_host()[j];
        if (gs[nfa_index]->symbol_sets->get_host()[vertex].test(first_symbol)) {
          if (gs[nfa_index]->node_attrs->get_host()[vertex] & 0b10) {
            h_results_v[h_results_size] = vertex;
            h_results_i[h_results_size] = input_index * multi_ss_size;
            h_results_size++;
          }
          h_buffer[h_buffer_size] = vertex;
          h_buffer_idx[h_buffer_size] = input_index * multi_ss_size;
          h_buffer_size++;
        }
      }
    }

    // if (i == 0) {
    //   for (int j = 0; j < graph.startActiveNum; j++) {
    //     int vertex = graph.start_active_nodes->get_host()[j];
    //     h_buffer[h_buffer_size] = vertex;
    //     h_buffer_idx[h_buffer_size] = i * multi_ss_size;
    //     h_buffer_size++;
    //   }
    // }
    if (i < block_num / 2) {
      if (plo->use_uvm) {
        memcpy((d_buffer + i * buffer_capacity_per_block), h_buffer,
               sizeof(int) * buffer_capacity_per_block);
        memcpy((d_buffer_idx + i * buffer_capacity_per_block), h_buffer_idx,
               sizeof(int) * buffer_capacity_per_block);
      } else {
        CHECK_ERROR(cudaMemcpy(
            (void *)(d_buffer + i * buffer_capacity_per_block), h_buffer,
            sizeof(int) * buffer_capacity_per_block, cudaMemcpyHostToDevice));
        CHECK_ERROR(
            cudaMemcpy((void *)(d_buffer_idx + i * buffer_capacity_per_block),
                       h_buffer_idx, sizeof(int) * buffer_capacity_per_block,
                       cudaMemcpyHostToDevice));
      }

    } else {
      if (plo->use_uvm) {
        memcpy((d_buffer2 + (i - block_num / 2) * buffer_capacity_per_block),
               h_buffer, sizeof(int) * buffer_capacity_per_block);

        memcpy(
            (d_buffer_idx2 + (i - block_num / 2) * buffer_capacity_per_block),
            h_buffer_idx, sizeof(int) * buffer_capacity_per_block);
      } else {
        CHECK_ERROR(
            cudaMemcpy((void *)(d_buffer2 + (i - block_num / 2) *
                                                buffer_capacity_per_block),
                       h_buffer, sizeof(int) * buffer_capacity_per_block,
                       cudaMemcpyHostToDevice));
        CHECK_ERROR(
            cudaMemcpy((void *)(d_buffer_idx2 + (i - block_num / 2) *
                                                    buffer_capacity_per_block),
                       h_buffer_idx, sizeof(int) * buffer_capacity_per_block,
                       cudaMemcpyHostToDevice));
      }
    }
    CHECK_ERROR(cudaMemcpy((void *)(d_buffer_end + i), &h_buffer_size,
                           sizeof(uint), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy((void *)(d_buffer_end_tmp + i), &h_buffer_size,
                           sizeof(uint), cudaMemcpyHostToDevice));
    int tmp_newest_idx = input_index * multi_ss_size + add_aas_start;
    CHECK_ERROR(cudaMemcpy((void *)(d_newest_idx + i), &tmp_newest_idx,
                           sizeof(uint), cudaMemcpyHostToDevice));
  }

  // CHECK_ERROR(
  //     cudaMalloc((void **)&d_results, sizeof(uint64_t) * results_capacity));
  // CHECK_ERROR(
  //     cudaMemset((void *)d_results, 0, sizeof(uint64_t) * results_capacity));
  CHECK_ERROR(
      cudaMalloc((void **)&d_results_size, sizeof(unsigned long long int)));
  CHECK_ERROR(
      cudaMemset((void *)d_results_size, 0, sizeof(unsigned long long int)));
  CHECK_ERROR(cudaMemcpy((void *)d_results_size, &h_results_size,
                         sizeof(unsigned long long int),
                         cudaMemcpyHostToDevice));
  if (plo->use_uvm) {
    CHECK_ERROR(cudaMallocManaged((void **)&d_results_v,
                                  sizeof(uint32_t) * results_capacity));
    CHECK_ERROR(cudaMallocManaged((void **)&d_results_i,
                                  sizeof(uint32_t) * results_capacity));
    memcpy(d_results_v, h_results_v, h_results_size * sizeof(int));
    memcpy(d_results_i, h_results_i, h_results_size * sizeof(int));
  } else {
    CHECK_ERROR(
        cudaMalloc((void **)&d_results_v, sizeof(uint32_t) * results_capacity));
    CHECK_ERROR(
        cudaMalloc((void **)&d_results_i, sizeof(uint32_t) * results_capacity));
    CHECK_ERROR(cudaMemset((void *)d_results_v, 0,
                           sizeof(uint32_t) * results_capacity));
    CHECK_ERROR(cudaMemset((void *)d_results_i, 0,
                           sizeof(uint32_t) * results_capacity));
    CHECK_ERROR(cudaMemcpy((void *)d_results_v, h_results_v,
                           sizeof(uint32_t) * results_capacity,
                           cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy((void *)d_results_i, h_results_i,
                           sizeof(uint32_t) * results_capacity,
                           cudaMemcpyHostToDevice));
  }

  CHECK_ERROR(cudaMalloc((void **)&d_symbol_table,
                         sizeof(int) * input_total_size * group_num));
  CHECK_ERROR(cudaMemset((void *)d_symbol_table, 0,
                         sizeof(int) * input_total_size * group_num));

  // CHECK_ERROR(cudaMalloc((void **)&preresult, sizeof(int) *
  // buffer_capacity)); CHECK_ERROR(
  //     cudaMalloc((void **)&preresult_iter, sizeof(int) * buffer_capacity));
  // CHECK_ERROR(cudaMalloc((void **)&preresult_size, sizeof(int)));
  // CHECK_ERROR(cudaMalloc((void **)&preresult_end, sizeof(int)));
  // CHECK_ERROR(cudaMemset((void *)preresult, 0, sizeof(int) *
  // buffer_capacity)); CHECK_ERROR(
  //     cudaMemset((void *)preresult_iter, 0, sizeof(int) * buffer_capacity));
  // CHECK_ERROR(cudaMemset((void *)preresult_size, 0, sizeof(int)));
  // CHECK_ERROR(cudaMemset((void *)preresult_end, 0, sizeof(int)));

  delete[] h_buffer;
  // delete[] h_results;
  delete[] h_results_v;
  delete[] h_results_i;
  delete[] h_buffer_idx;
}

__host__ void NonBlockingBuffer::release(bool isGroup) {
  for (int i = 0; i < precompute_depth; i++) {
    h_pts[i].releaseDevice();
    h_pts[i].releaseHost();
  }
  if (precompute_depth > 0)
    delete[] h_pts;

  if (!isGroup) {
    CHECK_ERROR(cudaFree((void *)d_pts));
  }
  CHECK_ERROR(cudaFree((void *)d_buffer));
  CHECK_ERROR(cudaFree((void *)d_buffer2));
  CHECK_ERROR(cudaFree((void *)d_buffer_idx));
  CHECK_ERROR(cudaFree((void *)d_buffer_idx2));
  // CHECK_ERROR(cudaFree((void *)d_results));
  CHECK_ERROR(cudaFree((void *)d_results_size));
  CHECK_ERROR(cudaFree((void *)d_results_v));
  CHECK_ERROR(cudaFree((void *)d_results_i));
  CHECK_ERROR(cudaFree((void *)d_newest_idx));
  CHECK_ERROR(cudaFree((void *)d_fakeiter2));
  CHECK_ERROR(cudaFree((void *)d_fakeiter_size2));
  CHECK_ERROR(cudaFree((void *)cutoffnum));
  CHECK_ERROR(cudaFree((void *)d_buffer_start));
  CHECK_ERROR(cudaFree((void *)d_buffer_end));
  CHECK_ERROR(cudaFree((void *)d_buffer_end_tmp));
  CHECK_ERROR(cudaFree((void *)d_symbol_table));
}

__host__ void NonBlockingBuffer::reset(Array2<uint8_t> *input_stream,
                                       int input_total_size, int multi_ss_size,
                                       int group_num, std::vector<Graph *> gs,
                                       ngap_option *plo) {
  bool isDup = false;
  if (plo->duplicate_input_stream > 1 &&
      (plo->split_chunk_size == -1 ||
       plo->split_chunk_size == plo->input_len)) {
    isDup = true;
  }

  int block_num = group_num * num_seg;
  CHECK_ERROR(cudaMemset((void *)d_fakeiter_size2, 0, sizeof(int) * block_num));
  CHECK_ERROR(cudaMemset((void *)d_buffer_start, 0, sizeof(uint) * block_num));

  int *h_buffer = new int[buffer_capacity_per_block];
  int *h_buffer_idx = new int[buffer_capacity_per_block];
  unsigned long long int h_results_size = 0;
  uint32_t *h_results_v = new uint32_t[results_capacity];
  uint32_t *h_results_i = new uint32_t[results_capacity];
  for (int i = 0; i < block_num; i++) {
    int nfa_index = i % group_num;
    int input_index = i / group_num;

    uint h_buffer_size = 0;
    memset(h_buffer, 0, sizeof(int) * buffer_capacity_per_block);
    memset(h_buffer_idx, 0, sizeof(int) * buffer_capacity_per_block);
    // Fake nodes are use for activate the always active node for one iter
    if (gs[nfa_index]->alwaysActiveNum > 0) {
      for (int fake = 0; fake < add_aas_start; fake++) {
        // TODO(tge): if length is 0, don't add it to buffer.
        h_buffer[h_buffer_size] = -1;
        h_buffer_idx[h_buffer_size] = input_index * multi_ss_size + fake;
        h_buffer_size++;
      }
    }

    // start nodes
    if (input_index == 0 || isDup) {
      uint8_t first_symbol =
          input_stream->get_host()[input_index * multi_ss_size];
      for (int j = 0; j < gs[nfa_index]->startActiveNum; j++) {
        int vertex = gs[nfa_index]->start_active_nodes->get_host()[j];
        if (gs[nfa_index]->symbol_sets->get_host()[vertex].test(first_symbol)) {
          if (gs[nfa_index]->node_attrs->get_host()[vertex] & 0b10) {
            h_results_v[h_results_size] = vertex;
            h_results_i[h_results_size] = input_index * multi_ss_size;
            h_results_size++;
          }
          h_buffer[h_buffer_size] = vertex;
          h_buffer_idx[h_buffer_size] = input_index * multi_ss_size;
          h_buffer_size++;
        }
      }
    }

    if (i < block_num / 2) {
      CHECK_ERROR(cudaMemcpy((void *)(d_buffer + i * buffer_capacity_per_block),
                             h_buffer, sizeof(int) * buffer_capacity_per_block,
                             cudaMemcpyHostToDevice));
      CHECK_ERROR(cudaMemcpy(
          (void *)(d_buffer_idx + i * buffer_capacity_per_block), h_buffer_idx,
          sizeof(int) * buffer_capacity_per_block, cudaMemcpyHostToDevice));
    } else {
      CHECK_ERROR(cudaMemcpy(
          (void *)(d_buffer2 + (i - block_num / 2) * buffer_capacity_per_block),
          h_buffer, sizeof(int) * buffer_capacity_per_block,
          cudaMemcpyHostToDevice));
      CHECK_ERROR(
          cudaMemcpy((void *)(d_buffer_idx2 +
                              (i - block_num / 2) * buffer_capacity_per_block),
                     h_buffer_idx, sizeof(int) * buffer_capacity_per_block,
                     cudaMemcpyHostToDevice));
    }
    CHECK_ERROR(cudaMemcpy((void *)(d_buffer_end + i), &h_buffer_size,
                           sizeof(uint), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy((void *)(d_buffer_end_tmp + i), &h_buffer_size,
                           sizeof(uint), cudaMemcpyHostToDevice));
    int tmp_newest_idx = input_index * multi_ss_size + add_aas_start;
    CHECK_ERROR(cudaMemcpy((void *)(d_newest_idx + i), &tmp_newest_idx,
                           sizeof(uint), cudaMemcpyHostToDevice));
  }
  CHECK_ERROR(
      cudaMemset((void *)d_results_size, 0, sizeof(unsigned long long int)));
  CHECK_ERROR(cudaMemcpy((void *)d_results_size, &h_results_size,
                         sizeof(unsigned long long int),
                         cudaMemcpyHostToDevice));
  if (plo->use_uvm) {
    memcpy(d_results_v, h_results_v, h_results_size * sizeof(int));
    memcpy(d_results_i, h_results_i, h_results_size * sizeof(int));
  } else {
    CHECK_ERROR(cudaMemset((void *)d_results_v, 0,
                           sizeof(uint32_t) * results_capacity));
    CHECK_ERROR(cudaMemset((void *)d_results_i, 0,
                           sizeof(uint32_t) * results_capacity));
    CHECK_ERROR(cudaMemcpy((void *)d_results_v, h_results_v,
                           sizeof(uint32_t) * results_capacity,
                           cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy((void *)d_results_i, h_results_i,
                           sizeof(uint32_t) * results_capacity,
                           cudaMemcpyHostToDevice));
  }
  CHECK_ERROR(cudaMemset((void *)d_symbol_table, 0,
                         sizeof(int) * input_total_size * group_num));
  delete[] h_buffer;
  // delete[] h_results;
  delete[] h_results_v;
  delete[] h_results_i;
  delete[] h_buffer_idx;
}
