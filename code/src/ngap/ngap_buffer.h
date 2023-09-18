#ifndef NGAP_BUFFER_H_
#define NGAP_BUFFER_H_

#include "graph.h"
#include "my_bitset.h"

#include "ngap_option.h"
#include "precompute_table.h"
#include "utils.h"

// #define DEBUG_PL_FILTER
// #define DEBUG_PL_ADVANCE
// #define DEBUG_SHADOW_BUFFER
// #define DEBUG_PL_FILTER_ITER
#define DEBUG_ITER 10000
// #define DEBUG_MAX_BUFFER_SIZE  // set data_buffer_stream_size large enough
// #define DEBUG_PL_KERNEL_LAUNCH

#define DEBUG_PL_ADVANCE_CONCAT // do not comment it out

// #define DEBUG_FRONTIER_SIZE
// #define MULTI_BLOCKS
#define USE_CSR
// #define USE_PRECOMP_ONCE
// #define USE_PRECOMP_TWICE

#ifndef DATA_BUFFER_SIZE
// #define DATA_BUFFER_SIZE 300000000
#define DATA_BUFFER_SIZE 1000000000LL
#endif

#ifndef DATA_BUFFER_SIZE_FRONTIER
#define DATA_BUFFER_SIZE_FRONTIER 2000000000
#endif

#ifndef RESULTS_SIZE
#define RESULTS_SIZE 80000000
#endif

#define MAX_THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_MP 16

#define BLOCK_SIZE 256

// #define PRINT_INDEX_QUEUE

class BlockingBuffer {
public:
  int buffer_capacity;
  int buffer_capacity_per_block;
  unsigned long long int results_capacity;
  bool unique;

  int *d_buffer;
  int *d_buffer_idx;
  int *d_buffer_size;
  uint64_t *d_results;
  unsigned long long int *d_results_size;

  int *d_froniter_length;
  int *d_froniter_end;

  int *d_froniter_divergence_end;
  int *d_froniter_divergence_advance;
  int *d_froniter_divergence_filter;
  int *d_froniter_workload_end;
  int *d_froniter_workload;

  int group_num;
  int num_seg;

  bool motivate_worklist_length;

  bool report_off;

  __host__ void init(Array2<uint8_t> *input_stream, int input_total_size,
                     int input_num, int multi_ss_size, Graph &graph,
                     ngap_option *plo);
  __host__ void init_nfagroups(Array2<uint8_t> *input_stream,
                               int input_total_size, int input_num,
                               int multi_ss_size, std::vector<Graph *> gs,
                               ngap_option *plo);

  __host__ void release();
};

class NonBlockingBuffer {
public:
  long long int buffer_capacity;
  long long int buffer_capacity_per_block;
  unsigned long long int results_capacity;
  int data_buffer_fetch_size = 64;

  int add_aas_start = 1000;
  int add_aas_interval;

  int active_threshold;

  bool unique;
  int unique_frequency;

  int *d_buffer;
  int *d_buffer_idx;
  int *d_buffer2;
  int *d_buffer_idx2;

  int *d_buffer_test;
  int *d_buffer_idx_test;
  uint *d_buffer_end_tmp_test;

  uint *d_buffer_start;
  uint *d_buffer_end;
  uint *d_buffer_end_tmp;
  uint64_t *d_results;
  uint32_t *d_results_v;
  uint32_t *d_results_i;
  unsigned long long int *d_results_size;
  int *d_symbol_table;
  int *d_newest_idx;

  int *prec_once_offset;
  int *prec_once;
  int *prec_twice_offset;
  int *prec_twice;
  int *prec_once_report_offset;
  int *prec_once_report;
  int *prec_twice_report_offset;
  int *prec_twice_report;

  int *preresult;
  int *preresult_iter;
  int *preresult_size;
  int *preresult_end;

  int *d_fakeiter;
  int *d_fakeiter_size;
  int *d_fakeiter2;
  int *d_fakeiter_size2;
  int d_fakeiter_capacity;
  int *cutoffnum;

  int *d_froniter_length;
  int *d_froniter_end;

  // O3
  PrecTable *h_pts;
  PrecTable *d_pts;
  int precompute_depth = 0;
  int precompute_cutoff;

  int group_num;
  int num_seg;

  bool report_off;

  __host__ void init(Array2<uint8_t> *input_stream, int input_total_size,
                     int input_num, int multi_ss_size, Graph &graph,
                     ngap_option *plo);
  __host__ void init_nfagroups(Array2<uint8_t> *input_stream,
                               int input_total_size, int input_num,
                               int multi_ss_size, std::vector<Graph *> gs,
                               ngap_option *plo);

  __host__ void release(bool isGroup = false);

  __host__ void reset(Array2<uint8_t> *input_stream, int input_total_size,
                      int multi_ss_size, int group_num, std::vector<Graph *> gs,
                      ngap_option *plo);
};

#endif