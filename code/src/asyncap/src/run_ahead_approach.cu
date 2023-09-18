#include "NFA.h"
#include "SymbolStream.h"
#include "abstract_gpunfa.h"
#include "array2.h"
#include "common.h"
#include "run_ahead_approach.h"
#include "run_ahead_kernels.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <set>
#include <vector>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

#include "nfa_utils.h"
#include "report_formatter.h"
#include <unordered_map>

#include <numeric>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

/*#include <thrust/unique.h>
#include <thrust/execution_policy.h>

#include <thrust/device_ptr.h>
*/

#include <thread>

using std::unordered_map;

using namespace cub;

#define RUNAHEAD_SHR_LENGTH_KERNEL_(x, report_on)                              \
  {                                                                            \
  case (x):                                                                    \
    scanning_input_kernel3<ACTIVE_STATE_ARRAY_SIZE, (x), false, (report_on)>   \
        <<<blocksPerGrid, threadsPerBlock, 0>>>(                               \
            list_of_tt_of_nfa[cc_id]->get_dev(), input_stream->get_dev(),      \
            input_stream->size(), start_node_ids_arr2->get_dev(),              \
            start_node_ids_arr2->size(), this->is_report[cc_id]->get_dev(),    \
            arr_of_report_buffer[i]                                            \
                ->intermediate_output_array_offset->get_dev(),                 \
            arr_of_report_buffer[i]->intermediate_output_array_sid->get_dev(), \
            this->tail_of_intermediate_output_array->get_dev() + cc_id,        \
            arr_of_report_buffer[i]->real_output_array->get_dev(),             \
            this->tail_of_real_output_array->get_dev() + cc_id, this->R);      \
    break;                                                                     \
  }

struct CSR_NFA {
public:
  Array2<int> *start_ids;
  Array2<int> *degrees;
  Array2<int> *edge_list;
  Array2<matchset_t> *mss;
  Array2<int> *is_report;
  Array2<int> *start_type;

  CSR_NFA(int V, int E) {
    edge_list = new Array2<int>(E);
    start_ids = new Array2<int>(V);
    degrees = new Array2<int>(V);
    mss = new Array2<matchset_t>(V);
    is_report = new Array2<int>(V);
    start_type = new Array2<int>(V);
  }

  ~CSR_NFA() {
    delete edge_list;
    delete start_ids;
    delete degrees;
    delete mss;
    delete is_report;
    delete start_type;
  }
};

run_ahead_alg::run_ahead_alg(NFA *nfa, int num_streams0)
    : abstract_algorithm(nfa), R(3), print_intermediate_reports(false),
      CAP_OUTPUT_BUFFER_FOR_EACH_EXECUTION_GROUP(4000000), // 4MB
      irhandle_threshold(4), fullsort(false), num_streams(4),
      merge_cc_to_one(0) {
  this->record_ir = 1;
  this->blockDimX = -1;
  this->shrmem_wl = 0;

  num_execution_group = num_streams0;
  num_streams = num_streams0;

  // streams = new cudaStream_t[num_streams];

  // for (int i = 0; i < num_streams; i++) {
  //     // cudaStreamCreate(&streams[i]);
  // 	cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  // }

  this->reorder_nodeids = 0;
}

run_ahead_alg::~run_ahead_alg() {}

void run_ahead_alg::preprocessing() {

  cout << "merge_cc_to_one = " << merge_cc_to_one << endl;
  nfa->mark_cc_id();

  if (merge_cc_to_one == 0) {
    this->ccs = nfa_utils::split_nfa_by_ccs(*nfa);
    nfa_utils::add_fake_start_node_for_ccs(this->ccs);
  } else {
    vector<NFA *> ccs1 = nfa_utils::split_nfa_by_ccs(*nfa);
    nfa_utils::add_fake_start_node_for_ccs(ccs1);

    for (int i = 0; i < ccs1.size(); i += merge_cc_to_one) {
      vector<NFA *> current_merge;
      for (int ii = 0; ii < merge_cc_to_one && i + ii < ccs1.size(); ii++) {
        int idx = i + ii;
        current_merge.push_back(ccs1[idx]);
      }

      this->ccs.push_back(nfa_utils::merge_nfas(current_merge));
    }

    for (int i = 0; i < ccs1.size(); i++) {
      delete ccs1[i];
    }
  }

  auto succ_fail = nfa_utils::limit_out_degree_on_ccs(this->ccs, 4, quit_degree,
                                                      remove_degree);
  cout << "limit4 success = " << succ_fail.first
       << " fail = " << succ_fail.second << endl;

  for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
    auto cc = ccs[cc_id];
    cc->calc_scc();
    cc->topo_sort();
    // cout << "cc_size_original(" << cc_id << ") = " << cc->size() << endl;
  }

  if (reorder_nodeids == 1) {
    cout << "reorder_nodeids " << endl;
    for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
      auto cc = ccs[cc_id];
      ccs[cc_id] =
          nfa_utils::remap_intid_of_nfa(cc, [](const Node *a, const Node *b) {
            if (a->bfs_layer < b->bfs_layer) {
              return true;
            }

            if (a->bfs_layer > b->bfs_layer) {
              return false;
            }

            if (a->is_start()) {
              return true;
            }

            return false;
          });
    }
  }

  for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
    auto cc = ccs[cc_id];

    // cout << "dag_cc(" << cc_id << ") = " << cc->get_dag() << endl;
    // cout << "cc_size_limit4(" << cc_id << ") = " << cc->size() << endl;
    // cout << "has_self_loop_plus_large_matchset(" << cc_id << ") = " <<
    // cc->has_self_loop_plus_large_matchset() << endl;
  }
}

Array2<char> *run_ahead_alg::get_is_report_array(NFA *cc) {
  Array2<char> *res = new Array2<char>(cc->size());
  for (int i = 0; i < cc->size(); i++) {

    auto node = cc->get_node_by_int_id(i);

    char cur = node->is_report() ? 1 : 0;

    // if (node->is_start_always_enabled()) {
    // 	cur |= (1 << 1);
    // }

    // cout << "cur = " << (int)cur << endl;
    res->set(i, cur);
  }
  return res;
}

Array2<uint8_t> *run_ahead_alg::get_array2_of_input_stream0() {
  int total_num_of_symbol = 0;
  for (auto ss : this->symbol_streams) {
    total_num_of_symbol += ss.size();
  }

  Array2<uint8_t> *res =
      new Array2<uint8_t>(total_num_of_symbol, "input_streams_one_array");

  int tt = 0;
  for (auto ss : this->symbol_streams) {
    for (int i = 0; i < ss.size(); i++) {
      res->set(tt++, ss.get_position(i));
    }
  }

  return res;
}

void run_ahead_alg::device_sort_kv(int *d_k, int *d_v, int N) {

  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;

  // is double buffer really needed? ...
  CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                          d_k, d_k, d_v, d_v, N));
  CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                          d_k, d_k, d_v, d_v, N));
}

void run_ahead_alg::print_report_to_file(
    string filename,
    const vector<std::pair<int, match_pair *>> &real_report_for_each_cc) {

  report_formatter rf;
  assert(ccs.size() == real_report_for_each_cc.size());
  for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
    int num_report_cur_cc = real_report_for_each_cc[cc_id].first;
    match_pair *arr = real_report_for_each_cc[cc_id].second;

    if (num_report_cur_cc > 0) {
      assert(arr != NULL);

      for (int i = 0; i < num_report_cur_cc; i++) {
        match_pair mp = arr[i];
        int offset = mp.symbol_offset;
        int sid = mp.state_id;

        auto node = ccs[cc_id]->get_node_by_int_id(sid);
        string str_id = node->original_id;

        report r(offset, str_id, cc_id, 1);
        rf.add_report(r);
      }
    }
  }

  rf.print_to_file(filename, true);
}

unsigned long long int run_ahead_alg::print_report_to_file(
    string filename, const unordered_map<int, std::pair<int, match_pair *>>
                         &real_report_for_each_cc) {

  report_formatter rf;
  unsigned long long int result_num = 0;
  assert(ccs.size() == real_report_for_each_cc.size());
  for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
    assert(real_report_for_each_cc.find(cc_id) !=
           real_report_for_each_cc.end());
    auto it = real_report_for_each_cc.find(cc_id);

    unsigned long long int num_report_cur_cc = it->second.first;

    match_pair *arr = it->second.second;

    if (num_report_cur_cc > 0) {
      if (this->report_on) {
        assert(arr != NULL);
        for (unsigned long long int i = 0; i < num_report_cur_cc; i++) {
          match_pair mp = arr[i];
          int offset = mp.symbol_offset;
          int sid = mp.state_id;
          // printf("sid = %d ccid = %d ccsize = %d\n", sid, cc_id,
          // ccs[cc_id]->size());

          auto node = ccs[cc_id]->get_node_by_int_id(sid);
          string str_id = node->original_id;

          report r(offset, str_id, cc_id, 1);
          rf.add_report(r);
        }
      } else {
        result_num += num_report_cur_cc;
      }
    }
  }

  if (this->report_on) {
    rf.print_to_file(filename, true);
  }
  return result_num;
}

// ***** seems we have alternatives in utils.h

void run_ahead_alg::prepare_report_arrays() {
  this->is_report = new Array2<char> *[ccs.size()];

  for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
    list_of_tt_of_nfa[cc_id]->copy_to_device();
    is_report[cc_id] = this->get_is_report_array(ccs[cc_id]);
    is_report[cc_id]->copy_to_device();
  }
}

void run_ahead_alg::prepare_tail_pointers() {
  this->tail_of_real_output_array =
      new Array2<unsigned long long int>(ccs.size());
  this->tail_of_intermediate_output_array = new Array2<int>(ccs.size());

  tail_of_real_output_array->fill(0);
  tail_of_real_output_array->copy_to_device();

  tail_of_intermediate_output_array->fill(0);
  tail_of_intermediate_output_array->copy_to_device();
}

void run_ahead_alg::prepare_transition_table_for_scanning_kernel() {
  list_of_tt_of_nfa = new Array2<OutEdges> *[ccs.size()];
  for (int i = 0; i < ccs.size(); i++) {
    list_of_tt_of_nfa[i] = nfa_utils::create_int4_tt_for_nfa(ccs[i]);
  }
}

void run_ahead_alg::prepare_nodelist_for_each_cc() {
  this->list_of_stes_of_nfa = new Array2<STE_dev<4>> *[ccs.size()];
  for (int i = 0; i < ccs.size(); i++) {
    list_of_stes_of_nfa[i] = nfa_utils::create_list_of_STE_dev(ccs[i]);
    list_of_stes_of_nfa[i]->copy_to_device();
  }
}

void run_ahead_alg::group_ccs_to_execution_group() {
  int num_of_ccs = ccs.size();

  int j = 0;
  for (int i = 0; i < num_of_ccs; i++) {
    execution_groups[j].push_back(i);
    j = (j + 1) % num_execution_group;
  }
}

void run_ahead_alg::call_single_kernel(int i, Array2<uint8_t> *input_stream) {
  for (int j = 0; j < execution_groups[i].size(); j++) {

    int cc_id = execution_groups[i][j];

    arr_of_report_buffer[i]->set_tail_of_intermediate(
        this->tail_of_intermediate_output_array, cc_id);
    arr_of_report_buffer[i]->set_tail_of_real(this->tail_of_real_output_array,
                                              cc_id);

    // int start_node_id = nfa_utils::get_start_node_id(ccs[cc_id]);
    vector<int> start_node_ids = nfa_utils::get_all_start_node_id(ccs[cc_id]);
    Array2<int> *start_node_ids_arr2 = new Array2<int>(start_node_ids.size());
    for (int ii = 0; ii < start_node_ids.size(); ii++) {
      start_node_ids_arr2->set(ii, start_node_ids[ii]);
    }
    start_node_ids_arr2->copy_to_device();

    int blockX = input_stream->size() / block_size + 1;
    if (this->blockDimX != -1) {
      blockX = this->blockDimX;
    }
    // cout << "blockX = " << blockX << endl;

    dim3 blocksPerGrid(blockX, 1, 1);
    dim3 threadsPerBlock(block_size, 1);
    assert(this->R > input_stream->size());
    if (record_ir == 0) {
      if (shr_wl_len == 0) {
        if (this->report_on) {
          scanning_input_kernel2<ACTIVE_STATE_ARRAY_SIZE, false,
                                 true><<<blocksPerGrid, threadsPerBlock, 0>>>(
              list_of_tt_of_nfa[cc_id]->get_dev(), input_stream->get_dev(),
              input_stream->size(), start_node_ids_arr2->get_dev(),
              start_node_ids_arr2->size(), this->is_report[cc_id]->get_dev(),
              // intermediate_output_array->get_dev(),
              arr_of_report_buffer[i]
                  ->intermediate_output_array_offset->get_dev(),
              arr_of_report_buffer[i]->intermediate_output_array_sid->get_dev(),
              this->tail_of_intermediate_output_array->get_dev() + cc_id,
              arr_of_report_buffer[i]->real_output_array->get_dev(),
              this->tail_of_real_output_array->get_dev() + cc_id, this->R);
        } else {
          scanning_input_kernel2<ACTIVE_STATE_ARRAY_SIZE, false,
                                 false><<<blocksPerGrid, threadsPerBlock, 0>>>(
              list_of_tt_of_nfa[cc_id]->get_dev(), input_stream->get_dev(),
              input_stream->size(), start_node_ids_arr2->get_dev(),
              start_node_ids_arr2->size(), this->is_report[cc_id]->get_dev(),
              // intermediate_output_array->get_dev(),
              arr_of_report_buffer[i]
                  ->intermediate_output_array_offset->get_dev(),
              arr_of_report_buffer[i]->intermediate_output_array_sid->get_dev(),
              this->tail_of_intermediate_output_array->get_dev() + cc_id,
              arr_of_report_buffer[i]->real_output_array->get_dev(),
              this->tail_of_real_output_array->get_dev() + cc_id, this->R);
        }

      } else if (shr_wl_len > 0) {
        assert(block_size == 128);
        if (this->report_on) {
          switch (this->shr_wl_len) {
            RUNAHEAD_SHR_LENGTH_KERNEL_(1, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(2, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(3, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(4, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(5, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(6, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(7, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(8, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(9, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(10, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(11, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(12, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(13, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(14, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(15, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(16, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(17, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(18, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(19, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(20, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(21, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(22, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(23, true);
            RUNAHEAD_SHR_LENGTH_KERNEL_(24, true);

          default:
            cout << "not supported " << endl;
          }
        } else {
          switch (this->shr_wl_len) {
            RUNAHEAD_SHR_LENGTH_KERNEL_(1, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(2, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(3, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(4, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(5, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(6, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(7, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(8, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(9, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(10, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(11, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(12, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(13, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(14, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(15, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(16, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(17, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(18, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(19, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(20, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(21, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(22, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(23, false);
            RUNAHEAD_SHR_LENGTH_KERNEL_(24, false);

          default:
            cout << "not supported " << endl;
          }
        }
      }
    } else {
      assert(shrmem_wl == 0);
      scanning_input_kernel2<ACTIVE_STATE_ARRAY_SIZE, true>
          <<<blocksPerGrid, threadsPerBlock, 0>>>(
              list_of_tt_of_nfa[cc_id]->get_dev(), input_stream->get_dev(),
              input_stream->size(), start_node_ids_arr2->get_dev(),
              start_node_ids_arr2->size(), this->is_report[cc_id]->get_dev(),
              // intermediate_output_array->get_dev(),
              arr_of_report_buffer[i]
                  ->intermediate_output_array_offset->get_dev(),
              arr_of_report_buffer[i]->intermediate_output_array_sid->get_dev(),
              this->tail_of_intermediate_output_array->get_dev() + cc_id,
              arr_of_report_buffer[i]->real_output_array->get_dev(),
              this->tail_of_real_output_array->get_dev() + cc_id, this->R);
    }

    if (false && record_ir != 0) {
      arr_of_report_buffer[i]->copy_intermediate_tail_to_host();

      int num_intermediate_report =
          *arr_of_report_buffer[i]->h_tail_of_intermediate;
      if (num_intermediate_report > 0) {
        int num_block_sort = num_intermediate_report / BLOCKSIZE_SORT;
        if (num_intermediate_report % BLOCKSIZE_SORT != 0) {
          num_block_sort += 1;
        }

        dim3 blocksPerGrid1(num_block_sort, 1, 1);
        dim3 threadsPerBlock1(BLOCKSIZE_SORT, 1);

        int smemsize = sizeof(int) * ccs[cc_id]->size() * 2;

        if (!fullsort) {
          ir_handle_stage1<int, BLOCKSIZE_SORT, 1, int>
              <<<blocksPerGrid1, threadsPerBlock1, 0>>>(
                  arr_of_report_buffer[i]
                      ->intermediate_output_array_offset->get_dev(),
                  arr_of_report_buffer[i]
                      ->intermediate_output_array_sid->get_dev(),
                  this->tail_of_intermediate_output_array->get_dev() + cc_id,
                  arr_of_report_buffer[i]->real_output_array->get_dev(),
                  this->tail_of_real_output_array->get_dev() + cc_id,
                  list_of_tt_of_nfa[cc_id]->get_dev(),
                  this->is_report[cc_id]->get_dev(), input_stream->get_dev(),
                  input_stream->size(), ccs[cc_id]->size());

        } else {
          device_sort_kv(
              arr_of_report_buffer[i]
                  ->intermediate_output_array_offset->get_dev(),
              arr_of_report_buffer[i]->intermediate_output_array_sid->get_dev(),
              num_intermediate_report);
        }

        ir_handle_stage2<<<blocksPerGrid1, threadsPerBlock1, smemsize>>>(
            arr_of_report_buffer[i]
                ->intermediate_output_array_offset->get_dev(),
            arr_of_report_buffer[i]->intermediate_output_array_sid->get_dev(),
            this->tail_of_intermediate_output_array->get_dev() + cc_id,
            arr_of_report_buffer[i]->real_output_array->get_dev(),
            this->tail_of_real_output_array->get_dev() + cc_id,
            list_of_tt_of_nfa[cc_id]->get_dev(),
            this->is_report[cc_id]->get_dev(), input_stream->get_dev(),
            input_stream->size(), ccs[cc_id]->size());
      }
    }

    arr_of_report_buffer[i]->copy_real_tail_to_host();
    unsigned long long int num_real_report =
        *arr_of_report_buffer[i]->h_tail_of_real;
    // printf("num_real_report = %llu\n", num_real_report);

    // cout << "num_real_report_for_ccid = " << cc_id << " = " <<
    // num_real_report << endl;

    if (num_real_report == 0 || !this->report_on) {
      real_report_for_each_cc[cc_id] =
          std::make_pair(num_real_report, (match_pair *)NULL);
    } else {
      real_report_for_each_cc[cc_id] = std::make_pair(
          num_real_report,
          arr_of_report_buffer[i]->real_output_array->copy_to_host_async(
              num_real_report));
      ;
    }

    delete start_node_ids_arr2;
  }

  auto cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    printf("  cudaGetLastError() returned %d: %s\n", cudaError,
           cudaGetErrorString(cudaError));
  }
}

void run_ahead_alg::launch_kernel() {
  cout << "launch kernel host side multithreaded.  " << endl;
  cout << "shr_wl_len = " << this->shr_wl_len << endl;

  preprocessing(); // for NFA...

  group_ccs_to_execution_group();

  // constants : transition table; is_report arrays.
  prepare_transition_table_for_scanning_kernel();
  prepare_report_arrays();

  Array2<uint8_t> *input_stream = get_array2_of_input_stream0();
  input_stream->copy_to_device();

  prepare_tail_pointers();

  prepare_report_buffers();

  // prepare_nodelist_for_each_cc();

  for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
    real_report_for_each_cc[cc_id] = std::make_pair(0, (match_pair *)NULL);
  }

  cudaDeviceSynchronize();

  cout << "num_execution_group = " << num_execution_group << endl;
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  std::thread *threads = new std::thread[num_execution_group];

  auto t1 = high_resolution_clock::now();
  for (int i = 0; i < num_execution_group; i++) {
    threads[i] =
        std::thread(&run_ahead_alg::call_single_kernel, this, i, input_stream);
  }

  for (int i = 0; i < num_execution_group; i++) {
    threads[i].join();
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  auto t2 = high_resolution_clock::now();

  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Elapsed time : %f ms\n", elapsedTime);
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << "cpu_time_double = " << ms_double.count() << "ms" << endl;

  float sec = elapsedTime / 1000.0;
  cout << "throughput = " << std::fixed
       << (symbol_streams[0].get_length() * symbol_streams.size()) / 1000000.0 /
              sec
       << endl;

  unsigned long long int mc =
      print_report_to_file("report.txt", real_report_for_each_cc);
  printf("result number = %llu\n", mc);
  if (validation) {
    if (!this->report_on) {
      int dupnum = this->dup_input_stream;
      if (quick_validation >= 0 && quick_validation * dupnum <= mc) {
        if (quick_validation * dupnum == mc) {
          tge_log("Quick Validation PASS! (report off, perfect)", BOLDGREEN);
        } else {
          tge_log("Quick Validation PASS! (report off, not perfect)",
                  BOLDGREEN);
        }
      } else {
        tge_log("Quick Validation FAILED! (report off)", BOLDRED);
      }
    } else {
      if (quick_validation >= 0) {
        int dupnum = this->dup_input_stream;
        if (quick_validation * dupnum == mc) {
          tge_log("Quick Validation PASS!", BOLDGREEN);
        } else {
          tge_log("Quick Validation FAILED!", BOLDRED);
        }
      } else {
        printf("CPU validation not supported\n");

        // 		std::vector<uint64_t> ref_results, ref_db_results;
        // 		int nfa_num = 10;

        // 		auto old_grouped_nfas =
        // nfa_utils::group_nfas_by_num(nfa_num, old_ccs);
        // std::vector<Graph *> gs; 		for (auto nfa :
        // old_grouped_nfas) { 				Graph *g = new Graph();
        // 				g->ReadNFA(nfa);
        // 				g->copyToDevice();
        // 				gs.push_back(g);
        // 		}
        // 		auto compareResult = [](uint64_t r1, uint64_t r2) ->
        // bool { 				uint32_t input_1, state_1,
        // input_2, state_2; 				input_1 =
        // (uint32_t)(0xffffffff & r1); 				state_1
        // = (uint32_t)(r1 >> 32); 				input_2 =
        // (uint32_t)(0xffffffff & r2); 				state_2
        // = (uint32_t)(r2 >> 32);
        // 				// printf("{%u, %u}, ", state, input);
        // 				if (input_1 == input_2)
        // 						return state_1 <
        // state_2; 				else
        // return input_1 < input_2;
        // 		};

        // 		GroupCsr gcsr;
        // 		gcsr.init(gs);
        // 		automata_utils::automataGroupsReference(
        // 						gs,
        // arr_input_streams->get_host(),
        // get_num_streams(), symbol_streams[0].get_length(),
        // &ref_results, &ref_db_results, 0, gcsr);

        // 		printf("\n############ Validate result "
        // 										"############\n
        // ");
        // 		// automata_utils::automataValidation(&results,
        // &ref_results, true); 		std::sort(ref_results.begin(),
        // ref_results.end(), compareResult);
        // ref_results.erase(unique(ref_results.begin(), ref_results.end()),
        // 								ref_results.end());
        // 		printf("ref_results size: %zu\n", ref_results.size());

        // 		if (rf.size() == ref_results.size()) {
        // 				tge_log("Validation PASS!", BOLDGREEN);
        // 		} else {
        // 				tge_log("Validation FAILED!", BOLDRED);
        // 		}
      }
    }
  }

  delete input_stream;
}

void run_ahead_alg::prepare_report_buffers() {

  this->arr_of_report_buffer.clear();
  for (int i = 0; i < num_execution_group; i++) {
    arr_of_report_buffer.push_back(
        new report_buffer(CAP_OUTPUT_BUFFER_FOR_EACH_EXECUTION_GROUP));
  }

  // meiwan
}

report_buffer::report_buffer(int CAP) {
  this->intermediate_output_array_offset = new Array2<int>(CAP);
  this->intermediate_output_array_sid = new Array2<int>(CAP);
  this->real_output_array = new Array2<match_pair>(CAP);
}

report_buffer::~report_buffer() {
  delete this->intermediate_output_array_offset;
  delete this->intermediate_output_array_sid;
  delete this->real_output_array;
}

void report_buffer::set_tail_of_real(
    Array2<unsigned long long int> *tail_of_real_array, int offset) {
  // int *h_tail_of_real, *d_tail_of_real;

  this->h_tail_of_real = tail_of_real_array->get_host() + offset;
  this->d_tail_of_real = tail_of_real_array->get_dev() + offset;
}

void report_buffer::set_tail_of_intermediate(
    Array2<int> *tail_of_intermediate_array, int offset) {
  this->h_tail_of_intermediate =
      tail_of_intermediate_array->get_host() + offset;
  this->d_tail_of_intermediate = tail_of_intermediate_array->get_dev() + offset;
}

void report_buffer::copy_real_tail_to_host() {
  // let's to synchronous version first.

  auto errcode = cudaMemcpy(h_tail_of_real, d_tail_of_real, sizeof(int),
                            cudaMemcpyDeviceToHost);

  if (errcode != cudaSuccess) {
    cerr << "cannot copy to device   @ copy_real_tail_to_host = " << errcode
         << endl;
    exit(-1);
  }
}

void report_buffer::copy_intermediate_tail_to_host() {
  auto errcode = cudaMemcpy(h_tail_of_intermediate, d_tail_of_intermediate,
                            sizeof(int), cudaMemcpyDeviceToHost);

  if (errcode != cudaSuccess) {
    cerr << "cannot copy to device   @ copy_intermediate_tail_to_host "
         << errcode << endl;
    exit(-1);
  }
}
