#include "graph.h"
#include "group_graph.h"
#include "kernel.h"
#include "nfa_utils.h"
#include "ngap.h"
#include "ngap_buffer.h"
#include "precompute_table.h"
#include "utils.h"
#include "validate.h"

#include "omp.h"
#include <chrono>
#include <cmath>
#include <execution>
#include <fstream>
#include <iostream>

// #define DEBUG_AM
bool compareResult(uint64_t r1, uint64_t r2) {
  uint32_t input_1, state_1, input_2, state_2;
  input_1 = (uint32_t)(0xffffffff & r1);
  state_1 = (uint32_t)(r1 >> 32);
  input_2 = (uint32_t)(0xffffffff & r2);
  state_2 = (uint32_t)(r2 >> 32);
  // printf("{%u, %u}, ", state, input);
  if (input_1 == input_2)
    return state_1 < state_2;
  else
    return input_1 < input_2;
};

ngap::ngap(NFA *nfa, Graph &g)
    : abstract_algorithm(nfa), graph(g), active_state_array_size(256),
      num_segment_per_ss(1) {}

ngap::~ngap() {}

void ngap::set_block_size(int blocksize) { this->block_size = blocksize; }

void ngap::set_active_state_array_size(int active_state_array_size) {
  this->active_state_array_size = active_state_array_size;
}

void ngap::set_alphabet(set<uint8_t> alphabet) { this->alphabet = alphabet; }

void ngap::prepare_original_input_streams(SymbolStream &ss) {
  int length = ss.get_length();
  arr_input_streams = new Array2<uint8_t>(length);
  int t = 0;
  for (int p = 0; p < ss.get_length(); p++) {
    arr_input_streams->set(t++, ss.get_position(p));
  }
}

void ngap::prepare_outputs() {
  match_array = new Array2<match_entry>(this->output_buffer_size);
  match_count = new Array2<unsigned int>(1);

  match_array->clear_to_zero();
  match_count->clear_to_zero();
}

void ngap::preprocessing() {
  nfa->mark_cc_id();
  ccs = nfa_utils::split_nfa_by_ccs(*nfa);

  for (auto cc : ccs) {
    cc->calc_scc();
    cc->topo_sort();
  }

  std::pair<int, int> degree_result = nfa_utils::limit_out_degree_on_ccs(
      this->ccs, 4, plo->quit_degree, plo->remove_degree);
  printf("limit_out_degree_on_ccs: num_succeed %d, num_fail %d\n",
         degree_result.first, degree_result.second);

  if (this->max_cc_size_limit != -1) {
    vector<NFA *> tmp_ccs;
    for (int i = 0; i < this->ccs.size(); i++) {
      if (ccs[i]->size() <= max_cc_size_limit) {
        tmp_ccs.push_back(ccs[i]);
      } else {
        cout << "max_cc_size_limit " << max_cc_size_limit
             << ": remove_ccid = " << i << " "
             << ccs[i]->get_node_by_int_id(0)->str_id << endl;
        delete ccs[i];
      }
    }
    this->ccs = tmp_ccs;
  }

  // auto cg_calc_start = high_resolution_clock::now();
  // group_nfas(); // calculate compatible groups here.
  // auto cg_calc_end = high_resolution_clock::now();

  // auto cg_calc_duration = duration_cast<seconds>(cg_calc_end -
  // cg_calc_start); cout << "time_calc_cg_sec = " << cg_calc_duration.count()
  // << endl;

  // prepare_state_start_position_tb();
  // calc_str_id_to_compatible_group_per_block();

  // prepare_compatible_grps();

  // prepare_transition_table();
  // prepare_states_status();
  // prepare_initial_active_state_array();

  // prepare_always_enabled_frontier();//TGE: only algo=graph
  // prepare_input_streams();
  // prepare_outputs();
}

// O0
void ngap::launch_blocking_groups() {
  tge_log("automata blocking::launch!", BOLDBLUE);

  Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
  input_stream->copy_to_device();
  int multi_ss_size = symbol_streams[0].get_length();

  // Csr
  Csr csr(graph);
  csr.fromCoo(graph.edge_pairs->get_host());
  csr.moveToDevice();
  Matchset ms = graph.get_matchset_device(plo->use_soa);

  GroupCsr gcsr;
  GroupMatchset gms;
  GroupNodeAttrs gna;
  GroupAAS gaas;
  gcsr.init(gs);
  gms.init(gs, plo->use_soa);
  gna.init(gs);
  gaas.init(gs);

  BlockingBuffer blb;

  blb.init_nfagroups(input_stream, input_stream->size(), num_seg, multi_ss_size,
                     gs, plo);

  dim3 blocksPerGrid(plo->group_num, num_seg, 1);
  dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);

  calculateTheoreticalOccupancy2(advanceAndFilterBlockingGroups, BLOCK_SIZE);

  CHECK_LAST_ERROR

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  advanceAndFilterBlockingGroups<<<blocksPerGrid, threadsPerBlock>>>(
      blb, input_stream->get_dev(), multi_ss_size, gms, gna, gaas, gcsr);
  // cudaDeviceSynchronize();

  CHECK_LAST_ERROR
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double throughput = (double)input_stream->size() / (milliseconds * 1000);
  std::cout << "ngap elapsed time: " << milliseconds / 1000.0
            << " seconds, throughput = " << throughput << " MB/s " << std::endl;

  unsigned long long int *h_results_size = new unsigned long long int;
  CHECK_ERROR(cudaMemcpy((void *)h_results_size, blb.d_results_size,
                         sizeof(unsigned long long int),
                         cudaMemcpyDeviceToHost));
  std::cout << "Results number: " << *h_results_size << std::endl;

  bool validation = plo->validation;
  if (validation) {
    if (plo->report_off) {
      int dupnum =
          plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
      if (plo->quick_validation >= 0 &&
          plo->quick_validation * dupnum <= *h_results_size) {
        tge_log("Quick Validation PASS!", BOLDGREEN);
      } else {
        tge_log("Quick Validation FAILED!", BOLDRED);
      }
    } else {
      uint64_t *h_results;
      if (plo->use_uvm) {
        h_results = blb.d_results;
      } else {
        h_results = new uint64_t[*h_results_size];
        CHECK_ERROR(cudaMemcpy((void *)h_results, blb.d_results,
                               sizeof(uint64_t) * *h_results_size,
                               cudaMemcpyDeviceToHost));
      }

      std::vector<uint64_t> results, ref_results, db_results, ref_db_results;
      // for (int i = 0; i < *h_results_size; i++)
      //   results.push_back(h_results[i]);
      int mc = *h_results_size;
      results.resize(mc);
      std::for_each(std::execution::par_unseq, std::begin(results),
                    std::end(results), [&](uint64_t &r) {
                      u_int32_t i = &r - &results[0];
                      assert(i < mc);
                      results[i] = h_results[i];
                    });
      std::sort(std::execution::par_unseq, results.begin(), results.end(),
                compareResult);
      results.erase(std::unique(std::execution::par_unseq, results.begin(),
                                results.end()),
                    results.end());
      std::cout << "Unique results number: " << results.size() << std::endl;
      printf("validation start.\n");

      if (plo->quick_validation >= 0) {
        int dupnum =
            plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
        if (plo->quick_validation * dupnum == results.size()) {
          tge_log("Quick Validation PASS!", BOLDGREEN);
        } else {
          tge_log("Quick Validation FAILED!", BOLDRED);
        }
      } else {

        bool isDup = false;
        if (plo->duplicate_input_stream > 1 &&
            (plo->split_chunk_size == -1 ||
             plo->split_chunk_size == plo->input_len)) {
          isDup = true;
        }
        automataGroupsReference(gs, input_stream->get_host(), num_seg,
                                multi_ss_size, &ref_results, &ref_db_results,
                                DEBUG_ITER, gcsr, isDup);
        printf("\n############ Validate result ############ \n");
        automataValidation(&results, &ref_results, true);
      }

      if (!plo->use_uvm) {
        delete[] h_results;
      }
    }
  }

  // if (plo->motivate_worklist_length) {
  //   int *h_froniter_end = new int;
  //   CHECK_ERROR(cudaMemcpy((void *)h_froniter_end, blb.d_froniter_end,
  //                          sizeof(int), cudaMemcpyDeviceToHost));

  //   int *h_froniter_length = new int[*h_froniter_end];
  //   CHECK_ERROR(cudaMemcpy((void *)h_froniter_length, blb.d_froniter_length,
  //                          sizeof(int) * *h_froniter_end,
  //                          cudaMemcpyDeviceToHost));

  //   for (int i = 0; i < *h_froniter_end; i++) {
  //     printf("%d\n", h_froniter_length[i]);
  //   }

  //   std::string nfa_name =
  //       plo->nfa_filename.substr(plo->nfa_filename.find_last_of("/\\") + 1);
  //   std::string path = "../../froniter_length/" + nfa_name + ".txt";
  //   std::ofstream froniter_length(path);
  //   printf("Save froniter length file to %s\n", path.c_str());
  //   if (froniter_length.is_open()) {
  //     for (int i = 0; i < *h_froniter_end; i++) {
  //       froniter_length << h_froniter_length[i] << "\n";
  //     }
  //     froniter_length.close();
  //   } else
  //     assert(false);
  // }

  blb.release();
  csr.release();
  csr.releaseDevice();
  ms.release();

  gcsr.release();
  gms.release();
  gna.release();
  gaas.release();

  delete h_results_size;
}

// NAP
void ngap::launch_non_blocking_nap_groups() {
  tge_log("automata nonblocking  nap ::launch!", BOLDBLUE);

  Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
  input_stream->copy_to_device();
  int multi_ss_size = symbol_streams[0].get_length();

  // Csr
  Csr csr(graph);
  csr.fromCoo(graph.edge_pairs->get_host());
  csr.moveToDevice();
  Matchset ms = graph.get_matchset_device(plo->use_soa);

  GroupCsr gcsr;
  GroupMatchset gms;
  GroupNodeAttrs gna;
  GroupAAS gaas;
  gcsr.init(gs);
  gms.init(gs, plo->use_soa);
  gna.init(gs);
  gaas.init(gs);

  NonBlockingBuffer nblb;

  assert(plo->add_aas_start > 0);
  if (plo->add_aas_start > 0) {
    printf("add_aas_start should be set to 0\n");
    exit(-1);
  }

  nblb.init_nfagroups(input_stream, input_stream->size(), num_seg,
                      multi_ss_size, gs, plo);

  dim3 blocksPerGrid(plo->group_num, num_seg, 1);
  dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
  if (nblb.unique)
    if (plo->motivate_worklist_length)
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingNAPGroups<true, true>, BLOCK_SIZE);
    else
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingNAPGroups<true, false>, BLOCK_SIZE);
  else if (plo->motivate_worklist_length)
    calculateTheoreticalOccupancy2(
        advanceAndFilterNonBlockingNAPGroups<false, true>, BLOCK_SIZE);
  else
    calculateTheoreticalOccupancy2(
        advanceAndFilterNonBlockingNAPGroups<false, false>, BLOCK_SIZE);
  CHECK_LAST_ERROR

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  if (nblb.unique) {
    if (plo->motivate_worklist_length)
      advanceAndFilterNonBlockingNAPGroups<true, true>
          <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                               multi_ss_size, gms, gna, gaas,
                                               gcsr);
    else
      advanceAndFilterNonBlockingNAPGroups<true, false>
          <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                               multi_ss_size, gms, gna, gaas,
                                               gcsr);
  } else {
    if (plo->motivate_worklist_length)
      advanceAndFilterNonBlockingNAPGroups<false, true>
          <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                               multi_ss_size, gms, gna, gaas,
                                               gcsr);
    else
      advanceAndFilterNonBlockingNAPGroups<false, false>
          <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                               multi_ss_size, gms, gna, gaas,
                                               gcsr);
  }

  CHECK_LAST_ERROR
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double throughput = (double)input_stream->size() / (milliseconds * 1000);
  std::cout << "ngap elapsed time: " << milliseconds / 1000.0
            << " seconds, throughput = " << throughput << " MB/s " << std::endl;

  uint *h_buffer_end = new uint[num_seg];
  uint buffer_total_size = 0;
  CHECK_ERROR(cudaMemcpy((void *)h_buffer_end, nblb.d_buffer_end,
                         sizeof(uint) * num_seg, cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_seg; i++) {
    buffer_total_size += h_buffer_end[i];
  }
  // printf("buffer_total_size: %u \n", buffer_total_size);

  unsigned long long int *h_results_size = new unsigned long long int;
  CHECK_ERROR(cudaMemcpy((void *)h_results_size, nblb.d_results_size,
                         sizeof(unsigned long long int),
                         cudaMemcpyDeviceToHost));
  std::cout << "Results number: " << *h_results_size << std::endl;

  if (plo->motivate_worklist_length) {
    int *h_froniter_end = new int;
    CHECK_ERROR(cudaMemcpy((void *)h_froniter_end, nblb.d_froniter_end,
                           sizeof(int), cudaMemcpyDeviceToHost));

    int *h_froniter_length = new int[*h_froniter_end];
    CHECK_ERROR(cudaMemcpy((void *)h_froniter_length, nblb.d_froniter_length,
                           sizeof(int) * *h_froniter_end,
                           cudaMemcpyDeviceToHost));

    std::string path = "/home/tge/workspace/gpunfa-ngap/froniter_length/nap/" +
                       plo->app_name + ".txt";
    std::ofstream froniter_length_file(path);
    printf("Save froniter length file to %s\n", path.c_str());
    if (froniter_length_file.is_open()) {
      for (int i = 0; i < *h_froniter_end; i++) {
        froniter_length_file << h_froniter_length[i] << "\n";
      }
      froniter_length_file.close();
    } else
      assert(false);
    delete[] h_froniter_length;
  }

  bool validation = plo->validation;
  if (validation) {

    if (plo->report_off) {
      int dupnum =
          plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
      if (plo->quick_validation >= 0 &&
          plo->quick_validation * dupnum <= *h_results_size) {
        tge_log("Quick Validation PASS!", BOLDGREEN);
      } else {
        tge_log("Quick Validation FAILED!", BOLDRED);
      }
    } else {
      uint32_t *h_results_v;
      uint32_t *h_results_i;
      uint64_t *h_results = new uint64_t[*h_results_size];
      if (plo->use_uvm) {
        h_results_v = nblb.d_results_v;
        h_results_i = nblb.d_results_i;
      } else {
        h_results_v = new uint32_t[*h_results_size];
        h_results_i = new uint32_t[*h_results_size];
        CHECK_ERROR(cudaMemcpy((void *)h_results_v, nblb.d_results_v,
                               sizeof(uint32_t) * *h_results_size,
                               cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy((void *)h_results_i, nblb.d_results_i,
                               sizeof(uint32_t) * *h_results_size,
                               cudaMemcpyDeviceToHost));
      }

      auto addResult = [](uint32_t node, uint32_t index) {
        uint64_t r = 0;
        r = (uint32_t)node;
        r = r << 32;
        r = r | (uint32_t)index;
        return r;
      };
      for (int i = 0; i < *h_results_size; i++)
        h_results[i] = addResult(h_results_v[i], h_results_i[i]);
      std::vector<uint64_t> results, ref_results, db_results, ref_db_results;
      // for (int i = 0; i < *h_results_size; i++)
      //   results.push_back(h_results[i]);
      int mc = *h_results_size;
      results.resize(mc);
      std::for_each(std::execution::par_unseq, std::begin(results),
                    std::end(results), [&](uint64_t &r) {
                      u_int32_t i = &r - &results[0];
                      assert(i < mc);
                      results[i] = h_results[i];
                    });
      std::sort(std::execution::par_unseq, results.begin(), results.end(),
                compareResult);
      results.erase(std::unique(std::execution::par_unseq, results.begin(),
                                results.end()),
                    results.end());
      std::cout << "Unique results number: " << results.size() << std::endl;
      printf("validation start.\n");

      if (plo->quick_validation >= 0) {
        int dupnum =
            plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
        if (plo->quick_validation * dupnum == results.size()) {
          tge_log("Quick Validation PASS!", BOLDGREEN);
        } else {
          tge_log("Quick Validation FAILED!", BOLDRED);
        }
      } else {

        bool isDup = false;
        if (plo->duplicate_input_stream > 1 &&
            (plo->split_chunk_size == -1 ||
             plo->split_chunk_size == plo->input_len)) {
          isDup = true;
        }
        automataGroupsReference(gs, input_stream->get_host(), num_seg,
                                multi_ss_size, &ref_results, &ref_db_results,
                                DEBUG_ITER, gcsr, isDup);
        printf("\n############ Validate result ############ \n");
        automataValidation(&results, &ref_results, true);
      }

      if (!plo->use_uvm) {
        delete[] h_results_v;
        delete[] h_results_i;
      }
      delete[] h_results;
    }
  }

  nblb.release(true);
  csr.release();
  csr.releaseDevice();
  ms.release();

  gcsr.release();
  gms.release();
  gna.release();
  gaas.release();

  delete h_results_size;
}

// O1
void ngap::launch_non_blocking_groups() {
  tge_log("automata nonblocking::launch!", BOLDBLUE);

  Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
  input_stream->copy_to_device();
  int multi_ss_size = symbol_streams[0].get_length();

  // Csr
  Csr csr(graph);
  csr.fromCoo(graph.edge_pairs->get_host());
  csr.moveToDevice();
  Matchset ms = graph.get_matchset_device(plo->use_soa);

  GroupCsr gcsr;
  GroupMatchset gms;
  GroupNodeAttrs gna;
  GroupAAS gaas;
  gcsr.init(gs);
  gms.init(gs, plo->use_soa);
  gna.init(gs);
  gaas.init(gs);

  NonBlockingBuffer nblb;

  nblb.init_nfagroups(input_stream, input_stream->size(), num_seg,
                      multi_ss_size, gs, plo);

  dim3 blocksPerGrid(plo->group_num, num_seg, 1);
  dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
  if (nblb.unique)
    if (plo->motivate_worklist_length)
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingGroups<true, true>, BLOCK_SIZE);
    else
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingGroups<true, false>, BLOCK_SIZE);
  else if (plo->motivate_worklist_length)
    calculateTheoreticalOccupancy2(
        advanceAndFilterNonBlockingGroups<false, true>, BLOCK_SIZE);
  else
    calculateTheoreticalOccupancy2(
        advanceAndFilterNonBlockingGroups<false, false>, BLOCK_SIZE);
  CHECK_LAST_ERROR

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  if (nblb.unique) {
    if (plo->motivate_worklist_length)
      advanceAndFilterNonBlockingGroups<true, true>
          <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                               multi_ss_size, gms, gna, gaas,
                                               gcsr);
    else
      advanceAndFilterNonBlockingGroups<true, false>
          <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                               multi_ss_size, gms, gna, gaas,
                                               gcsr);
  } else {
    if (plo->motivate_worklist_length)
      advanceAndFilterNonBlockingGroups<false, true>
          <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                               multi_ss_size, gms, gna, gaas,
                                               gcsr);
    else
      advanceAndFilterNonBlockingGroups<false, false>
          <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                               multi_ss_size, gms, gna, gaas,
                                               gcsr);
  }

  CHECK_LAST_ERROR
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double throughput = (double)input_stream->size() / (milliseconds * 1000);
  std::cout << "ngap elapsed time: " << milliseconds / 1000.0
            << " seconds, throughput = " << throughput << " MB/s " << std::endl;

  uint *h_buffer_end = new uint[num_seg];
  uint buffer_total_size = 0;
  CHECK_ERROR(cudaMemcpy((void *)h_buffer_end, nblb.d_buffer_end,
                         sizeof(uint) * num_seg, cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_seg; i++) {
    buffer_total_size += h_buffer_end[i];
  }
  // printf("buffer_total_size: %u \n", buffer_total_size);

  unsigned long long int *h_results_size = new unsigned long long int;
  CHECK_ERROR(cudaMemcpy((void *)h_results_size, nblb.d_results_size,
                         sizeof(unsigned long long int),
                         cudaMemcpyDeviceToHost));
  std::cout << "Results number: " << *h_results_size << std::endl;

  if (plo->motivate_worklist_length) {
    int *h_froniter_end = new int;
    CHECK_ERROR(cudaMemcpy((void *)h_froniter_end, nblb.d_froniter_end,
                           sizeof(int), cudaMemcpyDeviceToHost));

    int *h_froniter_length = new int[*h_froniter_end];
    CHECK_ERROR(cudaMemcpy((void *)h_froniter_length, nblb.d_froniter_length,
                           sizeof(int) * *h_froniter_end,
                           cudaMemcpyDeviceToHost));

    // for (int i = 0; i < *h_froniter_end; i++) {
    //   printf("%d\n", h_froniter_length[i]);
    // }

    // auto find_n_of = [](string &s, int n, char symbol) -> int {
    //   int count = 0;
    //   auto it =
    //       std::find_if(s.begin(), s.end(), [&count, &s, &n, &symbol](char c)
    //       {
    //         return c == symbol && ++count == n;
    //       });
    //   return std::distance(std::begin(s), it);
    // };
    // int name_start = find_n_of(plo->nfa_filename, 6, '/') + 1;
    // int name_end = find_n_of(plo->nfa_filename, 7, '/');
    // std::string nfa_name =
    //     plo->nfa_filename.substr(name_start, name_end - name_start);
    std::string path =
        "/home/tge/workspace/gpunfa-ngap/froniter_length/early_start/" +
        plo->app_name + ".txt";
    std::ofstream froniter_length_file(path);
    printf("Save froniter length file to %s\n", path.c_str());
    if (froniter_length_file.is_open()) {
      for (int i = 0; i < *h_froniter_end; i++) {
        froniter_length_file << h_froniter_length[i] << "\n";
      }
      froniter_length_file.close();
    } else
      assert(false);
    delete[] h_froniter_length;
  }

  bool validation = plo->validation;
  if (validation) {

    if (plo->report_off) {
      int dupnum =
          plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
      if (plo->quick_validation >= 0 &&
          plo->quick_validation * dupnum <= *h_results_size) {
        tge_log("Quick Validation PASS!", BOLDGREEN);
      } else {
        tge_log("Quick Validation FAILED!", BOLDRED);
      }
    } else {
      uint32_t *h_results_v;
      uint32_t *h_results_i;
      uint64_t *h_results = new uint64_t[*h_results_size];
      if (plo->use_uvm) {
        h_results_v = nblb.d_results_v;
        h_results_i = nblb.d_results_i;
      } else {
        h_results_v = new uint32_t[*h_results_size];
        h_results_i = new uint32_t[*h_results_size];
        CHECK_ERROR(cudaMemcpy((void *)h_results_v, nblb.d_results_v,
                               sizeof(uint32_t) * *h_results_size,
                               cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy((void *)h_results_i, nblb.d_results_i,
                               sizeof(uint32_t) * *h_results_size,
                               cudaMemcpyDeviceToHost));
      }

      auto addResult = [](uint32_t node, uint32_t index) {
        uint64_t r = 0;
        r = (uint32_t)node;
        r = r << 32;
        r = r | (uint32_t)index;
        return r;
      };
      for (int i = 0; i < *h_results_size; i++)
        h_results[i] = addResult(h_results_v[i], h_results_i[i]);
      std::vector<uint64_t> results, ref_results, db_results, ref_db_results;
      // for (int i = 0; i < *h_results_size; i++)
      //   results.push_back(h_results[i]);
      int mc = *h_results_size;
      results.resize(mc);
      std::for_each(std::execution::par_unseq, std::begin(results),
                    std::end(results), [&](uint64_t &r) {
                      u_int32_t i = &r - &results[0];
                      assert(i < mc);
                      results[i] = h_results[i];
                    });
      std::sort(std::execution::par_unseq, results.begin(), results.end(),
                compareResult);
      results.erase(std::unique(std::execution::par_unseq, results.begin(),
                                results.end()),
                    results.end());
      std::cout << "Unique results number: " << results.size() << std::endl;
      printf("validation start.\n");

      if (plo->quick_validation >= 0) {
        int dupnum =
            plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
        if (plo->quick_validation * dupnum == results.size()) {
          tge_log("Quick Validation PASS!", BOLDGREEN);
        } else {
          tge_log("Quick Validation FAILED!", BOLDRED);
        }
      } else {

        bool isDup = false;
        if (plo->duplicate_input_stream > 1 &&
            (plo->split_chunk_size == -1 ||
             plo->split_chunk_size == plo->input_len)) {
          isDup = true;
        }
        automataGroupsReference(gs, input_stream->get_host(), num_seg,
                                multi_ss_size, &ref_results, &ref_db_results,
                                DEBUG_ITER, gcsr, isDup);
        printf("\n############ Validate result ############ \n");
        automataValidation(&results, &ref_results, true);
      }

      if (!plo->use_uvm) {
        delete[] h_results_v;
        delete[] h_results_i;
      }
      delete[] h_results;
    }
  }

  nblb.release(true);
  csr.release();
  csr.releaseDevice();
  ms.release();

  gcsr.release();
  gms.release();
  gna.release();
  gaas.release();

  delete h_results_size;
}

// O3
void ngap::launch_non_blocking_prec_groups() {
  tge_log("automata nonblocking_po::launch!", BOLDBLUE);

  Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
  input_stream->copy_to_device();
  int multi_ss_size = symbol_streams[0].get_length();

  // Csr
  Csr csr(graph);
  csr.fromCoo(graph.edge_pairs->get_host());
  csr.moveToDevice();
  Matchset ms_aos = graph.get_matchset_device(false);
  Matchset ms_soa = graph.get_matchset_device(true);

  GroupCsr gcsr;
  GroupMatchset gms;
  GroupNodeAttrs gna;
  GroupAAS gaas;
  initGroupCsrWithPrec(gcsr, gs, plo->precompute_depth,
                       plo->compress_prec_table);
  // gcsr.init(gs);
  gms.init(gs, plo->use_soa);
  gna.init(gs);
  gaas.init(gs);
  // return;

  NonBlockingBuffer nblb;
  nblb.init_nfagroups(input_stream, input_stream->size(), num_seg,
                      multi_ss_size, gs, plo);

  dim3 blocksPerGrid(plo->group_num, num_seg, 1);
  dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);

  if (nblb.unique) {
    switch (plo->precompute_depth) {
    case 0:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingPrecGroups<true, 0, false>, BLOCK_SIZE);
      break;
    case 1:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingPrecGroups<true, 1, false>, BLOCK_SIZE);
      break;
    case 2:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingPrecGroups<true, 2, false>, BLOCK_SIZE);
      break;
    case 3:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingPrecGroups<true, 3, false>, BLOCK_SIZE);
      break;
    default:
      break;
    }
  } else {
    switch (plo->precompute_depth) {
    case 0:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingPrecGroups<false, 0, false>, BLOCK_SIZE);
      break;
    case 1:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingPrecGroups<false, 1, false>, BLOCK_SIZE);
      break;
    case 2:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingPrecGroups<false, 2, false>, BLOCK_SIZE);
      break;
    case 3:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingPrecGroups<false, 3, false>, BLOCK_SIZE);
      break;
    default:
      break;
    }
  }

  CHECK_LAST_ERROR

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  if (plo->motivate_worklist_length) {
    if (nblb.unique) {
      switch (plo->precompute_depth) {
      case 0:
        advanceAndFilterNonBlockingPrecGroups<true, 0, true>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 1:
        advanceAndFilterNonBlockingPrecGroups<true, 1, true>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 2:
        advanceAndFilterNonBlockingPrecGroups<true, 2, true>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 3:
        advanceAndFilterNonBlockingPrecGroups<true, 3, true>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      default:
        break;
      }
    } else {
      switch (plo->precompute_depth) {
      case 0:
        advanceAndFilterNonBlockingPrecGroups<false, 0, true>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 1:
        advanceAndFilterNonBlockingPrecGroups<false, 1, true>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 2:
        advanceAndFilterNonBlockingPrecGroups<false, 2, true>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 3:
        advanceAndFilterNonBlockingPrecGroups<false, 3, true>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      default:
        break;
      }
    }
  } else {
    if (nblb.unique) {
      switch (plo->precompute_depth) {
      case 0:
        advanceAndFilterNonBlockingPrecGroups<true, 0, false>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 1:
        advanceAndFilterNonBlockingPrecGroups<true, 1, false>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 2:
        advanceAndFilterNonBlockingPrecGroups<true, 2, false>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 3:
        advanceAndFilterNonBlockingPrecGroups<true, 3, false>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      default:
        break;
      }
    } else {
      switch (plo->precompute_depth) {
      case 0:
        advanceAndFilterNonBlockingPrecGroups<false, 0, false>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 1:
        advanceAndFilterNonBlockingPrecGroups<false, 1, false>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 2:
        advanceAndFilterNonBlockingPrecGroups<false, 2, false>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      case 3:
        advanceAndFilterNonBlockingPrecGroups<false, 3, false>
            <<<blocksPerGrid, threadsPerBlock>>>(nblb, input_stream->get_dev(),
                                                 multi_ss_size, gms, gna, gaas,
                                                 gcsr);
        break;
      default:
        break;
      }
    }
  }

  // cudaDeviceSynchronize();

  CHECK_LAST_ERROR
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double throughput = (double)input_stream->size() / (milliseconds * 1000);
  std::cout << "ngap elapsed time: " << milliseconds / 1000.0
            << " seconds, throughput = " << throughput << " MB/s " << std::endl;

  uint *h_buffer_end = new uint[num_seg];
  uint buffer_total_size = 0;
  CHECK_ERROR(cudaMemcpy((void *)h_buffer_end, nblb.d_buffer_end,
                         sizeof(uint) * num_seg, cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_seg; i++) {
    buffer_total_size += h_buffer_end[i];
  }
  // printf("buffer_total_size: %u \n", buffer_total_size);

  unsigned long long int *h_results_size = new unsigned long long int;
  CHECK_ERROR(cudaMemcpy((void *)h_results_size, nblb.d_results_size,
                         sizeof(unsigned long long int),
                         cudaMemcpyDeviceToHost));
  std::cout << "Results number: " << *h_results_size << std::endl;

  if (plo->motivate_worklist_length) {
    int *h_froniter_end = new int;
    CHECK_ERROR(cudaMemcpy((void *)h_froniter_end, nblb.d_froniter_end,
                           sizeof(int), cudaMemcpyDeviceToHost));

    int *h_froniter_length = new int[*h_froniter_end];
    CHECK_ERROR(cudaMemcpy((void *)h_froniter_length, nblb.d_froniter_length,
                           sizeof(int) * *h_froniter_end,
                           cudaMemcpyDeviceToHost));

    std::string path =
        "/home/tge/workspace/gpunfa-ngap/froniter_length/precompute/" +
        plo->app_name + ".txt";
    std::ofstream froniter_length_file(path);
    printf("Save froniter length file to %s\n", path.c_str());
    if (froniter_length_file.is_open()) {
      for (int i = 0; i < *h_froniter_end; i++) {
        froniter_length_file << h_froniter_length[i] << "\n";
      }
      froniter_length_file.close();
    } else
      assert(false);
    delete[] h_froniter_length;
  }

  bool validation = plo->validation;
  if (validation) {

    if (plo->report_off) {
      int dupnum =
          plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
      if (plo->quick_validation >= 0 &&
          plo->quick_validation * dupnum <= *h_results_size) {
        tge_log("Quick Validation PASS!", BOLDGREEN);
      } else {
        tge_log("Quick Validation FAILED!", BOLDRED);
      }
    } else {
      uint32_t *h_results_v;
      uint32_t *h_results_i;
      uint64_t *h_results = new uint64_t[*h_results_size];
      if (plo->use_uvm) {
        h_results_v = nblb.d_results_v;
        h_results_i = nblb.d_results_i;
      } else {
        h_results_v = new uint32_t[*h_results_size];
        h_results_i = new uint32_t[*h_results_size];
        CHECK_ERROR(cudaMemcpy((void *)h_results_v, nblb.d_results_v,
                               sizeof(uint32_t) * *h_results_size,
                               cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy((void *)h_results_i, nblb.d_results_i,
                               sizeof(uint32_t) * *h_results_size,
                               cudaMemcpyDeviceToHost));
      }
      auto addResult = [](uint32_t node, uint32_t index) {
        uint64_t r = 0;
        r = (uint32_t)node;
        r = r << 32;
        r = r | (uint32_t)index;
        return r;
      };
      for (int i = 0; i < *h_results_size; i++)
        h_results[i] = addResult(h_results_v[i], h_results_i[i]);
      std::vector<uint64_t> results, ref_results, db_results, ref_db_results;
      // for (int i = 0; i < *h_results_size; i++)
      //   results.push_back(h_results[i]);
      int mc = *h_results_size;
      results.resize(mc);
      std::for_each(std::execution::par_unseq, std::begin(results),
                    std::end(results), [&](uint64_t &r) {
                      u_int32_t i = &r - &results[0];
                      assert(i < mc);
                      results[i] = h_results[i];
                    });
      std::sort(std::execution::par_unseq, results.begin(), results.end(),
                compareResult);
      results.erase(std::unique(std::execution::par_unseq, results.begin(),
                                results.end()),
                    results.end());
      std::cout << "Unique results number: " << results.size() << std::endl;
      printf("validation start.\n");

      if (plo->quick_validation >= 0) {
        int dupnum =
            plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
        if (plo->quick_validation * dupnum == results.size()) {
          tge_log("Quick Validation PASS!", BOLDGREEN);
        } else {
          tge_log("Quick Validation FAILED!", BOLDRED);
        }
      } else {

        bool isDup = false;
        if (plo->duplicate_input_stream > 1 &&
            (plo->split_chunk_size == -1 ||
             plo->split_chunk_size == plo->input_len)) {
          isDup = true;
        }
        automataGroupsReference(gs, input_stream->get_host(), num_seg,
                                multi_ss_size, &ref_results, &ref_db_results,
                                DEBUG_ITER, gcsr, isDup);
        printf("\n############ Validate result ############ \n");
        automataValidation(&results, &ref_results, true);
      }
      if (!plo->use_uvm) {
        delete[] h_results_v;
        delete[] h_results_i;
      }
      delete[] h_results;
    }
  }

  nblb.release(true);
  csr.release();
  csr.releaseDevice();
  ms_aos.release();
  ms_soa.release();

  gcsr.release();
  gms.release();
  gna.release();
  gaas.release();

  delete h_results_size;
}

// O4
void ngap::launch_non_blocking_r1_groups() {
  tge_log("automata nonblocking::launch!", BOLDBLUE);

  Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
  input_stream->copy_to_device();
  int multi_ss_size = symbol_streams[0].get_length();

  // Csr
  Csr csr(graph);
  csr.fromCoo(graph.edge_pairs->get_host());
  csr.moveToDevice();
  Matchset ms = graph.get_matchset_device(plo->use_soa);

  GroupCsr gcsr;
  GroupMatchset gms;
  GroupNodeAttrs gna;
  GroupAAS gaas;
  gcsr.init(gs);
  gms.init(gs, plo->use_soa);
  gna.init(gs);
  gaas.init(gs);

  NonBlockingBuffer nblb;

  nblb.init_nfagroups(input_stream, input_stream->size(), num_seg,
                      multi_ss_size, gs, plo);

  // NonBlockingBuffer *d_nblb;
  // CHECK_ERROR(cudaMalloc(&d_nblb, sizeof(NonBlockingBuffer)));
  // CHECK_ERROR(cudaMemcpy(d_nblb, &nblb, sizeof(NonBlockingBuffer),
  //                        cudaMemcpyHostToDevice));

  dim3 blocksPerGrid(plo->group_num, num_seg, 1);
  dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
  if (nblb.unique)
    calculateTheoreticalOccupancy2(advanceAndFilterNonBlockingR1Groups<true>,
                                   BLOCK_SIZE);
  else
    calculateTheoreticalOccupancy2(advanceAndFilterNonBlockingR1Groups<false>,
                                   BLOCK_SIZE);
  CHECK_LAST_ERROR

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  if (nblb.unique)
    advanceAndFilterNonBlockingR1Groups<true>
        <<<blocksPerGrid, threadsPerBlock>>>(
            nblb, input_stream->get_dev(), multi_ss_size, gms, gna, gaas, gcsr);
  else
    advanceAndFilterNonBlockingR1Groups<false>
        <<<blocksPerGrid, threadsPerBlock>>>(
            nblb, input_stream->get_dev(), multi_ss_size, gms, gna, gaas, gcsr);
  // cudaDeviceSynchronize();

  CHECK_LAST_ERROR
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double throughput = (double)input_stream->size() / (milliseconds * 1000);
  std::cout << "ngap elapsed time: " << milliseconds / 1000.0
            << " seconds, throughput = " << throughput << " MB/s " << std::endl;

  uint *h_buffer_end = new uint[num_seg];
  uint buffer_total_size = 0;
  CHECK_ERROR(cudaMemcpy((void *)h_buffer_end, nblb.d_buffer_end,
                         sizeof(uint) * num_seg, cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_seg; i++) {
    buffer_total_size += h_buffer_end[i];
  }
  // printf("buffer_total_size: %u \n", buffer_total_size);

  unsigned long long int *h_results_size = new unsigned long long int;
  CHECK_ERROR(cudaMemcpy((void *)h_results_size, nblb.d_results_size,
                         sizeof(unsigned long long int),
                         cudaMemcpyDeviceToHost));
  std::cout << "Results number: " << *h_results_size << std::endl;

  bool validation = plo->validation;
  if (validation) {
    if (plo->report_off) {
      int dupnum =
          plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
      if (plo->quick_validation >= 0 &&
          plo->quick_validation * dupnum <= *h_results_size) {
        tge_log("Quick Validation PASS!", BOLDGREEN);
      } else {
        tge_log("Quick Validation FAILED!", BOLDRED);
      }
    } else {
      uint32_t *h_results_v;
      uint32_t *h_results_i;
      uint64_t *h_results = new uint64_t[*h_results_size];
      if (plo->use_uvm) {
        h_results_v = nblb.d_results_v;
        h_results_i = nblb.d_results_i;
      } else {
        h_results_v = new uint32_t[*h_results_size];
        h_results_i = new uint32_t[*h_results_size];
        CHECK_ERROR(cudaMemcpy((void *)h_results_v, nblb.d_results_v,
                               sizeof(uint32_t) * *h_results_size,
                               cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy((void *)h_results_i, nblb.d_results_i,
                               sizeof(uint32_t) * *h_results_size,
                               cudaMemcpyDeviceToHost));
      }

      auto addResult = [](uint32_t node, uint32_t index) {
        uint64_t r = 0;
        r = (uint32_t)node;
        r = r << 32;
        r = r | (uint32_t)index;
        return r;
      };
      for (int i = 0; i < *h_results_size; i++)
        h_results[i] = addResult(h_results_v[i], h_results_i[i]);
      std::vector<uint64_t> results, ref_results, db_results, ref_db_results;
      // for (int i = 0; i < *h_results_size; i++)
      //   results.push_back(h_results[i]);
      int mc = *h_results_size;
      results.resize(mc);
      std::for_each(std::execution::par_unseq, std::begin(results),
                    std::end(results), [&](uint64_t &r) {
                      u_int32_t i = &r - &results[0];
                      assert(i < mc);
                      results[i] = h_results[i];
                    });
      std::sort(std::execution::par_unseq, results.begin(), results.end(),
                compareResult);
      results.erase(std::unique(std::execution::par_unseq, results.begin(),
                                results.end()),
                    results.end());
      std::cout << "Unique results number: " << results.size() << std::endl;
      printf("validation start.\n");

      if (plo->quick_validation >= 0) {
        int dupnum =
            plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
        if (plo->quick_validation * dupnum == results.size()) {
          tge_log("Quick Validation PASS!", BOLDGREEN);
        } else {
          tge_log("Quick Validation FAILED!", BOLDRED);
        }
      } else {

        bool isDup = false;
        if (plo->duplicate_input_stream > 1 &&
            (plo->split_chunk_size == -1 ||
             plo->split_chunk_size == plo->input_len)) {
          isDup = true;
        }
        automataGroupsReference(gs, input_stream->get_host(), num_seg,
                                multi_ss_size, &ref_results, &ref_db_results,
                                DEBUG_ITER, gcsr, isDup);
        printf("\n############ Validate result ############ \n");
        automataValidation(&results, &ref_results, true);
      }

      if (!plo->use_uvm) {
        delete[] h_results_v;
        delete[] h_results_i;
      }
      delete[] h_results;
    }
  }

  nblb.release(true);
  csr.release();
  csr.releaseDevice();
  ms.release();

  gcsr.release();
  gms.release();
  gna.release();
  gaas.release();

  delete h_results_size;
}

void ngap::launch_non_blocking_r2_groups() {
  tge_log("automata nonblocking::launch!", BOLDBLUE);

  Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
  input_stream->copy_to_device();
  int multi_ss_size = symbol_streams[0].get_length();

  // Csr
  Csr csr(graph);
  csr.fromCoo(graph.edge_pairs->get_host());
  csr.moveToDevice();
  Matchset ms = graph.get_matchset_device(plo->use_soa);

  GroupCsr gcsr;
  GroupMatchset gms;
  GroupNodeAttrs gna;
  GroupAAS gaas;
  gcsr.init(gs);
  gms.init(gs, plo->use_soa);
  gna.init(gs);
  gaas.init(gs);

  NonBlockingBuffer nblb;

  nblb.init_nfagroups(input_stream, input_stream->size(), num_seg,
                      multi_ss_size, gs, plo);

  // NonBlockingBuffer *d_nblb;
  // CHECK_ERROR(cudaMalloc(&d_nblb, sizeof(NonBlockingBuffer)));
  // CHECK_ERROR(cudaMemcpy(d_nblb, &nblb, sizeof(NonBlockingBuffer),
  //                        cudaMemcpyHostToDevice));

  dim3 blocksPerGrid(plo->group_num, num_seg, 1);
  dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
  if (nblb.unique)
    calculateTheoreticalOccupancy2(advanceAndFilterNonBlockingR2Groups<true>,
                                   BLOCK_SIZE);
  else
    calculateTheoreticalOccupancy2(advanceAndFilterNonBlockingR2Groups<false>,
                                   BLOCK_SIZE);
  CHECK_LAST_ERROR

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  if (nblb.unique)
    advanceAndFilterNonBlockingR2Groups<true>
        <<<blocksPerGrid, threadsPerBlock>>>(
            nblb, input_stream->get_dev(), multi_ss_size, gms, gna, gaas, gcsr);
  else
    advanceAndFilterNonBlockingR2Groups<false>
        <<<blocksPerGrid, threadsPerBlock>>>(
            nblb, input_stream->get_dev(), multi_ss_size, gms, gna, gaas, gcsr);
  // cudaDeviceSynchronize();

  CHECK_LAST_ERROR
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double throughput = (double)input_stream->size() / (milliseconds * 1000);
  std::cout << "ngap elapsed time: " << milliseconds / 1000.0
            << " seconds, throughput = " << throughput << " MB/s " << std::endl;

  uint *h_buffer_end = new uint[num_seg];
  uint buffer_total_size = 0;
  CHECK_ERROR(cudaMemcpy((void *)h_buffer_end, nblb.d_buffer_end,
                         sizeof(uint) * num_seg, cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_seg; i++) {
    buffer_total_size += h_buffer_end[i];
  }
  // printf("buffer_total_size: %u \n", buffer_total_size);

  unsigned long long int *h_results_size = new unsigned long long int;
  CHECK_ERROR(cudaMemcpy((void *)h_results_size, nblb.d_results_size,
                         sizeof(unsigned long long int),
                         cudaMemcpyDeviceToHost));
  std::cout << "Results number: " << *h_results_size << std::endl;

  bool validation = plo->validation;
  if (validation) {
    if (plo->report_off) {
      int dupnum =
          plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
      if (plo->quick_validation >= 0 &&
          plo->quick_validation * dupnum <= *h_results_size) {
        tge_log("Quick Validation PASS!", BOLDGREEN);
      } else {
        tge_log("Quick Validation FAILED!", BOLDRED);
      }
    } else {
      uint32_t *h_results_v;
      uint32_t *h_results_i;
      uint64_t *h_results = new uint64_t[*h_results_size];
      if (plo->use_uvm) {
        h_results_v = nblb.d_results_v;
        h_results_i = nblb.d_results_i;
      } else {
        h_results_v = new uint32_t[*h_results_size];
        h_results_i = new uint32_t[*h_results_size];
        CHECK_ERROR(cudaMemcpy((void *)h_results_v, nblb.d_results_v,
                               sizeof(uint32_t) * *h_results_size,
                               cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy((void *)h_results_i, nblb.d_results_i,
                               sizeof(uint32_t) * *h_results_size,
                               cudaMemcpyDeviceToHost));
      }

      auto addResult = [](uint32_t node, uint32_t index) {
        uint64_t r = 0;
        r = (uint32_t)node;
        r = r << 32;
        r = r | (uint32_t)index;
        return r;
      };
      for (int i = 0; i < *h_results_size; i++)
        h_results[i] = addResult(h_results_v[i], h_results_i[i]);
      std::vector<uint64_t> results, ref_results, db_results, ref_db_results;
      // for (int i = 0; i < *h_results_size; i++)
      //   results.push_back(h_results[i]);
      int mc = *h_results_size;
      results.resize(mc);
      std::for_each(std::execution::par_unseq, std::begin(results),
                    std::end(results), [&](uint64_t &r) {
                      u_int32_t i = &r - &results[0];
                      assert(i < mc);
                      results[i] = h_results[i];
                    });
      std::sort(std::execution::par_unseq, results.begin(), results.end(),
                compareResult);
      results.erase(std::unique(std::execution::par_unseq, results.begin(),
                                results.end()),
                    results.end());
      std::cout << "Unique results number: " << results.size() << std::endl;
      printf("validation start.\n");

      if (plo->quick_validation >= 0) {
        int dupnum =
            plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
        if (plo->quick_validation * dupnum == results.size()) {
          tge_log("Quick Validation PASS!", BOLDGREEN);
        } else {
          tge_log("Quick Validation FAILED!", BOLDRED);
        }
      } else {

        bool isDup = false;
        if (plo->duplicate_input_stream > 1 &&
            (plo->split_chunk_size == -1 ||
             plo->split_chunk_size == plo->input_len)) {
          isDup = true;
        }
        automataGroupsReference(gs, input_stream->get_host(), num_seg,
                                multi_ss_size, &ref_results, &ref_db_results,
                                DEBUG_ITER, gcsr, isDup);
        printf("\n############ Validate result ############ \n");
        automataValidation(&results, &ref_results, true);
      }

      if (!plo->use_uvm) {
        delete[] h_results_v;
        delete[] h_results_i;
      }
      delete[] h_results;
    }
  }

  nblb.release(true);
  csr.release();
  csr.releaseDevice();
  ms.release();

  gcsr.release();
  gms.release();
  gna.release();
  gaas.release();

  delete h_results_size;
}

// OA
void ngap::launch_non_blocking_all_groups() {
  tge_log("automata nonblocking all ::launch!", BOLDBLUE);

  auto start1 = std::chrono::high_resolution_clock::now();

  Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
  input_stream->copy_to_device();
  int multi_ss_size = symbol_streams[0].get_length();

  // Csr
  Csr csr(graph);
  csr.fromCoo(graph.edge_pairs->get_host());
  csr.moveToDevice();
  Matchset ms = graph.get_matchset_device(plo->use_soa);

  GroupCsr gcsr;
  GroupMatchset gms;
  GroupNodeAttrs gna;
  GroupAAS gaas;
  initGroupCsrWithPrec(gcsr, gs, plo->precompute_depth,
                       plo->compress_prec_table);
  // gcsr.init(gs);
  gms.init(gs, plo->use_soa);
  gna.init(gs);
  gaas.init(gs);

  NonBlockingBuffer nblb;
  nblb.init_nfagroups(input_stream, input_stream->size(), num_seg,
                      multi_ss_size, gs, plo);

  dim3 blocksPerGrid(plo->group_num, num_seg, 1);
  dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
  auto end1 = std::chrono::high_resolution_clock::now();
  auto duration1 =
      std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  printf("prepare time = %f s\n", duration1.count() / 1000000.0);
  if (nblb.unique) {
    switch (plo->precompute_depth) {
    case 0:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingAllGroups<true, 0, false, false>,
          BLOCK_SIZE);
      break;
    case 1:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingAllGroups<true, 1, false, false>,
          BLOCK_SIZE);
      break;
    case 2:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingAllGroups<true, 2, false, false>,
          BLOCK_SIZE);
      break;
    case 3:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingAllGroups<true, 3, false, false>,
          BLOCK_SIZE);
      break;
    default:
      break;
    }
  } else {
    switch (plo->precompute_depth) {
    case 0:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingAllGroups<false, 0, false, false>,
          BLOCK_SIZE);
      break;
    case 1:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingAllGroups<false, 1, false, false>,
          BLOCK_SIZE);
      break;
    case 2:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingAllGroups<false, 2, false, false>,
          BLOCK_SIZE);
      break;
    case 3:
      calculateTheoreticalOccupancy2(
          advanceAndFilterNonBlockingAllGroups<false, 3, false, false>,
          BLOCK_SIZE);
      break;
    default:
      break;
    }
  }

  auto startNonBlockAutomata = [&](bool &passValidation) -> double {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    if (plo->adaptive_aas) {
      printf("Use adaptive aas\n");
      if (plo->motivate_worklist_length) {
        if (nblb.unique) {
          switch (plo->precompute_depth) {
          case 0:
            advanceAndFilterNonBlockingAllGroups<true, 0, true, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 1:
            advanceAndFilterNonBlockingAllGroups<true, 1, true, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 2:
            advanceAndFilterNonBlockingAllGroups<true, 2, true, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 3:
            advanceAndFilterNonBlockingAllGroups<true, 3, true, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          default:
            break;
          }
        } else {
          switch (plo->precompute_depth) {
          case 0:
            advanceAndFilterNonBlockingAllGroups<false, 0, true, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 1:
            advanceAndFilterNonBlockingAllGroups<false, 1, true, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 2:
            advanceAndFilterNonBlockingAllGroups<false, 2, true, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 3:
            advanceAndFilterNonBlockingAllGroups<false, 3, true, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          default:
            break;
          }
        }
      } else {
        if (nblb.unique) {
          switch (plo->precompute_depth) {
          case 0:
            advanceAndFilterNonBlockingAllGroups<true, 0, false, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 1:
            advanceAndFilterNonBlockingAllGroups<true, 1, false, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 2:
            advanceAndFilterNonBlockingAllGroups<true, 2, false, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 3:
            advanceAndFilterNonBlockingAllGroups<true, 3, false, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          default:
            break;
          }
        } else {
          switch (plo->precompute_depth) {
          case 0:
            advanceAndFilterNonBlockingAllGroups<false, 0, false, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 1:
            advanceAndFilterNonBlockingAllGroups<false, 1, false, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 2:
            advanceAndFilterNonBlockingAllGroups<false, 2, false, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 3:
            advanceAndFilterNonBlockingAllGroups<false, 3, false, true>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          default:
            break;
          }
        }
      }

    } else {
      if (plo->motivate_worklist_length) {
        if (nblb.unique) {
          switch (plo->precompute_depth) {
          case 0:
            advanceAndFilterNonBlockingAllGroups<true, 0, true, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 1:
            advanceAndFilterNonBlockingAllGroups<true, 1, true, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 2:
            advanceAndFilterNonBlockingAllGroups<true, 2, true, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 3:
            advanceAndFilterNonBlockingAllGroups<true, 3, true, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          default:
            break;
          }
        } else {
          switch (plo->precompute_depth) {
          case 0:
            advanceAndFilterNonBlockingAllGroups<false, 0, true, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 1:
            advanceAndFilterNonBlockingAllGroups<false, 1, true, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 2:
            advanceAndFilterNonBlockingAllGroups<false, 2, true, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 3:
            advanceAndFilterNonBlockingAllGroups<false, 3, true, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          default:
            break;
          }
        }
      } else {
        if (nblb.unique) {
          switch (plo->precompute_depth) {
          case 0:
            advanceAndFilterNonBlockingAllGroups<true, 0, false, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 1:
            advanceAndFilterNonBlockingAllGroups<true, 1, false, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 2:
            advanceAndFilterNonBlockingAllGroups<true, 2, false, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 3:
            advanceAndFilterNonBlockingAllGroups<true, 3, false, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          default:
            break;
          }
        } else {
          switch (plo->precompute_depth) {
          case 0:
            advanceAndFilterNonBlockingAllGroups<false, 0, false, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 1:
            advanceAndFilterNonBlockingAllGroups<false, 1, false, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 2:
            advanceAndFilterNonBlockingAllGroups<false, 2, false, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          case 3:
            advanceAndFilterNonBlockingAllGroups<false, 3, false, false>
                <<<blocksPerGrid, threadsPerBlock>>>(
                    nblb, input_stream->get_dev(), multi_ss_size, gms, gna,
                    gaas, gcsr);
            break;
          default:
            break;
          }
        }
      }
    }

    // cudaDeviceSynchronize();

    CHECK_LAST_ERROR

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double throughput = (double)input_stream->size() / (milliseconds * 1000);

    uint *h_buffer_end = new uint[num_seg];
    uint buffer_total_size = 0;
    CHECK_ERROR(cudaMemcpy((void *)h_buffer_end, nblb.d_buffer_end,
                           sizeof(uint) * num_seg, cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_seg; i++) {
      buffer_total_size += h_buffer_end[i];
    }
    // printf("buffer_total_size: %u \n", buffer_total_size);

    unsigned long long int *h_results_size = new unsigned long long int;
    CHECK_ERROR(cudaMemcpy((void *)h_results_size, nblb.d_results_size,
                           sizeof(unsigned long long int),
                           cudaMemcpyDeviceToHost));
    std::cout << "Results number: " << *h_results_size << std::endl;
    if (plo->motivate_worklist_length) {
      int *h_froniter_end = new int;
      CHECK_ERROR(cudaMemcpy((void *)h_froniter_end, nblb.d_froniter_end,
                             sizeof(int), cudaMemcpyDeviceToHost));

      int *h_froniter_length = new int[*h_froniter_end];
      CHECK_ERROR(cudaMemcpy((void *)h_froniter_length, nblb.d_froniter_length,
                             sizeof(int) * *h_froniter_end,
                             cudaMemcpyDeviceToHost));

      std::string path =
          "/home/tge/workspace/gpunfa-ngap/froniter_length/all/" +
          plo->app_name + ".txt";
      std::ofstream froniter_length_file(path);
      printf("Save froniter length file to %s\n", path.c_str());
      if (froniter_length_file.is_open()) {
        for (int i = 0; i < *h_froniter_end; i++) {
          froniter_length_file << h_froniter_length[i] << "\n";
        }
        froniter_length_file.close();
      } else
        assert(false);
      delete[] h_froniter_length;
    }

    bool validation = plo->validation;
    if (validation) {

      if (plo->report_off) {
        unsigned long long int dupnum =
            plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
        unsigned long long int validation_num = plo->quick_validation * dupnum;
        if (plo->quick_validation >= 0 && validation_num <= *h_results_size) {
          if (validation_num == *h_results_size) {
            tge_log("Quick Validation PASS! (report off, perfect)", BOLDGREEN);
          } else {
            tge_log("Quick Validation PASS! (report off, not perfect)",
                    BOLDGREEN);
          }
          passValidation = true;
          // return throughput;
        } else {
          tge_log("Quick Validation FAILED! (report off)", BOLDRED);
          passValidation = false;
          // return -1;
        }
      } else {
        uint32_t *h_results_v;
        uint32_t *h_results_i;
        uint64_t *h_results = new uint64_t[*h_results_size];
        if (plo->use_uvm) {
          h_results_v = nblb.d_results_v;
          h_results_i = nblb.d_results_i;
        } else {
          h_results_v = new uint32_t[*h_results_size];
          h_results_i = new uint32_t[*h_results_size];
          CHECK_ERROR(cudaMemcpy((void *)h_results_v, nblb.d_results_v,
                                 sizeof(uint32_t) * *h_results_size,
                                 cudaMemcpyDeviceToHost));
          CHECK_ERROR(cudaMemcpy((void *)h_results_i, nblb.d_results_i,
                                 sizeof(uint32_t) * *h_results_size,
                                 cudaMemcpyDeviceToHost));
        }

        auto addResult = [](uint32_t node, uint32_t index) {
          uint64_t r = 0;
          r = (uint32_t)node;
          r = r << 32;
          r = r | (uint32_t)index;
          return r;
        };
        for (unsigned long long int i = 0; i < *h_results_size; i++)
          h_results[i] = addResult(h_results_v[i], h_results_i[i]);

        std::vector<uint64_t> results, ref_results, db_results, ref_db_results;
        // for (int i = 0; i < *h_results_size; i++)
        //   results.push_back(h_results[i]);
        unsigned long long int mc = *h_results_size;
        results.resize(mc);
        std::for_each(std::execution::par_unseq, std::begin(results),
                      std::end(results), [&](uint64_t &r) {
                        u_int32_t i = &r - &results[0];
                        assert(i < mc);
                        results[i] = h_results[i];
                      });
        std::sort(std::execution::par_unseq, results.begin(), results.end(),
                  compareResult);
        results.erase(std::unique(std::execution::par_unseq, results.begin(),
                                  results.end()),
                      results.end());
        std::cout << "Unique results number: " << results.size() << std::endl;
        printf("validation start.\n");

        if (plo->quick_validation >= 0) {
          unsigned long long int dupnum =
              plo->duplicate_input_stream > 1 ? plo->duplicate_input_stream : 1;
          unsigned long long int validation_num =
              plo->quick_validation * dupnum;
          if (validation_num == results.size()) {
            tge_log("Quick Validation PASS!", BOLDGREEN);
            passValidation = true;
            // return throughput;
          } else {
            tge_log("Quick Validation FAILED!", BOLDRED);
            passValidation = false;
            // return -1;
          }
        } else {

          bool isDup = false;
          if (plo->duplicate_input_stream > 1 &&
              (plo->split_chunk_size == -1 ||
               plo->split_chunk_size == plo->input_len)) {
            isDup = true;
          }
          automataGroupsReference(gs, input_stream->get_host(), num_seg,
                                  multi_ss_size, &ref_results, &ref_db_results,
                                  DEBUG_ITER, gcsr, isDup);
          printf("\n############ Validate result ############ \n");
          if (automataValidation(&results, &ref_results, true)) {
            passValidation = true;
            // return throughput;
          } else {
            passValidation = false;
            // return -1;
          }
        }

        if (!plo->use_uvm) {
          delete[] h_results_v;
          delete[] h_results_i;
        }
        delete[] h_results;
      }
    } else {
      passValidation = true;
    }
    delete h_results_size;
    if (passValidation) {
      std::cout << "ngap elapsed time: " << milliseconds / 1000.0
                << " seconds, throughput = " << throughput << " MB/s "
                << std::endl;
    } else {
      std::cout << "ngap elapsed time: " << milliseconds / 1000.0
                << " seconds, WRONGTHRPUT(" << throughput << ") MB/s "
                << std::endl;
    }
    return throughput;
  };

  CHECK_LAST_ERROR

  if (plo->tuning == true) {
    std::vector<int> tuning_fetch_sizes{512,    5120,    10240,   25600,
                                        256000, 2560000, 25600000};
    std::vector<int> tuning_add_aas_intervals{4096, 1024, 2048, 1539, 512};
    std::vector<int> tuning_active_thresholds{0, 8, 16, 24, 32};
    double max_throughput = -1;
    std::vector<int> best_choice(3);

    auto tuning_total_start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < tuning_fetch_sizes.size(); i++) {
      for (int k = 0; k < tuning_active_thresholds.size(); k++) {
        for (int j = 0; j < tuning_add_aas_intervals.size(); j++) {
          auto tuning_start_time = std::chrono::high_resolution_clock::now();
          int fetch_size = tuning_fetch_sizes[i];
          int add_aas_interval = tuning_add_aas_intervals[j];
          int active_threshold = tuning_active_thresholds[k];
          nblb.data_buffer_fetch_size = fetch_size;
          nblb.add_aas_interval = add_aas_interval;
          nblb.active_threshold = active_threshold;
          printf("[Tuning] Try choice: \n   fetch_size=%d, "
                 "add_aas_interval=%d, active_threshold=%d\n",
                 fetch_size, add_aas_interval, active_threshold);
          bool passValidation = true;
          double throughput = startNonBlockAutomata(passValidation);
          if (passValidation == true && throughput > 0 &&
              throughput > max_throughput) {
            max_throughput = throughput;
            best_choice = {fetch_size, add_aas_interval, active_threshold};
            printf(
                "[Tuning] Update best choice: \n   fetch_size=%d, "
                "add_aas_interval=%d, active_threshold=%d max_throughput=%f\n",
                fetch_size, add_aas_interval, active_threshold, max_throughput);
          } else {
            printf(
                "[Tuning] Keep best choice: \n   fetch_size=%d, "
                "add_aas_interval=%d, active_threshold=%d max_throughput=%f\n",
                best_choice[0], best_choice[1], best_choice[2], max_throughput);
          }
          nblb.reset(input_stream, input_stream->size(), multi_ss_size,
                     plo->group_num, gs, plo);
          CHECK_LAST_ERROR
          auto tuning_end_time = std::chrono::high_resolution_clock::now();
          auto tuning_duration =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  tuning_end_time - tuning_start_time);
          std::cout << "[Tuning] "
                    << (double)tuning_duration.count() / 1000000.0
                    << " seconds\n";
        }
      }
    }
    auto tuning_total_end_time = std::chrono::high_resolution_clock::now();
    auto tuning_total_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            tuning_total_end_time - tuning_total_start_time);
    std::cout << "[Tuning] Total time: "
              << (double)tuning_total_duration.count() / 1000000.0
              << " seconds\n";
    printf("[Tuning] Tuning completed: the best choice is\n   fetch_size=%d, "
           "add_aas_interval=%d, active_threshold=%d\n",
           best_choice[0], best_choice[1], best_choice[2]);
  } else {
    bool passValidation = true;
    startNonBlockAutomata(passValidation);
    if (!passValidation && plo->try_adaptive_aas) {
      plo->adaptive_aas = true;
      printf("Try adaptive aas\n");
      nblb.release(true);
      nblb.init_nfagroups(input_stream, input_stream->size(), num_seg,
                          multi_ss_size, gs, plo);
      startNonBlockAutomata(passValidation);
    }
  }

  nblb.release(true);
  csr.release();
  csr.releaseDevice();
  ms.release();

  gcsr.release();
  gms.release();
  gna.release();
  gaas.release();
}

void ngap::launch_kernel() {}

void ngap::automataReference(Graph &g, uint8_t *input_str, int num_seg,
                             int input_length, std::vector<uint64_t> *results,
                             std::vector<uint64_t> *db_results, int debug_iter,
                             Csr csr) {

  auto addResult = [](uint32_t node, uint32_t index,
                      std::vector<uint64_t> *results) {
    uint64_t r = 0;
    r = (uint32_t)node;
    r = r << 32;
    r = r | (uint32_t)index;
#pragma omp critical
    results->push_back(r);
  };

  auto start_time = std::chrono::high_resolution_clock::now();
  omp_set_num_threads(num_seg);
#pragma omp parallel
  {
    std::vector<int> frontiers[2];
    int ns = omp_get_thread_num();
    int iter = ns * input_length;
    frontiers[0].clear();
    frontiers[1].clear();
    if (ns == 0)
      for (int i = 0; i < g.startActiveNum; i++) {
        frontiers[0].push_back(g.start_active_nodes->get_host()[i]);
      }
    for (int i = 0; i < g.alwaysActiveNum; i++) {
      frontiers[0].push_back(g.always_active_nodes->get_host()[i]);
    }

    while (!frontiers[iter % 2].empty() && iter < input_length * (ns + 1)) {
      // auto printFrontier = [iter](std::vector<int> frontier,
      //                             const char *message) {
      //   printf("[CPU automata]:%d %s frontier:", iter, message);
      //   for (auto v : frontier) {
      //     printf("%d, ", v);
      //   }
      //   printf("\n");
      // };
      char current_symbol = input_str[iter];
      auto &curr_frontier = frontiers[iter % 2];
      auto &next_frontier = frontiers[(iter + 1) % 2];
      next_frontier.clear();
      for (auto v : curr_frontier) {
#ifdef DEBUG_PL_FILTER_ITER
        // debug per iteration
        if (iter == debug_iter) {
          addResult(v, iter, db_results);
        }
#endif
        if (g.symbol_sets->get_host()[v].test(current_symbol)) {
          // report
          if (g.node_attrs->get_host()[v] & 0b10) {
            addResult(v, iter, results);
          }
          int e_start = csr.GetNeighborListOffset(v);
          int e_end = e_start + csr.GetNeighborListLength(v);
          for (int e = e_start; e < e_end; e++) {
            int u = csr.GetEdgeDest(e);
            next_frontier.push_back(u);
          }
        }
      }
      if (plo->unique) {
        std::sort(next_frontier.begin(), next_frontier.end());
        next_frontier.erase(
            std::unique(next_frontier.begin(), next_frontier.end()),
            next_frontier.end());
      }
#ifdef DEBUG_PL_ADVANCE_CONCAT
      for (int i = 0; i < g.alwaysActiveNum; i++) {
        next_frontier.push_back(g.always_active_nodes->get_host()[i]);
      }
#endif
      iter++;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  double throughput =
      (double)input_length * num_seg / (duration.count() * 1000);
  std::cout << "Reference CPU elapsed time: " << duration.count() << " ms,"
            << " " << throughput << " MB/s" << std::endl;
}

void ngap::automataGroupsReference(std::vector<Graph *> &gs, uint8_t *input_str,
                                   int num_seg, int input_length,
                                   std::vector<uint64_t> *results,
                                   std::vector<uint64_t> *db_results,
                                   int debug_iter, GroupCsr gcsr, bool isDup) {

  auto addResult = [](uint32_t node, uint32_t index, uint32_t nfa_index,
                      std::vector<uint64_t> *results) {
    uint64_t r = 0;
    // assert(node < 0xffffff);
    node = (node | (nfa_index << 22));
    r = (uint32_t)node;
    r = r << 32;
    r = r | (uint32_t)index;
#pragma omp critical
    results->push_back(r);
  };

  auto start_time = std::chrono::high_resolution_clock::now();
  omp_set_num_threads(num_seg * gs.size());

#pragma omp parallel
  {
    std::vector<int> frontiers[2];
    int ns = omp_get_thread_num();
    int nfa_index = ns / num_seg;
    int input_index = ns % num_seg;
    Graph *g = gs[nfa_index];
    Csr csr = gcsr.h_groups_csr[nfa_index];

    int iter = input_index * input_length;
    frontiers[0].clear();
    frontiers[1].clear();
    if (input_index == 0 || isDup)
      for (int i = 0; i < g->startActiveNum; i++) {
        frontiers[0].push_back(g->start_active_nodes->get_host()[i]);
      }
    for (int i = 0; i < g->alwaysActiveNum; i++) {
      frontiers[0].push_back(g->always_active_nodes->get_host()[i]);
    }

    while (!frontiers[iter % 2].empty() &&
           iter < input_length * (input_index + 1)) {
      // auto printFrontier = [iter](std::vector<int> frontier,
      //                             const char *message) {
      //   printf("[CPU automata]:%d %s frontier:", iter, message);
      //   for (auto v : frontier) {
      //     printf("%d, ", v);
      //   }
      //   printf("\n");
      // };
      char current_symbol = input_str[iter];
      auto &curr_frontier = frontiers[iter % 2];
      auto &next_frontier = frontiers[(iter + 1) % 2];
      next_frontier.clear();
      for (auto v : curr_frontier) {
#ifdef DEBUG_PL_FILTER_ITER
        // debug per iteration
        if (iter == debug_iter) {
          addResult(v, iter, nfa_index, db_results);
        }
#endif
        if (g->symbol_sets->get_host()[v].test(current_symbol)) {
          // report
          if (g->node_attrs->get_host()[v] & 0b10) {
            addResult(v, iter, nfa_index, results);
          }
          int e_start = csr.GetNeighborListOffset(v);
          int e_end = e_start + csr.GetNeighborListLength(v);
          for (int e = e_start; e < e_end; e++) {
            int u = csr.GetEdgeDest(e);
            next_frontier.push_back(u);
          }
        }
      }
      if (plo->unique) {
        std::sort(next_frontier.begin(), next_frontier.end());
        next_frontier.erase(
            std::unique(next_frontier.begin(), next_frontier.end()),
            next_frontier.end());
      }
#ifdef DEBUG_PL_ADVANCE_CONCAT
      for (int i = 0; i < g->alwaysActiveNum; i++) {
        next_frontier.push_back(g->always_active_nodes->get_host()[i]);
      }
#endif
      iter++;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  double throughput =
      (double)input_length * num_seg / (duration.count() * 1000);
  std::cout << "Reference CPU elapsed time: " << duration.count() << " ms,"
            << " " << throughput << " MB/s" << std::endl;
}

bool ngap::automataValidation(std::vector<uint64_t> *results,
                              std::vector<uint64_t> *ref_results,
                              bool ifPrintBoth) {
  auto printBoth = [](std::vector<uint64_t> *results,
                      std::vector<uint64_t> *ref_results) {
    auto printVector = [](std::vector<uint64_t> *v) {
      if (v->size() > 40) {
        // if (false) {
        for (int i = 0; i < 40; i++)
          printf("0x%lx, ", (*v)[i]);
        printf("...\n");
      } else {
        for (int i = 0; i < v->size(); i++)
          printf("0x%lx, ", (*v)[i]);
        printf("\n");
      }
    };
    printf("Result(%zu): \n", results->size());
    printVector(results);
    printf("Reference result(%zu): \n", ref_results->size());
    printVector(ref_results);
    printf("\n");
  };

  auto compareResult = [](uint64_t r1, uint64_t r2) -> bool {
    uint32_t input_1, state_1, input_2, state_2;
    input_1 = (uint32_t)(0xffffffff & r1);
    state_1 = (uint32_t)(r1 >> 32);
    input_2 = (uint32_t)(0xffffffff & r2);
    state_2 = (uint32_t)(r2 >> 32);
    // printf("{%u, %u}, ", state, input);
    if (input_1 == input_2)
      return state_1 < state_2;
    else
      return input_1 < input_2;
  };
  // int *new_end = thrust::unique(thrust::host, A, A + N);
  // std::sort(std::execution::par_unseq, results->begin(), results->end(),
  // compareResult); results->erase(std::unique(std::execution::par_unseq,
  // results->begin(), results->end()), results->end());
  std::sort(std::execution::par_unseq, ref_results->begin(), ref_results->end(),
            compareResult);
  ref_results->erase(std::unique(std::execution::par_unseq,
                                 ref_results->begin(), ref_results->end()),
                     ref_results->end());

  int size = results->size();
  int ref_size = ref_results->size();
  if (size != ref_size) {
    tge_log("Validation FAILED!", BOLDRED);
    std::cerr << BOLDRED << "Validation FAILED!" << RESET << "\n";
    int bound = std::min(size, ref_size);
    for (int i = 0; i < bound; i++) {
      if ((*results)[i] != (*ref_results)[i]) {
        printf("0x%lx != 0x%lx\n", (*results)[i], (*ref_results)[i]);
        break;
      }
    }
    printBoth(results, ref_results);
    return false;
  }
  for (int i = 0; i < size; i++) {
    if ((*results)[i] != (*ref_results)[i]) {
      tge_log("Validation FAILED!", BOLDRED);
      std::cerr << BOLDRED << "Validation FAILED!" << RESET << "\n";
      printBoth(results, ref_results);
      return false;
    }
  }
  tge_log("Validation PASS!", BOLDGREEN);
  // std::cerr<<"Validation PASS!\n";
  if (ifPrintBoth)
    printBoth(results, ref_results);
  return true;
}

/**
 * --------------------------
 * prefix|  precompute result
 * ------|-------------------
 * aa    |
 * ab    |
 * ...   |
 * zy    |
 * zz    |
 * --------------------------
 */

void ngap::recursivePrecomputeForK(Csr &csr, Graph *g, PrecTable *pts, int k,
                                   int max_depth, bool compressPrecTable) {
  assert(k >= 1);
  if (k >= max_depth)
    return;

  PrecTable *last_pt = pts + (k - 1);
  PrecTable *current_pt = pts + k;
  auto start = std::chrono::high_resolution_clock::now();
  current_pt->allocate(256 * last_pt->size, k, compressPrecTable);
  printf("    depth %d has %zu entries\n", k - 1, last_pt->size);
  if (compressPrecTable) {
    for (uint64_t i = 0; i < last_pt->nonzeroVerticesNum; i++) {
      int *vertices = last_pt->vertices[i];
      uint64_t vl = last_pt->vertices_length[i];
      // prefix + any symbol
      for (uint32_t symbol = 0; symbol < 256; symbol++) {
        std::vector<int> vk, rk;
        for (uint64_t j = 0; j < vl; j++) {
          int vertex = vertices[j];
          int e_start = csr.GetNeighborListOffset(vertex);
          int e_end = e_start + csr.GetNeighborListLength(vertex);
          for (int e = e_start; e < e_end; e++) { // advance
            int child = csr.GetEdgeDest(e);
            if (g->symbol_sets->get_host()[child].test(symbol)) { // filter
              vk.push_back(child);
              if (g->node_attrs->get_host()[child] & 0b10) { // report
                rk.push_back(child);
              }
            }
          }
        }
        current_pt->setVertices(last_pt->nonzeroVerticesMap[i] * 256 + symbol,
                                vk);
        current_pt->setResults(last_pt->nonzeroVerticesMap[i] * 256 + symbol,
                               rk);
      }
    }
    // }

  } else {
    for (long unsigned int i = 0; i < last_pt->size; i++) {
      int *vertices = last_pt->vertices[i];
      int vertices_length = last_pt->vertices_length[i];
      // prefix + any symbol
      if (vertices_length == 0)
        continue;
      for (uint symbol = 0; symbol < 256; symbol++) {
        std::vector<int> vk, rk;
        for (int j = 0; j < vertices_length; j++) {
          int vertex = vertices[j];
          int e_start = csr.GetNeighborListOffset(vertex);
          int e_end = e_start + csr.GetNeighborListLength(vertex);
          for (int e = e_start; e < e_end; e++) { // advance
            int child = csr.GetEdgeDest(e);
            if (g->symbol_sets->get_host()[child].test(symbol)) { // filter
              vk.push_back(child);
              if (g->node_attrs->get_host()[child] & 0b10) { // report
                rk.push_back(child);
              }
            }
          }
        }
        assert((entry_index * 256 + symbol) < current_pt->size);
        current_pt->setVertices(i * 256 + symbol, vk);
        current_pt->setResults(i * 256 + symbol, rk);
      }
    }

    //     uint64_t entryPerThread = last_pt->size / 64;
    //     omp_set_num_threads(64);
    // #pragma omp parallel
    //     {
    //       int thread_rank = omp_get_thread_num();
    //       for (uint64_t i = 0; i < entryPerThread; i++) {
    //         uint64_t entry_index = thread_rank * entryPerThread + i;
    //         assert(entry_index < last_pt->size);
    //         int *vertices = last_pt->vertices[entry_index];
    //         int vl = last_pt->vertices_length[entry_index];
    //         // prefix + any symbol
    //         for (uint symbol = 0; symbol < 256; symbol++) {
    //           std::vector<int> vk, rk;
    //           for (int j = 0; j < vl; j++) {
    //             int vertex = vertices[j];
    //             int e_start = csr.GetNeighborListOffset(vertex);
    //             int e_end = e_start + csr.GetNeighborListLength(vertex);
    //             for (int e = e_start; e < e_end; e++) { // advance
    //               int child = csr.GetEdgeDest(e);
    //               if (g->symbol_sets->get_host()[child].test(symbol)) { //
    //               filter
    //                 vk.push_back(child);
    //                 if (g->node_attrs->get_host()[child] & 0b10) { // report
    //                   rk.push_back(child);
    //                 }
    //               }
    //             }
    //           }
    //           assert((entry_index * 256 + symbol) < current_pt->size);
    //           current_pt->setVertices(entry_index * 256 + symbol, vk);
    //           current_pt->setResults(entry_index * 256 + symbol, rk);
    //         }
    //       }
    //     }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  current_pt->printHistogram();
  printf("    table_%d_time = %f s\n", k, duration.count() / 1000000.0);
  recursivePrecomputeForK(csr, g, pts, k + 1, max_depth, compressPrecTable);
}

int ngap::getPrecomputeResultsForKGroupsInCsr(Csr &csr, Graph *g, int max_depth,
                                              bool compressPrecTable) {
  int total_size = 0;
  if (max_depth <= 0)
    return 0;
  PrecTable *pts = new PrecTable[max_depth];
  pts[0].allocate(256, 0, compressPrecTable);
  for (uint32_t symbol = 0; symbol < 256; symbol++) {
    std::vector<int> v0, r0;
    for (uint32_t n = 0; n < g->alwaysActiveNum; n++) {
      int vertex = g->always_active_nodes->get_host()[n];
      if (g->symbol_sets->get_host()[vertex].test(symbol)) { // filter
        v0.push_back(vertex);
        if (g->node_attrs->get_host()[vertex] & 0b10) { // report
          r0.push_back(vertex);
        }
      }
    }
    pts[0].setVertices(symbol, v0);
    pts[0].setResults(symbol, r0);
  }
  pts[0].printHistogram();
  recursivePrecomputeForK(csr, g, pts, 1, max_depth, compressPrecTable);

  for (int i = 0; i < max_depth; i++) {
    pts[i].toDevice(plo->pc_use_uvm);
    // pts[i].calcCutoffMedian();
    pts[i].calcCutoff();
    if (plo->precompute_cutoff >= 0) {
      printf("Use user-defined precompute_cutoff %d\n", plo->precompute_cutoff);
      pts[i].cutoff = plo->precompute_cutoff;
    }
    // total_size += pts[i].printHistogram();
    pts[i].releaseHost();
  }

  csr.h_pts = pts;
  CHECK_ERROR(cudaMalloc((void **)&csr.d_pts, sizeof(PrecTable) * max_depth));
  CHECK_ERROR(cudaMemcpy((void *)csr.d_pts, pts, sizeof(PrecTable) * max_depth,
                         cudaMemcpyHostToDevice));
  csr.precompute_depth = max_depth;
  // printf("csr.precompute_depth=%d\n", csr.precompute_depth);
  return total_size;
}

int ngap::getPrecomputeResultsForKGroupsInCsrFake(Csr &csr, Graph *g,
                                                  int max_depth,
                                                  bool compressPrecTable) {
  // int total_size = 0;
  if (max_depth <= 0)
    return 0;

  //   // step 1
  //   {
  //     long long unsigned nonzero_vertex_num = 0;
  //     long long unsigned nonzero_result_num = 0;
  //     long long unsigned matched_num = 0;
  //     long long unsigned report_num = 0;
  //     long long unsigned size = std::pow(256, 1);
  //     long long unsigned entryPerThread = size / 64;
  //     omp_set_num_threads(64);
  // #pragma omp parallel
  // {
  //       int thread_rank = omp_get_thread_num();
  //       for (uint32_t symbol = thread_rank * entryPerThread;
  //            symbol < (thread_rank + 1) * entryPerThread; symbol++) {
  //         long long matched = 0;
  //         long long report = 0;
  //         for (uint32_t n = 0; n < g->alwaysActiveNum; n++) {
  //           int vertex = g->always_active_nodes->get_host()[n];
  //           if (g->symbol_sets->get_host()[vertex].test(symbol)) { // filter
  //             matched++;
  //             if (g->node_attrs->get_host()[vertex] & 0b10) { // report
  //               report++;
  //             }
  //           }
  //         }

  //         if (matched > 0) {
  //           #pragma omp atomic
  //           nonzero_vertex_num++;
  //           #pragma omp atomic
  //           matched_num += matched;
  //         }
  //         if (report > 0) {
  //           #pragma omp atomic
  //           nonzero_result_num++;
  //           #pragma omp atomic
  //           report_num += report;
  //         }
  //       }
  // }
  // {
  //       double tablem;
  //       tablem = (nonzero_vertex_num + nonzero_result_num) * (4) / 1000000.0;
  //       double vltm = matched_num * 4.0 / 1000000;
  //       double rltm = report_num * 4.0 / 1000000;
  //       printf("    table_%dc_index = %f MB\n", 0, tablem);
  //       printf("    table_%dc_content = %f MB\n", 0, vltm + rltm);
  //       printf("    total_%dc_size = %f MB\n", 0, vltm + rltm + tablem);
  // }
  // {
  //       double tablem;
  //       // printf("matched_num=%llu %llu\n", matched_num, report_num);
  //       tablem = 2.0 * size * 4 / 1000000;
  //       double vltm = matched_num * 4.0 / 1000000;
  //       double rltm = report_num * 4.0 / 1000000;
  //       printf("    table_%dd_index = %f MB\n", 0, tablem);
  //       printf("    table_%dd_content = %f MB\n", 0, vltm + rltm);
  //       printf("    total_%dd_size = %f MB\n", 0, vltm + rltm + tablem);
  // }
  //   }

  //   // step 2
  //     {
  //     long long unsigned nonzero_vertex_num = 0;
  //     long long unsigned nonzero_result_num = 0;
  //     long long unsigned matched_num = 0;
  //     long long unsigned report_num = 0;
  //     long long unsigned size = std::pow(256, 2);
  //      long long unsigned entryPerThread = size / 64;
  //     omp_set_num_threads(64);
  // #pragma omp parallel
  // {
  //       int thread_rank = omp_get_thread_num();
  //       for (long long unsigned index = thread_rank * entryPerThread;
  //            index < (thread_rank + 1) * entryPerThread; index++) {
  //       uint32_t symbol1 = index/std::pow(256, 1);
  //       uint32_t symbol2 = index % 256;
  //       long long matched = 0;
  //       long long report = 0;
  //       for (uint32_t n = 0; n < g->alwaysActiveNum; n++) {
  //         int vertex = g->always_active_nodes->get_host()[n];
  //         if (g->symbol_sets->get_host()[vertex].test(symbol1)) { // filter
  //           int e_start = csr.GetNeighborListOffset(vertex);
  //           int e_end = e_start + csr.GetNeighborListLength(vertex);
  //           for (int e = e_start; e < e_end; e++) { // advance
  //             int child = csr.GetEdgeDest(e);
  //             if (g->symbol_sets->get_host()[child].test(symbol2)) { //
  //             filter
  //               matched++;
  //               if (g->node_attrs->get_host()[child] & 0b10) { // report
  //                 report++;
  //               }
  //             }
  //           }
  //         }
  //       }
  //       if (matched > 0) {
  //           #pragma omp atomic
  //           nonzero_vertex_num++;
  //           #pragma omp atomic
  //           matched_num += matched;
  //         }
  //         if (report > 0) {
  //           #pragma omp atomic
  //           nonzero_result_num++;
  //           #pragma omp atomic
  //           report_num += report;
  //         }
  //     }
  // }
  // {
  //       double tablem;
  //       tablem = (nonzero_vertex_num + nonzero_result_num) * (4) / 1000000.0;
  //       double vltm = matched_num * 4.0 / 1000000;
  //       double rltm = report_num * 4.0 / 1000000;
  //       printf("    table_%dc_index = %f MB\n", 1, tablem);
  //       printf("    table_%dc_content = %f MB\n", 1, vltm + rltm);
  //       printf("    total_%dc_size = %f MB\n", 1, vltm + rltm + tablem);
  // }
  // {
  //       double tablem;
  //       tablem = 2.0 * size * 4 / 1000000;
  //       double vltm = matched_num * 4.0 / 1000000;
  //       double rltm = report_num * 4.0 / 1000000;
  //       printf("    table_%dd_index = %f MB\n", 1, tablem);
  //       printf("    table_%dd_content = %f MB\n", 1, vltm + rltm);
  //       printf("    total_%dd_size = %f MB\n", 1, vltm + rltm + tablem);
  // }
  //   }

  //   // step 3
  //     {
  //     long long unsigned nonzero_vertex_num = 0;
  //     long long unsigned nonzero_result_num = 0;
  //     long long unsigned matched_num = 0;
  //     long long unsigned report_num = 0;
  //     long long unsigned size = std::pow(256, 3);
  //      long long unsigned entryPerThread = size / 64;
  //     omp_set_num_threads(64);
  // #pragma omp parallel
  // {
  //       int thread_rank = omp_get_thread_num();
  //       for (long long unsigned  index = thread_rank * entryPerThread;
  //            index < (thread_rank + 1) * entryPerThread; index++) {
  //       // for (long long unsigned index = 0;
  //       //      index < size; index++) {
  //       //   if(index% 10000 == 0)
  //       //   printf("%llu %llu\n", index, size);
  //       uint32_t symbol1 = index / std::pow(256, 2);
  //       uint32_t symbol2 = index / std::pow(256, 1);
  //       uint32_t symbol3 = index % 256;
  //       long long matched = 0;
  //       long long report = 0;
  //       for (uint32_t n = 0; n < g->alwaysActiveNum; n++) {
  //         int vertex = g->always_active_nodes->get_host()[n];
  //         if (g->symbol_sets->get_host()[vertex].test(symbol1)) { // filter
  //           int e_start = csr.GetNeighborListOffset(vertex);
  //           int e_end = e_start + csr.GetNeighborListLength(vertex);
  //           for (int e = e_start; e < e_end; e++) { // advance
  //             int child = csr.GetEdgeDest(e);
  //             if (g->symbol_sets->get_host()[child].test(symbol2)) { //
  //             filter
  //               int e_start2 = csr.GetNeighborListOffset(child);
  //               int e_end2 = e_start2 + csr.GetNeighborListLength(child);
  //               for (int e2 = e_start2; e2 < e_end2; e2++) { // advance
  //                 int childchild = csr.GetEdgeDest(e2);
  //                 if (g->symbol_sets->get_host()[childchild].test(symbol3)) {
  //                 // filter
  //                   matched++;
  //                   if (g->node_attrs->get_host()[childchild] & 0b10) { //
  //                   report
  //                     report++;
  //                   }
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //       if (matched > 0) {
  //           #pragma omp atomic
  //           nonzero_vertex_num++;
  //           #pragma omp atomic
  //           matched_num += matched;
  //         }
  //         if (report > 0) {
  //           #pragma omp atomic
  //           nonzero_result_num++;
  //           #pragma omp atomic
  //           report_num += report;
  //         }
  //     }
  // }
  // {
  //       double tablem;
  //       tablem = (nonzero_vertex_num + nonzero_result_num) * (4) / 1000000.0;
  //       double vltm = matched_num * 4.0 / 1000000;
  //       double rltm = report_num * 4.0 / 1000000;
  //       printf("    table_%dc_index = %f MB\n", 2, tablem);
  //       printf("    table_%dc_content = %f MB\n", 2, vltm + rltm);
  //       printf("    total_%dc_size = %f MB\n", 2, vltm + rltm + tablem);
  // }
  // {
  //       double tablem;
  //       tablem = 2.0 * size * 4 / 1000000;
  //       double vltm = matched_num * 4.0 / 1000000;
  //       double rltm = report_num * 4.0 / 1000000;
  //       printf("    table_%dd_index = %f MB\n", 2, tablem);
  //       printf("    table_%dd_content = %f MB\n", 2, vltm + rltm);
  //       printf("    total_%dd_size = %f MB\n", 2, vltm + rltm + tablem);
  // }
  //   }

  // step 4
  {
    long long unsigned nonzero_vertex_num = 0;
    long long unsigned nonzero_result_num = 0;
    long long unsigned matched_num = 0;
    long long unsigned report_num = 0;
    long long unsigned size = std::pow(256, 4);

    long long unsigned entryPerThread = size / 64;
    omp_set_num_threads(64);
#pragma omp parallel
    {
      int thread_rank = omp_get_thread_num();
      for (long long unsigned index = thread_rank * entryPerThread;
           index < (thread_rank + 1) * entryPerThread; index++) {
        // for (long long unsigned index = 0; index < size; index++) {
        //   if (index % 10000 == 0)
        //     printf("%llu %f\n", index, 1.0*index/size);
        uint32_t symbol1 = index / std::pow(256, 3);
        uint32_t symbol2 = index / std::pow(256, 2);
        uint32_t symbol3 = index / std::pow(256, 1);
        uint32_t symbol4 = index % 256;
        long long matched = 0;
        long long report = 0;
        for (uint32_t n = 0; n < g->alwaysActiveNum; n++) {
          int vertex = g->always_active_nodes->get_host()[n];
          if (g->symbol_sets->get_host()[vertex].test(symbol1)) { // filter
            int e_start = csr.GetNeighborListOffset(vertex);
            int e_end = e_start + csr.GetNeighborListLength(vertex);
            for (int e = e_start; e < e_end; e++) { // advance
              int child = csr.GetEdgeDest(e);
              if (g->symbol_sets->get_host()[child].test(symbol2)) { // filter
                int e_start2 = csr.GetNeighborListOffset(child);
                int e_end2 = e_start2 + csr.GetNeighborListLength(child);
                for (int e2 = e_start2; e2 < e_end2; e2++) { // advance
                  int childchild = csr.GetEdgeDest(e2);
                  if (g->symbol_sets->get_host()[childchild].test(
                          symbol3)) { // filter
                    int e_start3 = csr.GetNeighborListOffset(childchild);
                    int e_end3 =
                        e_start3 + csr.GetNeighborListLength(childchild);
                    for (int e3 = e_start3; e3 < e_end3; e3++) { // advance
                      int childchildchild = csr.GetEdgeDest(e3);
                      if (g->symbol_sets->get_host()[childchildchild].test(
                              symbol4)) { // filter
                        matched++;
                        if (g->node_attrs->get_host()[childchildchild] &
                            0b10) { // report
                          report++;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if (matched > 0) {
#pragma omp atomic
          nonzero_vertex_num++;
#pragma omp atomic
          matched_num += matched;
        }
        if (report > 0) {
#pragma omp atomic
          nonzero_result_num++;
#pragma omp atomic
          report_num += report;
        }
      }
    }
    {
      double tablem;
      tablem = (nonzero_vertex_num + nonzero_result_num) * (4) / 1000000.0;
      double vltm = matched_num * 4.0 / 1000000;
      double rltm = report_num * 4.0 / 1000000;
      printf("    table_%dc_index = %f MB\n", 3, tablem);
      printf("    table_%dc_content = %f MB\n", 3, vltm + rltm);
      printf("    total_%dc_size = %f MB\n", 3, vltm + rltm + tablem);
    }
    {
      double tablem;
      tablem = 2.0 * size * 4 / 1000000;
      double vltm = matched_num * 4.0 / 1000000;
      double rltm = report_num * 4.0 / 1000000;
      printf("    table_%dd_index = %f MB\n", 3, tablem);
      printf("    table_%dd_content = %f MB\n", 3, vltm + rltm);
      printf("    total_%dd_size = %f MB\n", 3, vltm + rltm + tablem);
    }
  }
  return 0;
}

void ngap::getPrecomputeResultsForK(Csr &csr, NonBlockingBuffer &plb,
                                    int max_depth) {
  if (max_depth <= 0)
    return;
  PrecTable *pts = new PrecTable[max_depth];
  pts[0].allocate(256, 0);
  for (uint symbol = 0; symbol < 256; symbol++) {
    std::vector<int> v0, r0;
    for (uint n = 0; n < graph.alwaysActiveNum; n++) {
      int vertex = graph.always_active_nodes->get_host()[n];
      if (graph.symbol_sets->get_host()[vertex].test(symbol)) { // filter
        v0.push_back(vertex);
        if (graph.node_attrs->get_host()[vertex] & 0b10) { // report
          r0.push_back(vertex);
        }
      }
    }
    pts[0].setVertices(symbol, v0);
    pts[0].setResults(symbol, r0);
  }
  recursivePrecomputeForK(csr, &graph, pts, 1, max_depth, false);

  for (int i = 0; i < max_depth; i++) {
    // pts[i].calcCutoffMedian();
    pts[i].calcCutoff();
    if (plo->precompute_cutoff >= 0) {
      printf("Use user-defined precompute_cutoff %d\n", plo->precompute_cutoff);
      pts[i].cutoff = plo->precompute_cutoff;
    }
    pts[i].toDevice(plo->pc_use_uvm);
  }
  plb.h_pts = pts;
  CHECK_ERROR(cudaMalloc((void **)&plb.d_pts, sizeof(PrecTable) * max_depth));
  CHECK_ERROR(cudaMemcpy((void *)plb.d_pts, pts, sizeof(PrecTable) * max_depth,
                         cudaMemcpyHostToDevice));
  plb.precompute_depth = max_depth;
  printf("plb.precompute_depth=%d\n", plb.precompute_depth);
}

void ngap::getPrecomputeResults(Csr &csr, NonBlockingBuffer &plb) {
  std::vector<int> vv1[256];
  std::vector<int> vv2[256][256];
  std::vector<int> vr1[256];
  std::vector<int> vr2[256][256];
  int prec_once_length = 0;
  int prec_twice_length = 0;
  int prec_once_report_length = 0;
  int prec_twice_report_length = 0;
  for (uint symbol1 = 0; symbol1 < 256; symbol1++) {
    std::vector<int> v1, r1;
    for (uint n = 0; n < graph.alwaysActiveNum; n++) {
      int vertex = graph.always_active_nodes->get_host()[n];
      if (graph.symbol_sets->get_host()[vertex].test(symbol1)) {
        prec_once_length++;
        v1.push_back(vertex);
        if (graph.node_attrs->get_host()[vertex] & 0b10) {
          prec_once_report_length++;
          r1.push_back(vertex);
        }
      }
    }
    vv1[symbol1] = v1;
    vr1[symbol1] = r1;
    for (uint symbol2 = 0; symbol2 < 256; symbol2++) {
      std::vector<int> v2, r2;
      for (uint n = 0; n < graph.alwaysActiveNum; n++) {
        int vertex = graph.always_active_nodes->get_host()[n];
        if (graph.symbol_sets->get_host()[vertex].test(symbol1)) { // filter 1
          int e_start = csr.GetNeighborListOffset(vertex);
          int e_end = e_start + csr.GetNeighborListLength(vertex);
          for (int e = e_start; e < e_end; e++) { // advance 1
            int vertex2 = csr.GetEdgeDest(e);
            if (graph.symbol_sets->get_host()[vertex2].test(
                    symbol2)) { // filter 2
              prec_twice_length++;
              v2.push_back(vertex2);
              if (graph.node_attrs->get_host()[vertex2] & 0b10) { // report 2
                prec_twice_report_length++;
                r2.push_back(vertex2);
              }
            }
          }
        }
      }
      vv2[symbol1][symbol2] = v2;
      vr2[symbol1][symbol2] = r2;
      // if(v2.size()==0)
      // printf("zero symbol %d\n", symbol1);
    }
  }

  int once_offset_length = 256 + 1;
  int twice_offset_length = 256 * 256 + 1;
  volatile int *prec_once_offset = new int[once_offset_length];
  volatile int *prec_once = new int[prec_once_length];
  volatile int *prec_twice_offset = new int[twice_offset_length];
  volatile int *prec_twice = new int[prec_twice_length];
  volatile int *prec_once_report_offset = new int[once_offset_length];
  volatile int *prec_once_report = new int[prec_once_report_length];
  volatile int *prec_twice_report_offset = new int[twice_offset_length];
  volatile int *prec_twice_report = new int[prec_twice_report_length];

  int tmp_prec_once = 0;
  int tmp_prec_twice = 0;
  int tmp_prec_once_report = 0;
  int tmp_prec_twice_report = 0;
  int max_vsize1 = -1;
  int min_vsize1 = csr.nodesNum + 1;
  for (uint symbol1 = 0; symbol1 < 256; symbol1++) {
    prec_once_offset[symbol1] = tmp_prec_once;
    prec_once_report_offset[symbol1] = tmp_prec_once_report;
    int vsize1 = vv1[symbol1].size();
    max_vsize1 = std::max(max_vsize1, vsize1);
    min_vsize1 = std::min(min_vsize1, vsize1);
    // printf("vsize=%d\n", vsize1);
    for (int j = 0; j < vsize1; j++) {
      *(prec_once + tmp_prec_once + j) = vv1[symbol1][j];
      // printf("vertex=%d\n",vv1[symbol1][j]);
    }
    int rsize1 = vr1[symbol1].size();
    for (int j = 0; j < rsize1; j++) {
      *(prec_once_report + tmp_prec_once_report + j) = vr1[symbol1][j];
    }
    tmp_prec_once += vsize1;
    tmp_prec_once_report += rsize1;
    for (uint symbol2 = 0; symbol2 < 256; symbol2++) {
      prec_twice_offset[symbol1 * 256 + symbol2] = tmp_prec_twice;
      prec_twice_report_offset[symbol1 * 256 + symbol2] = tmp_prec_twice_report;

      int vsize2 = vv2[symbol1][symbol2].size();
      // if(vsize2!=0)
      // printf("vsize2[%d][%d]=%d\n",symbol1,symbol2,vsize2);
      for (int j = 0; j < vsize2; j++) {
        *(prec_twice + tmp_prec_twice + j) = vv2[symbol1][symbol2][j];
      }
      int rsize2 = vr2[symbol1][symbol2].size();
      for (int j = 0; j < rsize2; j++) {
        *(prec_twice_report + tmp_prec_twice_report + j) =
            vr2[symbol1][symbol2][j];
      }
      tmp_prec_twice += vsize2;
      tmp_prec_twice_report += rsize2;
    }
  }
  prec_once_offset[256] = prec_once_length;
  prec_once_report_offset[256] = prec_once_report_length;
  prec_twice_offset[256 * 256] = prec_twice_length;
  prec_twice_report_offset[256 * 256] = prec_twice_report_length;
  // printf("min_vsize1=%d, max_vsize1=%d \n", min_vsize1, max_vsize1);

  // for(int jj=0 ;jj<prec_length;jj++)
  //   printf("%d ", prec[jj]);

#define cudaMallocAndCpy(buffer, data, length)                                 \
  CHECK_ERROR(cudaMalloc((void **)&buffer, sizeof(int) * length));             \
  CHECK_ERROR(cudaMemcpy((void *)buffer, (int *)data, sizeof(int) * length,    \
                         cudaMemcpyHostToDevice));

  cudaMallocAndCpy(plb.prec_once, prec_once, prec_once_length);
  cudaMallocAndCpy(plb.prec_once_offset, prec_once_offset, once_offset_length);
  cudaMallocAndCpy(plb.prec_twice, prec_twice, prec_twice_length);
  cudaMallocAndCpy(plb.prec_twice_offset, prec_twice_offset,
                   twice_offset_length);
  cudaMallocAndCpy(plb.prec_once_report, prec_once_report,
                   prec_once_report_length);
  cudaMallocAndCpy(plb.prec_once_report_offset, prec_once_report_offset,
                   once_offset_length);
  cudaMallocAndCpy(plb.prec_twice_report, prec_twice_report,
                   prec_twice_report_length);
  cudaMallocAndCpy(plb.prec_twice_report_offset, prec_twice_report_offset,
                   twice_offset_length);

  CHECK_LAST_ERROR

  printf("Precompute memory usage:\n");
  printf("    Precompute once vertices: number:%d, size:%.1f MB\n",
         prec_once_length, sizeof(int) * prec_once_length / std::pow(1024, 2));
  printf("    Precompute once reports: number:%d, size:%.1f MB\n",
         prec_once_report_length,
         sizeof(int) * prec_once_report_length / std::pow(1024, 2));
  printf("    Precompute twice vertex: number:%d, size:%.1f MB\n",
         prec_twice_length,
         sizeof(int) * prec_twice_length / std::pow(1024, 2));
  printf("    Precompute twice reports: number:%d, size:%.1f MB\n",
         prec_twice_report_length,
         sizeof(int) * prec_twice_report_length / std::pow(1024, 2));

  printf("    Total memory: size:%.1f MB\n",
         sizeof(int) *
             (prec_once_length + prec_twice_length + prec_once_report_length +
              prec_twice_report_length) /
             std::pow(1024, 2));

  delete[] prec_once_offset;
  delete[] prec_once;
  delete[] prec_once_report_offset;
  delete[] prec_once_report;
  delete[] prec_twice_offset;
  delete[] prec_twice;
  delete[] prec_twice_report_offset;
  delete[] prec_twice_report;
}

template <typename T>
void ngap::calculateTheoreticalOccupancy2(T func, int block_size) {
  // calculate theoretical occupancy
  int maxActiveBlocks1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks1, func,
                                                block_size, 0);
  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  float occupancy_filter =
      (maxActiveBlocks1 * block_size / props.warpSize) /
      (float)(props.maxThreadsPerMultiProcessor / props.warpSize);

  cudaFuncAttributes funcAttrib1;
  // int bytes = 0;
  // int carveout = 50;
  // cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize,
  //                      bytes);
  // cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout,
  //                      carveout);
  CHECK_ERROR(cudaFuncGetAttributes(&funcAttrib1, func));
  printf("Device info:\n");
  printf("    Device name: %s\n", props.name);
  printf("    compute capability: %d.%d\n", props.major, props.minor);
  printf("    Concurrent kernels: %s\n",
         props.concurrentKernels ? "yes" : "no");
  printf("    multiProcessorCount: %d\n", props.multiProcessorCount);

  printf("    maxThreadsPerMultiProcessor: %d\n",
         props.maxThreadsPerMultiProcessor);
  printf("    maxThreadsPerBlock: %d\n", props.maxThreadsPerBlock);

  printf("    regsPerMultiprocessor: %d\n", props.regsPerMultiprocessor);
  printf("    regsPerBlock: %d\n", props.regsPerBlock);
  printf("    constSizeBytes: %lu\n", funcAttrib1.constSizeBytes);
  printf("    localSizeBytes: %lu\n", funcAttrib1.localSizeBytes);
  printf("    maxDynamicSharedSizeBytes: %d\n",
         funcAttrib1.maxDynamicSharedSizeBytes);
  printf("    preferredShmemCarveout: %d\n",
         funcAttrib1.preferredShmemCarveout);
  printf("    ptxVersion: %d\n", funcAttrib1.ptxVersion);
  printf("    sharedSizeBytes: %lu\n", funcAttrib1.sharedSizeBytes);

  printf("Theoretical occupancy info:\n");
  printf("    kernel: numRegs:%d\n", funcAttrib1.numRegs);
  printf("    kernel: blocks size:%d, maxActiveBlocks:%d, theoretical "
         "occupancy:%.3f\n",
         block_size, maxActiveBlocks1, occupancy_filter);
  // printf("binaryVersion: %d\n", funcAttrib1.binaryVersion);
  // printf("cacheModeCA: %d\n", funcAttrib1.cacheModeCA);
}

void ngap::initGroupCsrWithPrec(GroupCsr &gscr, std::vector<Graph *> &gs,
                                int max_depth, bool compressPrecTable) {
  int total_size = 0;
  gscr.size = gs.size();
  gscr.h_groups_csr = new Csr[gscr.size];
  CHECK_ERROR(cudaMalloc(&gscr.groups_csr, sizeof(Csr) * gscr.size));

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < gscr.size; i++) {
    Graph *graph = gs[i];
    Csr csr(*graph);
    csr.fromCoo(graph->edge_pairs->get_host());
    if (max_depth > 0)
      total_size += getPrecomputeResultsForKGroupsInCsr(csr, graph, max_depth,
                                                        compressPrecTable);
    csr.moveToDevice();
    gscr.h_groups_csr[i] = csr;
    CHECK_ERROR(cudaMemcpy((void *)(gscr.groups_csr + i), (Csr *)&csr,
                           sizeof(Csr), cudaMemcpyHostToDevice));
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  printf("all table_time = %f s\n", duration.count() / 1000000.0);
  printf("precompute table total size = %d MB\n", total_size);
}

void ngap::initGroupCsrWithPrecFake(GroupCsr &gscr, std::vector<Graph *> &gs,
                                    int max_depth, bool compressPrecTable) {
  int total_size = 0;
  gscr.size = gs.size();
  gscr.h_groups_csr = new Csr[gscr.size];
  CHECK_ERROR(cudaMalloc(&gscr.groups_csr, sizeof(Csr) * gscr.size));
  for (int i = 0; i < gscr.size; i++) {
    Graph *graph = gs[i];
    Csr csr(*graph);
    csr.fromCoo(graph->edge_pairs->get_host());
    if (max_depth > 0)
      total_size += getPrecomputeResultsForKGroupsInCsrFake(
          csr, graph, max_depth, compressPrecTable);
    // csr.moveToDevice();
    // gscr.h_groups_csr[i] = csr;
    // CHECK_ERROR(cudaMemcpy((void *)(gscr.groups_csr + i), (Csr *)&csr,
    //                        sizeof(Csr), cudaMemcpyHostToDevice));
  }
  printf("precompute table total size = %d MB\n", total_size);
}