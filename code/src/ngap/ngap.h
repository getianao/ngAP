#ifndef NGAP_H_
#define NGAP_H_

#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <vector>

#include "NFA.h"
#include "SymbolStream.h"
#include "abstract_gpunfa.h"
#include "array2.h"
#include "common.h"
#include "compatible_group_helper.h"
#include "graph.h"
#include "group_graph.h"
#include "ngap_buffer.h"

#include "ngap_option.h"
#include "precompute_table.h"
#include "utils.h"

class ngap : public abstract_algorithm {
public:
  ngap(NFA *nfa, Graph &g);
  ~ngap();

  void set_block_size(int blocksize);
  void set_active_state_array_size(int active_state_array_size);
  void set_alphabet(set<uint8_t> alphabet);
  void set_data_buffer_stream_size(u_int32_t data_buffer_stream_size);
  void set_ngap_option(ngap_option *plo) { this->plo = plo; };
  void set_nfa_group(std::vector<Graph *> gs) { this->gs = gs; };

  void group_nfas();

  virtual void preprocessing() override;

  int get_num_states_gpu() const;

  void prepare_transition_table();

  void prepare_states_status();

  void prepare_initial_active_state_array();
  void prepare_always_enabled_frontier();
  void prepare_state_start_position_tb();
  void prepare_compatible_grps();

  void prepare_input_streams();
  void prepare_original_input_streams(SymbolStream &ss);

  void prepare_outputs();

  void launch();
  void launch_kernel();
  void launch_ngap();

  void launch_non_blocking_save_advance();
  void launch_blocking_ch();
  // O0
  void launch_blocking();
  void launch_blocking_groups();
  // NAP
  void launch_non_blocking_nap_groups();
  // O1
  void launch_non_blocking();
  void launch_non_blocking_groups();
  // O3
  void launch_non_blocking_prec();
  void launch_non_blocking_prec_groups();
  // O4
  void launch_non_blocking_r1();
  void launch_non_blocking_r2();
  void launch_non_blocking_r1_groups();
  void launch_non_blocking_r2_groups();
  // OA
  void launch_non_blocking_all();
  void launch_non_blocking_all_groups();

  void launch_pcsize_groups();

  void launch_kernel_readinputchunk();

  void automataReference(Graph &g, uint8_t *input_str, int num_seg,
                         int input_length, std::vector<uint64_t> *results,
                         std::vector<uint64_t> *db_results, int debug_iter,
                         Csr csr);
  void automataGroupsReference(std::vector<Graph *> &gs, uint8_t *input_str,
                               int num_seg, int input_length,
                               std::vector<uint64_t> *results,
                               std::vector<uint64_t> *db_results,
                               int debug_iter, GroupCsr gcsr,
                               bool isDup = false);
  bool automataValidation(std::vector<uint64_t> *results,
                          std::vector<uint64_t> *ref_results,
                          bool ifPrintBoth = false);

  void print_reports(string filename);

  void set_num_segment_per_ss(int nn) { this->num_segment_per_ss = nn; }
  int num_seg;

  int get_num_segment_per_ss() const { return num_segment_per_ss; }

  void recursivePrecomputeForK(Csr &csr, Graph *g, PrecTable *pts, int k,
                               int max_depth, bool compressPrecTable);

  void getPrecomputeResults(Csr &csr, NonBlockingBuffer &plb);
  void getPrecomputeResultsForK(Csr &csr, NonBlockingBuffer &plb,
                                int max_depth);
  int getPrecomputeResultsForKGroupsInCsr(Csr &csr, Graph *g, int max_depth,
                                          bool compressPrecTable);
  int getPrecomputeResultsForKGroupsInCsrFake(Csr &csr, Graph *g, int max_depth,
                                              bool compressPrecTable);

  void calculateTheoreticalOccupancy();
  template <typename T>
  void calculateTheoreticalOccupancy2(T func, int block_size);
  void printKernelInfo();

  void initGroupCsrWithPrec(GroupCsr &gscr, std::vector<Graph *> &gs,
                            int max_depth = 0, bool compressPrecTable = true);
  void initGroupCsrWithPrecFake(GroupCsr &gscr, std::vector<Graph *> &gs,
                                int max_depth = 0,
                                bool compressPrecTable = true);

private:
  // for debug
  Graph graph;
  NFA *select_one_nfa_by_id(string str_id);

  void calc_str_id_to_compatible_group_per_block();

  int active_state_array_size;

  map<int, vector<int>> nfa_group_tb;
  int num_nfa_chunk;
  map<int, int> num_compatible_groups_cc;

  Array2<int> *state_start_position_tb;

  Array2<int> *num_state_tb;
  Array2<int> *array_compatible_group;
  Array2<int4> *trans_table;

  Array2<int8_t> *states_status;
  Array2<int> *initial_active_state_array;

  // input
  Array2<uint8_t> *arr_input_streams;

  // output
  Array2<match_entry> *match_array;
  Array2<unsigned int> *match_count;

  map<string, int> str_id_to_compatible_group;
  // per cc
  map<string, int> str_id_to_compatible_group_per_block;
  // per block

  vector<NFA *> nfa_in_tb;

  bool no_cg;

  bool profile;

  int num_segment_per_ss;

  ngap_option *plo;

  std::vector<Graph *> gs; // nfa_group
};

#endif
