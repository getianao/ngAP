#ifndef RUN_AHEAD_H
#define RUN_AHEAD_H

#include "NFA.h"
#include "SymbolStream.h"
#include "abstract_gpunfa.h"
#include "array2.h"
#include "common.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

using std::unordered_map;

#define RESET "\033[0m"
#define BLACK "\033[30m"              /* Black */
#define RED "\033[31m"                /* Red */
#define GREEN "\033[32m"              /* Green */
#define YELLOW "\033[33m"             /* Yellow */
#define BLUE "\033[34m"               /* Blue */
#define MAGENTA "\033[35m"            /* Magenta */
#define CYAN "\033[36m"               /* Cyan */
#define WHITE "\033[37m"              /* White */
#define BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"   /* Bold White */
#define tge_log(Message, color) std::cout << color << Message << RESET << "\n";

class report_buffer {
public:
  report_buffer(int CAP);
  ~report_buffer();

  Array2<int> *intermediate_output_array_offset;
  Array2<int> *intermediate_output_array_sid;
  Array2<match_pair> *real_output_array;

  unsigned long long int *h_tail_of_real, *d_tail_of_real;
  int *h_tail_of_intermediate, *d_tail_of_intermediate;

  void set_tail_of_real(Array2<unsigned long long> *tail_of_real_array,
                        int offset);
  void set_tail_of_intermediate(Array2<int> *tail_of_intermediate_array,
                                int offset);

  void copy_real_tail_to_host();
  void copy_intermediate_tail_to_host();

  int CAP;

}; // for execution groups;

class run_ahead_alg : public abstract_algorithm {
public:
  std::string app_name;
  bool validation;
  bool quit_degree;
  bool remove_degree;
  int dup_input_stream;
  long long int quick_validation;
  run_ahead_alg(NFA *nfa, int num_streams0 = 16);

  virtual ~run_ahead_alg();

  void device_sort_kv(int *d_key, int *d_v, int N);

  void device_sort_unique_kv(int *d_key, int *d_v, int N);

  void preprocessing();

  void launch_kernel() override;

  void set_report_off(bool report_off) { this->report_on = !report_off; }

  void set_report_off(bool &report_off, int result_capacity,
                      long long int quick_result_number) {
    if (quick_result_number >= 0 &&
        result_capacity <= quick_result_number * 1.5) {
      if (!report_off) {
        printf("Warning: The number of results may exceed the "
               "capacity limit. "
               "Set report_off=true.\n");
        report_off = true;
      }
    }
    this->report_on = !report_off;
    if (this->report_on) {
      printf("Report on.\n");
    } else {
      printf("Report off.\n");
    }
  }

  /*currently for multithreading. */
  void call_single_kernel(int exec_group, Array2<uint8_t> *input_stream);

  /*
   *   per NFA transition table for scanning kernel.
   *
   */
  void prepare_transition_table_for_scanning_kernel();

  void set_R(int R) { this->R = R; }

  void set_print_intermediate_reports(bool s) {
    this->print_intermediate_reports = s;
  }

  void print_report_to_file(
      string filename,
      const vector<std::pair<int, match_pair *>> &real_report_for_each_cc);

  unsigned long long int
  print_report_to_file(string filename,
                       const unordered_map<int, std::pair<int, match_pair *>>
                           &real_report_for_each_cc);

  void group_ccs_to_execution_group();

  // bool place_active_array_to_reg;

  static const int ACTIVE_STATE_ARRAY_SIZE = 8192;

  void set_cap_output(int cap_output) {
    CAP_OUTPUT_BUFFER_FOR_EACH_EXECUTION_GROUP = cap_output;
  }

  void set_irhandle_threshold(int irhandle_threshold) {
    this->irhandle_threshold = irhandle_threshold;
  }

  void set_fullsort(bool fs) { fullsort = fs; }

  int record_ir;
  int blockDimX;
  int shrmem_wl;
  int reorder_nodeids;
  int num_streams;

  int merge_cc_to_one;

  int shr_wl_len;

private:
  Array2<char> *get_is_report_array(NFA *cc);

  bool fullsort;

  int irhandle_threshold;

  unordered_map<int, vector<int>> execution_groups;

  int R;

  void prepare_report_arrays();

  void prepare_report_buffers();

  void prepare_tail_pointers();

  void prepare_nodelist_for_each_cc();

  // transition table
  // TT[k] the kth NFA;
  // TT[k]['']

  Array2<OutEdges> **list_of_tt_of_nfa;
  Array2<STE_dev<4>> **list_of_stes_of_nfa;

  Array2<char> **is_report;
  Array2<unsigned long long int> *tail_of_real_output_array;
  Array2<int> *tail_of_intermediate_output_array;

  Array2<uint8_t> *get_array2_of_input_stream0();
  // int get_len_of_input_stream0();
  int get_start_node_id(NFA *cc);

  // Array2<int4> get_tt_of_nfa()

  bool print_intermediate_reports;

  const static int BLOCKSIZE_SORT = 256;

  int num_execution_group, num_cc_in_group;

  int CAP_OUTPUT_BUFFER_FOR_EACH_EXECUTION_GROUP;

  vector<report_buffer *> arr_of_report_buffer;

  unordered_map<int, std::pair<int, match_pair *>>
      real_report_for_each_cc; // size, arr

  // const int static
  // cudaStream_t * streams; // [num_streams];
};

#endif