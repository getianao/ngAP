#include "NFA.h"
#include "NFALoader.h"
#include "SymbolStream.h"
#include "nfa_utils.h"
#include "node.h"
#include "run_ahead_approach.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <stdlib.h>
#include <string>

using std::cout;
using std::endl;
using std::set;
using std::string;
using std::unique_ptr;

int main(int argc, char *argv[]) {
  printf("Command: ");
  for (int i = 0; i < argc; i++)
    printf("%s ", argv[i]);
  printf("\n");

  const int32_t S_algorithm = 1002;
  const int32_t S_one_output_capacity = 1004;
  const int32_t S_block_size = 1007;
  const int32_t S_output_file_name = 1013;
  const int32_t S_lookahead_range = 1015;
  const int32_t S_print_intermediate_report = 1016;
  const int32_t S_irhandle_threshold = 1017;
  const int32_t S_print_ccid = 1018;
  const int32_t S_select_ccid = 1019;

  const int32_t S_duplicate_input_stream = 2015;

  const int32_t S_report_off = 2016;
  const int32_t S_input_len = 2017;
  const int32_t S_record_ir = 2018;
  const int32_t S_blockDimX = 2019;
  const int32_t S_reorder_nodeids = 2020;
  const int32_t S_shr_wl = 2021;
  const int32_t S_num_streams = 2022;
  const int32_t S_merge_cc_to_one = 2023;
  const int32_t S_shr_wl_len = 2024;

  const int32_t S_quick_validation = 2026;
  const int32_t S_validation = 2027;

  const int32_t S_app_name = 2028;

  const int32_t S_ccstate = 3001;
  const int32_t S_remove_degree = 3002;
  const int32_t S_quit_degree = 3003;

  struct option long_opt[] = {
      {"automata", required_argument, NULL, 'a'},
      {"input", required_argument, NULL, 'i'},
      {"algorithm", required_argument, NULL, S_algorithm},
      {"one-output-capacity", required_argument, NULL, S_one_output_capacity},
      {"block-size", required_argument, NULL, S_block_size},
      {"output-file", required_argument, NULL, S_output_file_name},
      {"scanning-R", required_argument, NULL, S_lookahead_range},
      {"print-intermediate-reports", no_argument, NULL,
       S_print_intermediate_report},
      {"ir-handle-threshold", required_argument, NULL, S_irhandle_threshold},
      {"print-cc-id", required_argument, NULL, S_print_ccid},
      {"only-exec-ccid", required_argument, NULL, S_select_ccid},
      {"duplicate-input-stream", required_argument, NULL,
       S_duplicate_input_stream},
      {"report-off", required_argument, NULL, S_report_off},
      {"input-len", required_argument, NULL, S_input_len},
      {"record-ir", required_argument, NULL, S_record_ir},
      {"blockDimX", required_argument, NULL, S_blockDimX},
      {"reorder-nodeids", required_argument, NULL, S_reorder_nodeids},
      {"shrmem-wl", required_argument, NULL, S_shr_wl},
      {"num-streams", required_argument, NULL, S_num_streams},
      {"merge-cc", required_argument, NULL, S_merge_cc_to_one},
      {"shr_wl_len", required_argument, NULL, S_shr_wl_len},
      {"quick-validation", required_argument, NULL, S_quick_validation},
      {"validation", required_argument, NULL, S_validation},
      {"app-name", required_argument, NULL, S_app_name},
      {"only-exec-cc-by-state", required_argument, NULL, S_ccstate},
      {"quit-degree", required_argument, NULL, S_quit_degree},
      {"remove-degree", required_argument, NULL, S_remove_degree},

      {NULL, 0, NULL, 0}};

  int num_streams = 16;

  int shrmem_wl = 0;
  int only_exec_cc = -1;

  int irhandle_threshold = 4;

  int scanning_R = 3;

  bool print_intermediate_report = false;

  int c;
  const char *short_opt = "a:i:";

  string automata_filename = "";
  string input_filename = "";
  string to_gr_file_path = "";

  string algo = "donothing";
  string output_file_name = "reports.txt";
  string app_name;

  bool validation = true;
  bool quit_degree;
  bool remove_degree;
  long long int quick_result_number = -1;

  int one_output_capacity = 80000000;

  int block_size = 256;

  int reorder_nodeids = 0;
  // bool output_report = false;

  int cc_id_print = -1;
  int dup_input_stream = 1;

  int long_ind;

  bool report_off = false;

  int input_len = -1;
  int record_ir = 1;
  int blockDimX = -1;

  int merge_cc_to_one = 1;
  int shr_wl_len = 0;

  string onlyexec_cc_w_state = "";

  while ((c = getopt_long(argc, argv, short_opt, long_opt, &long_ind)) != -1) {
    switch (c) {
    case -1: /* no more arguments */
    case 0:  /* long options toggles */
      break;

    case S_ccstate:
      onlyexec_cc_w_state = optarg;
      break;

    case S_merge_cc_to_one:
      merge_cc_to_one = atoi(optarg);
      break;

    case S_num_streams:
      num_streams = atoi(optarg);
      break;

    case S_shr_wl:
      shrmem_wl = atoi(optarg);
      break;

    case S_blockDimX:
      blockDimX = atoi(optarg);
      break;

    case S_reorder_nodeids:
      reorder_nodeids = atoi(optarg);
      break;

    case S_report_off: {
      std::string report_off_tmp = optarg;
      if (report_off_tmp.compare("true") == 0) {
        report_off = true;
      } else {
        report_off = false;
      }

      break;
    }

    case S_record_ir:
      record_ir = atoi(optarg);
      break;

    case S_input_len:
      input_len = atoi(optarg);
      break;

    case S_duplicate_input_stream:
      dup_input_stream = atoi(optarg);
      break;

    case S_select_ccid:
      only_exec_cc = atoi(optarg);
      break;

    case S_print_ccid:
      cc_id_print = atoi(optarg);
      break;

    case S_irhandle_threshold:
      irhandle_threshold = atoi(optarg);
      break;

    case S_print_intermediate_report:
      print_intermediate_report = true;
      break;

    case S_lookahead_range:
      scanning_R = atoi(optarg);
      break;

    case S_output_file_name:
      output_file_name = optarg;
      break;

    case 'a':
      automata_filename = optarg;
      break;

    case 'i':
      input_filename = optarg;
      break;

    case S_algorithm:
      algo = optarg;
      break;

    case S_one_output_capacity:
      one_output_capacity = atoi(optarg);
      break;

    case S_block_size:
      block_size = atoi(optarg);
      break;

    case S_shr_wl_len:
      shr_wl_len = atoi(optarg);
      break;

    case S_validation: {
      std::string validation_tmp = optarg;
      if (validation_tmp.compare("true") == 0) {
        validation = true;
      } else {
        validation = false;
      }
    } break;
    case S_quit_degree: {
      std::string validation_tmp = optarg;
      if (validation_tmp.compare("true") == 0) {
        quit_degree = true;
      } else {
        quit_degree = false;
      }
    } break;
    case S_remove_degree: {
      std::string validation_tmp = optarg;
      if (validation_tmp.compare("true") == 0) {
        remove_degree = true;
      } else {
        remove_degree = false;
      }
    } break;
    case S_quick_validation:
      quick_result_number = std::stoll(optarg);
      break;
    case S_app_name:
      app_name = optarg;
      break;

    default:
      fprintf(stderr, "%s: invalid option -- %c\n", argv[0], c);
      exit(-1);
    };
  };

  if (shrmem_wl == 1 && shr_wl_len == 0) {
    shr_wl_len = 4;
  }

  if (shrmem_wl == 0 && shr_wl_len > 0) {
    shrmem_wl = 1;
  }

  // DisplayHeader();

  SymbolStream ss;
  ss.readFromFile(input_filename);

  if (input_len != -1) {
    cout << "input_len = " << input_len << endl;
    ss = ss.slice(0, input_len);
  }

  auto ab = ss.calc_alphabet();

  auto nfa = load_nfa_from_anml(automata_filename);

  nfa_utils::print_starting_node_info(nfa);

  if (to_gr_file_path != "") {
    nfa_utils::dump_to_gr_file(*nfa, to_gr_file_path);
  }

  if (cc_id_print != -1) {
    nfa_utils::print_cc(nfa, cc_id_print);
  }

  if (only_exec_cc != -1) {
    cout << "only execute ccid = " << only_exec_cc << endl;
    nfa->mark_cc_id();
    auto ccs = nfa_utils::split_nfa_by_ccs(*nfa);
    assert(only_exec_cc >= 0 && only_exec_cc < ccs.size());
    nfa = ccs[only_exec_cc];
    for (int i = 0; i < ccs.size(); i++) {
      if (i != only_exec_cc) {
        delete ccs[i];
      }
    }
  }

  if (onlyexec_cc_w_state != "") {
    nfa->mark_cc_id();
    auto ccs = nfa_utils::split_nfa_by_ccs(*nfa);
    int cc_id =
        nfa_utils::search_state_id_in_nfa_vector(ccs, onlyexec_cc_w_state);

    assert(cc_id != -1);

    delete nfa;

    nfa = ccs[cc_id];
    for (int i = 0; i < ccs.size(); i++) {
      if (i != cc_id) {
        delete ccs[i];
      }
    }
  }

  cout << "algorithm = " << algo << endl;

  if (algo == "runahead") {

    run_ahead_alg ra(nfa, num_streams);
    ra.quit_degree = quit_degree;
    ra.remove_degree = remove_degree;
    ra.set_irhandle_threshold(irhandle_threshold);
    ra.set_cap_output(one_output_capacity);
    ra.set_R(scanning_R);
    ra.set_block_size(block_size);

    for (int rep = 0; rep < dup_input_stream; rep++) {
      ra.add_symbol_stream(ss);
    }
    ra.dup_input_stream = dup_input_stream;
    ra.validation = validation;
    ra.quick_validation = quick_result_number;

    ra.set_report_off(report_off, one_output_capacity, quick_result_number);
    ra.set_print_intermediate_reports(print_intermediate_report);

    ra.reorder_nodeids = reorder_nodeids;
    ra.record_ir = record_ir;
    ra.blockDimX = blockDimX;
    ra.shrmem_wl = shrmem_wl;

    ra.merge_cc_to_one =
        merge_cc_to_one; // every ``merge_cc_to_one'' CCs are merged to one CC.
    ra.shr_wl_len = shr_wl_len;

    ra.launch_kernel();

  } else {
    cout << "not supported algoritm " << algo << endl;
  }

  // DisplayHeader();

  // delete nfa;
  cout << "FINISHED\n";
  return 0;
}
