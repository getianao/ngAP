#include <string>

#include "NFA.h"
#include "NFALoader.h"
#include "SymbolStream.h"
#include "graph.h"
#include "kernel.h"
#include "nfa_utils.h"
#include "ngap.h"
#include "ngap_buffer.h"

#include "ngap_option.h"
#include "node.h"
#include "utils.h"

int main(int argc, char *argv[]) {
  printf("Command: ");
  for (int i = 0; i < argc; i++)
    printf("%s ", argv[i]);
  printf("\n");

  ngap_option opt;

  auto result = opt.parse(argc, argv);

  if (!result) {
    std::cerr << "Error in command line: " << result.errorMessage()
              << std::endl;
    exit(1);
  }

  if (opt.showHelp) {
    cout << opt.getHelp();
  }

  std::string automata_filename = opt.nfa_filename;
  std::string input_filename = opt.input_filename;
  int start_pos = opt.input_start_pos, input_length = opt.input_len;
  std::string algo = opt.algorithm;
  std::string output_file_name = opt.report_filename;
  int dup_input_stream = opt.duplicate_input_stream;
  unsigned long long int one_output_capacity = opt.output_capacity;
  int block_size = opt.block_size;
  int max_size_of_cc = opt.max_nfa_size;
  int split_entire_inputstream_to_chunk_size = opt.split_chunk_size;

  SymbolStream ss, old_ss;
  old_ss.readFromFile(input_filename);
  if (start_pos != -1 && input_length != -1) {
    assert(start_pos >= 0);
    old_ss = old_ss.slice(start_pos, input_length);
  }
  // cout << "input_stream_size = " << ss.size() << endl;
  auto ab = old_ss.calc_alphabet();

  auto nfa = load_nfa_from_file(automata_filename);
  nfa_utils::print_nfa_info(nfa);

  Graph g;
  // g.ReadANML(automata_filename);
  g.ReadNFA(nfa);
  printf("ReadANML finish \n");
  g.copyToDevice();

  ngap pl(nfa, g);
  pl.set_ngap_option(&opt);
  pl.set_max_cc_size_limit(max_size_of_cc);
  pl.preprocessing();
  auto grouped_nfas = nfa_utils::group_nfas_by_num(opt.group_num, pl.ccs);
  printf("grouped_nfas.size = %zu pl.num_seg=%d\n", grouped_nfas.size(),
         pl.num_seg);
  std::vector<Graph *> gs;
  for (auto nfa : grouped_nfas) {
    Graph *g = new Graph();
    g->ReadNFA(nfa);
    g->copyToDevice();
    gs.push_back(g);
  }
  assert(gs.size() == opt.group_num);

  cout << "Input Stream Info:\n";
  cout << "    input_start_pos = " << start_pos << endl;
  cout << "    input_length = " << input_length << endl;
  cout << "    split_entire_inputstream_to_chunk_size = "
       << split_entire_inputstream_to_chunk_size << endl;
  cout << "    dup_input_stream = " << dup_input_stream << endl;

  for (int i = 0; i < dup_input_stream; i++) {
    ss.concat(old_ss);
  }
  if (split_entire_inputstream_to_chunk_size > 0) {
    int sslen = ss.size();
    int num_seg = sslen / split_entire_inputstream_to_chunk_size;
    pl.num_seg = num_seg;
    // cout << "num_seg_" << i << " = " << num_seg << endl;
    for (int j = 0; j < num_seg; j++) {
      int start_pos1 = j * split_entire_inputstream_to_chunk_size;
      auto ss_seg =
          ss.slice(start_pos1, split_entire_inputstream_to_chunk_size);
      pl.add_symbol_stream(ss_seg);
    }
  }

  pl.set_nfa_group(gs);
  pl.set_report_off(opt.report_off, opt.output_capacity,
                    opt.duplicate_input_stream * opt.quick_validation);
  pl.set_output_file(output_file_name);
  pl.set_num_segment_per_ss(1);
  pl.set_output_buffer_size(one_output_capacity);
  pl.set_block_size(block_size);
  pl.set_alphabet(ab);
  pl.prepare_original_input_streams(ss);

  if (algo == "blockinggroups") {
    pl.launch_blocking_groups(); // BAP
  } else if (algo == "NAPgroups") {
    pl.launch_non_blocking_nap_groups(); // NAP
  } else if (algo == "nonblockinggroups") {
    pl.launch_non_blocking_groups(); // O1
  } else if (algo == "nonblockingr1groups") {
    pl.launch_non_blocking_r1_groups(); // O4
  } else if (algo == "nonblockingr2groups") {
    pl.launch_non_blocking_r2_groups(); // O4
  } else if (algo == "nonblockingpcgroups") {
    pl.launch_non_blocking_prec_groups(); // O3
  } else if (algo == "nonblockingallgroups") {
    pl.launch_non_blocking_all_groups(); // OA
  } else {
    cout << "not supported algoritm " << algo << endl;
  }

  delete nfa;
  for (auto g : gs)
    delete g;
  printf("FINISHED!\n");
  return 0;
}
