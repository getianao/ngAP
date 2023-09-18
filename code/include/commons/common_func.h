#pragma once

#include "NFA.h"
#include "SymbolStream.h"
#include "clara/clara.hpp"
#include "compatible_group_helper.h"
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tools {
void create_path_if_not_exists(string path);

}

using namespace clara;

class common_gpunfa_options {
public:
  int block_size;
  string algorithm;
  bool write_output_file;
  string report_filename;
  int max_nfa_size;
  int duplicate_input_stream;
  int input_start_pos;
  int input_len;
  unsigned long long int output_capacity;
  bool report_off;

  bool showHelp;

  int split_chunk_size;
  int only_exec_cc;

  string input_filename;
  string nfa_filename;
  string select_cc_by_state;
  string app_name;

  int padding_input_stream_to_base;
  long long int quick_validation;
  int unique_frequency;

  bool use_uvm;
  bool quit_degree;
  bool remove_degree;

  common_gpunfa_options()
      : block_size(256), algorithm(""), write_output_file(true),
        report_filename("report.txt"), max_nfa_size(-1),
        duplicate_input_stream(1), input_start_pos(0), input_len(-1),
        report_off(false), output_capacity(54619400), showHelp(false),
        split_chunk_size(-1), only_exec_cc(-1),
        padding_input_stream_to_base(256), quick_validation(-1),
        unique_frequency(0), use_uvm(false), quit_degree(true),
        remove_degree(false)

  {

    parser =
        Help(showHelp) |
        Opt(algorithm,
            "algorithm")["--algorithm"]["-g"](" the algorithm to test ") |
        Opt(block_size,
            "blocksize")["--block-size"]("the block size in the CUDA kernel") |
        Opt(max_nfa_size, "max nfa size")["--max-nfa-size"](
            "any NFA with larger than the specified max "
            "NFA size will be filtered out. ") |
        Opt(report_off,
            "true/false")["--report-off"]("record reporting states") |
        Opt(report_filename, "report_file_name")["--report-filename"](
            "the file name to store the reports") |
        Opt(output_capacity, "output_capacity")["--result-capacity"](
            "specify the array capacity "
            "storing reports on GPU") |
        Opt(input_start_pos,
            "start_pos")["--input-start-pos"]("Input starting position") |
        Opt(input_len,
            "input_len")["--input-len"]("specify the length of an input") |
        Opt(input_filename,
            "input_filename")["--input"]["-i"]("the input stream file path")
            .required() |
        Opt(nfa_filename, "nfa_filename")["--automata"]["-a"](
            "the automata file path. (currently support anml file")
            .required() |
        Opt(split_chunk_size,
            "split_chunk_size")["--split-entire-inputstream-to-chunk-size"](
            "Split the entire input stream to equal chunks, and treat them "
            "as different input streams") |
        Opt(select_cc_by_state,
            "select_cc_by_state")["--only-exec-cc-with-state-id"](
            "For debug purpose, only execute one CC with the specified state "
            "name. ") |
        Opt(only_exec_cc, "only_exec_cc")["--only-exec-ccid"](
            "For debug purpose, only execute the specified CC ") |
        Opt(duplicate_input_stream,
            "duplicate_input_stream")["--duplicate-input-stream"](
            "duplicate the input stream") |
        Opt(quick_validation, "reporting number")["--quick-validation"](
            "reference reporting number for quick validation") |
        Opt(unique_frequency, "unique frequency")["--unique-frequency"](
            "frequency for unique in the iteration") |
        Opt(padding_input_stream_to_base,
            "padding_input_stream_to_base")["--padding"](
            "padding input stream such "
            "that the length of input stream can divide the padding "
            "factor. ") |
        Opt(app_name, "app_name")["--app-name"]("app name") |
        Opt(use_uvm, "true/false")["--use-uvm"]("use uvm to store result.") |
        Opt(quit_degree,
            "true/false")["--quit-degree"]("quit if the degree of NFA > 4") |
        Opt(remove_degree, "true/false")["--remove-degree"](
            "remove the NFA if its degree > 4");
  }

  virtual detail::InternalParseResult parse(int argc, char *argv[]) {
    return parser.parse(Args({argc, argv}));
  }

  virtual std::string getHelp() {
    std::stringstream ss;
    parser.writeToStream(ss);
    return ss.str();
  }

protected:
  Parser parser;
};
