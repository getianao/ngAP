#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "NFA.h"
#include "NFALoader.h"
#include <string>
#include <getopt.h>
#include <memory>
#include <set>
#include "SymbolStream.h"
#include "ppopp12.h"
#include "utils.h"
#include "node.h"
#include "nfa_utils.h"
#include <clara/clara.hpp>
#include "moderngpu/context.hxx"
#include "moderngpu/util.hxx"
#include "ppopp12_option.h"


using namespace clara;

using std::set;
using std::unique_ptr;
using std::string;
using std::cout;
using std::endl;


int main(int argc, char *argv[])
{
    printf("Command: ");
    for (int i = 0; i < argc; i++)
        printf("%s ", argv[i]);
    printf("\n");

    ppopp12_config cfg;

    auto result = cfg.parse( argc, argv );

    if( !result )
    {
        std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
        exit(1);
    }

    if (cfg.showHelp) {
        cout << cfg.getHelp();
    }

    string automata_filename = cfg.nfa_filename;
    string input_filename = cfg.input_filename;
    int start_pos = cfg.input_start_pos, input_length = cfg.input_len;
    string algo = cfg.algorithm;
    string output_file_name = cfg.report_filename;
    int dup_input_stream = cfg.duplicate_input_stream;
    unsigned long long int one_output_capacity = cfg.output_capacity;
    int block_size = cfg.block_size;
    int max_size_of_cc = cfg.max_nfa_size;
    int split_entire_inputstream_to_chunk_size = cfg.split_chunk_size;

    SymbolStream ss;
    ss.readFromFile(input_filename);
    
    if (start_pos != -1 && input_length != -1) {
        assert(start_pos >= 0);
        ss = ss.slice(start_pos, input_length);
    }

    //cout << "input_stream_size = " << ss.size() << endl;
    auto ab = ss.calc_alphabet();

    auto nfa = load_nfa_from_file(automata_filename);

    cout << "nfa_size_original = " << nfa->size() << endl;

    nfa_utils::print_starting_node_info(nfa);
    
    int active_state_array_size = block_size;

    ppopp12 p12(nfa);

    cout << "dup_input_stream = " << dup_input_stream << endl;
    cout << "split_entire_inputstream_to_chunk_size = " << split_entire_inputstream_to_chunk_size << endl;

    p12.set_max_cc_size_limit(max_size_of_cc);
    
    for (int i = 0; i < dup_input_stream; i++) {
        if (split_entire_inputstream_to_chunk_size == -1) {
            p12.add_symbol_stream(ss);    
        } else {
            assert(split_entire_inputstream_to_chunk_size > 0);
            int sslen = ss.size();
            int num_seg = sslen / split_entire_inputstream_to_chunk_size;

            cout << "num_seg_" << i << " = " << num_seg << endl; 

            for (int j = 0; j < num_seg; j++) {
                int start_pos1 = j * split_entire_inputstream_to_chunk_size;
                auto ss_seg = ss.slice(start_pos1, split_entire_inputstream_to_chunk_size);
                p12.add_symbol_stream(ss_seg);
            }
        }
    }

    p12.set_report_off(cfg.report_off, cfg.output_capacity,
                    cfg.duplicate_input_stream * cfg.quick_validation);
    p12.set_output_file(output_file_name);
    p12.set_num_segment_per_ss(1);
    p12.set_output_buffer_size(one_output_capacity);
    p12.set_block_size(block_size);
    p12.set_active_state_array_size(active_state_array_size);
    p12.set_alphabet(ab);
    p12.set_option(cfg);
    p12.validation = cfg.validation;

    
    p12.preprocessing();


    if (algo == "ppopp12") {
        p12.launch_kernel();
    } else if (algo == "ppopp12_inputshropt") {
        p12.launch_kernel_readinputchunk();
    }
    else {
        cout <<"not supported algoritm " << algo << endl;

    }

    delete nfa;
    cout<< "FINISHED\n";
    return 0;
}




