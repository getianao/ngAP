#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "commons/NFA.h"
#include "commons/NFALoader.h"
#include <string>
#include <getopt.h>
#include <memory>
#include <set>
#include "commons/SymbolStream.h"
#include "infant.h"
#include "gpunfautils/utils.h"
#include "commons/node.h"
#include "commons/nfa_utils.h"
#include "commons/common_func.h"
#include "infant_config.h"

#include "moderngpu/context.hxx"
#include "moderngpu/util.hxx"

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
    mgpu::standard_context_t context;

    infant_config cfg;

    auto result = cfg.parse( argc, argv );

    if( !result )
    {
        std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
        exit(1);
    }

    if (cfg.showHelp) {
        cout << cfg.getHelp();
    }


    int max_size_of_cc = cfg.max_nfa_size;
    string automata_filename = cfg.nfa_filename;
    string input_filename = cfg.input_filename;

    int start_pos = cfg.input_start_pos, input_length = cfg.input_len + 1;

    string algo = cfg.algorithm;
    
    string output_file_name = cfg.report_filename;

    unsigned long long int  one_output_capacity = cfg.output_capacity;

    int only_exec_cc = cfg.only_exec_cc;

    int split_entire_inputstream_to_chunk_size = cfg.split_chunk_size + 1;

    int block_size = cfg.block_size; 

    int dup_input_stream = cfg.duplicate_input_stream;
    

    int num_state_per_group = -1;

    SymbolStream ss;
    ss.readFromFile(input_filename);
    
    if (start_pos != -1 && input_length != -1) {
        assert(start_pos >= 0);
        ss = ss.slice(start_pos, input_length);
    }
    

    set<uint8_t> ab;
    for (int i = 0; i < 256; i++) {
        ab.insert((uint8_t) i);
    }

    auto nfa = load_nfa_from_file(automata_filename);

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
    } /*else if (select_cc_by_state != "") {
        cout << "only execute CC with state id = " << select_cc_by_state << endl;
        nfa->mark_cc_id();
        auto ccs = nfa_utils::split_nfa_by_ccs(*nfa);
        int cc_id = nfa_utils::search_state_id_in_nfa_vector(ccs, select_cc_by_state) ;

        assert(cc_id != -1);

        nfa = ccs[cc_id];
        for (int i = 0; i < ccs.size(); i++) {
            if (i != cc_id) {
                delete ccs[i];
            }
        }
    }*/

    if (max_size_of_cc != -1) {
        cout << "max_size_of_cc = " << max_size_of_cc << endl;
        nfa->mark_cc_id();
        auto ccs = nfa_utils::split_nfa_by_ccs(*nfa);
        
        vector<NFA*> ccs1;
        for (auto cc : ccs) {
            if (cc->size() <= max_size_of_cc) {
                ccs1.push_back(cc);
            }
        }

        assert(ccs1.size() > 0);

        delete nfa;

        nfa = NULL;
        nfa = nfa_utils::merge_nfas(ccs1);

        for (auto cc : ccs) {
            delete cc;
        }
    }

    iNFAnt infant(nfa); 
    cout << "nfa_size_original = " << nfa->size() << endl;
    cout << "dup_input_stream = " << dup_input_stream << endl;
    cout << "split_entire_inputstream_to_chunk_size = " << split_entire_inputstream_to_chunk_size << endl;

    for (int i = 0; i < dup_input_stream; i++) {
        if (split_entire_inputstream_to_chunk_size == -1) {
            infant.add_symbol_stream(ss);    
        } else {
            assert(split_entire_inputstream_to_chunk_size > 0);
            int sslen = ss.size();
            int num_seg = sslen / split_entire_inputstream_to_chunk_size;

            cout << "num_seg_" << i << " = " << num_seg << endl; 

            for (int j = 0; j < num_seg; j++) {
                int start_pos1 = j * split_entire_inputstream_to_chunk_size;
                auto ss_seg = ss.slice(start_pos1, split_entire_inputstream_to_chunk_size);
                infant.add_symbol_stream(ss_seg);
            }
        }
    }
   
    infant.set_output_buffer_size(one_output_capacity);

    infant.set_block_size(block_size);

    if (num_state_per_group == -1) {
        num_state_per_group = block_size;
    }

    infant.set_num_state_per_group(num_state_per_group);

    infant.set_report_off(cfg.report_off, cfg.output_capacity,
                    cfg.duplicate_input_stream * cfg.quick_validation);

    infant.set_output_file(output_file_name);
    infant.set_alphabet(ab);
    infant.set_option(cfg);
    infant.validation = cfg.validation;
    
    if (algo == "infant") {  
        infant.launch_kernel();   
    }
    else {
        cout << "not supported algoritm " << algo << endl;

    }
  	cout << "FINISHED" << endl;
    delete nfa;
    return 0;
}




