#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "commons/NFA.h"
#include "commons/NFALoader.h"
#include <string>
#include <memory>
#include <set>
#include "commons/SymbolStream.h"
#include <gpunfautils/utils.h>
#include "commons/node.h"
#include "one_byte_at_a_time.h"
#include "commons/nfa_utils.h"
#include "commons/common_func.h"

#include <clara/clara.hpp>
#include "option_config.h"

#include "moderngpu/context.hxx"
#include "moderngpu/util.hxx"

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
    
    mgpu::standard_context_t context;

    obat_config cfg;
    auto result = cfg.parse( argc, argv );
    if( !result ) {
        std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
        exit(1);
    }

    if (cfg.showHelp) {
        cout << cfg.getHelp();
    }

    SymbolStream ss;
    ss.readFromFile(cfg.input_filename);
    ss.padding_to_base(cfg.padding_input_stream_to_base);

    if (cfg.input_start_pos != -1 && cfg.input_len != -1) {
        ss = ss.slice(cfg.input_start_pos, cfg.input_len);
    }

    auto nfa = load_nfa_from_file(cfg.nfa_filename);

    if (cfg.only_exec_cc != -1) {
        cout << "only execute ccid = " << cfg.only_exec_cc << endl;
        nfa->mark_cc_id();
        auto ccs = nfa_utils::split_nfa_by_ccs(*nfa);
        assert(cfg.only_exec_cc >= 0 && cfg.only_exec_cc < ccs.size());

        delete nfa;

        nfa = ccs[cfg.only_exec_cc];
        for (int i = 0; i < ccs.size(); i++) {
            if (i != cfg.only_exec_cc) {
                delete ccs[i];
            }
        }
    } else if (cfg.select_cc_by_state != "") {
        cout << "only execute CC with state id = " << cfg.select_cc_by_state << endl;
        nfa->mark_cc_id();
        auto ccs = nfa_utils::split_nfa_by_ccs(*nfa);
        int cc_id = nfa_utils::search_state_id_in_nfa_vector(ccs, cfg.select_cc_by_state) ;

        assert(cc_id != -1);

        delete nfa;

        nfa = ccs[cc_id];
        for (int i = 0; i < ccs.size(); i++) {
            if (i != cc_id) {
                delete ccs[i];
            }
        }
    }

    map<string, double> freq_map;

    one_byte_at_a_time ra(nfa);

    ra.set_read_input(true);
    ra.set_cold_threshold(cfg.cold_threshold);
    ra.set_output_file(cfg.report_filename);
    ra.set_option(cfg);

    ra.packing = cfg.packing;
    ra.packing_filename = cfg.packing_activation_file;

    cout << "dup_input_stream = " << cfg.duplicate_input_stream << endl;
    cout << "split_entire_inputstream_to_chunk_size = " << cfg.split_chunk_size << endl;


    for (int i = 0; i < cfg.duplicate_input_stream; i++) {
        if (cfg.split_chunk_size == -1) {
            ra.add_symbol_stream(ss);
        } else {
            assert(cfg.split_chunk_size > 0);
            int sslen = ss.size();
            int num_seg = sslen / cfg.split_chunk_size;
            //cout << "num_seg_" << i << " = " << num_seg << endl;
            for (int j = 0; j < num_seg; j++) {
                int start_pos1 = j * cfg.split_chunk_size;
                auto ss_seg = ss.slice(start_pos1, cfg.split_chunk_size);
                ra.add_symbol_stream(ss_seg);
            }
        }
    }

    if (cfg.hotcold_filter_filename != "") {
        freq_map = nfa_utils::read_freq_map(cfg.hotcold_filter_filename);
        ra.set_node_active_freq_map(freq_map);
    }

    if (cfg.bfs_hot_ratio > 0.0) {
        cfg.hot_n_state_limit = nfa->size() * cfg.bfs_hot_ratio;
    }


    if ( cfg.hot_n_state_limit > 0) {
        ra.set_hot_limit_by_bfs_layer(cfg.hot_n_state_limit);
    }

    //if (cc_id_print != -1) {
    //    nfa_utils::print_cc(nfa, cc_id_print);
    //}

    ra.set_output_buffer_size(cfg.output_capacity);
    ra.set_block_size(cfg.block_size);
    ra.set_report_off(cfg.report_off, cfg.output_capacity,
                      cfg.duplicate_input_stream * cfg.quick_validation);
    ra.set_active_queue_size(cfg.active_queue_size);
    ra.set_max_cc_size_limit(cfg.max_nfa_size);
    ra.hot_stage_only = cfg.hot_stage_only;
    ra.remap_input_stream = cfg.remap_input_stream;
    ra.validation = cfg.validation;

    auto algo = cfg.algorithm;
    cout << "algorithm = " << algo << endl;

    if (algo == "obat2") {
        // This is the NewTran implementation in our paper;
        ra.OBAT_baseline_2();
    }

    else if (algo == "matchset_compression_new_imp" || algo == "obat_MC" ) {
        ra.obat_MC();
    }

    else if (algo == "hotcold_nodup_queue_mc_CaH") {
        // This implementation Was experimented in Section 5.3
        ra.test_hotcold_nodup_queue_mc_CaH();
    }

    else if (algo == "hotstart_ea") {
        ra.hotstart_ea();
    }

    else if (algo == "hotstart_ea_no_MC2") {
        // This is the hot start + newTran implementation in our paper.
        ra.hotstart_ea_without_MC2();
    }

    else if (algo == "hotstart_aa") {
        // This is the HotStartTT implementation in our paper.
        ra.hotstart_aa();

    }
    else if (algo == "only_read_inputstream") {
        ra.test_data_movement_read_input_stream_only(cfg.num_of_blocks_read_input_only);
    }

    else if (algo == "only_read_inputstream2") {
        ra.test_data_movement_read_input_stream_only2(cfg.num_of_blocks_read_input_only);
    }

    else {
        cout << "not supported algorithm" << endl;
    }

    cout<< "FINISHED\n";
    return 0;
}



