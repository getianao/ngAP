#ifndef OBAT_CONFIG
#define OBAT_CONFIG

#include "commons/common_func.h"

using namespace clara;

class obat_config : public common_gpunfa_options {
public:
    obat_config() : common_gpunfa_options(),
                    bfs_hot_ratio(0.0),
                    hotcold_filter_filename(""),
                    hot_n_state_limit(-1),
                    hot_stage_only(false),
                    remap_input_stream(false),
                    active_queue_size(1024),
                    packing(0),
                    cold_threshold(0.001),
                    num_of_blocks_read_input_only(1),
                    validation{true}
                    {

        auto additional_parser =
                Opt(bfs_hot_ratio, "bfs_hot_ratio")["--hot-limit-by-bfs-ratio"]
                        ("The ratio of states to be fixed mapped to threads that offloaed by bfs-layer")
                | Opt(hotcold_filter_filename, "hot cold filter filename")
                ["--hot-cold-filter"]
                ("The file specifies which states are hot and thereby fixed mapped to threads.")

                | Opt(active_queue_size, "active_queue_size")["--active-queue-size"]["-q"]("worklist size in shared memory")

                | Opt(hot_n_state_limit, "hot_n_state_limit")["--hot-limit-by-bfs"]("hot-limit-by-bfs")
                | Opt(hot_stage_only, "hot_stage_only")["--hot-stage-only"]("only execute hot stage. "
                                                                            "Only works in hotstart_ea and hotstart_aa")
                | Opt(remap_input_stream, "remap_input_stream")["--remap-input-stream"]
                                       ("remap input stream to thread block. (Testing now only applicable to hotstart ea)")
                | Opt(packing, "packing")["--packing"]("The way of packing NFAs to thread blocks. Default: 0. Random 1. ")
                | Opt(packing_activation_file, "packing_activation_file")["--packing-file"]("packing activation ratio file")
                | Opt(cold_threshold, "cold_threshold")["--cold-threshold"]("If we use profiling, what ratio can be considered to be cold")
                | Opt(num_of_blocks_read_input_only, "num_of_blocks_read_input_only")["--num-of-blocks-read-input-only"]("only for characterization. ")
                | Opt(validation,  "validation")["--validation"]("fake validation (do nothing)");
        parser = parser | additional_parser;

    }

    bool hot_stage_only;
    bool remap_input_stream;
    double bfs_hot_ratio;
    int hot_n_state_limit;
    string hotcold_filter_filename;
    int active_queue_size;

    int packing;
    string packing_activation_file;

    double cold_threshold;

    int num_of_blocks_read_input_only;

    bool validation;
};


#endif





