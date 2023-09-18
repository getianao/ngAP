#ifndef NGAP_OPTION_H_
#define NGAP_OPTION_H_

#include "commons/common_func.h"

class ngap_option : public common_gpunfa_options {
public:
  ngap_option() : common_gpunfa_options() {
    this->algorithm = "graph";
    this->num_state_per_group = this->block_size;

    auto additional_parser =
        Opt(add_aas_start, "start number")["--add-aan-start"](
            "the number of iteration to added always active state before "
            "execution") |
        Opt(add_aas_interval, "interval number")["--add-aas-interval"](
            "the number of iteration to added always active state during "
            "execution") |
        Opt(unique, "true/false")["--unique"]("unique during execution") |
        Opt(active_threshold, "active-threshold")["--active-threshold"](
            "the active thread number to enable work privatization") |
        Opt(validation, "true/false")["--validation"]("enable validation") |
        Opt(use_soa, "true/false")["--use-soa"](
            "change the data layout of NFA topology") |
        Opt(precompute_cutoff, "precompute-cutoff")["--precompute-cutoff"](
            "the threshold for table load balance") |
        Opt(precompute_depth, "precompute-depth")["--precompute-depth"](
            "the prefix length for the memiozation table") |
        Opt(data_buffer_fetch_size,
            "data-buffer-fetch-size")["--data-buffer-fetch-size"](
            "the number of states taken from the buffer in each iteration") |
        Opt(motivate_worklist_length,
            "true/false")["--motivate-worklist-length"](
            "record worklist length") |
        Opt(num_state_per_group,
            "num_state_per_group")["--num-state-per-group"](
            "number of state per group.") |
        Opt(group_num, "group_num")["--group-num"]("the group number for CCs") |
        Opt(tuning, "true/false")["--tuning"]("enable tuning") |
        Opt(pc_use_uvm, "true/false")["--pc-use-uvm"](
            "use uvm to store memiozation tables") |
        Opt(adaptive_aas, "true/false")["--adaptive-aas"](
            "use adaptive strategy for interval number") |
        Opt(try_adaptive_aas, "true/false")["--try-adaptive-aas"](
            "retry when adaptive strategy  failed") |
        Opt(compress_prec_table, "true/false")["--compress-prec-table"](
            "compress memiozation tables");
            
    parser = parser | additional_parser;
  }

  uint32_t data_buffer_fetch_size = 128;
  int add_aas_start = 0;
  int add_aas_interval = 1;
  bool unique = false;
  bool validation = true;
  int active_threshold = 20;
  bool use_soa = false;
  int precompute_cutoff = -1;
  int precompute_depth = 0;
  bool motivate_worklist_length = false;
  int num_state_per_group;
  int group_num = 10;
  bool compress_prec_table = true;
  bool tuning = false;
  bool pc_use_uvm = false;
  bool adaptive_aas = false;
  bool try_adaptive_aas = false;
};

#endif
