#ifndef ONEBYTE_AT_A_TIME
#define ONEBYTE_AT_A_TIME


#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <cassert>
#include <set>
#include "commons/NFA.h"
#include <gpunfautils/utils.h>
#include <gpunfautils/array2.h>
#include <gpunfautils/common.h>
#include "commons/SymbolStream.h"
#include <gpunfautils/abstract_gpunfa.h>
#include <unordered_map>
#include "commons/report_formatter.h"
#include "option_config.h"


using std::unordered_map;



class one_byte_at_a_time : public abstract_algorithm {
public:
	obat_config* opt;
	vector<NFA *> old_ccs;
	one_byte_at_a_time(NFA *nfa);
	virtual ~one_byte_at_a_time();

	void preprocessing_enable_active();

	void check_grouped_nfa_sizes();

	void preprocessing_active_active(); 

	void launch_kernel() override;

	void prepare_output_buffer();

	void print_reports(string filename, report_formatter &rf);

	void organize_reports2(Array2<match_entry> *output_buffer, int buffer_size, const vector<NFA*> &grouped_nfas1, report_formatter& rf);

	void remap_intid_of_nodes(remap_node_type tp);

	void remap_intid_of_nodes_with_boudary(remap_node_type tp, vector<NFA *> &grouped_nfa, const vector<int> &boundaries);

	void hotstart_aa();

    //OBAT series
    void OBAT_baseline_2();
    void obat_MC();

    // important. Activity based hot cold approach.
    void test_hotcold_nodup_queue_mc_CaH();

    // hotstart
    void hotstart_ea();
    void hotstart_ea_without_MC2();

	void set_node_active_freq_map(map<string, double> freq_map);
	void set_hot_limit_by_bfs_layer(int hot_limit_by_bfs_layer);
	void set_active_queue_size(int queuesize);

	void set_cold_threshold(double cold_threshold) {
		this->cold_thres = cold_threshold;
	}
	void set_option(obat_config &opt){
		this->opt = &opt;
	}

	void print_node_matchset_complete_info(const vector<NFA*> &ccs);

	void test_data_movement_read_input_stream_only(int num_tb_x);

	void test_data_movement_read_input_stream_only2(int num_tb_x);

    bool hot_stage_only;
    bool remap_input_stream;

    int packing;
    string packing_filename;

private:
	bool record_cold_warp_active_array;

	int history_queue_capacity;

	int active_queue_size;

	int hot_limit_by_bfs_layer;

	int max_indegree_of_cold_states; 

	double cold_thres;

	int profile_length;

	map<string, double> freq_map;

	vector<NFA *> grouped_nfas;

	Array2<match_entry> *real_output_array;
	Array2<unsigned long long int>     *tail_of_real_output_array;

	remap_node_type remap_node_id;


};



#endif
