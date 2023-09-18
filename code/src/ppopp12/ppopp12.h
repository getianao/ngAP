#ifndef PPOPP12_H_
#define PPOPP12_H_

#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <cassert>
#include <set>
#include "NFA.h"
#include "array2.h"
#include "utils.h"
#include "common.h"
#include "SymbolStream.h"
#include <cuda.h>
#include "abstract_gpunfa.h"
#include "ppopp12_option.h"
#include "compatible_group_helper.h"

using std::map;
using std::vector;
using std::fill;
using std::cout;
using std::endl;
using std::pair;
using std::set;
using std::make_pair;




class ppopp12 : public abstract_algorithm {
public:
	ppopp12_config* opt;
	vector<NFA *> old_ccs;
	ppopp12(NFA *nfa);
	~ppopp12();

	void set_block_size(int blocksize);
	void set_active_state_array_size(int active_state_array_size);
	void set_alphabet(set<uint8_t> alphabet);

	void group_nfas();

	virtual void preprocessing() override;

	int get_num_states_gpu() const;

	void prepare_transition_table();

	void prepare_states_status();

	void prepare_initial_active_state_array();

	void prepare_state_start_position_tb();
	void prepare_compatible_grps();

	void prepare_input_streams();

	void prepare_outputs();

	void launch_kernel();

	void launch_kernel_readinputchunk();
	
	void print_reports(string filename);

	void  set_num_segment_per_ss(int nn) {
		this->num_segment_per_ss = nn;
	}
	void set_option(ppopp12_config &opt){
		this->opt = &opt;
	}
	

	int get_num_segment_per_ss() const {
		return num_segment_per_ss;
	}
		
private:
	// for debug
	NFA* select_one_nfa_by_id(string str_id);
	
	void calc_str_id_to_compatible_group_per_block();

	int active_state_array_size;

	map<int, vector<int> > nfa_group_tb;
	int num_nfa_chunk;
	map<int, int> num_compatible_groups_cc;

	Array2<int> *state_start_position_tb; 

	Array2<int> *num_state_tb;
	Array2<int> *array_compatible_group;
	Array2<int4> *trans_table;

	Array2<int8_t> *states_status;
	Array2<int> *initial_active_state_array;

	// input
	Array2<uint8_t> *arr_input_streams;

	// output
    Array2<match_entry> *match_array;
    Array2<unsigned long long int > *match_count;

	map<string, int> str_id_to_compatible_group; 
	// per cc 
	map<string, int> str_id_to_compatible_group_per_block;
	// per block

	vector<NFA *> nfa_in_tb;

	bool no_cg;

	bool profile;

	int num_segment_per_ss;
}; 


#endif




























