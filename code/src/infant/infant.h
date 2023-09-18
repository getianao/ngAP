/*
 * infant.h
 *
 *  Created on: May 16, 2018
 *      Author: hyliu
 */

#ifndef INFANT_H_
#define INFANT_H_

#include <algorithm>
#include <memory>
#include "commons/NFA.h"
#include "commons/SymbolStream.h"
#include <string>

#include "gpunfautils/array2.h"
#include <fstream>
#include "infant_kernels.h"
#include "infant_config.h"
#include "gpunfautils/abstract_gpunfa.h"

using std::string;
using std::unique_ptr;
using std::pair;


class AlphabetBasedTransitionTable {
public:
	AlphabetBasedTransitionTable(const NFA& nfa, const set<uint8_t>& alphabet);
	~AlphabetBasedTransitionTable();

	void print_basic_stats();

	void init_state_vector();

	int get_transition_table_length() const;

	const pair<int, int> *get_transitions() const;

	const int *get_len() const;
	const int *get_index() const;

	const NFA& get_according_NFA() const;

	int get_length_of_state_bitvec() const;
	const int *get_enabled_bitvec() const;

	int get_max_edge_list_length_of_symbol() const {
		return max_edge_list_length_of_symbol;
	}

	double get_avg_edge_list_length_of_symbol() const {
		assert(alphabet.size() > 0);
		return this->sum_edge_list_length_of_symbol / alphabet.size();
	}

private:
	const NFA& nfa; 
	const set<uint8_t> &alphabet;
	
	pair<int, int> * transitions;
	int len[256];    //symbol_trans_len
	int index[256];  //symbol_trans_len

	int num_transitions;
	int transition_table_length;

	int V; 

	int *enabled_bitvec; 
	int state_bitvec_length; 

	bool *always_enabled;


//statistics
	int max_edge_list_length_of_symbol;
	double sum_edge_list_length_of_symbol;

};




class iNFAnt : public abstract_algorithm {
public:
	vector<NFA *> old_ccs;
	infant_config* opt;

	iNFAnt(NFA *nfa);
	~iNFAnt();

	int get_num_nfa() const;

	const AlphabetBasedTransitionTable& get_trans_table(int i) const;
	//void add_transition_table(AlphabetBasedTransitionTable *tt);
	void add_NFA(NFA *nfa);
	const NFA *get_NFA(int index) const;


	const SymbolStream& get_symbol_stream(int i) const;
	void add_symbol_stream(SymbolStream ss);

	void init_host_transition_tables();
	void prepare_host_state_info();
	void prepare_host_input_streams();
	
	void prepare_state_vector();

	void allocate_device_data_structures();

	void calc_state_bitvec_length();

	const AlphabetBasedTransitionTable *get_transition_table(int k) const;

	
	void set_alphabet(set<uint8_t> alphabet);

	void copy_to_device();

	void launch_kernel() override;

	void to_reports() const;

	void set_num_state_per_group(int num_state_per_group) {
		this->num_state_per_group = num_state_per_group;
	}
	void set_option(infant_config &opt){
		this->opt = &opt;
	}

private:
	int num_state_per_group;

	vector<NFA *> nfas;
	vector<AlphabetBasedTransitionTable * > transition_tables; // equals to num_nfa

	// --------------------------------------------------------------------
	Array2<int> *arr_src_table; 
	Array2<int> *arr_dst_table; 


	Array2<int> *arr_start_position_transition_tables; // length equals to num_nfa

    Array2<int>  *arr_symbol_trans_len;
	Array2<int>  *arr_symbol_trans_index;

    Array2<char> *arr_states_status; // 00    is always enabled; is output;  
    Array2<int>  *arr_state_start_position;
    Array2<int>  *num_of_state_per_tb;

	// input streams
	Array2<uint8_t> *arr_input_streams;
	Array2<uint8_t> *arr_input_streams2;
	//int input_stream_length,
	
	// state vector
	Array2<int> *arr_enabled_bitvec;
	int state_bitvec_length; // num of int per block
	
	// output processing
	Array2<int> 		*arr_match_count;


};


#endif /* INFANT_H_ */





