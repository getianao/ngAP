/*
 * infant.cpp
 *
 *
 *
 * implementation of key ideas in
 *
 * iNFAnt: MFA Pattern Matching on GPGPU Devices
 *
 *
 *  Created on: May 16, 2018
 *      Author: hyliu
 */

#include "infant.h"
#include "commons/NFA.h"
#include "gpunfautils/utils.h"
#include "commons/nfa_utils.h"
#include "commons/report_formatter.h"
#include "infant_kernels.h"
#include <algorithm>
#include <memory>
#include <iostream>
#include <cassert>
#include <bitset>
#include <numeric>
#include <fstream>
#include "gpunfautils/abstract_gpunfa.h"
#include "commons/validate.h"
#include <execution>

using std::bitset;
using std::cout;
using std::endl;
using std::unique_ptr;
using std::pair;


AlphabetBasedTransitionTable::AlphabetBasedTransitionTable(const NFA& nfa, const set<uint8_t>& alphabet) : 
nfa(nfa),
max_edge_list_length_of_symbol(-1),
sum_edge_list_length_of_symbol(0),
alphabet(alphabet)
{

	enabled_bitvec = NULL;

	std::fill(index, index + 256, -1);
	std::fill(len, len + 256, 0);

	V = nfa.size();


	num_transitions = 0;
	int num_transitions_in_alphabet_based_tables = 0;

	map<uint8_t, vector<pair<int, int> > > symbol_trans;

	assert(nfa.size() > 0);

	for (int i = 0; i < nfa.size(); i++) {
		auto n = nfa.get_node_by_int_id(i);
		assert(nfa.has_node(n->str_id));
		auto adjs = nfa.get_adj(n->str_id);
		for (auto it : adjs) {
			Node *to_node = nfa.get_node_by_str_id(it);
			auto to_intid = to_node->sid;
			auto trans1 = make_pair(i, to_intid);
			num_transitions++;

			for (auto symbol : alphabet) {
				if (to_node->match2(symbol)) {
					symbol_trans[symbol].push_back(trans1);
					len[symbol] ++;
					num_transitions_in_alphabet_based_tables++;
				}
			}
		}
	}

	for (auto symbol : alphabet) {
		if (len[symbol] > max_edge_list_length_of_symbol) {
			max_edge_list_length_of_symbol = len[symbol];
		}

		sum_edge_list_length_of_symbol += len[symbol];
	}

	assert(num_transitions > 0);

	this->transitions = new pair<int, int> [num_transitions_in_alphabet_based_tables];

	int t = 0;
	for (auto it : symbol_trans) {
		uint8_t symbol = it.first;
		assert(symbol < 256);

		index[symbol] = t;
		auto vec_transitions = it.second;
		for (auto trans : vec_transitions) {
			transitions[t++] = trans;
		}
	}


	this->transition_table_length = num_transitions_in_alphabet_based_tables;
	
	this->init_state_vector();

	//print_basic_stats();
}





AlphabetBasedTransitionTable::~AlphabetBasedTransitionTable() {
	delete[] transitions;
	delete[] enabled_bitvec;

	//TODO 
	//add deletes; 

}


const NFA& AlphabetBasedTransitionTable::get_according_NFA() const {
	return nfa;
}





void AlphabetBasedTransitionTable::print_basic_stats() {
	cout << "num_transitions_in_alphabet_based_transition_table = " << this->transition_table_length << endl;

	for (int i = 0; i < 256; i++) {
		//if (i >= 97 && i <= 122) {
			cout << "index[" << i << "] = " <<  index[i] << "    " << "len = " << len[i] <<  endl;
		//}
	}

}


void AlphabetBasedTransitionTable::init_state_vector() {
	int num_of_int = V / (sizeof(int) * 8) + 1;
	
	enabled_bitvec = new int[num_of_int];
	//active_bitvec = new int[num_of_int];

	std::fill(enabled_bitvec, enabled_bitvec + num_of_int, 0);
	//std::fill(active_bitvec, active_bitvec + num_of_int, 0);

	for (int i = 0; i < V; i++) {
		Node *node = nfa.get_node_by_int_id(i);
		//if (node->)
		if ( node->is_start() || node->is_start_always_enabled()) {
			bitvec::set_bit(enabled_bitvec, num_of_int, i);
		}
	}

	/*cout << "init_state_vector = ";
	for (int i = 0; i < num_of_int; i++) {
		bitset<32> x(enabled_bitvec[i]);
		std::cout << x;
	}
	cout << endl;*/

	
	//always_enabled = new bool[V];
	//std::fill(always_enabled, always_enabled + V, false);

	//for (int i = 0 ; i < V; i++) {
	//	Node *node = nfa.get_node_by_int_id(i);
	//	always_enabled[i] = node->is_start_always_enabled();
	//}

	this->state_bitvec_length = num_of_int;
}

int AlphabetBasedTransitionTable::get_length_of_state_bitvec() const {
	return this->state_bitvec_length;
}

const int *AlphabetBasedTransitionTable::get_enabled_bitvec() const {
	return enabled_bitvec;
}
	

int AlphabetBasedTransitionTable::get_transition_table_length() const {
	return this->transition_table_length;
}

const int *AlphabetBasedTransitionTable::get_len() const {
	return this->len;
}

const int *AlphabetBasedTransitionTable::get_index() const {
	return this->index;
}


const pair<int, int> *AlphabetBasedTransitionTable::get_transitions() const {
	return this->transitions;
}



const AlphabetBasedTransitionTable *iNFAnt::get_transition_table(int k) const {
	return transition_tables[k];
}




iNFAnt::iNFAnt(NFA *nfa):
arr_start_position_transition_tables(NULL),
abstract_algorithm(nfa)
{
	transition_tables.clear();
	
		nfa->mark_cc_id();
		old_ccs = nfa_utils::split_nfa_by_ccs(*nfa); 
    int nfasize = nfa->size();

    for (int i = 0; i < nfasize; i++) {
        Node *n  = nfa->get_node_by_int_id(i);

        if (n->is_start()) {
        	Node *cloned_start_node = new Node();
        	*cloned_start_node = *n;

        	cloned_start_node->str_id += "clone";
        	cloned_start_node->report = false;

        	nfa->addNode(cloned_start_node);
        	nfa->addEdge(cloned_start_node->str_id, n->str_id);

        	n->start = 0;
        }
    }
   	


    

}


iNFAnt::~iNFAnt() {
	if (arr_start_position_transition_tables != NULL) {
		delete arr_start_position_transition_tables;
	}



}


int iNFAnt::get_num_nfa() const {
	return nfas.size();
}


void iNFAnt::add_NFA(NFA *nfa) {
	nfas.push_back(nfa);
}


const NFA *iNFAnt::get_NFA(int index) const {
	assert(index >= 0 && index < get_num_nfa());

	return nfas[index];
}



const SymbolStream& iNFAnt::get_symbol_stream(int i) const {
	assert(i >= 0 && i < symbol_streams.size() );
	return symbol_streams[i];
}


void iNFAnt::add_symbol_stream(SymbolStream ss) {
	symbol_streams.push_back(ss);
}




void iNFAnt::init_host_transition_tables() {

	arr_start_position_transition_tables = new Array2<int>  (get_num_nfa());

	assert(get_num_nfa() > 0);
	assert(transition_tables.size() == get_num_nfa());

	int t = 0;
	int total_transition_table_length = 0;
	
	for (int i = 0; i < get_num_nfa(); i++) {

		if (get_transition_table(i)->get_transition_table_length() <= 0) {
			cout << "transition table length = 0??? " << endl;
			cout << "i = " << i << endl;
			cout << "length = " << get_transition_table(i)->get_transition_table_length() << endl;
			get_transition_table(i)->get_according_NFA().to_dot_file("problematic_nfa.dot");
			assert(get_transition_table(i)->get_transition_table_length() > 0);
		}

		arr_start_position_transition_tables->set(t++, total_transition_table_length);
		total_transition_table_length += get_transition_table(i)->get_transition_table_length();
	}


	arr_src_table = new Array2<int> (total_transition_table_length);
	arr_dst_table = new Array2<int> (total_transition_table_length);

	int current_index = 0;
	for (int tt = 0; tt < get_num_nfa(); tt++) {
		auto current_transition_table = get_transition_table(tt);
		for (int i = 0; i < current_transition_table->get_transition_table_length(); i++) {
			auto p = current_transition_table->get_transitions()[i];
			arr_src_table->set(current_index, p.first);
			arr_dst_table->set(current_index, p.second);
			current_index ++;
		}
	}

	int total_num_of_edges = 0;
	for (int i = 0; i < nfas.size(); i++) {
		auto cur_nfa = nfas[i];
		total_num_of_edges += cur_nfa->get_num_transitions();
	}

	cout << "total_num_of_edges = " << total_num_of_edges << endl;
	cout << "total_transition_table_length = " << total_transition_table_length << endl;
	// till now, we prepared h_src_table, h_dst_table, h_trans_table_start_position


	// next, we will prepare h_symbol_trans_len, and h_symbol_trans_index
	// h_symbol_trans_len
	// each NFA has 256 entries.
	// 

	arr_symbol_trans_len = new Array2<int> (get_num_nfa() * 256);
	arr_symbol_trans_index = new Array2<int> (get_num_nfa() * 256);

	t = 0;
	for (int i = 0 ; i < get_num_nfa(); i++) {
		const int *len = 	get_transition_table(i)-> get_len();
		const int *index = 	get_transition_table(i)-> get_index();
		for (int j = 0; j < 256; j++) {
			arr_symbol_trans_len->set(t, len[j]);
			arr_symbol_trans_index->set(t, index[j]);
			t++;
		}
	}

	//arr_symbol_trans_len->print();
	//arr_symbol_trans_index->print();
	//arr_src_table->print();
	//arr_dst_table->print();

	// here we finished prepare h_symbol_trans_len and h_symbol_trans_index
}


void iNFAnt::prepare_host_state_info() {
	
	arr_state_start_position = new Array2<int>   (get_num_nfa() );

	int total_num_of_states = 0;
	for (int i = 0; i < get_num_nfa(); i++) {
		arr_state_start_position->set(i, total_num_of_states);
		int n_state = nfas[i]->size();
		total_num_of_states += n_state;
	}

	arr_states_status = new Array2<char> (total_num_of_states);
	arr_states_status->clear_to_zero();

	num_of_state_per_tb = new Array2<int> (get_num_nfa());

	int n = 0;
	for (int i = 0; i < get_num_nfa(); i++) {
		
		num_of_state_per_tb->set(i, nfas[i]->size());
		for (int j = 0; j < nfas[i]->size(); j++) {
			Node *node = nfas[i]->get_node_by_int_id(j);
			
			char v = 0; 
			if (node->is_start_always_enabled()) {
				v = v | (1 << 1);
			}

			if (node-> is_report()) {
				v = v | 1;
			}

			arr_states_status->set(n++,  v);

			if (node->str_id == "145_star_0") {
        		cout << "145_star_0 " << (int)v << endl;

        	}
		}
	}




	// 
	// finish prepare h_states_stateus
	// finish prepare h_state_start_position
}



void iNFAnt::prepare_host_input_streams() {
	// prepare h_input_streams
	//uint8_t *h_input_streams; 
	

	int length = symbol_streams[0].get_length();

	for (auto ss : symbol_streams) {
		assert(length = ss.get_length());
	}// currently we require the length of each symbol stream with equal length

	arr_input_streams = new Array2<uint8_t> (get_num_streams() * length);
	arr_input_streams2 =
			new Array2<uint8_t>(get_num_streams() * (length - 1));

        int t = 0;
	for (auto ss : symbol_streams) {
		for (int p = 0; p < ss.get_length(); p++) {
			arr_input_streams->set(t++, ss.get_position(p) );
		}
	}
	 t = 0;
	for (auto ss : symbol_streams) {
		for (int p = 0; p < ss.get_length() - 1; p++) {
			arr_input_streams2->set(t++, ss.get_position(p) );
		}
	}

	// finish prepare h_input_streams;

}


void iNFAnt::calc_state_bitvec_length() {
	//
	//	int get_length_of_state_bitvec() const;
	// const int *get_enabled_bitvec() const;

	int max_length_of_state_bitvec = 0;
	for (int i = 0; i < get_num_nfa(); i++) {
		int t = 	get_transition_table(i)->get_length_of_state_bitvec();
		if (t > max_length_of_state_bitvec) {
			max_length_of_state_bitvec = t;
		}
	}

	//return max_length_of_state_bitvec;
	this->state_bitvec_length = max_length_of_state_bitvec;

}


void iNFAnt::prepare_state_vector() {
	this->calc_state_bitvec_length();

	this->arr_enabled_bitvec = new Array2<int>    (state_bitvec_length * get_num_nfa() );
	arr_enabled_bitvec->clear_to_zero();

	//cout << "state_bitvec_length = " << state_bitvec_length << endl;
	for (int i = 0; i < get_num_nfa(); i++) {
		
		for (int j = 0; j < nfas[i]->size(); j++) {
			Node *node = nfas[i]->get_node_by_int_id(j);
			if (node->is_start()) {
				//cout << "bit " << j << endl;
				bitvec::set_bit(arr_enabled_bitvec->get_host() + i * state_bitvec_length, state_bitvec_length, j);
			}
		}

	}

	//arr_enabled_bitvec->print();

}


void iNFAnt::set_alphabet(set<uint8_t> alphabet) {
	this->alphabet = alphabet;
}


void iNFAnt::copy_to_device() {
	arr_src_table->copy_to_device(); 
	arr_dst_table->copy_to_device(); 
	arr_start_position_transition_tables->copy_to_device(); 

    arr_symbol_trans_len->copy_to_device(); 
	arr_symbol_trans_index->copy_to_device(); 

    arr_states_status->copy_to_device(); 
    arr_state_start_position->copy_to_device(); 

	arr_input_streams->copy_to_device(); 

	arr_enabled_bitvec->copy_to_device(); 

	num_of_state_per_tb->copy_to_device();

}

void iNFAnt::launch_kernel() {

	  nfa->mark_cc_id();
    auto ccs = nfa_utils::split_nfa_by_ccs(*nfa);

    cout << "num_state_per_group = " << num_state_per_group << endl;
    auto grouped_nfas = nfa_utils::group_nfas(num_state_per_group, ccs); 
		this->old_ccs = grouped_nfas;

    for (auto g_nfa : grouped_nfas) {
        add_NFA(g_nfa);
    }

    for (auto cc : ccs) {
    	delete cc;
    }

    ccs.clear();

		for (auto g_nfa : nfas) {
			auto transtab = new AlphabetBasedTransitionTable(*g_nfa, this->alphabet);
			cout << "g_nfa_size = " << g_nfa->size() << " num_edges = " << g_nfa->get_num_transitions() << endl;
			cout << "max_edge_list_length_of_symbol = " << transtab->get_max_edge_list_length_of_symbol() << endl;
			cout << "avg_edge_list_length_of_symbol = " << std::fixed << transtab->get_avg_edge_list_length_of_symbol() << endl;
			transition_tables.push_back(transtab);
		}

    prepare_host_input_streams();
    init_host_transition_tables();
    prepare_host_state_info();
    prepare_state_vector();  
    copy_to_device();


    // output processing
    unsigned long long int match_array_capacity = this->output_buffer_size;
    cout << "match_array_capacity = " << match_array_capacity << endl;
    Array2<match_entry> *match_array = new Array2<match_entry> (match_array_capacity);
    Array2<unsigned long long int> *match_count = new Array2<unsigned long long int> (1);
    match_array->clear_to_zero();
    match_count->clear_to_zero();
    match_count->copy_to_device();
    //match_array->copy_to_device();
    // ends output


    cout << "launch_kernel_one_output " << endl;
    cout << "num_nfa = " << get_num_nfa() << " num_stream = " << get_num_streams() << endl;
	
	cout << "num_of_block = " << get_num_nfa() * get_num_streams() << endl;
	
	int shared_memory_size = state_bitvec_length * sizeof(int) * 2; //arr_enabled_bitvec->num_of_byte() * 2;
	cout << "state_bitvec_length = " << state_bitvec_length << endl;

	cout << "shared memory size per block = " << shared_memory_size << endl;

	dim3 blocksPerGrid(get_num_nfa(), get_num_streams()); 
 	dim3 threadsPerBlock(this->block_size, 1); /* NxN threads per block (2D) */ 

	cudaEvent_t start, stop;
  	float elapsedTime;

  	cudaEventCreate(&start);
  	cudaEventRecord(start,0);

	infant_kernel_one_output<<< blocksPerGrid, threadsPerBlock,  shared_memory_size >>>
	(arr_src_table->get_dev(),
	 arr_dst_table->get_dev(),
	 arr_start_position_transition_tables->get_dev(),
	 arr_symbol_trans_len->get_dev(),
	 arr_symbol_trans_index->get_dev(),
	 arr_states_status->get_dev(),
	 arr_state_start_position->get_dev(),
	 num_of_state_per_tb->get_dev(),
	 arr_input_streams->get_dev(),
	 symbol_streams[0].get_length(), // input stream length
	 arr_enabled_bitvec->get_dev(),
	 state_bitvec_length,

	 match_array->get_dev(),
	 match_array_capacity,
	 match_count->get_dev(),
	 this->report_on
	
	);

  	auto cudaError = cudaGetLastError();
  	if(cudaError != cudaSuccess)
  	{
   		 printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  	}

  	cudaError = cudaDeviceSynchronize();
  	if(cudaError != cudaSuccess)
  	{
   		 printf(" cudaDeviceSynchronize returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  	}

  	cudaEventCreate(&stop);
 	cudaEventRecord(stop,0);
 	cudaEventSynchronize(stop);

 	cudaEventElapsedTime(&elapsedTime, start,stop);
 	printf("Elapsed time : %f ms\n" ,elapsedTime);

 	float sec = elapsedTime / 1000.0;
 	cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * symbol_streams.size()) / 1000000.0 / sec  << endl;

  	match_count->copy_back();
    match_array->copy_back();
  	
  	cout << "count_of_match = " << match_count->get(0) << endl;


	// 	auto addResult = [](uint32_t node, uint32_t index, uint32_t nfa_index,
	// 									std::vector<uint64_t> *results) {
  //   uint64_t r = 0;
  //   node = (node | (nfa_index << 16));
  //   r = (uint32_t)node;
  //   r = r << 32;
  //   // index = (index | (nfa_index << 16));
  //   r = r | (uint32_t)index;
  //   results->push_back(r);
  // };

  	report_formatter rf;
		unsigned long long int mc = match_count->get(0);
	if(this->report_on) {
		rf.reports.resize(mc);
		std::for_each(std::execution::par_unseq, std::begin(rf.reports), std::end(rf.reports), [&](report& r) {
			u_int32_t i = &r - &rf.reports[0];
			assert(i< mc);
			match_entry mp = match_array->get(i);
  		auto sid = mp.state_id;
  		Node *n = nfas[mp.cc_id]->get_node_by_int_id(sid);
  		//cout << "report generated = " << n->str_id << " at " << mp.symbol_offset << endl;
  		report rp(mp.symbol_offset, n->original_id, mp.cc_id, mp.stream_id);
  		rf.reports[i]=rp;
			//  std::cout << i << "\n";
		});

		rf.print_to_file(output_file);
	}

		if (validation) {
			if (!this->report_on) {
					int dupnum = opt->duplicate_input_stream > 1
															? opt->duplicate_input_stream
															: 1;
					if (opt->quick_validation >= 0 &&
							opt->quick_validation * dupnum <= mc) {
						if (opt->quick_validation * dupnum == mc) {
            	tge_log("Quick Validation PASS! (report off, perfect)", BOLDGREEN);
						} else {
							tge_log("Quick Validation PASS! (report off, not perfect)",
											BOLDGREEN);
						}
					} else {
						tge_log("Quick Validation FAILED! (report off)", BOLDRED);
					}
			} else {
				if (opt->quick_validation >= 0) {
						int dupnum = opt->duplicate_input_stream > 1
															? opt->duplicate_input_stream
															: 1;
						if (opt->quick_validation * dupnum == rf.size()) {
										tge_log("Quick Validation PASS!", BOLDGREEN);
						} else {
										tge_log("Quick Validation FAILED!", BOLDRED);
						}
				} else {
						std::vector<uint64_t> ref_results, ref_db_results;
						int nfa_num = 10;
						auto old_grouped_nfas =
								nfa_utils::group_nfas_by_num(nfa_num, old_ccs);
						std::vector<Graph *> gs;
						for (auto nfa : old_grouped_nfas) {
										Graph *g = new Graph();
										g->ReadNFA(nfa);
										g->copyToDevice();
										gs.push_back(g);
						}

						auto compareResult = [](uint64_t r1,
																		uint64_t r2) -> bool {
							uint32_t input_1, state_1, input_2, state_2;
							input_1 = (uint32_t)(0xffffffff & r1);
							state_1 = (uint32_t)(r1 >> 32);
							input_2 = (uint32_t)(0xffffffff & r2);
							state_2 = (uint32_t)(r2 >> 32);
							// printf("{%u, %u}, ", state, input);
							if (input_1 == input_2)
								return state_1 < state_2;
							else
								return input_1 < input_2;
						};

						GroupCsr gcsr;
						gcsr.init(gs);
						automata_utils::automataGroupsReference(
								gs, arr_input_streams2->get_host(),
								get_num_streams(),
								symbol_streams[0].get_length() - 1, &ref_results,
								&ref_db_results, 0, gcsr);

						printf("\n############ Validate result "
										"############\n ");
						// automata_utils::automataValidation(&results,
						// &ref_results, true);
						std::sort(ref_results.begin(), ref_results.end(),
											compareResult);
						ref_results.erase(
								unique(ref_results.begin(), ref_results.end()),
								ref_results.end());
						printf("ref_results size: %zu\n", ref_results.size());

						if (rf.size() == ref_results.size()) {
										tge_log("Validation PASS!", BOLDGREEN);
						} else {
										tge_log("Validation FAILED!", BOLDRED);
						}
				}
			}
		}
		delete match_count;
		delete match_array;

}

