#pragma once


#include "NFA.h"
#include <string>
#include <vector>
#include "SymbolStream.h"
#include <map>
#include "compatible_group_helper.h"
#include <algorithm>
#include <cassert>
#include <functional>


using std::vector;
using std::string;
using std::pair;
using std::map;
using std::pair;

namespace nfa_utils {
	pair<int, int> get_num_hot_cold_from_map(const map<string, double>& freq_map, double cold_thres);

	void assign_hot_cold_by_hot_limit_by_bfs_layer(const vector<NFA *> &ccs, int hot_limit, int block_limit=256); 

	
	string from_range_to_symbol_set_str(int start, int end, bool complement);

	bitset<256> convert_from_range(pair<int, int> p);
	bitset<256> convert_from_range(int start, int end);
	
	void split_states_to_complete(NFA *nfa, int max_num_of_ranges = 3);

	vector<pair<int, int> > get_ranges_from_matchset(const bitset<256> &symbol_set);

	bool is_range_complete_matchset(bitset<256> symbol_set, int &start, int &end) ;

	void print_MC_complete_info(NFA *nfa, string suffix="");

	void add_fake_starters(NFA *nfa);

	
	void print_info_of_nfa(NFA *nfa, string suffix="", const std::function<bool(const Node&)>& f = {});


	void print_cg_info_of_ccs(const vector<NFA *> &ccs, AbstractCompatibleGroupHelper &agh, set<uint8_t> alphabet, string suffix="", string filename="");

	pair<int, int> print_starting_node_info(NFA *nfa);
	void print_nfa_info(NFA *nfa);

	int get_num_of_states(vector<NFA*> nfas) ;
	
	vector<NFA *> split_nfa_by_ccs(const NFA& nfa);

	vector<NFA *> group_nfas(int group_size, vector<NFA *> ccs);
	vector<NFA *> group_nfas_by_num(int &group_num, vector<NFA *> ccs);

	/** no ordering */
	vector<vector<int> > items_to_groups(const int group_size, const vector<int> item_sizes);

	NFA *merge_nfas(vector<NFA *> nfas); 

	vector<NFA*> merge_nfas_by_group(const vector<vector<int> > &grps, const vector<NFA *> &ccs);
	
	bool limit_out_degree_on_nfa(NFA* nfa, int limit);
	
	pair<int, int> limit_out_degree_on_ccs(vector<NFA *> &ccs, int limit, bool quit = true, bool remove_degree = false);

	/**
	 *
	 * for observing the topology
	 *
	 */
	void dump_to_gr_file(const NFA& nfa, string str);


	map<string, double> read_freq_map(string filename);

	map<string, int> get_state_hotcold_info(const SymbolStream& ss, const vector<NFA *> &ccs);

	void calc_compatible_groups(const vector<NFA*> &ccs, const set<uint8_t> &alphabet, AbstractCompatibleGroupHelper& cghelper, map<int, int> &ccid_cgsize, map<string, int> &nodestrid_cgid, string suffix=""); 


	int search_state_id_in_nfa_vector(const vector<NFA*> &ccs, string state_id);



	template<class Compare >
	NFA *remap_intid_of_nfa1(NFA *original_nfa, Compare comp, int boundary) {
		vector<Node*> nodes;
		for (int i = 0; i < original_nfa->size(); i++) {
			nodes.push_back(original_nfa->get_node_by_int_id(i));
		}

		assert(boundary > 0 && boundary <= original_nfa->size());

		std::stable_sort(nodes.begin(), nodes.begin() + boundary, comp);

		NFA *res = new NFA ();
		for (auto n : nodes) {
			Node *new_node = new Node();
			*new_node = *n;

			res->addNode(new_node);
		}

		for (auto n : nodes) {
			auto adjs = original_nfa->get_adj(n->str_id);
			for (auto to : adjs) {
				res->addEdge(n->str_id, to);
			}
		}

		return res;
	}


	template<class Compare >
	NFA *remap_intid_of_nfa(NFA *original_nfa, Compare comp ) {
		return remap_intid_of_nfa1(original_nfa, comp, original_nfa->size());
	}


	vector<int> get_hot_boundaries_of_grps(const vector<NFA *> &grouped_nfas, double cold_thres);

	void output_dot_files(string path, const vector<NFA*> &ccs);

	vector<vector<int> > group_nfas_by_hotcold(int block_size, vector<NFA *> ccs, double cold_thres);
	
	void print_nodes_to_file(string filename, const vector<NFA *> &ccs);
	
	void print_edge_and_matchset_to_file(string filename, const vector<NFA *> &ccs) ;

	vector<int> get_cc_ids_from_state_id(const vector<NFA *> ccs, const vector<string> state_id_representatives);

	vector<int> get_all_start_node_id(NFA *nfa);

}



class compatible_group {
public:
	compatible_group() : input_size(1000000) {
		
	}

	int cg_id;
	int cc_id;
	int global_id;

	int input_size;

	vector< std::pair<string, int> > stes; 
	
	void add(string str_id, int num_activation) {
		stes.push_back(std::make_pair(str_id, num_activation));
	}

	int size() const {
		return stes.size();
	}

	int get_cg_num_activations() const {
		int ss = 0;
		
		for (auto it : this->stes) {
			ss += it.second;
		}

		return ss;
	}

	int get_max_num_activations_of_states() const {
		int mm = 0;
		
		for (auto it : this->stes) {
			if (it.second > mm ) {
				mm = it.second;
			}
		}

		return mm;
	}

	int get_min_num_activations_of_states() const {
		int mm = 0x7fffffff;

		for (auto it : this->stes) {
			if (it.second < mm) {
				mm = it.second;
			}
		}

		return mm;
	}

	double get_avg_activations_of_states() const {
		double ss = 0;
		for (auto it : this->stes) {
			ss += it.second;
		}

		return ss / this->stes.size();
	}


	vector<int> get_type() const {

		// Type 1: Very important single state.  
		// The CG has only one state, and it was activated > 30%


		// Type 2: Important, but could be switched frequently. 
		// The CG has more than one states, and all states were activated > 30%. 


		// Type 3: Not important, single state. 
		// The CG has only one state, and it was activated < 3%


		// Type 4: Not important, multiple states. 
		// The CG has more than one states, and all of them were activated < 3%. 


		// Type 5: Bi-modal. 
		// The compatible group has one state was activated > 30%, but all others were activated less than 3% 


		vector<int> belongs_to_types; 
		
		if (this->size() == 1 && ((double)this->get_max_num_activations_of_states()) / input_size > 0.3 ) {
			belongs_to_types.push_back(1);
		}

		if (this->size() > 1) {
			if (((double)this->get_min_num_activations_of_states()) / input_size <= 0.03) {
				belongs_to_types.push_back(2);
			}
		}

		if (this->size() == 1 && ((double)this->get_max_num_activations_of_states()) / input_size <= 0.03  ) {
			belongs_to_types.push_back(3);
		}

		if (this->size() > 1 ) {
			if (((double)this->get_max_num_activations_of_states()) / input_size <= 0.03) {
				belongs_to_types.push_back(4);
			}
		}

		
		if (this->size() > 1 ) {
			int num_state_gt_03 = 0;
			int num_state_lt_003 = 0;
			for (auto it : this->stes) {
				if (it.second > 0.3 * input_size) {
					num_state_gt_03 ++;
				}

				if (it.second < 0.03 * input_size) {
					num_state_lt_003 ++;
				}
			}

			if (num_state_gt_03 == 1 && num_state_lt_003 >= 1) {
				belongs_to_types.push_back(5);
			}
		}


		return belongs_to_types;
	}



};

