/*
 * utils.cpp
 *
 *  Created on: May 16, 2018
 *      Author: hyliu
 */



#include "gpunfautils/utils.h"
#include "NFA.h"
#include "gpunfautils/common.h"
#include <unordered_map>
#include <iostream>
#include <string>
#include <algorithm>
#include <cassert>
#include <queue>
#include <stack>
#include <memory>
#include <map>
#include <vector>
#include <iomanip>
#include <fstream>
#include <climits>
#include "commons/nfa_utils.h"

#include <sys/types.h>
#include <sys/stat.h>



using std::ifstream;
using std::string;
using std::endl;
using std::cout;
using std::pair;


using namespace nfa_utils;





NFA *nfa_utils::filter_and_create_new_NFA(NFA *nfa, const std::set<string> &states_to_keep) {
	NFA *filtered_nfa = new NFA ();
	
	for (int i = 0; i < nfa->size(); i++) {
		auto n = nfa->get_node_by_int_id(i);
		if (states_to_keep.find(n->str_id) != states_to_keep.end()) {
			Node *nn = new Node();
			*nn = *n;

			filtered_nfa->addNode(nn);
		}
	}

	for (int i = 0; i < nfa->size(); i++) {
		auto n = nfa->get_node_by_int_id(i);
		if (states_to_keep.find(n->str_id) != states_to_keep.end()) {
			auto adjs = nfa->get_adj(n->str_id);
			for (auto to : adjs) {
				if (states_to_keep.find(to) != states_to_keep.end()) {
					filtered_nfa->addEdge(n->str_id, to);
				}
			}
		}
	}

	return filtered_nfa;
}
	

void nfa_utils::add_fake_start_node_for_ccs(vector<NFA *> ccs) {

    //cout << "ccs size add fake start node for ccs = " << ccs.size() << endl;

	for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
		auto cc = ccs[cc_id];

		for (int nodeid = 0; nodeid < cc->size(); nodeid++) {
			auto node = cc->get_node_by_int_id(nodeid);
			if (node->is_start_always_enabled() && cc->has_self_loop(nodeid) ) {
				//cout << "allinput and selfloop ccid " << cc_id << "  nodeid " << nodeid << endl;
				cc->remove_self_loop(nodeid);
			}
		}
	}


	for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
		vector<Node *> starting_nodes;
		auto cc = ccs[cc_id];
		
		for (int nodeid = 0; nodeid < cc->size(); nodeid++) {
			auto node = cc->get_node_by_int_id(nodeid);
			//cout << "node start = " << node->start << endl;
			if (node->is_start()) {
			    //cout << "here!" << endl;
				starting_nodes.push_back(node);
			}
		}

		if (starting_nodes.size() == 0) {
		    cout << "warning : " << " ccid = " << cc_id << " " << " does not have start states;" << endl;
		    continue;
		}
		// assert(starting_nodes.size() > 0);

		Node *fake_starting_node = new Node();
		fake_starting_node->start = NODE_START_ENUM::START_ALWAYS_ENABLED;
		fake_starting_node->str_id = "fakestart_cc_" + std::to_string(cc_id);
		fake_starting_node->cc_id = cc_id;
		fake_starting_node->symbol_set.set();
		fake_starting_node->hot_degree = 1.0;

		cc->addNode(fake_starting_node);

		for (auto start_node : starting_nodes) {
			cc->addEdge(fake_starting_node->str_id, start_node->str_id);
			start_node->start = 0;
		}
	}


}







set<uint8_t> nfa_utils::get_alphabet_from_nfa(NFA *nfa) {
	// NOT TESTED! 

	set<uint8_t> ab_set;
	for (uint8_t symbol = 0; (int) symbol < ALPHABET_SIZE; symbol++) {
		for (int node_id = 0; node_id < nfa->size(); node_id++) {
			auto node = nfa->get_node_by_int_id(node_id);
			if (node->match2(symbol)) {
				ab_set.insert(symbol);
				break;
			}
		}
	}
	return ab_set;
}


Array2<int4> *nfa_utils::create_int4_tt_for_nfa(NFA *nfa) {
	//cout << "create_int4_tt_for_nfa_size = " << nfa->size() << endl;

	auto trans_table = new Array2<int4> (nfa->size() * ALPHABET_SIZE);
	trans_table->fill({-1, -1, -1, -1});

	//auto alphabet = get_alphabet_from_nfa(nfa);

	for (int nodeid = 0; nodeid < nfa->size(); nodeid++) {
		Node *n = nfa->get_node_by_int_id(nodeid);
		auto adjs = nfa->get_adj(n->str_id);

		for (uint8_t symbol = 0; (int) symbol < 255; symbol++) {
			vector<int> vec_to_push_transition_table;
		
			for (auto adj : adjs) {
				Node *to_node = nfa->get_node_by_str_id(adj);
				if (to_node->match2(symbol)) {
					vec_to_push_transition_table.push_back(to_node->sid);
				}
			}

			assert(vec_to_push_transition_table.size() <= 4);
			int vec_size = vec_to_push_transition_table.size();
			for (int q = vec_size; q < 4; q ++) {
				vec_to_push_transition_table.push_back(-1);
			}
			assert(vec_to_push_transition_table.size() == 4);

			trans_table->set(nodeid * ALPHABET_SIZE + ((int) symbol), {vec_to_push_transition_table[0], vec_to_push_transition_table[1], vec_to_push_transition_table[2], vec_to_push_transition_table[3]});

			// row is state, column is symbol.
		}
	}

	return trans_table;
}



Array2<STE_dev<4> > *nfa_utils::create_list_of_STE_dev(NFA *nfa) {


	Array2< STE_dev<4> > *res = new Array2< STE_dev<4> > (nfa->size());

	for (int i = 0; i < nfa->size(); i++) {
		auto node = nfa->get_node_by_int_id(i);

		STE_dev<4> ste; 
		memset(&ste, 0, sizeof(ste));

		for (int s = 0; s < 256; s++) {
			if (node->match2((uint8_t) s )) {
				ste.ms[s / 32] |=  (1 << (s % 32));  
			}
		}

		auto adjs = nfa->get_adj(node->str_id);

		int degree = 0;
		for (auto adj : adjs) {
			Node *to_node = nfa->get_node_by_str_id(adj);
			int tonode_intid = to_node->sid;
			ste.edge_dst[degree++] = tonode_intid;
		}

		if (node->is_report()) {
			ste.attribute = 1;
		}

		if (node->is_start_always_enabled()) {
			ste.attribute |= (1 << 1);
		}

		
		if (node->is_start()) {
			ste.attribute |= (1 << 2);
		}

		ste.degree = degree;

		//https://stackoverflow.com/questions/5700204/c-does-implicit-copy-constructor-copy-array-member-variable
		// should be okay.
		res->set(i, ste);

		/*

			int32_t ms[8]; // 8 * 32 = 256;
			int   edge_dst[DEGREE_LIMIT];
			char     attribute;  // is report? 
		
		*/


	}


	return res;
}

// compress edge
Array2<STE_dev4 > *nfa_utils::create_STE_dev4_compressed_edges(NFA *nfa) {
	Array2<STE_dev4 > *res = new Array2 <STE_dev4> (nfa->size());

	for (int i = 0; i < nfa->size(); i++) {
		auto node = nfa->get_node_by_int_id(i);

		STE_dev4 ste; 
		memset(&ste, 0, sizeof(ste));

		for (int s = 0; s < 256; s++) {
			if (node->match2((uint8_t) s )) {
				ste.ms[s / 32] |=  (1 << (s % 32));  
			}
		}

		auto adjs = nfa->get_adj(node->str_id);
		
		assert(adjs.size() <= 4);

		unsigned long long edges = 0; // I will try 64 bit int first. 

		int degree = 0;
		
		for (auto adj : adjs) {
			Node *to_node = nfa->get_node_by_str_id(adj);
			unsigned int tonode_intid = to_node->sid;
			edges = (edges << 16) | tonode_intid;
			
			degree++;
			//cout << "tonode" << tonode_intid << endl;
		}

		//cout << "combined " << edges <<  " degree" << degree << endl;
 
		ste.edges = edges;

		/*for (int t = 0; t < degree; t++) {
			unsigned int uncombinedid = (edges >> (t * 16)) & 65535;
			Node *to_node = nfa->get_node_by_str_id(adjs[degree-t-1]);
			unsigned int tonode_intid = to_node->sid;

			assert(uncombinedid == tonode_intid);
		}*/

		if (node->is_report()) {
			ste.attribute = 1;
		}

		if (node->is_start_always_enabled()) {
			ste.attribute |= (1 << 1);
		}

		if (node->is_start()) {
			ste.attribute |= (1 << 2);
		}

		// least bit: report ;   second least bit : start-all-input;

		ste.degree = degree;

		res->set(i, ste);

	}


	return res;
}



Array2<unsigned long long> *nfa_utils::create_ull64_for_nfa(NFA *nfa) {
	auto trans_table = new Array2<unsigned long long> (nfa->size() * ALPHABET_SIZE);
	
	for (int nodeid = 0; nodeid < nfa->size(); nodeid++) {
		Node *n = nfa->get_node_by_int_id(nodeid);
		auto adjs = nfa->get_adj(n->str_id);

		for (int symbol = 0; symbol < 256; symbol++) {
			vector<int> vec_to_push_transition_table;
		
			for (auto adj : adjs) {
				Node *to_node = nfa->get_node_by_str_id(adj);
				if (to_node->match2((uint8_t) symbol)) {
				    assert(to_node->sid < EMPTY_ENTRY);
					vec_to_push_transition_table.push_back(to_node->sid);
				}
			}

			assert(vec_to_push_transition_table.size() <= 4);
			int vec_size = vec_to_push_transition_table.size();
			for (int q = vec_size; q < 4; q ++) {
				vec_to_push_transition_table.push_back(EMPTY_ENTRY);
			}

			assert(vec_to_push_transition_table.size() == 4);

			unsigned long long edges = 0;
			for (int j = 0; j < 4; j++) {
				edges = (edges << 16) | vec_to_push_transition_table[j]; 
			}


			trans_table->set(nodeid * ALPHABET_SIZE + ((int) symbol), edges);

			// row is state, column is symbol.
		}
	}

	return trans_table;
}



void nfa_utils::test_compress_matchset1(NFA *nfa) {
	int total = 0;

	int complete = 0;
	int complement = 0;

	int start, end;

	for (int i = 0; i < nfa->size(); i++) {
	 	auto node = nfa->get_node_by_int_id(i);
	 	if (nfa_utils::is_range_complete_matchset(node->symbol_set, start, end)) {
	 		//cout << "node  " << node->sid << "complete start = " << start << " " << end << endl;
	 		node->complete = true;
	 		node->match_set_range = end - start;
	 		total++;
	 		complete ++;
	 	} else {
	 		auto complement_matchset = ~node->symbol_set;
	 		if (nfa_utils::is_range_complete_matchset(complement_matchset, start, end)) {
	 			//cout << "node  " << node->sid << "complement complete start = " << start << " " << end << endl; 
	 			node->complete = true;
	 			node->match_set_range = end - start;
	 			total++;
	 			complement++;
	 		} else {
	 			node->complete = false;
	 		}
	 	}	
	}

	//cout << "num_of_states = " << nfa->size() << endl;
	//cout << "num_of_complete = " << complete << endl;
	//cout << "num_of_complement = " << complement << endl;
	//cout << "num_of_complete_or_complement = " << total << endl;
	 
	//cout << "comlete_complement_ratio = " << (total + 0.0) / nfa->size() << endl; 

}


bool nfa_utils::is_all_complete(NFA *nfa) {
	int total = 0;

	 int start, end;

	 for (int i = 0; i < nfa->size(); i++) {
	 	auto node = nfa->get_node_by_int_id(i);
	 	if (nfa_utils::is_range_complete_matchset(node->symbol_set, start, end)) {
	 		//cout << "node  " << node->sid << "complete start = " << start << " " << end << endl;
	 		node->complete = true;
	 		node->match_set_range = end -start;
	 		total++;
	 	} else {
	 		auto complement_matchset = ~node->symbol_set;
	 		if (nfa_utils::is_range_complete_matchset(complement_matchset, start, end)) {
	 			//cout << "node  " << node->sid << "complement complete start = " << start << " " << end << endl; 
	 			node->complete = true;
	 			node->match_set_range = end -start;
	 			total++;
	 		} else {
	 			node->complete = false;
	 		}
	 	}	
	 }
	 
	//cout << "comlete_complement_ratio = " << (total + 0.0) / nfa->size() << endl; 
	return total == nfa->size();
}



Array2<STE_dev4_compressed_matchset > *nfa_utils::create_STE_dev4_compressed_edges_and_compressed_matchset(NFA *nfa) {
	Array2<STE_dev4_compressed_matchset> *res = new Array2 <STE_dev4_compressed_matchset> (nfa->size());
	
	for (int i = 0; i < nfa->size(); i++) {
		auto node = nfa->get_node_by_int_id(i);

		STE_dev4_compressed_matchset ste; 

		memset(&ste, 0, sizeof(ste));

		for (int s = 0; s < 256; s++) {
			if (node->match2((uint8_t) s )) {
				ste.ms[s / 32] |=  (1 << (s % 32));  
			}
		}

		auto adjs = nfa->get_adj(node->str_id);
		
		assert(adjs.size() <= 4);

		unsigned long long edges = 0; // I will try 64 bit int first. 

		int degree = 0;
		
		for (auto adj : adjs) {
			Node *to_node = nfa->get_node_by_str_id(adj);
			unsigned int tonode_intid = to_node->sid;
			edges = (edges << 16) | tonode_intid;
			
			degree++;
			//cout << "tonode" << tonode_intid << endl;
		}

		//cout << "combined " << edges <<  " degree" << degree << endl;
 
		ste.edges = edges;

		/*for (int t = 0; t < degree; t++) {
			unsigned int uncombinedid = (edges >> (t * 16)) & 65535;
			Node *to_node = nfa->get_node_by_str_id(adjs[degree-t-1]);
			unsigned int tonode_intid = to_node->sid;

			assert(uncombinedid == tonode_intid);
		}*/

		if (node->is_report()) {
			ste.attribute = 1;
		}

		if (node->is_start_always_enabled()) {
			ste.attribute |= (1 << 1);
		}

		if (node->is_start()) {
			ste.attribute |= (1 << 2);
		}


		int start = -1, end = -1;
		if (nfa_utils::is_range_complete_matchset(node->symbol_set, start, end)) {
	 		//cout << "node  " << node->sid << "complete start = " << start << " " << end << endl;
	 		//total++;
	 		node->complete = true;

			ste.attribute |= (1 << 2);
			ste.start_end = 0;
			ste.start_end |= end;
			ste.start_end = (ste.start_end << 8) | start;

			//ste.start = start;
			//ste.end   = end;

			for (int tt = 0; tt < 256; tt++) {
				if (tt >= start && tt < end) {
					assert(node->symbol_set.test(tt));
				} else {
					assert(!node->symbol_set.test(tt));
				}
			}

			assert( (uint8_t)(ste.start_end & 255) == start );
			assert( (uint8_t)( (ste.start_end >> 8) & 255) == end );


	 	} else {
	 		auto complement_matchset = ~node->symbol_set;
	 		if (nfa_utils::is_range_complete_matchset(complement_matchset, start, end)) {
	 			//cout << "node  " << node->sid << "complement complete start = " << start << " " << end << endl; 
	 			//total++;
	 			node->complete = true;
	 			node->complement = true;
	 		
	 			ste.attribute |= (1 << 2); // complete
	 			ste.attribute |= (1 << 3); // complement

	 			ste.start_end = 0;
				ste.start_end |= end;
				ste.start_end = (ste.start_end << 8) | start;

				//ste.start = start;
				//ste.end   = end;

				for (int tt = 0; tt < 256; tt++) {
					if (tt >= start && tt < end) {
						assert(!node->symbol_set.test(tt));
					} else {
						assert(node->symbol_set.test(tt));
					}
				}

				assert( (uint8_t)( ste.start_end & 255) == start );
				assert( (uint8_t)( (ste.start_end >> 8) & 255) == end );


	 		}


	 	}


		// least bit: report ;   second least bit : start-all-input;

		ste.degree = degree;

		res->set(i, ste);

	}


	return res;
}


Array2<STE_dev4_compressed_matchset_allcomplete > *nfa_utils::create_STE_dev4_compressed_edges_and_compressed_matchset_allcomplete(NFA *nfa) {
		Array2<STE_dev4_compressed_matchset_allcomplete> *res = new Array2 <STE_dev4_compressed_matchset_allcomplete> (nfa->size());
	
	for (int i = 0; i < nfa->size(); i++) {
		auto node = nfa->get_node_by_int_id(i);

		STE_dev4_compressed_matchset_allcomplete ste; 

		memset(&ste, 0, sizeof(ste));
		auto adjs = nfa->get_adj(node->str_id);
		
		assert(adjs.size() <= 4);

		unsigned long long edges = 0; // I will try 64 bit int first. 

		int degree = 0;
		
		for (auto adj : adjs) {
			Node *to_node = nfa->get_node_by_str_id(adj);
			unsigned int tonode_intid = to_node->sid;
			edges = (edges << 16) | tonode_intid;
			
			degree++;
			//cout << "tonode" << tonode_intid << endl;
		}

		//cout << "combined " << edges <<  " degree" << degree << endl;
 
		ste.edges = edges;

		/*for (int t = 0; t < degree; t++) {
			unsigned int uncombinedid = (edges >> (t * 16)) & 65535;
			Node *to_node = nfa->get_node_by_str_id(adjs[degree-t-1]);
			unsigned int tonode_intid = to_node->sid;

			assert(uncombinedid == tonode_intid);
		}*/

		if (node->is_report()) {
			ste.attribute = 1;
		}

		if (node->is_start_always_enabled()) {
			ste.attribute |= (1 << 1);
		}

		if (node->is_start()) {
			ste.attribute |= (1 << 2);
		}

		int start = -1, end = -1;
		if (nfa_utils::is_range_complete_matchset(node->symbol_set, start, end)) {
	 		//cout << "node  " << node->sid << "complete start = " << start << " " << end << endl;
	 		//total++;
	 		node->complete = true;

			ste.attribute |= (1 << 3);   // problem!!! complete redefine to 3, complement redefine to 4
			ste.start_end = 0;
			ste.start_end |= end;
			ste.start_end = (ste.start_end << 8) | start;

			//ste.start = start;
			//ste.end   = end;

			for (int tt = 0; tt < 256; tt++) {
				if (tt >= start && tt < end) {
					assert(node->symbol_set.test(tt));
				} else {
					assert(!node->symbol_set.test(tt));
				}
			}

			assert( (uint8_t)(ste.start_end & 255) == start );
			assert( (uint8_t)( (ste.start_end >> 8) & 255) == end );


	 	} else {
	 		auto complement_matchset = ~node->symbol_set;
	 		if (nfa_utils::is_range_complete_matchset(complement_matchset, start, end)) {
	 			//cout << "node  " << node->sid << "complement complete start = " << start << " " << end << endl; 
	 			//total++;
	 			node->complete = true;
	 			node->complement = true;
	 		
	 			ste.attribute |= (1 << 3); // complete
	 			ste.attribute |= (1 << 4); // complement

	 			ste.start_end = 0;
				ste.start_end |= end;
				ste.start_end = (ste.start_end << 8) | start;

				//ste.start = start;
				//ste.end   = end;

				for (int tt = 0; tt < 256; tt++) {
					if (tt >= start && tt < end) {
						assert(!node->symbol_set.test(tt));
					} else {
						assert(node->symbol_set.test(tt));
					}
				}

				assert( (uint8_t)( ste.start_end & 255) == start );
				assert( (uint8_t)( (ste.start_end >> 8) & 255) == end );


	 		}


	 	}


		// least bit: report ;   second least bit : start-all-input;

		ste.degree = degree;

		res->set(i, ste);

	}


	return res;
}




Array2<STE_nodeinfo_new_imp> *nfa_utils::create_STE_nodeinfos_new(NFA *cc) {
	
	Array2<STE_nodeinfo_new_imp> *res = new Array2 <STE_nodeinfo_new_imp> (cc->size());
	
	for (int i = 0; i < cc->size(); i++) {
		auto node = cc->get_node_by_int_id(i);

		STE_nodeinfo_new_imp ste; 

		//cout << "sizeof_ste = " << sizeof(ste) << endl;
		memset(&ste, 0, sizeof(STE_nodeinfo_new_imp));

		auto adjs = cc->get_adj(node->str_id);
		
		assert(adjs.size() <= 4);

		unsigned long long edges = 0; // I will try 64 bit int first. 

		int degree = 0;
		
		for (auto adj : adjs) {
			Node *to_node = cc->get_node_by_str_id(adj);
			unsigned int tonode_intid = to_node->sid;
			edges = (edges << 16) | tonode_intid;
			
			degree++;
			//cout << "tonode" << tonode_intid << endl;
		}

		//cout << "combined " << edges <<  " degree" << degree << endl;
 
		ste.edges = edges;

		/*for (int t = 0; t < degree; t++) {
			unsigned int uncombinedid = (edges >> (t * 16)) & 65535;
			Node *to_node = cc->get_node_by_str_id(adjs[degree-t-1]);
			unsigned int tonode_intid = to_node->sid;

			assert(uncombinedid == tonode_intid);
		}*/

		ste.attribute = 0;

		if (node->is_report()) {
			ste.attribute = 1;
		}

		
		if (node->is_start_always_enabled()) {
			ste.attribute |= (1 << 1);
		}

		if (node->is_start()) {
			ste.attribute |= (1 << 2);
		}

		if (cc->get_indegree_of_node(node->str_id) == 1) {
			// we do not have to dedup in this case. 
			ste.attribute |= (1 << 5); // but it is not good. Looks. 
		}

		int start = -1, end = -1;
		if (nfa_utils::is_range_complete_matchset(node->symbol_set, start, end)) {
	 		//cout << "node  " << node->sid << "complete start = " << start << " " << end << endl;
	 		//total++;
	 		node->complete = true;
	 		node->complement = true;

			ste.attribute |= (1 << 3);   


			//ste.start_end = 0;
			//ste.start_end |= end;
			//ste.start_end = (ste.start_end << 8) | start;

			
			ste.start = start;
			ste.end = end - 1;

			//ste.start = start;
			//ste.end   = end;

			for (int tt = 0; tt < 256; tt++) {
				if (tt >= start && tt < end) {
					assert(node->symbol_set.test(tt));
				} else {
					assert(!node->symbol_set.test(tt));
				}
			}

			if ((uint8_t) ste.start  != start ) {
				cout << "ste.start " << ste.start << " " << start << endl;
				assert( (uint8_t) ste.start  == start );
				
			}

			if ( (uint8_t)ste.end != end - 1) {
				cout << "ste.end " << ste.end << " " << end << endl;
				assert(  (uint8_t)ste.end == end - 1 );
			}
			


	 	} else {
	 		auto complement_matchset = ~node->symbol_set;
	 		if (nfa_utils::is_range_complete_matchset(complement_matchset, start, end)) {
	 			//cout << "node  " << node->sid << "complement complete start = " << start << " " << end << endl; 
	 			//total++;
	 			node->complete = true;
	 		
	 			ste.attribute |= (1 << 3); // complete
	 			ste.attribute |= (1 << 4); // complement

	 			//ste.start_end = 0;
				//ste.start_end |= end;
				//ste.start_end = (ste.start_end << 8) | start;

				ste.start = start;
				ste.end   = end - 1;

				for (int tt = 0; tt < 256; tt++) {
					if (tt >= start && tt < end) {
						assert(!node->symbol_set.test(tt));
					} else {
						assert(node->symbol_set.test(tt));
					}
				}

				assert( (uint8_t)( ste.start ) == start );
				assert((uint8_t)(   ste.end ) == end - 1);
	 		}
	 	}


		// least bit: report ;   second least bit : start-all-input;

		ste.degree = degree;

		res->set(i, ste);

		// verify
		if (node->is_report()) {
			assert(ste.attribute & 1 == 1);
		} else {

			
			if (ste.attribute & 1 != 0) {
				//cout << "ste.attribute = " << ste.attribute << endl;
				assert(ste.attribute & 1 == 0);
			}
			
		}


	}


	return res;
}



Array2<STE_nodeinfo_new_imp2> *nfa_utils::create_STE_nodeinfos_new2(NFA *cc) {

    Array2<STE_nodeinfo_new_imp2> *res = new Array2 <STE_nodeinfo_new_imp2> (cc->size());

    for (int i = 0; i < cc->size(); i++) {
        auto node = cc->get_node_by_int_id(i);

        STE_nodeinfo_new_imp2 ste;

        //cout << "sizeof_ste = " << sizeof(ste) << endl;
        memset(&ste, 0, sizeof(STE_nodeinfo_new_imp));

        auto adjs = cc->get_adj(node->str_id);

        assert(adjs.size() <= 4);

        unsigned long long edges = 0; // I will try 64 bit int first.

        int degree = 0;

        for (auto adj : adjs) {
            Node *to_node = cc->get_node_by_str_id(adj);
            unsigned int tonode_intid = to_node->sid;
            edges = (edges << 16) | tonode_intid;

            degree++;
            //cout << "tonode" << tonode_intid << endl;
        }

        //cout << "combined " << edges <<  " degree" << degree << endl;

        ste.edges = edges;

        /*for (int t = 0; t < degree; t++) {
            unsigned int uncombinedid = (edges >> (t * 16)) & 65535;
            Node *to_node = cc->get_node_by_str_id(adjs[degree-t-1]);
            unsigned int tonode_intid = to_node->sid;

            assert(uncombinedid == tonode_intid);
        }*/

        ste.attribute = 0;

        if (node->is_report()) {
            ste.attribute = 1;
        }

        if (node->is_start_always_enabled()) {
            ste.attribute |= (1 << 1);
        }

        if (node->is_start()) {
            ste.attribute |= (1 << 2);
        }

        if (cc->get_indegree_of_node(node->str_id) == 1) {
            // we do not have to dedup in this case.
            ste.attribute |= (1 << 5); // but it is not good. Looks.
        }

        int start = -1, end = -1;
        if (nfa_utils::is_range_complete_matchset(node->symbol_set, start, end)) {
            //cout << "node  " << node->sid << "complete start = " << start << " " << end << endl;
            //total++;
            node->complete = true;
            node->complement = true;

            ste.attribute |= (1 << 3);


            //ste.start_end = 0;
            //ste.start_end |= end;
            //ste.start_end = (ste.start_end << 8) | start;


            //ste.start = start;
            //ste.end = end - 1;

            //ste.start = start;
            //ste.end   = end;

            for (int tt = 0; tt < 256; tt++) {
                if (tt >= start && tt < end) {
                    assert(node->symbol_set.test(tt));
                } else {
                    assert(!node->symbol_set.test(tt));
                }
            }



        } else {
            auto complement_matchset = ~node->symbol_set;
            if (nfa_utils::is_range_complete_matchset(complement_matchset, start, end)) {
                //cout << "node  " << node->sid << "complement complete start = " << start << " " << end << endl;
                //total++;
                node->complete = true;

                ste.attribute |= (1 << 3); // complete
                ste.attribute |= (1 << 4); // complement

                //ste.start_end = 0;
                //ste.start_end |= end;
                //ste.start_end = (ste.start_end << 8) | start;


                for (int tt = 0; tt < 256; tt++) {
                    if (tt >= start && tt < end) {
                        assert(!node->symbol_set.test(tt));
                    } else {
                        assert(node->symbol_set.test(tt));
                    }
                }

               // assert( (uint8_t)( ste.start ) == start );
               // assert((uint8_t)(   ste.end ) == end - 1);
            }
        }


        // least bit: report ;   second least bit : start-all-input;

        ste.degree = degree;

        res->set(i, ste);

        // verify
        if (node->is_report()) {
            assert(ste.attribute & 1 == 1);
        } else {


            if (ste.attribute & 1 != 0) {
                //cout << "ste.attribute = " << ste.attribute << endl;
                assert(ste.attribute & 1 == 0);
            }

        }


    }


    return res;
}




Array2<STE_nodeinfo_new_imp_withcg> *nfa_utils::create_STE_nodeinfos_new_withcg(NFA *cc) {
	/*

struct STE_nodeinfo_new_imp {
	unsigned long long edges;
	
	unsigned int attribute : 8; 
	unsigned int start : 8;
	unsigned int end : 8;
	unsigned int degree : 8;
};


struct STE_nodeinfo_new_imp_withcg {
	unsigned long long edges;
	
	unsigned int attribute : 8; 
	unsigned int start : 8;
	unsigned int end : 8;
	unsigned int degree : 8;

	// cg_id ---> write position in gpu kernel. 
	unsigned int cg_id : 8;

};
	*/

	//cout << "sizeof_cg_ste = " << sizeof(STE_nodeinfo_new_imp_withcg) << endl;

	Array2<STE_nodeinfo_new_imp_withcg> *res = new Array2 <STE_nodeinfo_new_imp_withcg> (cc->size());
	
	for (int i = 0; i < cc->size(); i++) {
		auto node = cc->get_node_by_int_id(i);

		STE_nodeinfo_new_imp_withcg ste; 

		//cout << "sizeof_ste = " << sizeof(ste) << endl;
		memset(&ste, 0, sizeof(STE_nodeinfo_new_imp_withcg));

		auto adjs = cc->get_adj(node->str_id);
		
		assert(adjs.size() <= 4);

		unsigned long long edges = 0; // I will try 64 bit int first. 

		int degree = 0;
		
		for (auto adj : adjs) {
			Node *to_node = cc->get_node_by_str_id(adj);
			unsigned int tonode_intid = to_node->sid;
			edges = (edges << 16) | tonode_intid;


			degree++;


			//cout << "tonode" << tonode_intid << endl;
		}

		//cout << "combined " << edges <<  " degree" << degree << endl;
 
		ste.edges = edges;

		/*for (int t = 0; t < degree; t++) {
			unsigned int uncombinedid = (edges >> (t * 16)) & 65535;
			Node *to_node = cc->get_node_by_str_id(adjs[degree-t-1]);

			unsigned int tonode_intid = to_node->sid;

			//cout << "tonode->str_id = " << to_node->str_id << " to_node->cg_id = " << to_node->cg_id << endl;
			ste.cg_of_to_edges[t] = to_node->cg_id;


			assert(uncombinedid == tonode_intid);
		}*/

		ste.attribute = 0;

		ste.cg_id = node->cg_id;

		if (node->is_report()) {
			ste.attribute = 1;
		}

		
		if (node->is_start_always_enabled()) {
			ste.attribute |= (1 << 1);
		}

		if (node->is_start()) {
			ste.attribute |= (1 << 2);
		}

		int start = -1, end = -1;
		if (nfa_utils::is_range_complete_matchset(node->symbol_set, start, end)) {
	 		//cout << "node  " << node->sid << "complete start = " << start << " " << end << endl;
	 		//total++;
	 		node->complete = true;
	 		node->complement = true;

			ste.attribute |= (1 << 3);   // problem!!! complete redefine to 3, complement redefine to 4
			


			//ste.start_end = 0;
			//ste.start_end |= end;
			//ste.start_end = (ste.start_end << 8) | start;

			
			ste.start = start;
			ste.end = end;

			//ste.start = start;
			//ste.end   = end;

			for (int tt = 0; tt < 256; tt++) {
				if (tt >= start && tt < end) {
					assert(node->symbol_set.test(tt));
				} else {
					assert(!node->symbol_set.test(tt));
				}
			}

			assert( ste.start  == start );
			assert( ste.end == end );


	 	} else {
	 		auto complement_matchset = ~node->symbol_set;
	 		if (nfa_utils::is_range_complete_matchset(complement_matchset, start, end)) {
	 			//cout << "node  " << node->sid << "complement complete start = " << start << " " << end << endl; 
	 			//total++;
	 			node->complete = true;
	 		
	 			ste.attribute |= (1 << 3); // complete
	 			ste.attribute |= (1 << 4); // complement

	 			//ste.start_end = 0;
				//ste.start_end |= end;
				//ste.start_end = (ste.start_end << 8) | start;

				ste.start = start;
				ste.end   = end;

				for (int tt = 0; tt < 256; tt++) {
					if (tt >= start && tt < end) {
						assert(!node->symbol_set.test(tt));
					} else {
						assert(node->symbol_set.test(tt));
					}
				}

				assert( (uint8_t)( ste.start ) == start );
				assert((uint8_t)(   ste.end ) == end );
	 		}
	 	}


		// least bit: report ;   second least bit : start-all-input;

		ste.degree = degree;

		res->set(i, ste);

		assert(ste.attribute <  (1 << 5 ) );


		// verify
		if (node->is_report()) {
			assert(ste.attribute & 1 == 1);
		} else {

			
			if (ste.attribute & 1 != 0) {
				//cout <<  "ste.attribute = " << ste.attribute << endl;
				assert(ste.attribute & 1 == 0);
			}
			
		}


		//if (node->str_id == "__58__") {
		//cout << "str_id = " << node->str_id <<    " ste.attribute = " << ste.attribute << endl;
		//}



	}


	return res;

}




Array2<STE_matchset_new_imp> *nfa_utils::create_STE_matchset_new (NFA *nfa) {

	Array2<STE_matchset_new_imp> *res = new Array2 <STE_matchset_new_imp> (nfa->size());
	
	for (int i = 0; i < nfa->size(); i++) {
		auto node = nfa->get_node_by_int_id(i);

		STE_matchset_new_imp ste; 

		memset(&ste, 0, sizeof(ste));

		for (int s = 0; s < 256; s++) {
			if (node->match2((uint8_t) s )) {
				ste.ms[s / 32] |=  (1 << (s % 32));  
			}
		}

		res->set(i, ste);
	}


	return res;

}








NFA *nfa_utils::focus_on_one_cc(NFA* bignfa, string str_id) {
	cout << "debug -- nfa_utils::focus_on_one_cc" << endl;
	bignfa-> mark_cc_id();
	auto ccs = split_nfa_by_ccs(*bignfa);

	for (int i = 0; i < ccs.size(); i++) {
		if (ccs[i]->has_node(str_id)) {
			return ccs[i];
		}
	}

	cout << "not found" << endl;
	return NULL;
}






vector<NFA *> nfa_utils::order_nfa_intid_by_hotcold(const vector<NFA *> &grouped_nfas) {
	vector<NFA *> res;

	for (int i = 0 ; i < grouped_nfas.size(); i++) {
		auto res_nfa = nfa_utils::remap_intid_of_nfa(grouped_nfas[i], [] (const Node *a, const Node *b) {	
			return (a->hot_degree - b->hot_degree) > 0;
		});


		res.push_back(res_nfa);
	}

	return res;
}



NFA *nfa_utils::cut_by_normalized_depth(NFA *nfa, double normalized_depth_limit) {

	nfa->mark_cc_id();

    auto ccs = nfa_utils::split_nfa_by_ccs(*nfa);
    
    for (auto cc : ccs) {
        cc->calc_scc();
        cc->topo_sort();
    }

    vector<NFA *> cut_nfas;

    for (auto cc : ccs) {
    	double depth = cc->get_num_topoorder();
    	NFA *cut_nfa = new NFA();
    	for (int i = 0; i < cc->size(); i++) {
    		auto node = cc->get_node_by_int_id(i);
    		if ((0.0 + node->topo_order) / depth < normalized_depth_limit) {
    			Node *new_node = new Node();
    			*new_node = *node;
    			cut_nfa->addNode(new_node);
    		}

    		auto adj = cc->get_adj(node->str_id);
    		for (auto to : adj) {
    			auto to_node = cc->get_node_by_str_id(to);
    			if ((to_node->topo_order + 0.0) / depth < normalized_depth_limit) {
    				cut_nfa->addEdge(node->str_id, to_node->str_id);
    			}
    		}

    	}

    	cut_nfas.push_back(cut_nfa);
    }

    NFA *res = merge_nfas(cut_nfas);
    
    for (auto it : cut_nfas) {
    	delete it;
    }

    for (auto cc : ccs) {
    	delete cc;
    }

    return res;
}



void nfa_utils::print_indegrees(const vector<NFA *> nfas) {
	map<int, int> indegress; 

	for (auto nfa : nfas) {
		for (int i = 0; i < nfa->size(); i++) {
			auto node = nfa->get_node_by_int_id(i);
			int ind = nfa->get_indegree_of_node(node->str_id);
			indegress[ind] += 1;
		}
	}

	int gt_than_4_indegree_states = 0;
	for (auto it : indegress) {
		if (it.first > 4) {
			gt_than_4_indegree_states += it.second;
		}
	}

	for (int i = 0; i <= 4; i++) {
		cout << "indegree_" << i << "_num_of_nodes = " << indegress[i] << endl;
	}

	cout << "indegree_" << "others = " << gt_than_4_indegree_states << endl;

}


/**
k_bit is counted from 0
*/
void bitvec::set_bit(int *arr, int len_arr, int k_bit) {
	int k_int = k_bit / (sizeof(int) * 8);
	assert(k_int >=0 && k_int < len_arr);


	int local_bit = k_bit % (sizeof(int) * 8);
	arr[k_int] = arr[k_int] | (1 << local_bit);



}




void nfa_utils::print_cc(NFA *nfa, int cc_id) {
	nfa->mark_cc_id();
	auto ccs = nfa_utils::split_nfa_by_ccs(*nfa);

	for (int i = 0; i < ccs.size(); i++) {
		auto cc = ccs[i];
		cc->calc_scc();
		cc->topo_sort();
	}

	assert(cc_id >= 0 && cc_id < ccs.size());

	ccs[cc_id] -> print();
	ccs[cc_id] -> to_dot_file("cc_" + std::to_string(cc_id) + ".dot");


	//add_fake_start_node_for_ccs(ccs);
	//nfa_utils::limit_out_degree_on_ccs(ccs, 4);  

	//ccs[cc_id] -> to_dot_file("cc_" + std::to_string(cc_id) + "_after" + ".dot");

	for (auto cc : ccs) {
		delete cc;
	}
}


Array2<uint8_t> *nfa_utils::get_array2_for_input_stream(const SymbolStream& symbolstream) {
	Array2<uint8_t> *res = new Array2<uint8_t> (symbolstream.get_length());

	for (int i = 0; i < symbolstream.get_length(); i++) {
		res->set(i, symbolstream.get_position(i));
	}

	return res;
}


	
Array2<int> *nfa_utils::get_nfa_size_array2(const vector<NFA*> &nfas) {
	if (nfas.size() == 0) {
		cout << "nfa_utils::get_nfa_size_array2 " << " empty nfas" << endl;
		return NULL;
	}

	Array2<int> *res = new Array2<int> (nfas.size());
	for (int i = 0; i < res->size(); i++) {
		res->set(i, nfas[i]->size());
	}

	return res;
}



Array2<char> *nfa_utils::get_attribute_array(NFA *cc) {
	Array2<char> *res = new Array2<char> (cc->size());
	for (int i = 0; i < cc->size(); i++) {
		
		auto node = cc->get_node_by_int_id(i);

		char cur = node->is_report() ? 1: 0;
		
		if (node->is_start_always_enabled()) {
			cur |= (1 << 1);
		}

		if (node->is_start()) {
			cur |= (1 << 2);
		}

		res->set(i, cur);
	}
	return res;
}



int nfa_utils::get_start_node_id(NFA *cc) {
	for (int i = 0; i < cc->size(); i++) {
		auto node = cc->get_node_by_int_id(i);
		if (node->is_start()) {
			return i;
		}
	}

	return -1;
}


Array2<int> * nfa_utils::get_allways_enabled_start_node_ids(NFA *cc) {
	vector<int> res; 

	for (int i = 0; i < cc->size(); i++) {
		auto node = cc->get_node_by_int_id(i);
		if (node->is_start_always_enabled()) {
			res.push_back(i);
		}
	}

	assert(res.size() > 0);
	
	Array2<int> *arr2 = new Array2<int> (res.size());
	
	for (int i = 0; i < res.size(); i++) {
		arr2->set(i, res[i]);
	}

	return arr2;
}




HotColdHelper::HotColdHelper() : threshold(0) {

}

void HotColdHelper::set_threshold(int t) {
	this->threshold = t;
}

bool HotColdHelper::is_hot(string sid) const {
	auto it = this->freq_map.find(sid);
	if (it == this->freq_map.end()) {
		cout << "freq map does not have the state id " << sid << endl;
		return true;
	}

	if (it->second > this->threshold) {
		return true;
	} else {
		return false;
	}

}

bool HotColdHelper::is_cold(string sid) const {
	return !this->is_hot(sid);
}





