#include "nfa_utils.h"
#include "NFA.h"
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
#include "compatible_group_helper.h"
#include "common_func.h"

#include <sys/types.h>
#include <functional>
#include <sys/stat.h>

using std::ifstream;
using std::string;
using std::endl;
using std::cout;
using std::pair;
using std::vector;
using std::make_pair;




bitset<256> nfa_utils::convert_from_range(pair<int, int> p) {
	return convert_from_range(p.first, p.second);
}

bitset<256> nfa_utils::convert_from_range(int start, int end) {
	bitset<256> res;
	res.reset();
	
	assert(start < end);
	assert(start >= 0);
	assert(end <= 256);

	for (int i = start; i < end; i++) {
		res.set(i, 1);
	}

	return res;
}


vector<NFA *> nfa_utils::split_nfa_by_ccs(const NFA& nfa) {
	int num_cc = nfa.get_num_cc();

	vector<NFA *> ccs; 

	for (int i = 0; i < num_cc; i++) {
		NFA *cc = new NFA();
		ccs.push_back(cc);
	}


	for (int i = 0; i < nfa.size(); i++) {
		Node *n = nfa.get_node_by_int_id(i);
		
		Node *node_for_cc = new Node();
		*node_for_cc = *n;

		node_for_cc->sid = n->cc_local_id;
		node_for_cc->cc_id = 0;
		node_for_cc->cc_local_id = 0;

		int current_ccid = n->cc_id;
		NFA *current_cc = ccs[current_ccid];

		current_cc->addNode(node_for_cc);
		
		auto adjs = nfa.get_adj(n->str_id);
		for (auto to : adjs) {
			Node *to_node = nfa.get_node_by_str_id(to);
			current_cc->addEdge(node_for_cc->str_id, to);
		}

	}

	return ccs;

}


vector<pair<int, int> > nfa_utils::get_ranges_from_matchset(const bitset<256> &symbol_set) {
	vector<pair<int, int> > res;

	if (symbol_set.count() == 0) {
		return res;
	}
	
	const static int NOT_IN_RANGE = 1, IN_RANGE = 2;

	int current_state = NOT_IN_RANGE;
	
	int start = -1, end = -1;

	for (int i = 0; i < 256; i++) {
		uint8_t ss = (uint8_t) i;
		if (current_state == NOT_IN_RANGE) {
			if (symbol_set.test(ss)) {
				current_state = IN_RANGE;
				start = i;
			}
		} else if (current_state == IN_RANGE) {
			if (!symbol_set.test(ss)) {
				current_state = NOT_IN_RANGE;
				end = i;
				res.push_back(make_pair(start, end));
			}
		}
	}

	if (current_state == IN_RANGE) {
		end = 256;
		res.push_back(make_pair(start, end));
	}

	return res;
}


bool nfa_utils::is_range_complete_matchset(bitset<256> symbol_set, int &start, int &end) {
	//if (symbol_set.count() == 0) {
		// it must be a * node, and the function is called in a complement situation. 
	//	return false;
	//}


	auto t = get_ranges_from_matchset(symbol_set);

	if (t.size() == 0) {
		//empty symbol set
		// the compliment should be complete.  
		return false; 
	}
	
	//assert(t.size() > 0);

	if (t.size() == 1) {
		start = t[0].first;
		end   = t[0].second;
		return true;
	} else {
		start = -1;
		end = -1;
		return false;
	}
	
}


void nfa_utils::print_MC_complete_info(NFA *nfa, string suffix) {
	int complete = 0;
	int complement = 0;
	
	for (int i = 0; i < nfa->size(); i++) {
	 	auto node = nfa->get_node_by_int_id(i);
	 	int start, end;
	 	if (nfa_utils::is_range_complete_matchset(node->symbol_set, start, end)) {
	 		//cout << "node  " << node->sid << "complete start = " << start << " " << end << endl;
	 		node->complete = true;
	 		node->match_set_range = end - start;
	 		complete ++;
	 	} else {
	 		auto complement_matchset = ~node->symbol_set;
	 		if (nfa_utils::is_range_complete_matchset(complement_matchset, start, end)) {
	 			//cout << "node  " << node->sid << "complement complete start = " << start << " " << end << endl; 
	 			node->complete = true;
	 			node->complement = true;
	 			node->match_set_range = end - start;
	 			complement++;
	 		} else {
	 			node->complete = false;
	 		}
	 	}	
	}

	if (suffix != "" && !suffix.rfind("_", 0) == 0) {
		suffix = "_" + suffix;
	}

	cout << "num_of_complete_state" << suffix << " = " << complete << endl;
	cout << "num_of_complement_state" << suffix << " = " << complement << endl;
	cout << "num_of_complete_or_complement_state" << suffix << " = " << complete + complement << endl;
}





void nfa_utils::print_info_of_nfa(NFA *nfa, string suffix, const std::function<bool(const Node&)>& f) {
	
	if (suffix != "" && suffix.find("_") != 0) {
		suffix = "_" + suffix;
	}

	cout << "num_of_state"  <<  suffix << " = " << nfa->size() << endl;

	nfa->mark_cc_id();

	int num_of_starting_state = 0;
	int num_of_start_always_enabled = 0;
	int num_of_reporting_state = 0;

	for (int i = 0; i < nfa->size(); i++) {
		auto node = nfa->get_node_by_int_id(i);

		//if (!f || f && f(*node)) {
			if (node->is_report()) {
				num_of_reporting_state ++;
			}

			if (node->is_start()) {
				num_of_starting_state ++;
			}

			if (node->is_start_always_enabled()) {
				num_of_start_always_enabled ++; 
			}
		//}
	}

	cout << "num_of_reporting_state" <<  suffix << " = " << num_of_reporting_state << endl;
	cout << "num_of_starting_state"  <<  suffix << " = " << num_of_starting_state << endl;
	cout << "num_of_start_always_enabled" <<  suffix << " = " << num_of_start_always_enabled << endl;

	int num_of_edges = 0;
	for (int i = 0; i < nfa->size(); i++) {
		auto node = nfa->get_node_by_int_id(i);
		num_of_edges += nfa->get_outdegree_of_node(node->str_id);
	}

	cout << "num_of_edges" << suffix << " = " << num_of_edges << endl;
	cout << "edge_to_complete_graph_rate = " << std::fixed << (num_of_edges + 0.0) / (1.0 * nfa->size() * nfa->size()) << endl;

	print_MC_complete_info(nfa, suffix);

	auto ccs = split_nfa_by_ccs(*nfa);
	assert(ccs.size() != 0);

	cout << "num_of_cc" <<  suffix << " = " << ccs.size() << endl;

	double avg_cc_size = 0;
	int max_cc_size = 0;
	int min_cc_size = 0x7fffffff;

	for (auto cc : ccs) {
		avg_cc_size += cc->size();

		if (cc->size() > max_cc_size) {
			max_cc_size = cc->size();
		}

		if (cc->size() < min_cc_size) {
			min_cc_size = cc->size();
		}
	}

	avg_cc_size /= ccs.size();

	cout << "max_cc_size"  <<  suffix << " = " << max_cc_size << endl;
	cout << "min_cc_size" <<  suffix << " = " << min_cc_size << endl;
	cout << "avg_cc_size" <<  suffix << " = " << std::fixed << avg_cc_size << endl;

	int num_cc_that_has_morethan_256_nodes = 0;
	int num_cc_that_has_morethan_512_nodes = 0;
	
	for (auto cc : ccs) {
		if (cc->size() > 256) {
			num_cc_that_has_morethan_256_nodes ++; 
		}

		if (cc->size() > 512) {
			num_cc_that_has_morethan_512_nodes ++;
		}
	}

	cout << "num_cc_that_has_morethan_256_nodes" << suffix << " = " << num_cc_that_has_morethan_256_nodes << endl;
	cout << "num_cc_that_has_morethan_512_nodes" << suffix << " = " << num_cc_that_has_morethan_512_nodes << endl;
	
	int max_in_degree = 0;
	int max_out_degree = 0;
	double avg_in_degree = 0;
	double avg_out_degree = 0;

	map<int, int> node_in_degree_dis;
	map<int, int> node_out_degree_dis;

	for (int i = 0; i < nfa->size(); i++) {
		auto node = nfa->get_node_by_int_id(i);

		if (!f || f && f(*node)) {
			max_in_degree = std::max(max_in_degree, nfa->get_indegree_of_node(node->str_id));
			max_out_degree = std::max(max_out_degree, nfa->get_outdegree_of_node(node->str_id));

			avg_in_degree += nfa->get_indegree_of_node(node->str_id);
			avg_out_degree += nfa->get_outdegree_of_node(node->str_id);

			node_in_degree_dis[nfa->get_indegree_of_node(node->str_id)] += 1;
			node_out_degree_dis[nfa->get_outdegree_of_node(node->str_id)] += 1;
		}
		
		

	}

	avg_in_degree /= nfa->size();
	avg_out_degree /= nfa->size();


	cout << "max_in_degree"  <<  suffix << " = " << max_in_degree << endl;
	cout << "max_out_degree" <<  suffix << " = " << max_out_degree << endl;
	cout << "avg_in_degree" << suffix << " = " << std::fixed << avg_in_degree << endl;
	cout << "avg_out_degree" << suffix << " = " << std::fixed << avg_out_degree << endl;

	for (auto it : node_in_degree_dis) {
		auto deg = it.first;
		auto num_of_node = it.second;
		cout << "num_of_node_has_in_degree_" << deg  <<  suffix << " = " << num_of_node << endl;
	}

	for (auto it : node_out_degree_dis) {
		auto deg = it.first;
		auto num_of_node = it.second;
		cout << "num_of_node_has_out_degree_" << deg  <<  suffix << " = "  << num_of_node << endl;
	}


}


vector<NFA *> nfa_utils::group_nfas(int group_size, vector<NFA *> ccs) {
	auto ccs1 = ccs;

	std::sort(ccs1.begin(), ccs1.end(), [](const NFA * a, const NFA * b) -> bool
	{
    	return a->size() < b->size(); 
	});

	//for (auto it : ccs1) {
	//	cout << it->size() << endl;
	//}


	vector<NFA *> res;

	int current_states = 0;
	vector<NFA *> nfas_to_merge;

	for (auto it : ccs1) {

		if (current_states + it->size() > group_size) {
			if (nfas_to_merge.size() > 0) {
				res.push_back(merge_nfas(nfas_to_merge));
			}

			nfas_to_merge.clear();
			nfas_to_merge.push_back(it);
			current_states = it->size();
		} else {
			nfas_to_merge.push_back(it);
			current_states += it->size();
		}
	}

	if (nfas_to_merge.size() > 0) {
		res.push_back(merge_nfas(nfas_to_merge));
	}

	return res;
}

vector<NFA *> nfa_utils::group_nfas_by_num(int &group_num, vector<NFA *> ccs) {
	auto ccs1 = ccs;

	// std::sort(ccs1.begin(), ccs1.end(), [](const NFA * a, const NFA * b) -> bool
	// {
  //   	return a->size() < b->size(); 
	// });

	//for (auto it : ccs1) {
	//	cout << it->size() << endl;
	//}


	vector<NFA *> res;

	int cc_num = ccs1.size();
	if (group_num > cc_num) {
			printf("warning: cc_num = %d is less than group_num = %d, new "
							"group_num will be %d\n",
							cc_num, group_num, cc_num);
			group_num = cc_num;
	}
	float cc_num_per_group = 1.0 * cc_num / group_num;
	vector<NFA *> nfas_to_merge;
	float bound = cc_num_per_group;

	for (int i = 0; i < cc_num; i++) {
			if (!nfas_to_merge.empty() && (i + 1) > bound &&
					res.size() + 1 < group_num) {
				// printf("%f %d %d\n", bound, cc_num, res.size());
				bound += cc_num_per_group;
				res.push_back(merge_nfas(nfas_to_merge));
				nfas_to_merge.clear();
			}
			nfas_to_merge.push_back(ccs1[i]);
	}

	if (nfas_to_merge.size() > 0) {
		res.push_back(merge_nfas(nfas_to_merge));
	}

	return res;
}



vector<vector<int> > nfa_utils::items_to_groups(const int group_size, const vector<int> item_sizes) {
	vector<vector<int> > res;
	
	auto ccs1 = item_sizes;

	int current_states = 0;
	vector<int> nfas_to_merge;

	for (int i = 0; i < ccs1.size(); i++) {

		int sz = ccs1[i];

		if (current_states + sz > group_size) {
			if (nfas_to_merge.size() > 0) {
				res.push_back(nfas_to_merge);
			}

			nfas_to_merge.clear();
			nfas_to_merge.push_back(i);

			assert(sz <= group_size);
			current_states = sz;

		} else {
			nfas_to_merge.push_back(i);
			current_states += sz;
		}
	}

	if (nfas_to_merge.size() > 0) {
		res.push_back(nfas_to_merge);
	}

	return res;
}



NFA *nfa_utils::merge_nfas(vector<NFA *> nfas) {
	NFA * res = new NFA();
	int t = 0;
	for (auto it : nfas) {
		for (int i = 0; i < it->size(); i++) {
			Node *n = it->get_node_by_int_id(i);
			Node *new_node = new Node();
			*new_node = *n;
			new_node->sid = t++;

			res->addNode(new_node);
			auto adjs = it->get_adj(n->str_id);
			for (auto to : adjs) {
				Node *to_node = it->get_node_by_str_id(to);
				res->addEdge(new_node->str_id, to);
			}
		}
	}

	return res;
}



vector<NFA *> nfa_utils::merge_nfas_by_group(const vector<vector<int> > &grps, const vector<NFA *> &ccs) {
	vector<NFA*> res;

	for (int i = 0 ; i < grps.size(); i++) {
		vector<NFA *> nfa_to_merge;
		
		for (auto ccid : grps[i]) {
			nfa_to_merge.push_back(ccs[ccid]);
		}

		res.push_back(merge_nfas(nfa_to_merge));
	}

	return res;
}


bool nfa_utils::limit_out_degree_on_nfa(NFA* cc, int limit) {
	int num_iteration = 0;
	
	int original_NFA_size = cc->size();
	int original_num_of_edges = cc->get_num_transitions();

	while (true) {
		num_iteration ++; 

		vector<string> big_node;
		for (int i = cc->size() - 1; i >= 0; i--) {
			Node *n = cc->get_node_by_int_id(i);
			auto adj = cc->get_adj(n->str_id);
			assert(cc->has_node(n->str_id));
			assert(cc->has_node(i));

			if (adj.size() > limit) {
				
				//cout << "n->str_id = " << n->str_id << " outdegree = " << adj.size() << "========== " ;
				//for (auto to : adj) {
				//	cout << to << "   " ;
				//}
				//cout << endl;

				big_node.push_back(n->str_id);
				//break;
			}
		}

		if (big_node.size() == 0) {
			break;
		}

		int min_in_degree = 0x7fffffff; 
		string min_in_degree_node_str_id = "";
		for (auto big_node_id : big_node) {
			if (cc->get_indegree_of_node(big_node_id) < min_in_degree) {
				min_in_degree = cc->get_indegree_of_node(big_node_id);
				min_in_degree_node_str_id = big_node_id;
			}
		}

		big_node.clear();
		big_node.push_back(min_in_degree_node_str_id);

		assert(big_node.size() == 1);

		for (auto it : big_node) {
			vector<string> from_nodes;
			vector<string> to_nodes;

			from_nodes = cc->get_from(it);
			to_nodes = cc->get_adj(it);

			assert(to_nodes.size() > limit);
			Node deletedNode = cc->remove_node_unsafe(it);

			//cout << "node to be split... it connects to ";
			//for (auto tt : to_nodes) {
			//	cout << tt << " ";
			//}
			//cout << endl;

			int p = 0; 

			int L = limit;
			int t = 0;

			auto it_self1 = std::find(to_nodes.begin(), to_nodes.end(), deletedNode.str_id);
			auto it_self2 = std::find(from_nodes.begin(), from_nodes.end(), deletedNode.str_id);

			bool self_loop = it_self1 != to_nodes.end();
			assert(self_loop == ( it_self2 != from_nodes.end() ) );

			if (it_self1 != to_nodes.end() && it_self2 != from_nodes.end()) {
				to_nodes.erase(it_self1);
				from_nodes.erase(it_self2);
			}

			//cout << "to_nodes.size() " << to_nodes.size() << endl;
			while (t < to_nodes.size()) {
				Node *next_node = new Node();
				*next_node = deletedNode;
				next_node->str_id += "_dup_" + std::to_string(p);

				if (p == 0) {
					cc->addNode(next_node, deletedNode.sid);
				} else {
					cc->addNode(next_node);
				}


				int L = limit;
				if (self_loop) {
					//cout << "self_loop " << endl;
					L = limit - 1;
					//cout << "addEdge self loop = " << next_node->str_id << " " << next_node->str_id << endl;
					cc->addEdge(next_node->str_id, next_node->str_id);
				}

				for (int kk = 0; kk < L && t + kk < to_nodes.size(); kk++) {
					assert(deletedNode.str_id != to_nodes[t+kk]);
					//cout << "addedge " <<  next_node->str_id << "   " << to_nodes[t + kk] << endl;
					cc->addEdge(next_node->str_id, to_nodes[t + kk]);	
					
				}

				for (auto from_str_id : from_nodes) {
					//cout << "addedge2 " <<  from_str_id << " " << next_node->str_id << endl;
					cc->addEdge(from_str_id, next_node->str_id);
				}

				//cout << "t = " << t << " p = " << p << " size = " << to_nodes.size() << " L = " << L << endl;
				p++;

				t += L;
			}

			assert(p >= 2);

			if (num_iteration >= original_NFA_size * 2) {   // not accurate. ---
				return false;
			}
		}			
	}

	return true;
}


pair<int, int> nfa_utils::limit_out_degree_on_ccs(vector<NFA *> &ccs, int limit , bool quit, bool remove_degree) {
	cout << "nfa_utils::limit_out_degree_on_ccs " << " ccs size = " << ccs.size() << " limit = " << limit << endl;

	// known : does not work when the graph is complete graph. 
	// guess : this function may not work when the graph is very dense. 
	
	int num_succeed = 0;
	int num_fail = 0;

	vector<NFA *> res;
	assert(ccs.size() > 0);

	for (int cc_id = 0; cc_id < ccs.size(); cc_id ++) {
		auto cc = ccs[cc_id];

		if (limit_out_degree_on_nfa(cc, limit)) {
			res.push_back(cc);
			num_succeed ++;
		} else {
				cout << "cannot_limit_degree_to_" << limit
							<< " on cc_id = " << cc_id << " quit=" << quit
							<< " remove_degree=" << remove_degree << endl;
				if (remove_degree) {
					if (quit) {
							printf("ERROR: Quit when degree > 4\n");
							exit(-1);
					}
					delete cc;
			}
			num_fail++; 
		}
	}

	ccs.clear();
	for (int i = 0; i < res.size(); i++) {
		ccs.push_back(res[i]);
	}
	cout << "    big_cc_number = " << num_fail << endl;
	return std::make_pair(num_succeed, num_fail);
}






void nfa_utils::dump_to_gr_file(const NFA& nfa, string fn) {
    std::ofstream out(fn);

    out << "p sp " << nfa.size() << " " << nfa.get_num_transitions() << endl;
	for (int i = 0; i < nfa.size(); i++) {
		Node *n = nfa.get_node_by_int_id(i);
		auto adjs = nfa.get_adj(n->str_id);

		for (auto it : adjs) {
			int to_id = nfa.get_int_id_by_str_id(it);

			int print_from_id = i+1;
			int print_to_id = to_id + 1;

			out << "a " << print_from_id << " " << print_to_id << " 1" << endl;

		}
	}


    out.close();
}




map<string, double> nfa_utils::read_freq_map(string filename) {
	
	map<string, double> res; 

	string line;
	double tt = 0;
 	ifstream myfile (filename);

  	if (myfile.is_open()) {
  		while (myfile >> line) {
  			if (line.empty()) {
  				break;
  			}
  			myfile >> tt;
  			//cout << "here" << line << " " <<  tt << endl;
  			
  			res[line] = tt;
  		}

    	myfile.close();
    	
  	}

  	else {
  		cout << "Unable to open file" << endl;
  	}

  	return res;
}





map<string, int> nfa_utils::get_state_hotcold_info(const SymbolStream& ss, const vector<NFA *> &ccs) {
	map<string, int> active_freq_of_nodes;

	for (auto cc : ccs) {
		for (int i = 0; i < cc->size(); i++) {
			auto n = cc->get_node_by_int_id(i);
			active_freq_of_nodes[n->str_id] = 0;
		}
	}


	set<string> always_enabled;	
	set<string> activated;
	set<string> next_activated;

	for (int cc_id = 0; cc_id < ccs.size(); cc_id ++) {
		auto nfa = ccs[cc_id];
		always_enabled.clear();

		for (int i = 0; i < nfa->size(); i++) {
			Node *n = nfa->get_node_by_int_id(i);
			if (n->is_start_always_enabled()) {
				always_enabled.insert(n->str_id);
			}
		}

		activated.clear();
		// initializatiion
		for (int i = 0; i < nfa->size(); i++) {
			Node *n = nfa->get_node_by_int_id(i);
			if (n->is_start()) {
				activated.insert(n->str_id);
			}
		}

		for (int p = 0; p < ss.size(); p++) {
			auto symbol = ss.get_position(p);
			
			next_activated.clear();

			for (auto active_id : activated) {
				Node *active_state = nfa->get_node_by_str_id(active_id);

				active_freq_of_nodes[active_state->str_id] += 1;
				
				for (auto adj : nfa->get_adj(active_state->str_id)) {
					Node *to_node = nfa->get_node_by_str_id(adj);
					if (to_node->match2(symbol)) {
						next_activated.insert(to_node->str_id);
					}
				}

			}

			activated.clear();
			activated.insert(next_activated.begin(), next_activated.end());
			activated.insert(always_enabled.begin(), always_enabled.end());
		}

		for (auto active_id : activated) { // for the last cycle. 
			active_freq_of_nodes[active_id] += 1;
		}
	}

	for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
		auto nfa = ccs[cc_id];

		for (int i = 0; i < nfa->size(); i++) {
			Node *n = nfa->get_node_by_int_id(i);
			if (n->is_start_always_enabled()) {
				assert(active_freq_of_nodes[n->str_id] > 0);
			}
		}
	}

	return active_freq_of_nodes;
}





void nfa_utils::calc_compatible_groups(const vector<NFA*> &ccs, const set<uint8_t> &alphabet, AbstractCompatibleGroupHelper& ph, map<int, int> &ccid_cgsize, map<string, int> &nodestrid_cgid, string suffix) {
	ccid_cgsize.clear();
	
	nodestrid_cgid.clear();

    int total_cg_size = 0;
    int total_size = 0;

    for (int i = 0; i < ccs.size(); i++) {
        auto cc = ccs[i];
        total_size += cc->size();
       	// cout << "ccsize = " << cc->size() << endl;
        
        ph.set_nfa(cc);
        ph.calc_incompatible_states(alphabet);
        ph.calc_compatible_groups();

        int num_compatible_groups = ph.num_compatible_grp();

        //cout << "ccsize = " << cc->size()  << "  num_compatible_groups = " << num_compatible_groups << endl;
        total_cg_size += num_compatible_groups;

        ccid_cgsize[i] = num_compatible_groups;
        

        for (int node_id = 0; node_id < cc->size(); node_id ++) {
        	auto node = cc->get_node_by_int_id(node_id);
        	auto node_str_id = node->str_id;
        	int cgid = ph.get_compatible_grp_by_intid(node_id);
        	nodestrid_cgid[node_str_id] = cgid;
        }  
    }

    if (suffix != "") {
    	suffix = "_" + suffix;
    }

    cout << "total_cg_size" << suffix << " = " << total_cg_size << endl;
    cout << "cg_ratio" << suffix << " = " <<  std::fixed << (total_cg_size + 0.0 ) / total_size << endl;
    cout << "avg_cg_size" << suffix << " = " << (total_size + 0.0 ) /  total_cg_size << endl;
}




int nfa_utils::get_num_of_states(vector<NFA*> nfas)  {
	int ss = 0;
	for (auto it : nfas) {
		ss += it->size();
	}
	return ss;
}




int nfa_utils::search_state_id_in_nfa_vector(const vector<NFA*> &ccs, string state_id) {
	for (int i = 0; i < ccs.size(); i++) {
		auto cc = ccs[i];
		if (cc->has_node(state_id)) {
			return i;
		}
	}

	return -1;
}


vector<int> nfa_utils::get_hot_boundaries_of_grps(const vector<NFA *> &grouped_nfas, double cold_thres) {
	vector<int> res;

	for (auto nfa : grouped_nfas) {
		int sss = 0;
		assert(nfa->size() > 0);

		assert(nfa->get_node_by_int_id(0)->hot_degree > 0);

		int boundary = -1;
		for (int i = 0 ; i < nfa->size(); i++) {
			auto node = nfa->get_node_by_int_id(i);

			if (node->hot_degree <= cold_thres) { // cold
				boundary = i;
				break;
			}

		}

		if (boundary == -1) {
			boundary = nfa->size();
		}
		
		res.push_back(boundary);
	}

	return res;

}



pair<int, int> nfa_utils::print_starting_node_info(NFA *nfa) {
	int num_start = 0;
	int num_all_input = 0;

	for (int i = 0; i < nfa->size(); i++) {
		auto node = nfa->get_node_by_int_id(i);
		if (node->is_start()) {
			num_start ++;
		}
		if (node->is_start_always_enabled()) {
			num_all_input ++;
		}
	}

	cout << "num_start = " << num_start << endl;
	cout << "num_all_input = " << num_all_input << endl;

	return std::make_pair(num_start, num_all_input);
}

void nfa_utils::print_nfa_info(NFA *nfa) {
	cout << "NFA Info:\n";
	cout << "    total_state_number = " << nfa->size() << endl;
	int num_start = 0;
	int num_all_input = 0;
	int num_report = 0;

	for (int i = 0; i < nfa->size(); i++) {
			auto node = nfa->get_node_by_int_id(i);
			if (node->is_start()) {
					num_start++;
			}
			if (node->is_start_always_enabled()) {
					num_all_input++;
			}
			if (node->is_report()) {
					num_report++;
			}
	}
	cout << "    start_state_number = " << num_start - num_all_input<< endl;
	cout << "    always_active_state_number = " << num_all_input << endl;
	cout << "    report_state_number = " << num_report << endl;

	nfa->mark_cc_id();
  vector<NFA *> ccs = nfa_utils::split_nfa_by_ccs(*nfa);
	cout << "    total_cc_number = " << ccs.size() << endl;
  int degree_cc_num = 0;
	int cc_256_num = 0;
	int max_cc_size = 0;
	int sum_cc_size = 0;
  for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
		NFA *cc = ccs[cc_id];
		sum_cc_size += cc->size();
		max_cc_size = std::max(max_cc_size, cc->size());
		if (cc->size() > 256)
			cc_256_num++;
	}
	cout << "    cc_256_number = " << cc_256_num << endl;
	cout << "    max_cc_size = " << max_cc_size << endl;
	cout << "    average_cc_size = " << 1.0*sum_cc_size/ccs.size() << endl;
}

void nfa_utils::print_cg_info_of_ccs(const vector<NFA *> &ccs, AbstractCompatibleGroupHelper &agh, 
										set<uint8_t> alphabet, string suffix, string filename) {
    map<int, int> ccid_cgsize; // a map between ccid to cgsize
    map<string, int> nodestrid_cgid; 


    nfa_utils::calc_compatible_groups(ccs, alphabet, agh, ccid_cgsize, nodestrid_cgid, suffix);

    /*int total_num_of_cg = 0;

    for (int cc_id = 0; cc_id < ccs.size(); cc_id ++) {
    	auto cc = ccs[cc_id];
    	CC_CG_helper ch(cc, nodestrid_cgid, 0.00);
    	total_num_of_cg += ch.get_num_of_cgs();

    }

    cout << "total_num_of_cg_" << suffix << " = " << total_num_of_cg << endl;
	*/

    if (filename != "") {
    	std::ofstream out(filename);

    	for (int i = 0; i < ccs.size(); i++) {	
    		
    		auto cc = ccs[i];

			out << "cc_id = " << i << " cc_size = " << cc->size() << endl;
    		
    		for (int j = 0; j < cc->size(); j++) {
    			auto node = cc->get_node_by_int_id(j);
    			out << "node_id = " << node->str_id << " cg_id = " << nodestrid_cgid[node->str_id] << endl;
    		}

    	}

	    out.close();
    }


}


void nfa_utils::print_nodes_to_file(string filename, const vector<NFA *> &ccs) {
	std::ofstream out(filename);

	out << "str_id" << "\t" << "matchset" << "\t" << "complete" << "\t" << "complement" <<  "\t" 
	<< "bfs_layer" << "\t" << "start" << "\t" << "alway_enabled" << "\t" << "report" << "\t"
	<< "in_degree" << "\t" << "out_degree" << 
	endl;
	for (int i = 0; i < ccs.size(); i++) {	
		auto cc = ccs[i];

		for (int j = 0; j < cc->size(); j++) {
			auto node = cc->get_node_by_int_id(j);
			out << node->str_id 
			<< "\t" << node->symbol_set 
			<< "\t" << node->complete 
			<< "\t" << node->complement 
			<< "\t" << node->bfs_layer 
			<< "\t" << node->is_start() 
			<< "\t" << node->is_start_always_enabled()
			<< "\t" << node->is_report() 
			<< "\t" << cc->get_indegree_of_node(node->str_id) 
			<< "\t" << cc->get_outdegree_of_node(node->str_id)
			<< 
			endl;
		}
	}

    out.close();


}



void nfa_utils::print_edge_and_matchset_to_file(string filename, const vector<NFA *> &ccs) {
	std::ofstream out(filename);

	out << "edge_from" << "\t" << "edge_to" << "\t" << "complete" << "\t" << "complement" <<  "\t"  << "matchset" << endl;

	for (int i = 0; i < ccs.size(); i++) {	
		auto cc = ccs[i];

		for (int node_id = 0; node_id < cc->size(); node_id++) {
			auto node = cc->get_node_by_int_id(node_id);
			auto to_nodes = cc->get_adj(node->str_id);
			
			for (auto to_node_str_id : to_nodes) {
				auto to_node = cc->get_node_by_str_id(to_node_str_id);
				
				out << node->str_id 
				<< "\t" << to_node->str_id 
				<< "\t" << to_node->complete
				<< "\t" << to_node->complement
				<< "\t" << to_node->symbol_set
				<< endl;
			
			}
		}
	}

    out.close();


}



void nfa_utils::add_fake_starters(NFA *nfa) {

	int nfasize = nfa->size();
    for (int i = 0; i < nfasize; i++) {
        Node *n  = nfa->get_node_by_int_id(i);

        if (n->is_start()) {
        	Node *cloned_start_node = new Node();
        	*cloned_start_node = *n;

        	cloned_start_node->str_id += "clone";
        	cloned_start_node->report = false;

        	cloned_start_node->symbol_set.set();

        	nfa->addNode(cloned_start_node);
        	nfa->addEdge(cloned_start_node->str_id, n->str_id);

        	n->start = 0;
        }
    }


}



void nfa_utils::split_states_to_complete(NFA *nfa, int max_num_of_ranges) {
	// Warning: may increase number of states and edges a lot. 

	assert(nfa != NULL);
	assert(nfa->size() > 0);

	int N = nfa->size();

	for (int i = 0; i < N; i++) {
		auto node = nfa->get_node_by_int_id(i);

		auto matchset = node->symbol_set;
		auto complement_matchset = ~node->symbol_set;

		auto ranges = get_ranges_from_matchset(matchset);
		auto complement_ranges = get_ranges_from_matchset(complement_matchset);

		if (ranges.size() == 1 || complement_ranges.size() == 1) {
			continue;
		}

		assert(ranges.size() > 1 && complement_ranges.size() > 1);

		if (ranges.size() > max_num_of_ranges && complement_ranges.size() > max_num_of_ranges) {
			continue;
		}

		bool self_loop = nfa->has_self_loop(node->str_id);
		
		if (self_loop) {
			// remove self loop to get indegrees and outdegrees without the self loop
			nfa->remove_self_loop(node->str_id);
		}

		auto to_nodes = nfa->get_adj(node->str_id);
		auto from_nodes = nfa->get_from(node->str_id);

		if (self_loop) {
			// recover self loop
			nfa->addEdge(node->str_id, node->str_id);
		}

		if (ranges.size() <= complement_ranges.size()) { // we use the matchset
			node->symbol_set = convert_from_range(ranges[0].first, ranges[0].second);

			node->symbol_set_str = from_range_to_symbol_set_str(ranges[0].first, ranges[0].second, false);

			for (int j = 1; j < ranges.size(); j++) {
				int start = ranges[j].first;
				int end = ranges[j].second;

				auto matchset_j = convert_from_range(start, end);

				Node *cloned_node_j = new Node();
        		*cloned_node_j = *node; 

        		cloned_node_j->str_id += "_mc_" + std::to_string(j);
        		cloned_node_j->symbol_set = matchset_j;

        		cloned_node_j->symbol_set_str = from_range_to_symbol_set_str(start, end, false);

        		nfa->addNode(cloned_node_j);
        		
        		for (auto from_node : from_nodes ) {
        			nfa->addEdge(from_node, cloned_node_j->str_id);
        		}

        		for (auto to_node : to_nodes) {
        			nfa->addEdge(cloned_node_j->str_id, to_node);
        		}

        		if (self_loop) {
        			nfa->addEdge(cloned_node_j->str_id, cloned_node_j->str_id);
        		}
			}

		} else {
			// require ~ matchset .
			
			node->symbol_set = ~ convert_from_range(complement_ranges[0].first, complement_ranges[0].second);
			
			for (int bit = complement_ranges[1].first; bit < 256; bit++) {
				node->symbol_set.set(bit, 0);
			}

			node->symbol_set_str = "na";

			for (int j = 1; j < complement_ranges.size(); j++) {
				int start = complement_ranges[j].first;
				int end = complement_ranges[j].second;

				auto matchset_j = ~ convert_from_range(start, end);

				Node *cloned_node_j = new Node();
        		*cloned_node_j = *node; 

        		cloned_node_j->str_id += "_mc_" + std::to_string(j);

        		int prev_boundary = complement_ranges[j-1].second;
        		for (int bit = 0; bit < prev_boundary; bit++) {
        			matchset_j.set(bit, 0);
        		}

        		int next_boundary = 0;
        		if (j < complement_ranges.size() - 1) {
        			next_boundary = complement_ranges[j+1].first;
        			for (int bit = next_boundary; bit < 256; bit ++) {
        				matchset_j.set(bit, 0);
        			}
        		}


        		cloned_node_j->symbol_set = matchset_j;

        		cloned_node_j->symbol_set_str = "na";

        		nfa->addNode(cloned_node_j);
        		
        		for (auto from_node : from_nodes ) {
        			nfa->addEdge(from_node, cloned_node_j->str_id);
        		}

        		for (auto to_node : to_nodes) {
        			nfa->addEdge(cloned_node_j->str_id, to_node);
        		}

        		if (self_loop) {
        			nfa->addEdge(cloned_node_j->str_id, cloned_node_j->str_id);
        		}
			
			}
			
		} // complement

	} 

}


string nfa_utils::from_range_to_symbol_set_str(int start, int end, bool complement) {
	string res = "";
	
	/*if (!complement) {		
		for (int i = start; i < end; i++) {
			res += (char) i; 
		}		
	} else {
		for (int i = start; i < end; i++) {
			if ()
		}
	}


	if (complement) {
		res = "^" + res + "";
	} else {
		res = "" + res + "";
	}*/

	return res;

}


void nfa_utils::output_dot_files(string path, const vector<NFA*> &ccs) {
	tools::create_path_if_not_exists(path + "/dot");
    
	for (int i = 0; i < ccs.size(); i++) {
		auto filename = path + "/dot/" + "cc_" + std::to_string(i) + ".dot";
		ccs[i]->to_dot_file(path + "/dot/" + "cc_" + std::to_string(i) + ".dot");
	}
}






vector<vector<int> > nfa_utils::group_nfas_by_hotcold(int block_size, vector<NFA *> ccs, double cold_thres) {
	vector<vector<int> > res;

	int current_states = 0;
	vector<int> current_group;

	for (int i = 0; i < ccs.size(); i++) {
		auto cc = ccs[i];
		int hot_size = 0;
		
		for (int ss = 0; ss < cc->size(); ss++) {
			auto node = cc->get_node_by_int_id(ss);
			if (node->is_start()) {
				node->hot_degree = std::max(node->hot_degree, cold_thres + 0.0000001);
			}
		}
		

		for (int ss = 0; ss < cc->size(); ss++) {
			auto node = cc->get_node_by_int_id(ss);
			if (node->hot_degree > cold_thres) {
				hot_size += 1;
			}
		}
		
		if (hot_size == 0) {
			cc->to_dot_file("problematic_hot_size_cc_" + std::to_string(i) + ".dot");
			for (int ss = 0; ss < cc->size(); ss++) {
				auto node = cc->get_node_by_int_id(ss);
				cout << "node_str_id = " << node->str_id << " " << std::fixed << node->hot_degree << endl;
			}

			assert(hot_size > 0);
		}
		if (hot_size > block_size) {
			cout << "problem occurs hot_size = " << hot_size << " block_size = " << block_size << " ccid = " << i << endl; 
			assert(hot_size <= block_size);
		}


		if (current_states + hot_size > block_size) {
			res.push_back(current_group);

			assert(current_group.size() > 0);
			current_group.clear();
			current_group.push_back(i);
			current_states = hot_size;

		} else {
			current_states += hot_size;
			current_group.push_back(i);
		}
	}

	if (current_group.size() > 0) {
		res.push_back(current_group);
	}

	//cout << "num_group_by_condition = " << res.size() << endl;

	return res;
}






void nfa_utils::assign_hot_cold_by_hot_limit_by_bfs_layer(const vector<NFA *> &ccs, int hot_limit, int block_limit) {
	// Each node in the ccs should only have 4 out edges, right?
	cout << "nfa_utils::assign_hot_cold_by_hot_limit_by_bfs_layer ccs size :  " << ccs.size() << " hot_limit = " << hot_limit << endl;
	cout << "block_limit = " << block_limit << endl;

	vector<vector<Node *> > all_nodes; 

	for (int i = 0; i < ccs.size(); i++ ) {
		vector<Node *> tmp;
		for (int nodeid = 0; nodeid < ccs[i]->size(); nodeid++) {
			Node *nn = ccs[i]->get_node_by_int_id(nodeid);
			nn->hot_degree = 0.0;
			tmp.push_back(nn);
		}

		std::sort(tmp.begin(), tmp.end(), [](const Node *a, const Node *b) -> bool {
			return a->bfs_layer < b->bfs_layer; 
		});

		all_nodes.push_back(tmp);
	}

	assert(all_nodes.size() == ccs.size());

	int current_hot = 0;
	int current_head = 0; 

	while (current_hot < hot_limit ) {
		bool added = false;
		for (int i = 0 ; i < all_nodes.size(); i++) {
			if (current_head < all_nodes[i].size() && current_head < block_limit) {
				all_nodes[i][current_head]->hot_degree = 1.0;
				current_hot ++;
				added = true;
				if (current_hot >= hot_limit) {
					return;
				}
			}
		}

		if (!added) {
			return;
		}

		current_head++;
	}

	
}



pair<int, int> nfa_utils::get_num_hot_cold_from_map(const map<string, double>& freq_map, double cold_thres) {
	int cold = 0;
	int hot = 0;
	for (auto it : freq_map) {
		double freq = it.second;
		if (freq <= cold_thres) {
			cold ++;
		} else {
			hot++;
		}
	}

	return make_pair(hot, cold);
}



vector<int>  nfa_utils::get_cc_ids_from_state_id(const vector<NFA *> ccs, const vector<string> state_id_representatives) {
    unordered_map<string, int> rstateid_to_ccid;
    for (int i = 0; i < ccs.size(); i++) {
        for (int node_id = 0; node_id < ccs[i]->size(); node_id++) {
            auto node = ccs[i]->get_node_by_int_id(node_id);
            rstateid_to_ccid[node->original_id] = i;
        }
    }

    vector<int> res;
    for (auto state_id : state_id_representatives) {
        assert(rstateid_to_ccid.find(state_id) != rstateid_to_ccid.end());
        auto cc_id = rstateid_to_ccid[state_id];
        res.push_back(cc_id);
    }

    return res;
}

vector<int> nfa_utils::get_all_start_node_id(NFA *cc) {
	vector<int> res;

	for (int i = 0; i < cc->size(); i++) {
		auto node = cc->get_node_by_int_id(i);
		if (node->is_start()) {
			res.push_back(i);
		}
	}

	return res;
}










