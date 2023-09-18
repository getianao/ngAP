/*
 * NFA.h
 *
 *  Created on: Apr 29, 2018
 *      Author: hyliu
 */

#ifndef NFA_H_
#define NFA_H_

#include <string>
#include <vector>
#include <map>
#include <list>
#include <unordered_map>
#include <bitset>
#include <memory>
#include <set>
#include "node.h"


using std::set;
using std::unique_ptr;
using std::bitset;
using std::string;
using std::map;
using std::vector;
using std::list;
using std::unordered_map;
using std::pair;
using std::make_pair;


class NFA {

public:
	NFA();
	NFA(int V);
	virtual ~NFA();

	void addNode(Node *n);

	void addNode(Node *n, int intid);

	void addEdge(string from_str_id, string to_str_id);

	int size() const;
	int edge_size() const;
	int always_active_nodes_num;
	int start_active_nodes_num;

	void mark_cc_id();
	int get_num_cc() const;

	Node* get_node_by_int_id(int iid) const;
	Node* get_node_by_str_id(string sid) const;

	int get_num_transitions() const;

	int get_int_id_by_str_id(string str_id) const;

	void print();

	void calc_scc();
	void topo_sort();

	int get_num_scc() const;

	vector<string> get_nodes_by_original_id(string original_id) const;

	vector<string> get_adj(string str) const;
	vector<string> get_from(string str_id) const;

	int get_indegree_of_node(string str_id) const;
	int get_outdegree_of_node(string str_id) const; 

	bool has_node(string str) const;
	bool has_node(int int_id) const;

	void to_dot_file(string dotfile) const;

	/** return the removed node's intid 
		
		This function must be called followed by an addNode(Node *n, int intid);
		where the intid is the previous one. 
		or there will be an inconsistency. 


	**/
	Node remove_node_unsafe(string str_id);

	void remove_edge(string from_node, string to_node);

	int get_num_topoorder() const;

	set<uint8_t> get_alphabet_in_nfa_wo_wildcard() const;

	set<uint8_t> get_alphabet_in_nodes_wo_wildcard_wo_nottype() const;

	int get_num_states_leq_topo(int topo);
	
	int get_dag();

	bool has_self_loop(int sid) const;
	bool has_self_loop(string str_id) const;
	
	void remove_self_loop(int sid);
	void remove_self_loop(string str_id);

	int has_self_loop_plus_large_matchset() const; 
	unordered_map<string, vector<string> > adj;


private:

	// ----- for separate CCs -------------------------------
	void calc_bidirected_graph();
	void clear_visit_flag();
	void dfs(int start_iid, int cc_id);

	unordered_map<string, vector<string> > from_node;

	unordered_map<string, int> strid_to_intid;
	
	unordered_map<int, Node * > node_pool;
	
	int V; // n nodes;
	int E;

	unordered_map<int, list<int>> bi_directed_eq_graph; 
	int num_cc;


	unordered_map<string, vector<string> > original_id_to_nodes;

};







#endif /* NFA_H_ */







