/*
 * NFA.cpp
 *
 *  Created on: Apr 29, 2018
 *      Author: hyliu
 */

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
#include <iomanip>      // std::setprecision
#include "node.h"
#include "graph_helper.h"
#include "vasim_helper.h"

#include <fstream>


using namespace VASim;

using std::vector;
using std::map;
using std::unique_ptr;
using std::make_pair;
using std::queue;
using std::string;
using std::cout;
using std::endl;
using std::unordered_map;
using std::pair;
using std::stack;



NFA::NFA() {
	V = 0;
	E = 0;
	always_active_nodes_num = 0;
	start_active_nodes_num = 0;

	num_cc = 0; // after running mark_cc_id, this should be at least 1. 

	original_id_to_nodes.clear();

}

NFA::NFA(int V) : NFA() {
	this->V = V;
}

NFA::~NFA() {
	for (auto it : node_pool) {
		delete it.second;
	}
}


void NFA::remove_edge(string from, string to) {
	

	//unordered_map<string, vector<string> > adj;
	//unordered_map<string, vector<string> > from_node;

	//assert(adj[from_node])
	adj[from].erase(std::remove(adj[from].begin(), adj[from].end(), to), adj[from].end());
	this->from_node[to].erase(std::remove(this->from_node[to].begin(), this->from_node[to].end(), from), this->from_node[to].end());
	
	//bi_directed_eq_graph
}

void NFA::addNode(Node *n) {
	assert(n != NULL);

	addNode(n, this->V);

	if (n->start == NODE_START_ENUM::START_ALWAYS_ENABLED)
					this->always_active_nodes_num++;
	else if (n->start == NODE_START_ENUM::START)
					this->start_active_nodes_num++;
	this->V++;

	/*
	if (strid_to_intid.find(n->str_id) != strid_to_intid.end()) {
		return;
	}

	n->sid = this->V;
	strid_to_intid[n->str_id] = this->V;

	node_pool[this->V] = n;

	this->V ++;
	*/

}


void NFA::addNode(Node *n, int intid) {
	assert( n != NULL );
	assert (strid_to_intid.find(n->str_id) == strid_to_intid.end());

	n->sid = intid;
	strid_to_intid[n->str_id] = intid;

	assert(node_pool.find(intid) == node_pool.end());
	
	node_pool[intid] = n;

	original_id_to_nodes[n->original_id].push_back(n->str_id);

}


void NFA::addEdge(string from_str_id, string to_str_id) {
	if (std::find(adj[from_str_id].begin(), adj[from_str_id].end(), to_str_id) == adj[from_str_id].end()) {
		adj[from_str_id].push_back(to_str_id);
		this->E++;
	}

	if (std::find(from_node[to_str_id].begin(), from_node[to_str_id].end(), from_str_id) == from_node[to_str_id].end()  ) {
		from_node[to_str_id].push_back(from_str_id);
	}
}


Node* NFA::get_node_by_int_id(int iid) const {
	auto iter = this->node_pool.find(iid);
	if (iter == node_pool.end()) {
		cout << "the required node does not exist intid = " << iid << endl;
		exit(-1);

	}


	return iter->second;
}


Node* NFA::get_node_by_str_id(string sid) const {


	if (!has_node(sid)) {
		cout << "the required node does not exist intid = " << sid << endl;
		exit(-1);
	}

	auto iter = strid_to_intid.find(sid);

	assert(iter != strid_to_intid.end());

	int intid = iter->second;

	return get_node_by_int_id(intid);

}


void NFA::calc_bidirected_graph() {
	// cout << "calc_bidirected_graph" << endl;
	for (int i = 0; i < V; i++) {
		Node* current_node = this->get_node_by_int_id(i);
		string node_id = current_node->str_id;

		for (auto iter = adj[node_id].begin(); iter != adj[node_id].end(); iter++) {
			int from = i;
			int to = strid_to_intid[*iter];

			//cout << "add edge u v " << from << " " << to << endl;
			this->bi_directed_eq_graph[from].push_back(to);
			this->bi_directed_eq_graph[to].push_back(from);
		}
	}
}

int NFA::get_num_cc() const {
	return num_cc;
}


void NFA::mark_cc_id() {

    // cout << "NFA::mark_cc_id() " << endl;

	calc_bidirected_graph();
	clear_visit_flag();

	if (V <= 1) {
		cout << "V = " << V << endl;
		assert(V > 0);
	}

	//assert(V >= 1);

	num_cc = 0;

	for (int i = 0; i < V; i++) {
		auto cur_node = this->get_node_by_int_id(i);
		if (!cur_node->visited) {
			cur_node->visited = true;
			dfs(i, num_cc);
			num_cc++;
		}
	}

	// cout << "num_of_cc = " << num_cc << endl;

	int *cc_max_id = new int[num_cc];
	std::fill(cc_max_id, cc_max_id + num_cc, 0);

	for (int i = 0; i < V; i++) {
		auto cur_node = get_node_by_int_id(i);
		cur_node->cc_local_id = cc_max_id[cur_node->cc_id];
		cc_max_id[cur_node->cc_id] ++;
		//groupby_ccids[cur_node->cc_id].push_back(cur_node->sid);
	}

	delete[] cc_max_id;
}


int NFA::get_dag() {
	MyGraph g(V);

	int n = 0;
	for (int i = 0; i < V; i++) {
		Node *ele = get_node_by_int_id(i);
		
		auto outputs = get_adj(ele->str_id);
		for (auto it : outputs) {
			int to_id = get_int_id_by_str_id(it);

			if (i != to_id) {
				g.addEdge(i, to_id);	
			}

		}
	}

	g.calc_SCC();

	set<int> scc_ids; 

	auto sccmap = g.get_scc();

	for (int i = 0; i < V; i++) {
		Node *n = get_node_by_int_id(i);
		n->scc_id = sccmap[i];
		scc_ids.insert(sccmap[i]);

		assert(n->scc_id != -1);

		//cout << n->str_id << "   " << n->scc_id << endl;
	}


	int dag = 0;

	if (scc_ids.size() == V) {
		dag = 1;
		bool selfloop = false;
		for (int i = 0; i < V; i++) {
			if (has_self_loop(i)) {
				selfloop = true;
				break;
			}
		}

		if (!selfloop) {
			dag = 2;
		}
	}

	return dag;


}


void NFA::calc_scc() {
	MyGraph g(V);

	int n = 0;
	for (int i = 0; i < V; i++) {
		Node *ele = get_node_by_int_id(i);
		
		auto outputs = get_adj(ele->str_id);
		for (auto it : outputs) {
			int to_id = get_int_id_by_str_id(it);

			if (i != to_id) {
				g.addEdge(i, to_id);	
			}

		}
	}

	g.calc_SCC();
	
	g.bfs();

	auto sccmap = g.get_scc();

	auto bfslayers = g.get_bfs_layers();

	for (int i = 0; i < V; i++) {
		Node *n = get_node_by_int_id(i);
		n->scc_id = sccmap[i];

		assert(n->scc_id != -1);

		n->bfs_layer = bfslayers[i];

		//cout << n->str_id << "   " << n->scc_id << " " << n->bfs_layer << endl;
	}

}





int NFA::get_num_scc() const {
	int max_scc = -1;

	for (int i = 0; i < V; i++) {
		Node *n = get_node_by_int_id(i);
	//	n->scc_id = sccmap[i];
	//	cout << n->str_id << "   " << n->scc_id << endl;
		if (n->scc_id > max_scc) {
			max_scc = n->scc_id;
		}

	}

	assert(max_scc != -1);

	return max_scc + 1;
}



void NFA::topo_sort() {
	int num_scc = get_num_scc();
	//cout << "num_scc = " << num_scc << " " << V << endl;

	DAG g(num_scc);

	for (int i = 0; i < V; i++) {
		Node *ele = get_node_by_int_id(i);
		auto adjs = get_adj(ele->str_id);
		for (auto to : adjs) {
			int from_id = ele->scc_id;
			Node * to_ele = get_node_by_str_id(to);
			int to_id = to_ele->scc_id;
			if (from_id != to_id) {
				g.addEdge(from_id, to_id);
			}
		}
	}

	g.topological_sort();
	auto torder = g.get_topo_order();

	for (int t_layer = 0; t_layer < torder.size(); t_layer++) {
		for (auto scc_id : torder[t_layer]) {
			for (int i = 0; i < V; i++) {
				Node *ele = get_node_by_int_id(i);
				if (ele->scc_id == scc_id) {
					ele->topo_order = t_layer;
				}

			}
		}
	}

	//for (int i = 0; i < V; i++) {
	//	Node *n = get_node_by_int_id(i);
		//cout << n->str_id << "  sccid =  " << n->scc_id << " topoorder = " << n->topo_order << endl;
	//}
}




void NFA::dfs(int start_iid, int cc_id) {
	//cout << "reach here " << start_iid << endl;
	auto cur_node = this->get_node_by_int_id(start_iid);
	cur_node->cc_id = cc_id;

	for (auto it : bi_directed_eq_graph[start_iid]) {
		int to_iid = it;
		auto to_node = get_node_by_int_id(to_iid);
		if (!to_node->visited) {
			to_node->visited = true;
			dfs(to_node->sid, cc_id);
		}
	}
}


void NFA::clear_visit_flag() {
	for (int i = 0; i < V; i++) {
		Node *n = this->get_node_by_int_id(i);
		n->visited = false;
	}
}

int NFA::size() const {
	return V;
}

int NFA::edge_size() const {
	return E;
}

/**
 * for debug.
 */
void NFA::print() {
	cout << "automata_size = " << size() << endl;
	cout << "num_cc = " << get_num_cc() << endl;

	for (int i = 0; i < V; i++) {
		Node *node = get_node_by_int_id(i);
		cout << "node id = " << node->str_id << " intid = " << node->sid << " cc_id = "
				<< node->cc_id << " cc_local_id = " << node->cc_local_id << " symbolset = " << node->symbol_set_str <<
				"  " << node->symbol_set << "  is_start = " << node->is_start() << " always_enabled = " << node->is_start_always_enabled() 
				<< " topo = " << node->topo_order << " hotdegree = " << node->hot_degree << 
				endl;

		cout << "connect to ";
		for (auto to : adj[node->str_id]) {
			cout << to << " ";
		}
		cout << endl;
	}

	cout << endl;
}




int NFA::get_int_id_by_str_id(string str_id) const {
	auto iter = this->strid_to_intid.find(str_id);
	if (iter == strid_to_intid.end()) {
		cout << "cannot find intid by strid = " << str_id << endl;
	}
	assert(iter != strid_to_intid.end());
	return iter->second;
}


int NFA::get_num_transitions() const {
	int E = 0;
	for (auto it : adj) {
		E += it.second.size();
	}

	return E;
}


vector<string> NFA::get_adj(string str) const {
	if (!has_node(str)) {
		cout << "has node str? = " << str << endl;
		assert(has_node(str));
		
	}
	auto iter = this->adj.find(str);
	if (iter != adj.end()) {
		return iter->second;
	} else {
		//cout << "node str_id = " << str << " does not exist " << endl;
		//exit(-1);
		return vector<string> ();
	}
}

vector<string> NFA::get_from(string str_id) const {
	if (!has_node(str_id)) {
		cout << "has node str? = " << str_id << endl;
		assert(has_node(str_id));	
	}

	auto iter = from_node.find(str_id);
	if (iter != from_node.end()) {
		return iter->second;
	} else {
		//cout << "node str_id = " << str << " does not exist " << endl;
		//exit(-1);
		return vector<string> ();
	}
}


int NFA::get_indegree_of_node(string str_id) const {
	return this->get_adj(str_id).size();
}


int NFA::get_outdegree_of_node(string str_id) const {
	return this->get_adj(str_id).size();
}



bool NFA::has_node(string str) const {
	auto iter = strid_to_intid.find(str);
	return iter != strid_to_intid.end();
}

bool NFA::has_node(int int_id) const {
	auto iter = node_pool.find(int_id);
	return iter != node_pool.end();
}



Node NFA::remove_node_unsafe(string str_id) {
	//cout << "remove_node_unsafe  = " << str_id << endl;
	assert(has_node(str_id));
	Node *n = this->get_node_by_str_id(str_id);

	// remove from other nodes' out degree
	auto from_nodes = get_from(str_id);
	for (auto fn : from_nodes) {
		if (adj.find(fn) == adj.end()) {
			cout << "assert(adj.find(fn) != adj.end()); "  << "to_delete = " << str_id << "   from_node = " << fn << endl;
			assert(adj.find(fn) != adj.end());
		}

		auto iter = std::find(adj[fn].begin(),adj[fn].end(), str_id);
		assert(iter != adj[fn].end());		
		adj[fn].erase(iter);
	}

	//remove from other nodes' in degree
	auto to_nodes = get_adj(str_id);
	for (auto to : to_nodes) {
		if (from_node.find(to) == from_node.end()) {
			cout << "assert(adj.find(fn) != adj.end()); "  << "to_delete = " << str_id << "   to_node = " << to << endl;
			assert(from_node.find(to) != from_node.end());
		}

		auto iter = std::find(from_node[to].begin(), from_node[to].end(), str_id);
		assert(iter != from_node[to].end());		
		from_node[to].erase(iter);
	}

	this->adj.erase(str_id);
	this->from_node.erase(str_id);

	assert(node_pool.find(n->sid) != node_pool.end());
	node_pool.erase(n->sid);

	assert(strid_to_intid.find(str_id) != strid_to_intid.end());
	strid_to_intid.erase(str_id);
	

	Node ret = *n;

	delete n;
	return ret;
}


/*
void Automata::automataToDotFile(string out_fn, int cc_id) {
    map<string, uint32_t> id_map;

    string str = "";
    str += "digraph G {\n";

    //add all nodes
    uint32_t id = 0;
    for (auto e : elements) {
    	if (cc_id != -1 && e.second->cc_id != cc_id) {
    		continue;
    	}

        // map ids to string names
        id_map[e.first] = id;

        //string fillcolor = "\"#ffffff\"";
        string fillcolor = "\"#add8e6\"";

        str += to_string(id);

        // label
        str.append("[label=\"") ;
        //str.append(e.first); 

        if(e.second->isSpecialElement()) {
           // str.append(e.first);
        	str.append(std::to_string(e.second->getIntId()));
            str.append("strid:");
            str.append(e.second->getId());
            str.append("pLayer" + std::to_string(e.second->getPartitionLayer()));
        } else {
            STE * ste = dynamic_cast<STE *>(e.second);
            //str.append(e.first);
            str.append(std::to_string(ste->getIntId()));
            str.append("strid:");
            str.append(ste->getId());
            str.append("pLayer" + std::to_string(e.second->getPartitionLayer()));
            str += ":" + ste->getSymbolSet();
        }

        // heatmap color
        str.append("\" style=filled fillcolor="); 

        if (e.second->isHasEnabled()) {
        	fillcolor= "\"#00FF00\"";  //"#00FF00";

            if (e.second->isBoundary()) {
            	fillcolor = "\"#FF0000\"";
            }

        } else {
        	fillcolor= "\"#FFFFFF\"";

            if (e.second->isFake()) {
            	fillcolor = "\"#F9E79F\"";
            }

            if (e.second->isBoundary()) {
            	fillcolor = "\"#EB984E\"";
            }
        }

        str.append(fillcolor); 

        //start state double circle/report double octagon
        if(!e.second->isSpecialElement()) {
            STE * ste = dynamic_cast<STE *>(e.second);
            if(ste->isStart()) {
                if(ste->isReporting()) {
                    str.append(" shape=doubleoctagon"); 
                }else {
                    str.append(" shape=doublecircle"); 
                } 
            } else {
                if(ste->isReporting()) {
                    str.append(" shape=octagon");
                }else{
                    str.append(" shape=circle");
                } 
            }

        } else {
            str.append(" shape=rectangle"); 
        }

        str.append(" ];\n"); 
        id++;
    }

    // id map <string element name> -> <dot id int>
    for(auto e : id_map) {
        uint32_t from = e.second;

        for(auto to : elements[Element::stripPort(e.first)]->getOutputs()) {
        	if (id_map.find(Element::stripPort(to)) != id_map.end()) {
        		str += to_string(from) + " -> " + to_string(id_map[Element::stripPort(to)]) + ";\n";
        	}
        }
    }

    str += "}\n";

    writeStringToFile(str, out_fn);
}*/


void NFA::to_dot_file(string dotfile) const {
	string dot = "digraph G {\n";


	for (int i = 0; i < V; i++) {
		Node *n = get_node_by_int_id(i);
		string current_n = std::to_string(n->sid);
		current_n += " [label=\""  + n->str_id + " intid " + std::to_string(n->sid) +   " " + n->symbol_set_str  + " "  
						+ std::to_string(n->num_of_1_in_matchset()) +  "\"]";

		if (n->is_report()) {
			current_n += " [shape=doubleoctagon]";
		}

		if (n->is_start_always_enabled()) {
			current_n += " [color=orange]"; 
		} else if (n->is_start()) {
			current_n += " [color=yellow]";
		}

		current_n += ";\n";

		dot += current_n;
	}


	for (int i = 0; i < V; i++) {
		Node *n = get_node_by_int_id(i);
		auto adjs = get_adj(n->str_id);
		for (auto to : adjs) {
			auto to_node = get_node_by_str_id(to);
			string current_edge = std::to_string(n->sid) + " -> " + std::to_string(to_node->sid) + ";\n";
			dot += current_edge;
		}	
	}

	dot += "}";

	std::ofstream out(dotfile);
	out << dot << endl;
	out.close();

}




int NFA::get_num_topoorder() const {
	int mm = -1;
	for (int i = 0; i < V; i++) {
		Node *n = get_node_by_int_id(i);
		if (n -> topo_order > mm) {
			mm = n->topo_order;
		}
	}

	return mm;
}



set<uint8_t> NFA::get_alphabet_in_nfa_wo_wildcard() const {
	set<uint8_t> s;

	for (int i = 0; i < V; i++) {
		auto node = get_node_by_int_id(i);
		if (!node->is_wildcard()) {
			for (int i = 0; i < 256; i++) {
				if (node->match2( (uint8_t) i)) {
					s.insert( (uint8_t) i);
				}
			}
		}
	}

	return s;

}




set<uint8_t> NFA::get_alphabet_in_nodes_wo_wildcard_wo_nottype() const {
	set<uint8_t> s;

	for (int i = 0; i < V; i++) {
		auto node = get_node_by_int_id(i);
		if (node->symbol_set.count() >= 3 && !node->symbol_set.all()) {
			for (int i = 0; i < 256; i++) {
				if (node->match2( (uint8_t) i)) {
					s.insert( (uint8_t) i);
				}
			}
		}
	}

	return s;

}



int NFA::get_num_states_leq_topo(int topo) {
	int s = 0;
	for (int i = 0; i < this->size(); i++) {
		auto node = this->get_node_by_int_id(i);
		if (node->topo_order <= topo) {
			s++;
		}
	}
	
	return s;
}




bool NFA::has_self_loop(int sid) const {
	Node *node = get_node_by_int_id(sid);
	
	string str_id = node->str_id;

	auto to_nodes = get_adj(str_id);
	
	auto it_self1 = std::find(to_nodes.begin(), to_nodes.end(), str_id);

	return it_self1 != to_nodes.end();
}

	
bool NFA::has_self_loop(string str_id) const {
	auto node = get_node_by_str_id(str_id);
	return has_self_loop(node->sid);
}

void NFA::remove_self_loop(string str_id) {
	auto node = get_node_by_str_id(str_id);
	remove_self_loop(node->sid);
}

void NFA::remove_self_loop(int sid) {
	if (!has_self_loop(sid)) {
		return;
	}

	//cout << "remove_self_loop  " << sid <<  endl;

	Node *node = get_node_by_int_id(sid);
	
	string str_id = node->str_id;

	auto it_self1 = std::find(this->adj[str_id].begin(), this->adj[str_id].end(), str_id);
	auto it_self2 = std::find(this->from_node[str_id].begin(), this->from_node[str_id].end(), str_id);

	bool self_loop = it_self1 != this->adj[str_id].end();
	assert(self_loop == ( it_self2 != this->from_node[str_id].end() ) );

	this->adj[str_id].erase(it_self1);
	this->from_node[str_id].erase(it_self2);
	/*

	if (it_self1 != to_nodes.end() && it_self2 != from_nodes.end()) {
		//cout << "remove" << endl;
		to_nodes.erase(it_self1);
		from_nodes.erase(it_self2);
	}*/ 

	assert(!has_self_loop(sid));


}



int NFA::has_self_loop_plus_large_matchset() const {
	int tt = 0;
	for (int i = 0; i < size(); i++) {
		auto node = get_node_by_int_id(i);
		int nsymbolmatch = node->num_of_1_in_matchset();
		if (has_self_loop(i) && nsymbolmatch > 252) {
			tt += 1;
		}
	}

	return tt;
}


vector<string> NFA::get_nodes_by_original_id(string original_id) const {
	//assert(this->original_id_to_nodes.find(original_id) != this->original_id_to_nodes.end());
	
	if (this->original_id_to_nodes.find(original_id) == this->original_id_to_nodes.end()) {
		return vector<string> ();
	}
	
	return this->original_id_to_nodes.find(original_id)->second;

}




