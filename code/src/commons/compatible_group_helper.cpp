#include "compatible_group_helper.h"
#include "NFA.h"
#include <vector>
#include <iostream>
#include <queue>
#include <algorithm>
#include <cassert>

using std::queue;
using std::cout;
using std::endl;
using std::vector;




AbstractCompatibleGroupHelper::AbstractCompatibleGroupHelper() : nfa(NULL) {
	for (int i = 0; i < 256; i++) {
		this->alphabet.insert( (uint8_t) i  );
	}
	// default alphabet
}


AbstractCompatibleGroupHelper::~AbstractCompatibleGroupHelper() {

}


vector<int> AbstractCompatibleGroupHelper::get_states_in_compatible_group(int group_id) const {
	
	auto it = compatible_group.find(group_id);
	if (it == compatible_group.end()) {
		vector<int> emptyvector;
		return emptyvector;
	} else {
		return it->second;
	}


}


int AbstractCompatibleGroupHelper::num_compatible_grp() const {
	return compatible_group.size();
}



int AbstractCompatibleGroupHelper::get_compatible_grp_by_intid(int intid) const {
	auto it = node_to_compatible_grp.find(intid);
	if (it == node_to_compatible_grp.end()) {
		cout << "cannot find id in compatible group list (id = " << intid << ")" << endl;
		exit(-1);
	}

	return it->second;
}





CompatibleGroupHelper::CompatibleGroupHelper() : AbstractCompatibleGroupHelper() {


}

CompatibleGroupHelper::~CompatibleGroupHelper() {}



void CompatibleGroupHelper::calc_incompatible_states(const set<uint8_t>& alphabet) {

	assert(nfa != NULL);

	incompatible.clear();

	int V = nfa->size();

	for (int i = 0; i < V; i++) {
		vector<bool> tmp;

		for (int j = 0; j < V; j++) {
			tmp.push_back(false);
		}

		incompatible.push_back(tmp);
	}


	// init
	queue<pair<int, int> > Q;
	for (int i = 0; i < V; i++) {
		Q.push(make_pair(i, i));
		incompatible[i][i] = true;
	}

	// if cc is split (e.g. by limit the degree. ), then we need to make start states of them as incompatible. 
	for (int i = 0; i < V - 1; i++) {
		for (int j = i + 1; j < V; j++) {
			auto n1 = nfa->get_node_by_int_id(i);
			auto n2 = nfa->get_node_by_int_id(j);
			if (n1->is_start() && n2->is_start()) {
				if (!incompatible[i][j]) {
					incompatible[i][j] = true;
					incompatible[j][i] = true;
					Q.push(make_pair(i, j));
				}
			}
		}
	}

	for (int i = 0; i < V; i++) {
		auto n = nfa->get_node_by_int_id(i);
		if (n->is_start() || n->is_report()) {
			for (int j = 0; j < V; j++) {
				if (!incompatible[i][j]) {
					incompatible[i][j] = true;
					incompatible[j][i] = true;	
					Q.push(make_pair(i, j));
				}
			}
		}
	}


	while (!Q.empty()) {
		auto incompatible_pair = Q.front();
		Q.pop();

		Node *a = nfa->get_node_by_int_id(incompatible_pair.first);
		Node *b = nfa->get_node_by_int_id(incompatible_pair.second);

		//cout << a->str_id << "   " << b->str_id << endl;

		set<int> all_places_can_reach;
		
		auto adj1 = nfa->get_adj(a->str_id);
		auto adj2 = nfa->get_adj(b->str_id);
			
		if (a->is_start_always_enabled()) {
			if (std::find(adj1.begin(), adj1.end(), a->str_id) == adj1.end()) {
				adj1.push_back(a->str_id);	
			}
		}

		if (b->is_start_always_enabled()) {
			if (std::find(adj2.begin(), adj2.end(), b->str_id) == adj2.end()) {
				adj2.push_back(b->str_id);	
			}
		}

		for (auto a_to : adj1) {
			int iid_a_to = nfa->get_int_id_by_str_id(a_to);
			all_places_can_reach.insert(iid_a_to);
		}

		for (auto b_to : adj2) {
			int iid_b_to = nfa->get_int_id_by_str_id(b_to);
			all_places_can_reach.insert(iid_b_to);
		}

		/*cout << "reach -- ";
		for (auto tttt : all_places_can_reach) {
			auto node1 = nfa.get_node_by_int_id(tttt);
			cout << node1->str_id << "  ";
		}
		cout << endl;*/

		for (auto symbol : alphabet) {
			set<int> could_active;
			for (auto p : all_places_can_reach) {
				auto to_node = nfa->get_node_by_int_id(p);
				if (to_node->match2(symbol)) {
					could_active.insert(to_node->sid);
				}
			}

			for (auto p1 : could_active) {
				for (auto p2 : could_active) {
					if (!incompatible[p1][p2]) {
						incompatible[p1][p2] = true;
						incompatible[p2][p1] = true;
						Q.push(make_pair(p1, p2));
					}
				}
			}		
		}
	}


}


void CompatibleGroupHelper::calc_compatible_groups() {
	node_to_compatible_grp.clear();
	compatible_group.clear();

	int V = nfa->size();
	MyGraph1 g(V);

	g.set_adj_matrix(incompatible);
	g.calc_independent_set();

	auto m = g.get_independent_sets();

	//cout << "V = " << V << " n_compatible_groups = " << g.get_num_independent_set() << endl;
	for (int i = 0; i < V; i++) {
		Node *n = nfa->get_node_by_int_id(i);
		//n->cg_id = m[i];
		//cout << "intid = " << n->sid << "  compatible_group_id = " << n->cg_id << endl;

		compatible_group[m[i]].push_back(i);
		node_to_compatible_grp[i] = m[i];
	}


}

/*
int CompatibleGroupHelper::num_possible_simultaneous_activation(int intid) {
	int s = 0;
	assert(intid < incompatible.size());
	for (int i = 0; i < incompatible.size(); i++) {
		if (incompatible[intid][i]) {
			s++;
		}
	}

	return s;
}



double  CompatibleGroupHelper::avg_possible_simultaneous_activation() {
	double s = 0;

	assert(nfa.size() > 0);

	for (int v = 0; v < nfa.size(); v++) {
		//auto node = nfa.get_node_by_int_id(v);
		
		int num_act = num_possible_simultaneous_activation(v);	
		
		s += num_act;
	}

	return s / nfa.size();
	
}


int CompatibleGroupHelper::max_num_possible_simultaneous_activation() {
	int s = 0;

	assert(nfa.size() > 0);

	for (int v = 0; v < nfa.size(); v++) {
		auto node = nfa.get_node_by_int_id(v);
		
		int tt = num_possible_simultaneous_activation(v);	

		if (!node->is_start()) {
			if (tt > s) {
				s = tt;
			}
		}

	}

	return s;


}

*/



MyGraph1::MyGraph1(int V) {
	for (int i = 0; i < V; i++) {
		this->node_list.insert(i);
	}

	n_independet_set = 0;
}



void MyGraph1::set_adj_matrix(vector<vector<bool> > adjmat) {
	this->adj_matrix = adjmat;
	update_degree_all();
}


int MyGraph1::deg(int node_id) const {

	if (node_degree.find(node_id) != node_degree.end()) {
		auto tmp = node_degree.find(node_id);
		return tmp->second;
	} else {
		return -1;
	}


	/*
	assert (node_list.find(node_id) != node_list.end());
	int d = 0;
	for (auto it : node_list) {
		if (this->adj_matrix[node_id][it]) {
			d++;
		}
	}
	*/

	//return d;
}


bool MyGraph1::is_residual_independent_set() {
	visited.clear();
	for (auto it : node_list) {
		visited[it] = false;
	}

	int num_cc = 0;
	for (auto it : node_list) {
		if (!visited[it]) {
			visited[it] = true;
			dfs(it);
			num_cc++;
 		}
	}

	return num_cc == node_list.size();
}



void MyGraph1::dfs(int start_id) {
	for (auto it : node_list) {
		if (adj_matrix[start_id][it] && !visited[it]) {
			visited[it] = true;
			dfs(it);
		}
	}
}



pair<int, int> MyGraph1::pick_edge() {
	int max_sum_degree = 0;
	int a = -1, b = -1;

	assert(node_list.size() >= 2);
	//cout << "node_list = " << node_list.size() << endl;


	for (auto it1 : node_list) {
		for (auto it2 : node_list) {
			if (it1 != it2) {
				if (deg(it1) + deg(it2) > max_sum_degree) {
					max_sum_degree = deg(it1) + deg(it2);
					a = it1;
					b = it2;
				}
			}
		}
	}

	assert(a != -1 && b != -1);
	return make_pair(a, b);
}


void MyGraph1::remove_edge_and_attached_nodes(int u, int v) {
	node_list.erase(u);
	node_list.erase(v);
}




void MyGraph1::incremental_update_related_degree(int v) {
	node_degree[v] = 0;

	for (auto node : node_list) {
		if (v != node && adj_matrix[v][node] ) {
			node_degree[node] -= 1;
		}
	}

}


int MyGraph1::independent_set_iteration() {

	vector<int> removed_nodes;

	int size_of_current_independent_set = 0;
	while (true) {
		auto picked_edge = pick_edge();

		removed_nodes.push_back(picked_edge.first);
		removed_nodes.push_back(picked_edge.second);

		incremental_update_related_degree(picked_edge.first);
		incremental_update_related_degree(picked_edge.second);

		remove_edge_and_attached_nodes(picked_edge.first, picked_edge.second);


		if (is_residual_independent_set()) {

			for (auto it : removed_nodes) {

				node_list.insert(it);

				if (!is_residual_independent_set()) {
					node_list.erase(it);
				} else {
					removed_nodes.erase(std::remove(removed_nodes.begin(), removed_nodes.end(), it), removed_nodes.end());
				}
			}

			for (auto it : node_list) {
				node_independent_set_number[it] = n_independet_set;
			}


			n_independet_set++;

			node_list.clear();

			break;
		}
	}


	for (auto it : removed_nodes) {
		node_list.insert(it);
	}

	update_degree_all();

	return size_of_current_independent_set;
}


void  MyGraph1::update_degree_all() {
	for (auto i : node_list) {
		node_degree[i] = 0;
	}

	for (auto i : node_list) {
		for (auto j : node_list) {
			if (i != j && adj_matrix[i][j]) {
				node_degree[i]++;
			}
		}
	}

}

void MyGraph1::calc_independent_set() {
	while (!is_residual_independent_set()) {
		independent_set_iteration();
	}

	if (node_list.size() > 0) {
		for (auto it : node_list) {
			node_independent_set_number[it] = n_independet_set;
		}

		n_independet_set ++;

	}
}



const map<int, int> &  MyGraph1::get_independent_sets() const {
	return this->node_independent_set_number;
}


int MyGraph1::get_num_independent_set() const {
	return this->n_independet_set;
}










CompatibleGroup_EA_Helper::CompatibleGroup_EA_Helper(): AbstractCompatibleGroupHelper() {

	// default alphabet
}



CompatibleGroup_EA_Helper::~CompatibleGroup_EA_Helper() {

}



void CompatibleGroup_EA_Helper::calc_incompatible_states(const set<uint8_t>& alphabet) {
	assert(nfa != NULL);

	this->clear_incompatible_matrix();

	queue<pair<int, int> > Q;
	
	int V = nfa->size();

	// (1) A state is incompatible with itself. 
	for (int i = 0; i < V; i++) {
		Q.push(make_pair(i, i));
		incompatible[i][i] = true;
	}

	
	// (2) If a cc has multiple starting states, then: 
	for (int i = 0; i < V - 1; i++) {
		for (int j = i + 1; j < V; j++) {
			auto n1 = nfa->get_node_by_int_id(i);
			auto n2 = nfa->get_node_by_int_id(j);
			if (n1->is_start() && n2->is_start()) {
				if (!incompatible[i][j]) {
					incompatible[i][j] = true;
					incompatible[j][i] = true;
					Q.push(make_pair(i, j));
				}
			}
		}
	}


	// (3) If a state is always enabled, then it is incompatible with any other nodes. 
	for (int i = 0; i < V; i++) {
		auto n = nfa->get_node_by_int_id(i);
		if (n->is_start() || n->is_report()) {
			for (int j = 0; j < V; j++) {
				if (!incompatible[i][j]) {
					incompatible[i][j] = true;
					incompatible[j][i] = true;	
					Q.push(make_pair(i, j));
				}
			}
		}
	}

	// (4) The states that have the same parent are incompatible. 
	for (int i = 0; i < V; i++) {
		auto n = nfa->get_node_by_int_id(i);
		auto adj = nfa->get_adj(n->str_id);

		for (int k1 = 0; k1 < adj.size(); k1++) {
			for (int k2 = 0; k2 < adj.size(); k2++) {

				string to_x = adj[k1];
				string to_y = adj[k2];

				//cout << "???" << to_x << " " << to_y << endl;

				auto aa = nfa->get_node_by_str_id(to_x);
				auto bb = nfa->get_node_by_str_id(to_y);

				if (!incompatible[aa->sid][bb->sid]) {
					//cout << "!!!" << aa->sid << " " << bb->sid << endl;
					incompatible[aa->sid][bb->sid] = true;
					incompatible[bb->sid][aa->sid] = true;
					Q.push(make_pair(aa->sid, bb->sid));
				}

			}
		}

	}


	vector<Node*> always_enabled_nodes_vec;
	for (int i = 0; i < nfa->size(); i++) {
		auto node = nfa->get_node_by_int_id(i);
		if (node->is_start_always_enabled()) {
			always_enabled_nodes_vec.push_back(node);
		}
	}


	while (!Q.empty()) {
		auto incompatible_pair = Q.front();
		Q.pop();

		Node *a = nfa->get_node_by_int_id(incompatible_pair.first);
		Node *b = nfa->get_node_by_int_id(incompatible_pair.second);

		//cout << a->str_id << " " << b->str_id << endl;
		//
		//to_node->match2(symbol)
		
		bool ab_active_together = false;
		for (auto symbol : alphabet) {
			if (a->match2(symbol) && b->match2(symbol)) {
				ab_active_together = true;
				break;
			}		
		}

		if (ab_active_together) {
			set<int> successors_of_a_and_b; 

			auto adjs_a = nfa->get_adj(a->str_id);
			auto adjs_b = nfa->get_adj(b->str_id);

			//cout << "adjs_a.size = " << adjs_a.size() << endl;
			//cout << "adjs_b.size = " << adjs_b.size() << endl;

			for (auto to_node_strid : adjs_a) {
				auto to_node = nfa->get_node_by_str_id(to_node_strid);
				successors_of_a_and_b.insert(to_node->sid);
			}

			for (auto to_node_strid : adjs_b) {
				auto to_node = nfa->get_node_by_str_id(to_node_strid);
				successors_of_a_and_b.insert(to_node->sid);
			}

			for (auto nd : always_enabled_nodes_vec) {
				successors_of_a_and_b.insert(nd->sid);
			}

			vector<int> successors_of_a_and_b_vec(successors_of_a_and_b.begin(), successors_of_a_and_b.end());

			//cout << "successors_of_a_and_b size = " << successors_of_a_and_b_vec.size() << endl;

			for (int i = 0; i < successors_of_a_and_b_vec.size(); i++) {
				for (int j = 0; j < successors_of_a_and_b_vec.size(); j++) {
					int x = successors_of_a_and_b_vec[i];
					int y = successors_of_a_and_b_vec[j];

					//cout << "i " << i << " " << j << endl;

					if (!incompatible[x][y]) {
						incompatible[x][y] = true;
						incompatible[y][x] = true;

						//cout << "push " << x << " " << y << endl;
						Q.push(make_pair(x, y));
					}
				}
			}
		}
	}

	/*for (int i = 0; i < V; i++) {
		for (int j = i+1; j<V; j++) {
			if (incompatible[i][j]) {
				cout << "incompatible " << i << " " << j << endl;
		
			}
		}
	}*/
	//cout << "here finishes the incompatible func" << endl;


}


void CompatibleGroup_EA_Helper::clear_incompatible_matrix() {
	this->incompatible.clear();

	for (int i = 0; i < nfa->size(); i++) {
		vector<bool> tmp;

		for (int j = 0; j < nfa->size(); j++) {
			tmp.push_back(false);
		}

		incompatible.push_back(tmp);
	}

}



void CompatibleGroup_EA_Helper::calc_compatible_groups() {
	assert(nfa != NULL);

	node_to_compatible_grp.clear();
	compatible_group.clear();

	int V = nfa->size();
	MyGraph1 g(V);

	g.set_adj_matrix(incompatible);
	g.calc_independent_set();

	auto m = g.get_independent_sets();

	//cout << "V = " << V << " n_compatible_groups = " << g.get_num_independent_set() << endl;
	for (int i = 0; i < V; i++) {
		Node *n = nfa->get_node_by_int_id(i);
		//n->cg_id = m[i];
		//cout << "intid = " << n->sid << "  compatible_group_id = " << n->cg_id << endl;

		compatible_group[m[i]].push_back(i);
		node_to_compatible_grp[i] = m[i];
	}


}


CC_CG_helper::CC_CG_helper(NFA *cc, double cold_thres)
: cc(cc),
cold_threshold(cold_thres)
{

}

CC_CG_helper::CC_CG_helper(NFA *cc, const map<string, int> &state_id_to_cgid_map, double cold_threshold)
: 
CC_CG_helper(cc, cold_threshold)
{
	this->state_id_to_cgid_map_thiscc.clear();

	for (int i = 0; i < cc->size(); i++) {
		auto node = cc->get_node_by_int_id(i);
		assert(state_id_to_cgid_map.find(node->str_id) != state_id_to_cgid_map.end());
		this->state_id_to_cgid_map_thiscc[node->str_id] = state_id_to_cgid_map.find(node->str_id)->second;
	}

	this->cg_id_to_states.clear();

	for (auto it : this->state_id_to_cgid_map_thiscc) {
		auto stateid = it.first;
		auto cgid    = it.second;

		auto node = cc->get_node_by_str_id(stateid);

		this->cg_id_to_states[cgid].push_back(node->str_id);
	}
}



CC_CG_helper::~CC_CG_helper() {

}


int CC_CG_helper::get_num_of_cgs() const {
	int n = this->cg_id_to_states.size();

	for (int i = 0; i < n; i++) {
		assert(this->cg_id_to_states.find(i) != this->cg_id_to_states.end());
	}

	return n;
}



vector<int> CC_CG_helper::get_state_int_ids_in_cg(int cgid) const {
	auto str_ids = get_state_str_ids_in_cg(cgid);

	vector<int> res;
	for (auto str_id : str_ids) {
		auto node = cc->get_node_by_str_id(str_id);
		res.push_back(node->sid);
	}

	return res;
}


vector<string> CC_CG_helper::get_state_str_ids_in_cg(int cgid) const {
	assert(this->cg_id_to_states.find(cgid) != this->cg_id_to_states.end());

	return this->cg_id_to_states.find(cgid)->second;
}



vector<int> CC_CG_helper::get_hot_cgs() const {
	return get_cgs_by_type(CC_CG_helper::HOT_CG);
}



vector<int> CC_CG_helper::get_cold_cgs() const {
	return get_cgs_by_type(CC_CG_helper::COLD_CG);
}


vector<int> CC_CG_helper::get_cgs_by_type(int ttid) const {
	vector<int> res;
	int n = get_num_of_cgs();
	for (int i = 0; i < n; i++) {
		if (get_cg_type(i) == ttid) {
			res.push_back(i);
		}
	}
	return res;
}


int CC_CG_helper::get_cg_type(int cg_id) const {	
	auto stes_in_cg = get_state_int_ids_in_cg(cg_id);

	int cg_size = stes_in_cg.size();

	int num_hot = 0;
	int num_cold = 0;

	for (auto ste_int_id : stes_in_cg) {
		auto node = cc->get_node_by_int_id(ste_int_id);
		if (node->hot_degree <= cold_threshold) {
			// cold
			num_cold ++;
		} else {
			// hot
			num_hot ++;
		}
	}

	if (num_hot == 0) {
		return CC_CG_helper::COLD_CG;
	}

	if (num_hot == get_cg_size(cg_id)) {
		return CC_CG_helper::HOT_CG;
	}

	return CC_CG_helper::MIXED_CG;
}


int CC_CG_helper::get_cg_size(int cg_id) const {
	return get_state_int_ids_in_cg(cg_id).size();
}






void CC_CG_helper::update_hot_cg_to_one_state_cgs(int cg_id) {
	assert(get_cg_type(cg_id) == CC_CG_helper::HOT_CG);
	assert(get_cg_size(cg_id) > 0);

	if (get_cg_size(cg_id) == 1) {
		return;
	}

	//

	//map<string, int> state_id_to_cgid_map_thiscc;

	//map<int, vector<int> > cg_id_to_states; 


	

	int max_cg_id = get_num_of_cgs() - 1;

	auto states_in_this_cg = get_state_int_ids_in_cg(cg_id);
	//cout << ">>>" << states_in_this_cg.size() << " " << get_cg_size(cg_id) << endl;
	assert(states_in_this_cg.size() > 1);

	
	for (int i = 1; i < states_in_this_cg.size(); i++) {
		auto state_int_id = states_in_this_cg[i];
		auto node = cc->get_node_by_int_id(state_int_id);

		state_id_to_cgid_map_thiscc[node->str_id] = max_cg_id + i;

		assert(cg_id_to_states.find(max_cg_id + i) == cg_id_to_states.end());

		cg_id_to_states[max_cg_id + i].push_back(node->str_id);
	}

	this->cg_id_to_states[cg_id].clear();

	auto first_node = cc->get_node_by_int_id(states_in_this_cg[0]);
	cg_id_to_states[cg_id].push_back(first_node->str_id);

	assert( get_num_of_cgs() == max_cg_id + states_in_this_cg.size());

	for (int i = max_cg_id + 1; i < get_num_of_cgs(); i++) {
		assert(get_cg_type(i) == CC_CG_helper::HOT_CG);
		if (get_cg_size(i) != 1) {
			//cout << "get_cg_size" << i << " " <<  get_cg_size(i) << endl;
			assert(get_cg_size(i) == 1);
		}
		
	}
}



void CC_CG_helper::update_mixed_cg_to_hot_cg_and_cold_cg(int cg_id) {
	assert(get_cg_type(cg_id) == CC_CG_helper::MIXED_CG);
	assert(get_cg_size(cg_id) > 1);

	auto states_in_this_cg = get_state_int_ids_in_cg(cg_id);

	vector<int> hot_states;
	vector<int> cold_states; 

	for (auto intid : states_in_this_cg) {
		auto node = cc->get_node_by_int_id(intid);
		if (node->hot_degree <= cold_threshold) {
			cold_states.push_back(intid);
		} else {
			hot_states.push_back(intid);
		}
	}

	this->cg_id_to_states[cg_id].clear();

	for (auto intid : cold_states) {
		auto node = cc->get_node_by_int_id(intid);

		this->cg_id_to_states[cg_id].push_back(node->str_id);
	}


	int new_hot_cg_id = get_num_of_cgs(); 

	for (auto intid : hot_states) {
		auto node = cc->get_node_by_int_id(intid);
		this->cg_id_to_states[new_hot_cg_id].push_back(node->str_id);

		this->state_id_to_cgid_map_thiscc[node->str_id] = new_hot_cg_id;
	}


	assert(get_num_of_cgs() == new_hot_cg_id + 1);
	assert(get_cg_type(new_hot_cg_id) == CC_CG_helper::HOT_CG);

}






int CC_CG_helper::get_cg_id_by_state_strid(string strid) const {
	assert(this->state_id_to_cgid_map_thiscc.find(strid) != this->state_id_to_cgid_map_thiscc.end());
	return this->state_id_to_cgid_map_thiscc.find(strid)->second;
}






void CC_CG_helper::insert_state_strid_and_cgid_pair(string state_str_id, int cg_id) {
	
	//map<string, int> state_id_to_cgid_map_thiscc;
	//map<int, vector<string> > cg_id_to_states; 

	assert(state_id_to_cgid_map_thiscc.find(state_str_id) == state_id_to_cgid_map_thiscc.end());
	assert(cc->has_node(state_str_id));

	state_id_to_cgid_map_thiscc[state_str_id] = cg_id;

	cg_id_to_states[cg_id].push_back(state_str_id);

}





CC_CG_helper cg::create_cc_cg_helper_grped(NFA *grped_nfa, const vector<int> &cc_idx_in_group, vector<NFA *> ccs, const map<string, int> &state_id_to_cgid_map, double cold_threshold) {

	CC_CG_helper ch_for_grped_nfa(grped_nfa, cold_threshold);

	assert(cc_idx_in_group.size() >= 1);

	int base_cg_id = 0;

	for (auto idx : cc_idx_in_group) {
		auto cc = ccs[idx];

		CC_CG_helper ch(cc, state_id_to_cgid_map, cold_threshold);

		// transform hot and cold cgs
		// We split hot cgs to be only one at a cg
		// we split mixed cgs to hot cg and cold cg, and perform (1) on the hot cg
    	auto mixed_cgs = ch.get_cgs_by_type(CC_CG_helper::MIXED_CG);
    	for (auto m : mixed_cgs) {
    		ch.update_mixed_cg_to_hot_cg_and_cold_cg(m);
    	}

    	auto hot_cgs = ch.get_cgs_by_type(CC_CG_helper::HOT_CG);
    	for (auto m : hot_cgs) {
    		if (ch.get_cg_size(m) > 1) {
    			ch.update_hot_cg_to_one_state_cgs(m);
    		}
    	}

    	// 
    	assert(ch.get_cgs_by_type(CC_CG_helper::MIXED_CG).size() == 0);
    	hot_cgs = ch.get_cgs_by_type(CC_CG_helper::HOT_CG);
    	for (auto m : hot_cgs) {
    		assert(ch.get_cg_size(m) == 1);
    	}
    	// -------------------------------------------------------------------------


    	for (int intid = 0; intid < cc->size(); intid++) {
    		auto node = cc->get_node_by_int_id(intid);
    		auto cg_id_in_the_grped_nfa = base_cg_id + ch.get_cg_id_by_state_strid(node->str_id);
    		
    		ch_for_grped_nfa.insert_state_strid_and_cgid_pair(node->str_id, cg_id_in_the_grped_nfa);
    	}


    	base_cg_id += ch.get_num_of_cgs();
	}

	return ch_for_grped_nfa;
}

