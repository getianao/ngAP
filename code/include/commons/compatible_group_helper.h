#pragma once

#include "NFA.h"
#include <vector>
#include <iostream>

using std::cout;
using std::endl;
using std::vector;



class AbstractCompatibleGroupHelper {
public:

	explicit AbstractCompatibleGroupHelper();
	virtual ~AbstractCompatibleGroupHelper();

	virtual void calc_compatible_groups() = 0;
	virtual void calc_incompatible_states(const set<uint8_t>& alphabet) = 0;

	virtual void calc_incompatible_states2() {
		this->calc_incompatible_states(this->alphabet);
	}


	virtual void set_alphabet(set<uint8_t> ab) {
		this->alphabet = ab;
	}

	vector<int> get_states_in_compatible_group(int group_id) const;

	virtual int num_compatible_grp() const;
	virtual int get_compatible_grp_by_intid(int intid) const;


	virtual void set_nfa(NFA *nfa) {
		this->nfa = nfa;
	}

protected:
	NFA *nfa;

	set<uint8_t> alphabet;

	vector<vector<bool> > incompatible;

	map<int, vector<int> > compatible_group;

	map<int, int> node_to_compatible_grp; 


};



class CompatibleGroupHelper : public AbstractCompatibleGroupHelper {
public:
	CompatibleGroupHelper();
	virtual ~CompatibleGroupHelper();

	virtual void calc_compatible_groups() override;
	virtual void calc_incompatible_states(const set<uint8_t>& alphabet) override;

	//int num_possible_simultaneous_activation(int intid);

	//double avg_possible_simultaneous_activation();
	//int max_num_possible_simultaneous_activation();

};


// New.
// Let's induce the definition of Compatible Group for EA style NFA processing;
// --- ** If two states cannot be ENABLED together, they are compatible. **
// --- let's see. 

class CompatibleGroup_EA_Helper : public AbstractCompatibleGroupHelper  {
public:
	CompatibleGroup_EA_Helper();
	virtual ~CompatibleGroup_EA_Helper();

	virtual void calc_compatible_groups() override;
	virtual void calc_incompatible_states(const set<uint8_t>& alphabet) override;


private:
	void clear_incompatible_matrix();

};





// for calculating independent set.
class MyGraph1 {
public:
	MyGraph1(int V);

	int deg(int node_id) const;

	void set_adj_matrix(vector<vector<bool> > adjmat);

	void calc_independent_set();

	const map<int, int> & get_independent_sets() const;

	int get_num_independent_set() const;


private:

	map<int, int> node_degree;

	void incremental_update_related_degree(int v);

	void update_degree_all();

	int independent_set_iteration();

	bool is_residual_independent_set();

	void dfs(int start_id);

	pair<int, int> pick_edge();

	void remove_edge_and_attached_nodes(int u, int v);

	vector<vector<bool> > adj_matrix;

	map<int, int> node_independent_set_number;

	map<int, bool> visited;

	int n_independet_set;

	set<int> node_list;
};




class CC_CG_helper {
public:
	CC_CG_helper(NFA *cc, double cold_thres);
	CC_CG_helper(NFA *cc, const map<string, int> &state_id_to_cgid_map, double cold_thres);
	virtual ~CC_CG_helper();

	int get_num_of_cgs() const;

	vector<int> get_state_int_ids_in_cg(int cgid) const;

	vector<string> get_state_str_ids_in_cg(int cgid) const;

	vector<int> get_hot_cgs() const;

	vector<int> get_cold_cgs() const;

	int get_cg_type(int cg_id) const;

	int get_cg_size(int cg_id) const;

	vector<int> get_cgs_by_type(int tt) const;
	
	void update_hot_cg_to_one_state_cgs(int cg_id);

	void update_mixed_cg_to_hot_cg_and_cold_cg(int cg_id);

	int get_cg_id_by_state_strid(string strid) const;

	void insert_state_strid_and_cgid_pair(string state_str_id, int cg_id);

	static const int HOT_CG = 1, COLD_CG = 2, MIXED_CG = 3;

private:
	NFA *cc;
	
	map<string, int> state_id_to_cgid_map_thiscc;

	map<int, vector<string> > cg_id_to_states; 

	const double cold_threshold; 


};



namespace cg {
	CC_CG_helper create_cc_cg_helper_grped(NFA *grped_nfa, const vector<int> &cc_idx_in_group, vector<NFA *> ccs, const map<string, int> &state_id_to_cgid_map, double cold_threshold);
}

/*
class CC_CG_helper_group {
public:
	CC_CG_helper_group(NFA *grped_nfa, const vector<int> &cc_idx_in_group, vector<NFA *> ccs, const map<string, int> &state_id_to_cgid_map, double cold_threshold);
	virtual ~CC_CG_helper_group();

	void insert_state_strid_and_cgid_pair(string state_str_id, int cg_id);


private:
	NFA *grped_nfa;

	map<string, int> state_id_to_cgid_map_thiscc;
	map<int, vector<string> > cg_id_to_states; 

	const double cold_thres;

};	
*/

// Actually the two classes have many similarities. They could be merged to one class. 


