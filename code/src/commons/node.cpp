#include "node.h"
#include <string>
#include <vector>
#include <map>
#include <list>
#include <unordered_map>
#include <bitset>
#include <memory>
#include <set>
#include <iostream>
#include "vasim_helper.h"

using std::cout;
using std::endl;

using namespace VASim;

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



Node::Node() {

	str_id = "";
	sid = -1;
	symbol_set.reset();
	symbol_set_str = "";
	start = 0;
	str_id = "";
	cc_id = 0;
	
	scc_id = -1;
	topo_order = -1;

	this->original_id = "undefined";

	report = false;

	complete = false;
	complement = false;


	this->hot_degree = 0.0;

	
	cg_id = -1;

	visited = false;
}

Node::~Node() {
}


bool Node::is_start_always_enabled() const {
	return (start == NODE_START_ENUM::START_ALWAYS_ENABLED);
}

bool Node::is_start() const {
	return (start == NODE_START_ENUM::START || start == NODE_START_ENUM::START_ALWAYS_ENABLED);
}

bool Node::is_report() const {
	return report;
}


void Node::symbol_set_to_bit() {
	parseSymbolSet(this->symbol_set, this->symbol_set_str);
}



bool Node::is_wildcard() const {
	return symbol_set.all();
}



// if the symbol set is a reverse of one symbol, we classify this to not type. 
bool Node::is_not_type_node() const {
	return (symbol_set.count() == 1);
} 



int Node::num_of_accept_symbol() const {
	return (symbol_set.count());
}




void Node::remap_alphabet(const map<int, int> &remap_table) {
	bitset<256> remapped_symbol_set;
	remapped_symbol_set.reset();
	for (auto it : remap_table) {
		int k = it.first;
		int v = it.second;
		//cout << "remap_alphabet_node " << k << " " << v << endl;

		if (this->symbol_set.test(k)) {
			remapped_symbol_set.set(v);
		}
	}

	this->symbol_set = remapped_symbol_set;

}



int Node::num_of_1_in_matchset() const {
	int n = 0; 
	for (int i = 0; i < 256; i++) {
		auto symbol = (uint8_t) i;
		if (this->match2(i)) {
			n++;
		}
	}
	return n;
}