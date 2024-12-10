#ifndef NODE_H_
#define NODE_H_

#include <string>
#include <vector>
#include <map>
#include <list>
#include <unordered_map>
#include <bitset>
#include <memory>
#include <set>
#include "vasim_helper.h"


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

enum NODE_START_ENUM {
	START=1, 
	START_ALWAYS_ENABLED=2
};


class Node {
public:
	Node();
	
	~Node();

	string original_id;

	string str_id;

	int sid;
	int cc_id;
	int cc_local_id;

	int scc_id;
	int topo_order;

	int bfs_layer;
	

	bitset<256> symbol_set;
	
	string symbol_set_str;

	bool complete; 
	bool complement;
	int match_set_range;

	int start;

	
	bool report = false;

	// new added for mnrl
	string report_code;
	bool report_eod = false; 


	bool visited = false;

	void symbol_set_to_bit();

    inline bool match2(uint8_t input) const {
        return symbol_set.test(input);
    }

    bool is_start_always_enabled() const;
    bool is_start() const;
    bool is_report() const;

    bool is_wildcard() const;

    // if the symbol set is a reverse of one symbol, we classify this to not type. 
    bool is_not_type_node() const; 

    int num_of_accept_symbol() const;

    void remap_alphabet(const map<int, int> &remap_table);

    int num_of_1_in_matchset() const; 


    double hot_degree; 

    int cg_id;

};







#endif /*NODE_H */

