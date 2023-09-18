#ifndef COMMON_H_
#define COMMON_H_

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
#include <bitset>

using std::vector;
using std::string;
using std::make_pair;
using std::pair;

const int ALPHABET_SIZE = 256;

const int EMPTY_ENTRY = 56789;

enum remap_node_type {
	NONE = 0,
	REPORT = 1,
	TOPO_ORDER = 2,
	BFS_LAYER = 3,
	OUTDEGREE = 4,
	COMPLETE = 5,
	COMPLETE_AND_TOP = 6,
	COMPLETE_AND_BFS = 7

};


struct match_pair {
	int symbol_offset;
	int state_id; 
	
	bool operator< (const match_pair& o) const {
		if (symbol_offset < o.symbol_offset) {
			return true;
		} else if (symbol_offset == o.symbol_offset) {
			if (state_id < o.state_id ) {
				return true;
			}
			return false;
		} else {
			return false;
		}
	}
};


struct match3 {
	int symbol_offset;
	int state_id; 
	int nfa; 

	bool operator< (const match_pair& o) const {
		if (symbol_offset < o.symbol_offset) {
			return true;
		} else if (symbol_offset == o.symbol_offset) {
			if (state_id < o.state_id ) {
				return true;
			}
			return false;
		} else {
			return false;
		}
	}

};



struct match_entry {
	int symbol_offset;
	int state_id;
	int cc_id;
	int stream_id;
};

std::ostream& operator<<(std::ostream& os, const match_pair &obj);



template<int DEGREE_LIMIT>
struct STE_dev {
	int32_t ms[8]; // 8 * 32 = 256; local memory. 
	
	int   edge_dst[DEGREE_LIMIT];
	
	char     attribute;  // is report? 
	int degree;

}; 



struct STE_dev4 {
	int32_t ms[8]; // 8 * 32 = 256; local memory. 

	unsigned long long edges;
	
	char     attribute;  // is report? 
	int degree;

}; 


struct STE_dev4_compressed_matchset {
	int32_t ms[8]; // 8 * 32 = 256; local memory. 


	unsigned long long edges;
	
	char     attribute;  // is report? 
	//       attribute has 8 bit..  

	//       complete; complement;  

	//uint8_t start;
	//uint8_t end; 

	unsigned int start_end;

	int degree;	

}; 



struct STE_dev4_compressed_matchset_allcomplete {
	unsigned long long edges;
	
	char     attribute;  // is report? 

	unsigned int start_end;

	int degree;	
}; 





// Revised implementation. 20190121

struct STE_nodeinfo_new_imp {
	unsigned long long edges;
	
	unsigned int attribute : 8; 
	unsigned int start : 8;
	unsigned int end : 8;
	unsigned int degree : 8;
};



struct STE_nodeinfo_new_imp2 {
    unsigned long long edges;
    unsigned int attribute : 8;
    unsigned int degree : 8;
};



struct STE_matchset_new_imp {
	int32_t ms[8]; // 8 * 32 = 256; local memory. 
};


struct STE_nodeinfo_new_imp_withcg {
	unsigned long long edges;

	unsigned int attribute : 8; 
	unsigned int start :  8;
	unsigned int end :  8;
	unsigned int degree : 8;

	// cg_id ---> write position in gpu kernel. 
	uint16_t cg_id;
	uint16_t cg_of_to_edges[4];
};

struct matchset_t {
	int32_t m[8];
};




#endif

































































