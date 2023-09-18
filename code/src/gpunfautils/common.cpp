#include "common.h"

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


using std::string;



std::ostream& operator<<(std::ostream& os, const match_pair &obj) {
    os << obj.symbol_offset << " " << obj.state_id << ' ';
    return os;
}

