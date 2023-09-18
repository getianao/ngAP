#pragma once


#include <string>
#include <bitset>



/**
 * helper functions
 * From VASim
 */
namespace VASim {
	void find_and_replace(std::string & source, std::string const & find, std::string const & replace);
	void setRange(std::bitset<256> &column, int start, int end, int value);
	void parseSymbolSet(std::bitset<256> &column, std::string symbol_set);
}



