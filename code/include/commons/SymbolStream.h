/*
 * SymbolStream.h
 *
 *  Created on: May 1, 2018
 *      Author: hyliu
 */

#ifndef SYMBOLSTREAM_H_
#define SYMBOLSTREAM_H_


#include <string>
#include <set>
#include <vector>

using std::string;
using std::set;
using std::vector;


class SymbolStream {
public:
	SymbolStream();

	virtual ~SymbolStream();
	void readFromFile(string filename);

	const set<uint8_t>& calc_alphabet();
	uint8_t get_position(int pos) const;
	void set_position(int pos, uint8_t c);

	void push_back(uint8_t c) {
		input.push_back(c);
	}

	void concat(SymbolStream &s) {
		input.insert(input.end(), s.input.begin(), s.input.end());
	}

	int get_length() const;

	int size() const {
		return input.size();
	}

	SymbolStream slice(int start, int len) const;

	void padding_to_base(int base);

private:
	vector<uint8_t> input;
	string fromFile;
	set<uint8_t> alphabet;

};



#endif /* SYMBOLSTREAM_H_ */
