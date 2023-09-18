#include "gpunfautils/abstract_gpunfa.h"

#include "gpunfautils/array2.h"



abstract_algorithm::abstract_algorithm(NFA *nfa) : 
nfa(nfa), 
padding_input_stream(4), 
max_cc_size_limit(-1),
read_input(true)
{

	block_size = 256;

	output_file = "reports.txt";
	
	this->report_on = true;
	
	// set default alphabet
	this->alphabet.clear();

	for (int i = 0; i < 256; i++) {
		this->alphabet.insert( (uint8_t) i );
	}
	
}


abstract_algorithm::~abstract_algorithm() {
}

void abstract_algorithm::set_alphabet(set<uint8_t> alphabet) {
	this->alphabet = alphabet;
}

const SymbolStream& abstract_algorithm::get_symbol_stream(int i) const {
	assert(i >= 0 && i < symbol_streams.size() );
	
	return symbol_streams[i];
}


void abstract_algorithm::add_symbol_stream(SymbolStream ss) {
	symbol_streams.push_back(ss);
}


void abstract_algorithm::set_block_size(int block_size) {
	this->block_size = block_size;

}


Array2<uint8_t> *abstract_algorithm::concat_input_streams_to_array2() {
  assert(symbol_streams.size() > 0);

  // cout << "padding_input_stream = " << this->padding_input_stream << endl;

  // for (int i = 0; i < this->symbol_streams.size(); i++) {
  //   symbol_streams[i].padding_to_base(this->padding_input_stream);
  // }

  int length = symbol_streams[0].get_length();

  for (auto ss : symbol_streams) {
    assert(length == ss.get_length());
  }

  auto input = new Array2<uint8_t>(symbol_streams.size() * length);

  int t = 0;

  for (auto ss : symbol_streams) {
    for (int p = 0; p < ss.get_length(); p++) {
      input->set(t++, ss.get_position(p));
    }
  }

  return input;
}






