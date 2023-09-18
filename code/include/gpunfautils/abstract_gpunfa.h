#ifndef ABSTRACT_NFA_PROCESSING_ALGORITHM
#define ABSTRACT_NFA_PROCESSING_ALGORITHM


#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <cassert>
#include <set>
#include "commons/NFA.h"
#include "array2.h"
#include "utils.h"
#include "common.h"
#include "commons/SymbolStream.h"

using std::map;
using std::vector;
using std::fill;
using std::cout;
using std::endl;
using std::pair;
using std::set;
using std::make_pair;



class abstract_algorithm {
public:
	explicit abstract_algorithm(NFA *nfa);
	virtual ~abstract_algorithm();

	virtual void preprocessing() {};
	virtual void launch_kernel() = 0;
	virtual void postprocessing() {};

	virtual void set_alphabet(set<uint8_t> alphabet);
	virtual const SymbolStream& get_symbol_stream(int i) const;
	virtual void add_symbol_stream(SymbolStream ss);

	virtual int get_num_streams() const {
		return symbol_streams.size();
	}

	void set_block_size(int block_size);

	void set_output_file(string output_filename) {
		this->output_file = output_filename;
	} 
	
	void set_output_buffer_size(unsigned long long int ob_size) {
		this->output_buffer_size = ob_size;
	}

	void set_NFA(NFA *nfa) {
		this->nfa = nfa;
	}

	// whether we want the algorithm to generate reports. 
	// If not, we can save time and space for the reports. 
	void turn_off_report() {
		this->report_on = false;
	}

	void turn_on_report() {
		this->report_on = true;
	}

	void set_report_off(bool &report_off,
											unsigned long long int result_capacity,
											long long int quick_result_number) {
			if (quick_result_number >= 0 &&
					result_capacity <= quick_result_number * 1.5) {
				if (!report_off) {
					printf("Warning: The number of results may exceed the "
									"capacity limit. "
									"Set report_off=true.\n");
					report_off = true;
				}
			}
			this->report_on = !report_off;
			if (this->report_on) {
					printf("Report on.\n");
			} else {
					printf("Report off.\n");
			}
	}

	Array2<uint8_t> *concat_input_streams_to_array2(); 
	
	void set_padding_input_stream(int pad) {
		this->padding_input_stream = pad;
	}


	void set_max_cc_size_limit(int max_cc_size_limit) {
		this->max_cc_size_limit = max_cc_size_limit;
	}
	
	void set_read_input(bool b) {
		this->read_input = b;
	}

	bool validation;
	NFA *nfa;
	vector<NFA *> ccs; 
	
protected:

	int max_cc_size_limit;

	int padding_input_stream;

	unsigned long long int  output_buffer_size;
	
	vector<SymbolStream> symbol_streams;

	set<uint8_t> alphabet;

	int block_size;

	bool report_on;       // decide whether generating reports. 

	string output_file;

	bool read_input;

};




#endif