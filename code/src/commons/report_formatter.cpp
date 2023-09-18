#include "report_formatter.h"

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <execution>

using std::cout;
using std::endl;
using std::vector;
using std::string;

report::report(int offset, string str_id, int cc, int input_stream_id):
offset(offset),
str_id(str_id),
cc(cc),
input_stream_id(input_stream_id)
{
}

report_formatter::report_formatter() {}

void report_formatter::add_report(report rp) {
	this->reports.push_back(rp);
}


void report_formatter::print_to_file(string filename, bool unique1) {
	cout << "report_fomatter_print_to_file_num_report = " << reports.size() << endl;
	std::sort(std::execution::par_unseq, reports.begin(), reports.end());

	if (unique1) {
		reports.erase( std::unique(std::execution::par_unseq, reports.begin(), reports.end() ), reports.end() );
		cout << "report_fomatter_print_to_file_unique_num_report = " << reports.size() << endl;	
	}

	// std::ofstream out(filename);
	// cout<<filename<<"\n";

	// for (auto it : reports) {
	// 	cout << it.offset << "\t" << it.str_id << endl; 
	// }


	// out.close();

}