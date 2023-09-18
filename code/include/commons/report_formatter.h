#ifndef REPORT_FORMATTER_H_
#define REPORT_FORMATTER_H_

#include <string>
#include <vector>

using std::vector;

using std::string;


class report {

public:
	 int offset;
	 string str_id;
	 int cc;
	 int input_stream_id;

	 bool operator < (const report &r ) const {
			if (input_stream_id < r.input_stream_id) {
				return true;
			} else if (input_stream_id == r.input_stream_id) {
				if (offset < r.offset) {
					return true;
				} else if (offset == r.offset) {
					if (cc < r.cc) {
						return true;
					} else if(str_id == r.str_id) {
						if (cc < r.cc) {
							return true;
						} else {
							return false;
						}
					} else {
						return false;
					}
				} else {
					return false;
				}
			} else {
				return false;
			}
		}

    bool operator == (const report &r) const {
    	return offset == r.offset && str_id == r.str_id && cc == r.cc && input_stream_id == r.input_stream_id;
    }
		report(){};

    report(int offset, string str_id, int cc, int input_stream_id);

};


class report_formatter {
public:
	report_formatter();

	void print_to_file(string filename, bool unique=true);

	void add_report(report rp);

	int size() const {
		return reports.size();
	}

// private:
	vector<report> reports;

};

#endif