#ifndef VALIDATE_H_
#define VALIDATE_H_

#include "graph.h"
#include "group_graph.h"

namespace automata_utils {
	  void automataGroupsReference(std::vector<Graph *> &gs,
																uint8_t *input_str, int num_seg,
																int input_length,
																std::vector<uint64_t> *results,
																std::vector<uint64_t> *db_results,
																int debug_iter, GroupCsr gcsr);
		void automataReference(Graph &g, uint8_t *input_str, int num_seg,
			int input_length,
			std::vector<uint64_t> *results,
			std::vector<uint64_t> *db_results,
			int debug_iter, Csr csr);

		bool automataValidation(std::vector<uint64_t> *results,
																std::vector<uint64_t> *ref_results,
																bool ifPrintBoth);
}

#endif