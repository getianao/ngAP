#include "ppopp12.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <cassert>
#include <set>
#include <execution>

#include <queue>
#include <string>
#include <cuda.h>
#include "nfa_utils.h"
#include "NFA.h"
#include "node.h"
#include "array2.h"
#include "utils.h"
#include "common.h"
#include "ppopp12_kernels.h"
#include "report_formatter.h"
#include "abstract_gpunfa.h"
#include "SymbolStream.h"
#include <chrono>
#include "compatible_group_helper.h"
#include "commons/validate.h"
using namespace std::chrono;

using std::map;
using std::vector;
using std::fill;
using std::cout;
using std::endl;
using std::pair;
using std::set;
using std::make_pair;
using std::queue;
using std::string;




ppopp12::ppopp12(NFA *nfa) : 
abstract_algorithm(nfa), 
active_state_array_size(256),
num_segment_per_ss(1)
{
		nfa->mark_cc_id();
    old_ccs = nfa_utils::split_nfa_by_ccs(*nfa); 

    int nfasize = nfa->size();
    for (int i = 0; i < nfasize; i++) {
        Node *n  = nfa->get_node_by_int_id(i);

        if (n->is_start()) {
        	Node *cloned_start_node = new Node();
        	*cloned_start_node = *n;

        	cloned_start_node->str_id += "clone";
        	cloned_start_node->report = false;

        	nfa->addNode(cloned_start_node);
        	nfa->addEdge(cloned_start_node->str_id, n->str_id);

        	n->start = 0;
        }
    }
   	
   	no_cg = false;
}



ppopp12::~ppopp12() {
	/*delete state_start_position_tb;     
	delete num_state_tb;
	delete array_compatible_group;
	delete trans_table;
	delete states_status;
	delete initial_active_state_array;
	delete arr_input_streams;
	delete match_array;
    delete match_count;*/


}


void ppopp12::set_block_size(int blocksize) {
	this->block_size = blocksize;
}


void ppopp12::set_active_state_array_size(int active_state_array_size) {
	this->active_state_array_size = active_state_array_size;
}

void ppopp12::set_alphabet(set<uint8_t> alphabet) {
	this->alphabet = alphabet;
}



void ppopp12::group_nfas() {
	cout << "group_nfas" << endl;

	int sum_cg_groups = 0;
	double avg_cg_ratio = 0;

	num_nfa_chunk = 0;
	nfa_group_tb.clear();

	this->str_id_to_compatible_group.clear();
	this->num_compatible_groups_cc.clear();

	int capacity = active_state_array_size;
	int current_used = 0;

	int max_size_of_compatible_group = 0;
	
	CompatibleGroupHelper *ph = new CompatibleGroupHelper();

	for (int i = 0; i < ccs.size(); i++) {
		auto cc = ccs[i];

		ph->set_nfa(cc);
		ph->calc_incompatible_states(alphabet);
        ph->calc_compatible_groups();
		int num_compatible_groups = ph->num_compatible_grp();
		//cout << "num_compatible_groups(" << i << ")" << " = " << num_compatible_groups << endl;

		max_size_of_compatible_group = std::max(max_size_of_compatible_group, num_compatible_groups);

		sum_cg_groups += num_compatible_groups;
		avg_cg_ratio += (num_compatible_groups + 0.0) / cc->size();
		num_compatible_groups_cc[i] = num_compatible_groups;
		

		for (int k = 0; k < cc->size(); k++) {
			Node *node = cc->get_node_by_int_id(k);
			int cg_id = ph->get_compatible_grp_by_intid(k);
			assert(str_id_to_compatible_group.find(node->str_id) == str_id_to_compatible_group.end());
			str_id_to_compatible_group[node->str_id] = cg_id; 
		}

		//cout << num_compatible_groups << endl;
		if (num_compatible_groups > capacity) {
			cout << "num_compatible_groups > capacity" << endl;
			exit(-1);
		}

		if (current_used + num_compatible_groups > capacity) {
			num_nfa_chunk ++;
			current_used = num_compatible_groups;
			nfa_group_tb[num_nfa_chunk].push_back(i);

		} else {
			current_used += num_compatible_groups;
			nfa_group_tb[num_nfa_chunk].push_back(i);
		}
	}

	cout << "max_size_of_compatible_group = " << max_size_of_compatible_group << endl;

	num_nfa_chunk ++; // just to make it normal like [start, end)
	

	/*int ss = 0; 
	cout << "group nfas" << endl;
	for (int i = 0; i < num_nfa_chunk; i++) {
		cout << "tb i = " << i << "  numnfas = " << nfa_group_tb[i].size() << endl;
		for (auto it : nfa_group_tb[i]) {
			cout << it << "  ";
		}
		ss += nfa_group_tb[i].size();
		cout << endl;
	}

	cout << "ss = " << ss << "  " << ccs.size() << endl;
	
	assert(ss == ccs.size());*/
	

	nfa_in_tb.clear();
	for (int i = 0; i < num_nfa_chunk; i++) {
		vector<NFA *> tmp;
		for (auto it : nfa_group_tb[i]) {
			tmp.push_back(ccs[it]);
		}

		nfa_in_tb.push_back(nfa_utils::merge_nfas(tmp));
	}


	cout << "num_nfa_chunk = " << num_nfa_chunk << endl;
	cout << "avg_cg_ratio = " << std::fixed << avg_cg_ratio / ccs.size() << endl;

	delete ph;

}


void ppopp12::prepare_state_start_position_tb() {
	state_start_position_tb = new Array2<int> (num_nfa_chunk); // not for compatible groups. 
	num_state_tb = new Array2<int> (num_nfa_chunk);
	int acc = 0;

	for (int i = 0; i < num_nfa_chunk; i++) {
		//cout << "i = " << i << endl;
		state_start_position_tb->set(i, acc);
		acc += nfa_in_tb[i]->size();
		num_state_tb->set(i, nfa_in_tb[i]->size());
	}

	cout << "num_of_states = " << acc << endl;
	assert(acc == get_num_states_gpu());
}



void ppopp12::calc_str_id_to_compatible_group_per_block() {
	this->str_id_to_compatible_group_per_block.clear();

	for (int i = 0; i < nfa_group_tb.size(); i++) {
		auto ccids = nfa_group_tb[i];
		int t_acc = 0;
		for (int ccid = 0; ccid < ccids.size(); ccid ++) {
			auto cc = ccs[ccids[ccid]];
			for (int nodeid = 0; nodeid < cc->size(); nodeid++) { // one cc
				auto node = cc->get_node_by_int_id(nodeid);
				int cg_id = str_id_to_compatible_group[node->str_id];
				str_id_to_compatible_group_per_block[node->str_id] = cg_id + t_acc;
			}

			t_acc += num_compatible_groups_cc[ccids[ccid]];
		}
	}
}
	
void ppopp12::prepare_compatible_grps() {
	cout << "prepare_compatible_grps" << endl;
	array_compatible_group = new Array2<int> (get_num_states_gpu());
	array_compatible_group->fill(-1);

	int cur = 0;
	for (int i = 0; i < num_nfa_chunk; i++) {
		auto nfa_chunk = nfa_in_tb[i];
		for (int nodeid = 0; nodeid < nfa_chunk->size(); nodeid++ ) {
			auto node = nfa_chunk->get_node_by_int_id(nodeid);
			assert(str_id_to_compatible_group_per_block.find(node->str_id) != str_id_to_compatible_group_per_block.end());
			int cgid = str_id_to_compatible_group_per_block[node->str_id];
			array_compatible_group->set(cur++ , cgid);
		}
	}

	//array_compatible_group->print();
}


int ppopp12::get_num_states_gpu() const {
	int acc = 0;
	assert(nfa_in_tb.size() == num_nfa_chunk);
	for (int i = 0; i < nfa_in_tb.size(); i++) {
		acc += nfa_in_tb[i]->size();
	}

	return acc;
}


void ppopp12::prepare_states_status() {
	cout << "prepare_states_status" << " get_num_states_gpu() = " << get_num_states_gpu() << endl;

	states_status = new Array2<int8_t> (get_num_states_gpu());
	
	states_status->clear_to_zero();

	int t = 0;
	for (int i = 0; i < num_nfa_chunk; i++ ) {
		auto nfa_chunk = nfa_in_tb[i];
		for (int node_id = 0; node_id < nfa_chunk->size(); node_id ++) {
			Node *n = nfa_chunk->get_node_by_int_id(node_id);
			int8_t val = 0;
			if (n->is_start_always_enabled()) {
				val = val | (1 << 1);
			}

			if (n->is_report()) {
				val = val | 1;
			}

			//cout << "t = " << t << endl;
			states_status->set(t++, val);
		}
	}

	//cout << "states_status " << endl;

	/*for (int i = 0; i < states_status->size(); i++) {
		cout << (int) states_status->get(i) << " ";
	}
	cout << endl;*/

}



void ppopp12::prepare_initial_active_state_array() {
	initial_active_state_array = new Array2<int> (active_state_array_size * num_nfa_chunk);
	initial_active_state_array->fill(-1);

	for (int i = 0; i < num_nfa_chunk; i++ ) {
		auto nfa_chunk = nfa_in_tb[i];
		for (int node_id = 0; node_id < nfa_chunk->size(); node_id ++) {
			Node *n = nfa_chunk->get_node_by_int_id(node_id);
			assert(node_id == n->sid);
			if (n->is_start()) {
				initial_active_state_array->set(i * active_state_array_size + array_compatible_group->get(state_start_position_tb->get(i) + node_id),  node_id);	
			}
		}
	}
}


void ppopp12::prepare_transition_table() {
	cout << "prepare_transition_table " << endl;

	cout << "get_num_states_gpu() = " << get_num_states_gpu() << endl;

	trans_table = new Array2<int4> (get_num_states_gpu() * 256LL);

	int4 initval;
	initval.x = -1;
	initval.y = -1;
	initval.z = -1;
	initval.w = -1;

	trans_table->fill(initval);

	int current_state = 0;
	for (int i = 0; i < num_nfa_chunk; i++) {
		auto nfa_chunk = nfa_in_tb[i];
		
		for (int nodeid = 0; nodeid < nfa_chunk->size(); nodeid++) {
			Node *n = nfa_chunk->get_node_by_int_id(nodeid);
			auto adjs = nfa_chunk->get_adj(n->str_id);

			for (auto symbol : alphabet) {
				//cout << "symbol = " << (int) symbol << endl;
				vector<int> vec_to_push_transition_table;
			
				for (auto adj : adjs) {
					Node *to_node = nfa_chunk->get_node_by_str_id(adj);
					if (to_node->match2(symbol)) {
						vec_to_push_transition_table.push_back(to_node->sid);
					}
				}

				//cout << "vec_to_push_transition_table.size() = " << vec_to_push_transition_table.size() << endl;
				assert(vec_to_push_transition_table.size() <= 4);
				int vec_size = vec_to_push_transition_table.size();
				for (int q = vec_size; q < 4; q ++) {
					vec_to_push_transition_table.push_back(-1);
				}
				assert(vec_to_push_transition_table.size() == 4);

				int4 tmp;
				tmp.x = vec_to_push_transition_table[0];
				tmp.y = vec_to_push_transition_table[1];
				tmp.z = vec_to_push_transition_table[2];
				tmp.w = vec_to_push_transition_table[3];

				trans_table->set(current_state * 256 + ((int) symbol), tmp);
			}

			current_state ++;
		}
	}

	assert(current_state == get_num_states_gpu());

	//trans_table->print();
	 
	/*
	int num_states_gpu = get_num_states_gpu();
	for (int s = 0; s < num_states_gpu; s++) {
		cout << s << endl;
		for (int c = 97; c < 97 + 26; c++) {
			//cout << "s * 256 + c = " << s * 256 + c << endl; 
			auto mm = trans_table->get(s * 256 + c);


			cout << s * 256 + c << "  " << (char) c << "  " << mm.x << " " << mm.y << " " << mm.z << " " << mm.w << endl;
		}
	}*/
	

/*
	for (int i = 0; i < nfa.get_num_state(); i++) {
		Node *n = nfa.get_node_by_int_id(i);
		assert(nfa.has_node(n->str_id));
		auto adjs = nfa.get_adj(n->str_id);
		for (auto it : adjs) {
			Node *to_node = nfa.get_node_by_str_id(it);
			auto to_intid = to_node->sid;
			auto trans1 = make_pair(i, to_intid);
			num_transitions++;

			for (auto symbol : alphabet) {
				if (n->match2(symbol)) {
					symbol_trans[symbol].push_back(trans1);
					len[symbol] ++;
					num_transitions_in_alphabet_based_tables++;
				}
			}
		}
	}
*/

}
	

void ppopp12::prepare_input_streams() {
	int length = symbol_streams[0].get_length();

	for (auto ss : symbol_streams) {
		assert(length = ss.get_length());
	}// currently we require the length of each symbol stream with equal length

	arr_input_streams = new Array2<uint8_t> (symbol_streams.size() * length);

	int t = 0;

	for (auto ss: symbol_streams) {
		for (int p = 0; p < ss.get_length(); p++) {
			arr_input_streams->set(t++, ss.get_position(p) );
		}
	}	
	
}


void ppopp12::prepare_outputs() {
	  match_array = new Array2<match_entry> (this->output_buffer_size);
    match_count = new Array2<unsigned long long int > (1);

    match_array->clear_to_zero();
    match_count->clear_to_zero();
}


void ppopp12::preprocessing() {
	nfa->mark_cc_id();
	ccs = nfa_utils::split_nfa_by_ccs(*nfa);
	
	for (auto cc : ccs) {
        cc->calc_scc();
        cc->topo_sort();
    }
	
	nfa_utils::limit_out_degree_on_ccs(this->ccs, 4, opt->quit_degree, opt->remove_degree);

	if (this->max_cc_size_limit != -1) {
		vector<NFA* > tmp_ccs;
		for (int i = 0; i < this->ccs.size(); i++) {
			if (ccs[i]->size() <= max_cc_size_limit) {
				tmp_ccs.push_back(ccs[i]);
			} else {
				cout << "remove_ccid = " << i << " " << ccs[i]->get_node_by_int_id(0)->str_id << endl;
				delete ccs[i];
			}
		}

		this->ccs = tmp_ccs;
	}

        auto cg_calc_start = high_resolution_clock::now();	
	group_nfas(); //calculate compatible groups here. 
	auto cg_calc_end = high_resolution_clock::now();

        auto cg_calc_duration = duration_cast<seconds>(cg_calc_end - cg_calc_start);
        cout << "time_calc_cg_sec = " << cg_calc_duration.count() << endl;


        prepare_state_start_position_tb();
	calc_str_id_to_compatible_group_per_block();


	prepare_compatible_grps();

	prepare_transition_table();
	prepare_states_status();
	prepare_initial_active_state_array();
	prepare_input_streams();
	prepare_outputs();



}


void ppopp12::launch_kernel() {
	//nfa 
	state_start_position_tb->copy_to_device();     
	array_compatible_group->copy_to_device();

	trans_table->copy_to_device();
	states_status->copy_to_device();
	initial_active_state_array->copy_to_device();

	//output
	//match_array->copy_to_device();
	match_count->copy_to_device();

	//input
	arr_input_streams->copy_to_device();

	// X num nfa chunk
	// Y num segments
	// Z num input streams

	dim3 blocksPerGrid(num_nfa_chunk, symbol_streams.size(), this->num_segment_per_ss); 

	dim3 threadsPerBlock(block_size, 1);
	
	cout << "symbol_streams.size() == " << symbol_streams.size() << endl;  

	cout << "num_of_block = " << num_nfa_chunk * symbol_streams.size() * this->num_segment_per_ss << endl;
	cout << "num_of_tb_x = " << num_nfa_chunk << endl; 

	int shared_memory_size = active_state_array_size * 2 * sizeof(int);

	cudaDeviceSynchronize();

	cudaEvent_t start, stop;
  	float elapsedTime;

  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
  	cudaEventRecord(start,0);

	ppopp12_kernel<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>> (
		trans_table->get_dev(),
		trans_table->size(),

		state_start_position_tb->get_dev(),
		array_compatible_group->get_dev(),
		initial_active_state_array->get_dev(),

		active_state_array_size, // currently it is the same as block size

		states_status->get_dev(), // 00    is always enabled; is output;  

		arr_input_streams->get_dev(),
		symbol_streams[0].size(),

		//output processing
		match_array->get_dev(),
		this->output_buffer_size,
		match_count->get_dev(),
		this->report_on
	);


 	auto cudaError = cudaGetLastError();
  	if(cudaError != cudaSuccess)
  	{
   		 printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  	}

  	cudaError = cudaDeviceSynchronize();
  	if(cudaError != cudaSuccess)
  	{
   		 printf(" cudaDeviceSynchronize returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  	}
  	
 	cudaEventRecord(stop,0);
 	cudaEventSynchronize(stop);

 	match_count->copy_back();
  	match_array->copy_back();

 	cudaEventElapsedTime(&elapsedTime, start,stop);
 	printf("Elapsed time : %f ms\n" ,elapsedTime);

 	
 	float sec = elapsedTime / 1000.0;
 	cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * symbol_streams.size())  / 1000000.0 / sec  << endl;


  	cout << "count_of_match = " << match_count->get(0) << endl;

  	report_formatter rf;
		unsigned long long int  mc = match_count->get(0);

		if(this->report_on) {
		rf.reports.resize(mc);
		std::for_each(std::execution::par_unseq, std::begin(rf.reports), std::end(rf.reports), [&](report& r) {
			u_int32_t i = &r - &rf.reports[0];
			assert(i< mc);
			match_entry mp = match_array->get(i);
  		auto sid = mp.state_id;
  		Node *n = nfa_in_tb[mp.cc_id]->get_node_by_int_id(sid);
  		//cout << "report generated = " << n->str_id << " at " << mp.symbol_offset << endl;
  		report rp(mp.symbol_offset, n->original_id, mp.cc_id, mp.stream_id);
  		rf.reports[i]=rp;
			//  std::cout << i << "\n";
		});

	rf.print_to_file(output_file);
		}


		if (validation) {
			if (!this->report_on) {
					int dupnum = opt->duplicate_input_stream > 1
															? opt->duplicate_input_stream
															: 1;
					if (opt->quick_validation >= 0 &&
							opt->quick_validation * dupnum <= mc) {
						if (opt->quick_validation * dupnum == mc) {
            	tge_log("Quick Validation PASS! (report off, perfect)", BOLDGREEN);
						} else {
							tge_log("Quick Validation PASS! (report off, not perfect)",
											BOLDGREEN);
						}
					} else {
						tge_log("Quick Validation FAILED! (report off)", BOLDRED);
					}
			} else {
				if (opt->quick_validation >= 0) {
						int dupnum = opt->duplicate_input_stream > 1
															? opt->duplicate_input_stream
															: 1;
						if (opt->quick_validation * dupnum == rf.size()) {
										tge_log("Quick Validation PASS!", BOLDGREEN);
						} else {
										tge_log("Quick Validation FAILED!", BOLDRED);
						}
				} else {
					std::vector<uint64_t> ref_results, ref_db_results;
					int nfa_num = 10;

					auto old_grouped_nfas = nfa_utils::group_nfas_by_num(nfa_num, old_ccs); 
					std::vector<Graph *> gs;
					for (auto nfa : old_grouped_nfas) {
							Graph *g = new Graph();
							g->ReadNFA(nfa);
							g->copyToDevice();
							gs.push_back(g);
					}
					auto compareResult = [](uint64_t r1, uint64_t r2) -> bool {
							uint32_t input_1, state_1, input_2, state_2;
							input_1 = (uint32_t)(0xffffffff & r1);
							state_1 = (uint32_t)(r1 >> 32);
							input_2 = (uint32_t)(0xffffffff & r2);
							state_2 = (uint32_t)(r2 >> 32);
							// printf("{%u, %u}, ", state, input);
							if (input_1 == input_2)
									return state_1 < state_2;
							else
									return input_1 < input_2;
					};

					GroupCsr gcsr;
					gcsr.init(gs);
					automata_utils::automataGroupsReference(
									gs, arr_input_streams->get_host(),
									get_num_streams(), symbol_streams[0].get_length(),
									&ref_results, &ref_db_results, 0, gcsr);

					printf("\n############ Validate result "
													"############\n ");
					// automata_utils::automataValidation(&results, &ref_results, true);
					std::sort(ref_results.begin(), ref_results.end(), compareResult);
			ref_results.erase(unique(ref_results.begin(), ref_results.end()),
											ref_results.end());
					printf("ref_results size: %zu\n", ref_results.size());

					if (rf.size() == ref_results.size()) {
							tge_log("Validation PASS!", BOLDGREEN);
					} else {
							tge_log("Validation FAILED!", BOLDRED);
					}
				}
			}
			
    } 



}




void ppopp12::launch_kernel_readinputchunk() {
	cout << "launch_kernel_readinputchunk " << endl;
	//nfa 
	state_start_position_tb->copy_to_device();     
	array_compatible_group->copy_to_device();

	trans_table->copy_to_device();
	states_status->copy_to_device();
	initial_active_state_array->copy_to_device();

	//output
	match_array->copy_to_device();
	match_count->copy_to_device();

	//input
	arr_input_streams->copy_to_device();

	// X num nfa chunk
	// Y num segments
	// Z num input streams

	dim3 blocksPerGrid(num_nfa_chunk, symbol_streams.size(), this->num_segment_per_ss); 

	dim3 threadsPerBlock(block_size, 1);
	
	cout << "symbol_streams.size() == " << symbol_streams.size() << endl;  

	cout << "num_of_block = " << num_nfa_chunk * symbol_streams.size() * this->num_segment_per_ss << endl;
	cout << "num_of_tb_x = " << num_nfa_chunk << endl; 

	int shared_memory_size = active_state_array_size * 2 * sizeof(int);
	shared_memory_size += this->block_size * sizeof(uint8_t);

	cudaDeviceSynchronize();

	cudaEvent_t start, stop;
  	float elapsedTime;

  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
  	cudaEventRecord(start,0);

	ppopp12_kernel_shrreadchunk<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>> (
		trans_table->get_dev(),
		trans_table->size(),

		state_start_position_tb->get_dev(),
		array_compatible_group->get_dev(),
		initial_active_state_array->get_dev(),

		active_state_array_size, // currently it is the same as block size

		states_status->get_dev(), // 00    is always enabled; is output;  

		arr_input_streams->get_dev(),
		symbol_streams[0].size(),

		//output processing
		match_array->get_dev(),
		this->output_buffer_size,
		match_count->get_dev(),
		this->report_on
	);


 	auto cudaError = cudaGetLastError();
  	if(cudaError != cudaSuccess)
  	{
   		 printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  	}

  	cudaError = cudaDeviceSynchronize();
  	if(cudaError != cudaSuccess)
  	{
   		 printf(" cudaDeviceSynchronize returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  	}
  	
 	cudaEventRecord(stop,0);
 	cudaEventSynchronize(stop);

 	match_count->copy_back();
  	match_array->copy_back();

 	cudaEventElapsedTime(&elapsedTime, start,stop);
 	printf("Elapsed time : %f ms\n" ,elapsedTime);

 	
 	float sec = elapsedTime / 1000.0;
 	cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * symbol_streams.size())  / 1000000.0 / sec  << endl;


  	cout << "count_of_match = " << match_count->get(0) << endl;

  	report_formatter rf;
		if (this->report_on) {
							for (unsigned long long int i = 0; i < match_count->get(0);
									i++) {
														match_entry mp = match_array->get(i);
														auto sid = mp.state_id;
														Node *n =
																nfa_in_tb[mp.cc_id]->get_node_by_int_id(
																		sid);
														// cout << "mp.cc_id = " << mp.cc_id << endl;
														report rp(mp.symbol_offset, n->original_id,
																			mp.cc_id, mp.stream_id);
														rf.add_report(rp);
							}

							rf.print_to_file(output_file);
		}

		cout << "FINISHED" << endl;


}


// for debug
NFA* ppopp12::select_one_nfa_by_id(string str_id) {
	for (int i = 0; i < nfa_in_tb.size(); i++ ) {
		if (nfa_in_tb[i]->has_node(str_id)) {
			return nfa_in_tb[i];
		}
	} 

	cout << "not found" << endl;
	return NULL;


}
