#include "one_byte_at_a_time.h"
#include <gpunfautils/utils.h>
#include "one_byte_a_time_kernels.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <cassert>
#include <set>
#include "commons/NFA.h"
#include <gpunfautils/array2.h>
#include <gpunfautils/common.h>
#include "commons/SymbolStream.h"
#include <gpunfautils/abstract_gpunfa.h>
#include <unordered_map>
#include "commons/report_formatter.h"
#include <cmath>
#include "commons/compatible_group_helper.h"
#include "commons/nfa_utils.h"
#include <random>
#include <cuda.h>
#include "commons/validate.h"
#include <execution>

using std::make_pair;


one_byte_at_a_time::one_byte_at_a_time(NFA *nfa): 
	abstract_algorithm(nfa),
	hot_limit_by_bfs_layer(-1),
	active_queue_size(128),
	record_cold_warp_active_array(false),
	history_queue_capacity(30000),
	max_indegree_of_cold_states(9999),
	profile_length(1000),
	cold_thres(0.0),
    hot_stage_only(false),
    remap_input_stream(false),
    remap_node_id(remap_node_type::NONE)
{
    nfa->mark_cc_id();
    old_ccs = nfa_utils::split_nfa_by_ccs(*nfa); 

}


one_byte_at_a_time::~one_byte_at_a_time() {

}


void one_byte_at_a_time::preprocessing_enable_active() {
	nfa->mark_cc_id();

	this->ccs = nfa_utils::split_nfa_by_ccs(*nfa);

	nfa_utils::limit_out_degree_on_ccs(this->ccs, 4, opt->quit_degree, opt->remove_degree);
	
	for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
		auto cc = ccs[cc_id];		
		cc->calc_scc();
		cc->topo_sort();
	}

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

    // Do packing

    if (packing == 1) {
        std::random_shuffle(this->ccs.begin(), this->ccs.end());

    } else if (packing == 2) {
        cout << "packing from a activation ratio file = " << packing_filename << endl;
        auto activation_ratio_per_cc = nfa_utils::read_freq_map(packing_filename);
        vector<pair<string, double> > act_ratio_per_cc_vec;
        for (auto it : activation_ratio_per_cc) {
            act_ratio_per_cc_vec.push_back(make_pair(it.first, it.second));
        }

        std::sort(act_ratio_per_cc_vec.begin(), act_ratio_per_cc_vec.end(),
                  [] (const  pair<string, double>  &a, const pair<string, double>  &b) -> bool {
                      return a.second - b.second > 0;
                  });

        vector<string> rstate_id_ordered;
        for (auto it : act_ratio_per_cc_vec) {
            rstate_id_ordered.push_back(it.first);
        }

        auto cc_ids = nfa_utils::get_cc_ids_from_state_id(ccs, rstate_id_ordered);

        vector<NFA* > tmp_ccs;
        tmp_ccs.resize(ccs.size());
        for (int i = 0; i < ccs.size(); i++) {
            tmp_ccs[i] = ccs[cc_ids[i]];
        }

        this->ccs = tmp_ccs;

    } else if (packing == 3) {

        cout << "packing from a activation ratio file = " << packing_filename << endl;
        cout << "dividing by the cc sizes. " << endl;

        auto activation_ratio_per_cc = nfa_utils::read_freq_map(packing_filename);

        vector<string> rstate_id;
        for (auto it : activation_ratio_per_cc) {
            rstate_id.push_back(it.first);
        }

        auto cc_ids = nfa_utils::get_cc_ids_from_state_id(ccs, rstate_id);

        vector<pair<string, double> > act_ratio_per_cc_vec;
        int t = 0;
        for (auto it : activation_ratio_per_cc) {
            act_ratio_per_cc_vec.push_back(make_pair(it.first, it.second / ccs[cc_ids[t++]]->size()));
        }

        std::sort(act_ratio_per_cc_vec.begin(), act_ratio_per_cc_vec.end(),
                  [] (const  pair<string, double>  &a, const pair<string, double>  &b) -> bool {
                      return a.second - b.second > 0;
                  });

        vector<string> rstate_id_ordered;
        for (auto it : act_ratio_per_cc_vec) {
            rstate_id_ordered.push_back(it.first);
        }

        cc_ids = nfa_utils::get_cc_ids_from_state_id(ccs, rstate_id_ordered);

        vector<NFA* > tmp_ccs;
        tmp_ccs.resize(ccs.size());
        for (int i = 0; i < ccs.size(); i++) {
            tmp_ccs[i] = ccs[cc_ids[i]];
        }

        this->ccs = tmp_ccs;
    }
    this->old_ccs = this->ccs;
	cout << "end one_byte_at_a_time::preprocessing_enable_active()  " << endl;
}


void one_byte_at_a_time::check_grouped_nfa_sizes() {
	for (int i = 0; i < this->grouped_nfas.size(); i++) {
		//cout << "group_nfas(" << i << ") = " << this->grouped_nfas[i]->size() << endl;
		if (this->grouped_nfas[i]->size() > this->block_size) {
			cout << "the NFA size could not > block_size"; 
			assert(this->grouped_nfas[i]->size() <= this->block_size);
		}
	}
}

void one_byte_at_a_time::preprocessing_active_active() {
	nfa->mark_cc_id();
	this->ccs = nfa_utils::split_nfa_by_ccs(*nfa);

	cout << "ccs size = " << ccs.size() << endl;

	nfa_utils::add_fake_start_node_for_ccs(this->ccs);

	nfa_utils::limit_out_degree_on_ccs(this->ccs, 4, opt->quit_degree, opt->remove_degree);  
	
	for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
		auto cc = ccs[cc_id];		
		cc->calc_scc();
		cc->topo_sort();
	}

	// add max_cc_size_limit --- 20190224 
	if (this->max_cc_size_limit != -1) {
	    cout << "filter out large CC --- max cc size = " << max_cc_size_limit << endl;
		vector<NFA* > tmp_ccs;
		for (int i = 0; i < this->ccs.size(); i++) {
			if (ccs[i]->size() <= max_cc_size_limit) {
				tmp_ccs.push_back(ccs[i]);
			} else {
				delete ccs[i];
			}
		}

		this->ccs = tmp_ccs;
	}


	this->grouped_nfas = nfa_utils::group_nfas(block_size, ccs);

	for (int i = 0; i < this->grouped_nfas.size(); i++) {
		//cout << "group_nfas(" << i << ") = " << this->grouped_nfas[i]->size() << endl;
		assert(grouped_nfas[i]->size() <= block_size);
	}
	
	//for (int cc_id = 0; cc_id < ccs.size(); cc_id++) {
	//	auto cc = ccs[cc_id];
		
		//cout << "dag_cc(" << cc_id << ") = " << cc->get_dag() << endl;
		//cout << "cc_size_limit4(" << cc_id << ") = " << cc->size() << endl;
		//cout << "has_self_loop_plus_large_matchset(" << cc_id << ") = " << cc->has_self_loop_plus_large_matchset() << endl;

	//}

}



void one_byte_at_a_time::remap_intid_of_nodes(remap_node_type tp) {
	if (tp == remap_node_type::NONE) {
	    cout << "remap nodes currently disabled. " << endl;
		return;
	}

	if (tp == remap_node_type::REPORT) {
		cout << "remap_node_type::REPORT" << endl;
		for (int i = 0; i < grouped_nfas.size(); i++) {
			auto res_nfa = nfa_utils::remap_intid_of_nfa(grouped_nfas[i], [](const Node *a, const Node *b) {
				 int rr = ( (int) a->is_report() ) - ((int) b->is_report());   
				 return rr > 0;
    		});

			delete grouped_nfas[i];
			grouped_nfas[i] = res_nfa;
		}
		
	} else if (tp == remap_node_type::TOPO_ORDER) {
		cout << "remap_node_type::TOPO_ORDER" << endl;
		for (int i = 0; i < grouped_nfas.size(); i++) {
			auto res_nfa = nfa_utils::remap_intid_of_nfa(grouped_nfas[i], [](const Node *a, const Node *b) {
				 if (a->topo_order < b->topo_order) {
				 	return true;
				 } else if (a->topo_order == b->topo_order) {
				 	return a->sid - b->sid < 0;
				 } else {
				 	return false;
				 }
    		});

			delete grouped_nfas[i];
			grouped_nfas[i] = res_nfa;
		}


	} else if (tp == remap_node_type::OUTDEGREE) {
		cout << "remap_node_type::OUTDEGREE" << endl;
		for (int i = 0; i < grouped_nfas.size(); i++) {

			auto res_nfa = nfa_utils::remap_intid_of_nfa(grouped_nfas[i], [&](const Node *a, const Node *b) {
				 int out1 = grouped_nfas[i]->get_outdegree_of_node(a->str_id);
				 int out2 = grouped_nfas[i]->get_outdegree_of_node(b->str_id);

				 return out1 - out2 <= 0;
    		});

			delete grouped_nfas[i];
			grouped_nfas[i] = res_nfa;
		}

	} else if (tp == remap_node_type::BFS_LAYER) {
		cout << "remap_node_type::BFS_LAYER" << endl;
		for (int i = 0; i < grouped_nfas.size(); i++) {
			auto res_nfa = nfa_utils::remap_intid_of_nfa(grouped_nfas[i], [](const Node *a, const Node *b) {
				 if (a->bfs_layer < b->bfs_layer) {
				 	return true;
				 } else if (a->bfs_layer == b->bfs_layer) {
				 	return a->sid - b->sid < 0;
				 } else {
				 	return false;
				 }
    		});

			delete grouped_nfas[i];
			grouped_nfas[i] = res_nfa;
		}

	} else if (tp == remap_node_type::COMPLETE) {
		cout << "remap_node_type::COMPLETE" << endl;

		for (int i = 0; i < grouped_nfas.size(); i++) {
			auto res_nfa = nfa_utils::remap_intid_of_nfa(grouped_nfas[i], [](const Node *a, const Node *b) {
				 if (a->complete - b->complete > 0) {
				 	return true;
				 } else if (a->complete == b->complete) {
				 	return a->match_set_range - b->match_set_range < 0;
				 } else {
				 	return false;
				 }
	    	});

			delete grouped_nfas[i];
			grouped_nfas[i] = res_nfa;
		}

		for (int i = 0; i < grouped_nfas.size(); i++) {
			cout << grouped_nfas[i]->size() << endl;
			for (int j = 0; j < grouped_nfas[i]->size(); j++) {
				auto node = grouped_nfas[i]->get_node_by_int_id(j);
				cout << node->original_id << endl;
			}

		}

	} else if (tp == remap_node_type::COMPLETE_AND_TOP) {
		cout << "remap_node_type::COMPLETE_AND_TOP" << endl;

		for (int i = 0; i < grouped_nfas.size(); i++) {
			auto res_nfa = nfa_utils::remap_intid_of_nfa(grouped_nfas[i], [](const Node *a, const Node *b) {
				 if (a->complete - b->complete > 0) {
				 	return true;
				 } else if (a->complete == b->complete) {
				 	return a->topo_order - b->topo_order < 0;
				 } else {
				 	return false;
				 }
	    	});

			delete grouped_nfas[i];
			grouped_nfas[i] = res_nfa;
		}



	} else if (tp == remap_node_type::COMPLETE_AND_BFS) {
		cout << "remap_node_type::COMPLETE_AND_BFS" << endl;

		for (int i = 0; i < grouped_nfas.size(); i++) {
			auto res_nfa = nfa_utils::remap_intid_of_nfa(grouped_nfas[i], [](const Node *a, const Node *b) {
				 if (a->complete - b->complete > 0) {
				 	return true;
				 } else if (a->complete == b->complete) {
				 	return a->bfs_layer - b->bfs_layer < 0;
				 } else {
				 	return false;
				 }
	    	});

			delete grouped_nfas[i];
			grouped_nfas[i] = res_nfa;
		}
	}
}


void one_byte_at_a_time::remap_intid_of_nodes_with_boudary(remap_node_type tp, vector<NFA *> &grouped_nfas, const vector<int> &boundaries) {
	

	if (tp == remap_node_type::TOPO_ORDER) {
		cout << "remap_node_type::TOPO_ORDER" << endl;
		for (int i = 0; i < grouped_nfas.size(); i++) {
			auto res_nfa = nfa_utils::remap_intid_of_nfa1(grouped_nfas[i], [](const Node *a, const Node *b) {
				 if (a->topo_order < b->topo_order) {
				 	return true;
				 } else if (a->topo_order == b->topo_order) {
				 	return a->sid - b->sid < 0;
				 } else {
				 	return false;
				 }
    		}, boundaries[i]);

			delete grouped_nfas[i];
			grouped_nfas[i] = res_nfa;
		}


	} else if (tp == remap_node_type::OUTDEGREE) {
		cout << "remap_node_type::OUTDEGREE" << endl;
		for (int i = 0; i < grouped_nfas.size(); i++) {

			auto res_nfa = nfa_utils::remap_intid_of_nfa1(grouped_nfas[i], [&](const Node *a, const Node *b) {
				 int out1 = grouped_nfas[i]->get_outdegree_of_node(a->str_id);
				 int out2 = grouped_nfas[i]->get_outdegree_of_node(b->str_id);

				 return out1 - out2 <= 0;
    		}, boundaries[i]);

			delete grouped_nfas[i];
			grouped_nfas[i] = res_nfa;
		}

	} else if (tp == remap_node_type::BFS_LAYER) {
		cout << "remap_node_type::BFS_LAYER" << endl;
		for (int i = 0; i < grouped_nfas.size(); i++) {
			auto res_nfa = nfa_utils::remap_intid_of_nfa1(grouped_nfas[i], [](const Node *a, const Node *b) {
				 if (a->bfs_layer < b->bfs_layer) {
				 	return true;
				 } else if (a->bfs_layer == b->bfs_layer) {
				 	return a->sid - b->sid < 0;
				 } else {
				 	return false;
				 }
    		}, boundaries[i]);

			delete grouped_nfas[i];
			grouped_nfas[i] = res_nfa;
		}
	}

}


void one_byte_at_a_time::prepare_output_buffer() {
	this->real_output_array = new Array2<match_entry> (this->output_buffer_size );
	this->tail_of_real_output_array = new Array2<unsigned long long int> (1);
	this->tail_of_real_output_array->fill(0);
}



void one_byte_at_a_time::launch_kernel() {

	this->preprocessing_enable_active(); // already limit 4 degree in this function. 

	this->grouped_nfas = nfa_utils::group_nfas(block_size, ccs);

	this->check_grouped_nfa_sizes();


	prepare_output_buffer();
	tail_of_real_output_array->copy_to_device();

	auto node_lists = nfa_utils::create_nodelist_for_nfa_groups(this->grouped_nfas, block_size, nfa_utils::create_list_of_STE_dev);
	
	node_lists->copy_to_device();


	// 
	Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
	input_stream->copy_to_device();

	cout << "input stream (concatenated) size = " << input_stream->size() << endl;
	//

	auto nfa_sizes = nfa_utils::get_nfa_size_array2(this->grouped_nfas);
  	nfa_sizes->copy_to_device();

  	dim3 blocksPerGrid1(grouped_nfas.size(), symbol_streams.size(), 1);
  	dim3 threadsPerBlock1(this->block_size, 1, 1);

  	cout << "num_of_block = " << grouped_nfas.size() * symbol_streams.size() * 1 << endl;

  	int smemsize = sizeof(bool) * block_size * 2;
	
  	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	one_byte_at_a_time_enable_active_fetch_multiple_symbols_all_together<<<blocksPerGrid1, threadsPerBlock1, smemsize >>> (
		this->real_output_array->get_dev(),
		this->tail_of_real_output_array->get_dev(),

		node_lists->get_dev(),
		
		input_stream->get_dev(),
		symbol_streams[0].get_length(),
		//input_stream->size(),
		
		nfa_sizes->get_dev(),

		this->report_on
	);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n" ,elapsedTime);

	 	
	float sec = elapsedTime / 1000.0;
	cout << "num_of_streams = " << symbol_streams.size() << endl;
	cout << "input_stream_size = " << symbol_streams[0].get_length() << endl;
	cout << "number_of_symbols = " << (symbol_streams[0].get_length() * symbol_streams.size())  << endl;
	cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * symbol_streams.size())  / 1000000.0 / sec  << endl;

	tail_of_real_output_array->copy_back();
	cout << "test_all_outputs =  " << tail_of_real_output_array->get(0) << endl; 

	real_output_array->copy_back();

	//for (int i = 0; i < tail_of_real_output_array->get(0) ; i++) {
	//	cout << real_output_array->get(i).symbol_offset << " " << real_output_array->get(i).state_id << endl;
	//}



    report_formatter rf;
    this->print_reports(this->output_file, rf);
    rf.print_to_file(this->output_file, true);


}


void one_byte_at_a_time::OBAT_baseline_2() {
    cout << "one_byte_at_a_time::OBAT_baseline_2" << endl;
    this->preprocessing_enable_active(); // already limit 4 degree in this function.
    this->grouped_nfas = nfa_utils::group_nfas(block_size, ccs);
    this->remap_intid_of_nodes(this->remap_node_id);
    this->check_grouped_nfa_sizes();

    prepare_output_buffer();
    tail_of_real_output_array->copy_to_device();

    auto node_info_lists = nfa_utils::create_nodelist_for_nfa_groups(this->grouped_nfas, block_size, nfa_utils::create_STE_nodeinfos_new2);
    auto node_ms_lists =  nfa_utils::create_nodelist_for_nfa_groups(this->grouped_nfas,  block_size, nfa_utils::create_STE_matchset_new);

    node_info_lists->copy_to_device();
    node_ms_lists->copy_to_device();

    Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();

    input_stream->copy_to_device();

    auto nfa_sizes = nfa_utils::get_nfa_size_array2(this->grouped_nfas);
    nfa_sizes->copy_to_device();

    cout << "num_of_block = " << grouped_nfas.size() * this->symbol_streams.size() * 1 << endl;
    cout << "num_of_tb_x = " << grouped_nfas.size() << endl;

    dim3 blocksPerGrid1(grouped_nfas.size(), this->symbol_streams.size(), 1);
    dim3 threadsPerBlock1(this->block_size, 1, 1);

    cudaDeviceSynchronize();

    int smemsize = sizeof(bool) * block_size * 2;

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    OBAT_baseline_kernel2_fix_<<<blocksPerGrid1, threadsPerBlock1, smemsize >>> (
            this->real_output_array->get_dev(),
                    this->tail_of_real_output_array->get_dev(),

                    node_info_lists->get_dev(),
                    node_ms_lists->get_dev(),

                    input_stream->get_dev(),
                    this->symbol_streams[0].size(),
                    //input_stream->size(),
                    nfa_sizes->get_dev(),

                    this->report_on
    );

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time : %f ms\n" ,elapsedTime);


    float sec = elapsedTime / 1000.0;
    cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * symbol_streams.size()) / 1000000.0  / sec  << endl;

    tail_of_real_output_array->copy_back();
    cout << "test_all_outputs =  " << tail_of_real_output_array->get(0) << endl;

    real_output_array->copy_back();

    report_formatter rf;
    if(this->report_on)
        this->print_reports(this->output_file, rf);

     if (validation) {
        if (!this->report_on) {
            int dupnum = opt->duplicate_input_stream > 1
                                                    ? opt->duplicate_input_stream
                                                    : 1;
            if (opt->quick_validation >= 0 &&
							opt->quick_validation * dupnum <= tail_of_real_output_array->get(0)) {
						if (opt->quick_validation * dupnum == tail_of_real_output_array->get(0)) {
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
        int dupnum = opt->duplicate_input_stream > 1 ? opt->duplicate_input_stream : 1;
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
                gs, input_stream->get_host(),
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

    delete node_info_lists;
    delete node_ms_lists;
    delete input_stream;
    delete nfa_sizes;
    delete tail_of_real_output_array;
    delete real_output_array;

    for (auto it : grouped_nfas) {
        delete it;
    }



}



void one_byte_at_a_time::obat_MC() {
	// 
	cout << "one_byte_at_a_time::obat_MC() --- OBAT MC!" << endl;

	this->preprocessing_enable_active(); // already limit 4 degree in this function. 

	this->grouped_nfas = nfa_utils::group_nfas(block_size, ccs);

	this->remap_intid_of_nodes(this->remap_node_id);

	this->check_grouped_nfa_sizes();

	prepare_output_buffer();
	tail_of_real_output_array->copy_to_device();

	auto node_info_lists = nfa_utils::create_nodelist_for_nfa_groups(this->grouped_nfas, block_size, nfa_utils::create_STE_nodeinfos_new);
	auto node_ms_lists =  nfa_utils::create_nodelist_for_nfa_groups(this->grouped_nfas,  block_size, nfa_utils::create_STE_matchset_new);

	node_info_lists->copy_to_device();
	node_ms_lists->copy_to_device();

	print_node_matchset_complete_info(this->grouped_nfas);

	Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
	input_stream->copy_to_device();

	
	auto nfa_sizes = nfa_utils::get_nfa_size_array2(this->grouped_nfas);
  	nfa_sizes->copy_to_device();

  	cout << "num_of_block = " << grouped_nfas.size() * this->symbol_streams.size() << endl;
  	cout << "num_of_tb_x = " << grouped_nfas.size() << endl; 

  	dim3 blocksPerGrid1(grouped_nfas.size(), this->symbol_streams.size(), 1);
  	dim3 threadsPerBlock1(this->block_size, 1, 1);

  	int smemsize = sizeof(bool) * block_size * 2;
	
	cudaDeviceSynchronize();

  	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	obat_matchset_compression_new_imp
	<<<blocksPerGrid1, threadsPerBlock1, smemsize >>> (
		this->real_output_array->get_dev(),
		this->tail_of_real_output_array->get_dev(),

		node_info_lists->get_dev(),
		node_ms_lists->get_dev(),
		
		input_stream->get_dev(),
		this->symbol_streams[0].size(),
		
		nfa_sizes->get_dev(),
		
		this->report_on
	);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n" ,elapsedTime);

	 	
	float sec = elapsedTime / 1000.0;
	cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * symbol_streams.size())  / 1000000.0 / sec  << endl;

	tail_of_real_output_array->copy_back();
	cout << "test_all_outputs =  " << tail_of_real_output_array->get(0) << endl; 

	real_output_array->copy_back();

	//for (unsigned long long int i = 0; i < tail_of_real_output_array->get(0) ; i++) {
	//	cout << real_output_array->get(i).symbol_offset << " " << real_output_array->get(i).state_id << endl;
	//}

    report_formatter rf;
    if(this->report_on)
        this->print_reports(this->output_file, rf);

     if (validation) {
        if (!this->report_on) {
            int dupnum = opt->duplicate_input_stream > 1
                                                    ? opt->duplicate_input_stream
                                                    : 1;
            if (opt->quick_validation >= 0 &&
							opt->quick_validation * dupnum <= tail_of_real_output_array->get(0)) {
						if (opt->quick_validation * dupnum == tail_of_real_output_array->get(0)) {
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
        int dupnum = opt->duplicate_input_stream > 1 ? opt->duplicate_input_stream : 1;
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
                gs, input_stream->get_host(),
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



void one_byte_at_a_time::print_reports(string filename, report_formatter& rf) {

	// TODO: extract this function. 


	// report_formatter rf;
	
    unsigned long long int mc = tail_of_real_output_array->get(0);
    rf.reports.resize(mc);
    std::for_each(std::execution::par_unseq, std::begin(rf.reports), std::end(rf.reports), [&](report& r) {
        u_int32_t i = &r - &rf.reports[0];
        assert(i< mc);
        match_entry mp = real_output_array->get(i);
        auto sid = mp.state_id;
        Node *n = grouped_nfas[mp.cc_id]->get_node_by_int_id(sid);
        //cout << "report generated = " << n->str_id << " at " << mp.symbol_offset << endl;
        report rp(mp.symbol_offset, n->original_id, mp.cc_id, mp.stream_id);
        rf.reports[i]=rp;
        //  std::cout << i << "\n";
    });
	
	rf.print_to_file(filename, true);

}


void one_byte_at_a_time::organize_reports2(Array2<match_entry> *output_buffer, int buffer_size, const vector<NFA*> &grouped_nfas1, report_formatter& rf) {


		rf.reports.resize(buffer_size);
		std::for_each(std::execution::par_unseq, std::begin(rf.reports), std::end(rf.reports), [&](report& r) {
			u_int32_t i = &r - &rf.reports[0];
			assert(i< buffer_size);
			match_entry mp = real_output_array->get(i);
  		auto sid = mp.state_id;
  		Node *n = grouped_nfas1[mp.cc_id]->get_node_by_int_id(sid);
  		//cout << "report generated = " << n->str_id << " at " << mp.symbol_offset << endl;
  		report rp(mp.symbol_offset, n->original_id, mp.cc_id, mp.stream_id);
  		rf.reports[i]=rp;
			//  std::cout << i << "\n";
		});

}



void one_byte_at_a_time::hotstart_ea() {
    cout << "one_byte_at_a_time::hotstart_ea" << endl;

    vector<NFA *> tmp_nfa_vec;
    tmp_nfa_vec.push_back(this->nfa);
    cout << "indegree_of_original_nfas" << endl;
    nfa_utils::print_indegrees(tmp_nfa_vec);
    cout << "end----indegree_of_original_nfas" << endl;

    cout << "cold_thres = " << std::fixed << this->cold_thres << endl;

    this->preprocessing_enable_active(); // already limit 4 degree in this function.

    nfa_utils::print_indegrees(this->ccs);

    prepare_output_buffer();
    tail_of_real_output_array->copy_to_device();

    if (this->hot_limit_by_bfs_layer > 0) {
        nfa_utils::assign_hot_cold_by_hot_limit_by_bfs_layer(this->ccs, this->hot_limit_by_bfs_layer, this->block_size);
    }

    //int warp_size = 32;

    for (auto cc : this->ccs) {
        for (int i = 0; i < cc->size(); i++) {
            auto node = cc->get_node_by_int_id(i);
            int indeg_of_node = cc->get_indegree_of_node(node->str_id);
            if (indeg_of_node > max_indegree_of_cold_states) {
                node->hot_degree = std::max(this->cold_thres + 0.00001, node->hot_degree);
                // if hot_degree > cold _thres then the node is hot.

                //std::max(4, node->hot_degree);
            }

        }
    }

    auto grps = nfa_utils::group_nfas_by_hotcold(this->block_size, this->ccs, this->cold_thres);


    /*for (int i = 0; i < grps.size(); i++) {
        cout << "group " << i << endl;
        for (auto it : grps[i]) {
            cout << it << " ";
        }
        cout << endl;
    }*/


    // so the max group size of hot states is this->block_size - warp_size
    // let's do transition table.

    auto grpd_nfas = nfa_utils::merge_nfas_by_group(grps, this->ccs);

    auto grpd_nfas1 = nfa_utils::order_nfa_intid_by_hotcold(grpd_nfas);

    /*for (auto cc : grpd_nfas) {
        delete cc;
    }

    grpd_nfas.clear(); */

    cout << "num_nfa_group = " << grpd_nfas1.size() << endl;

    auto grp_hot_boundaries = nfa_utils::get_hot_boundaries_of_grps(grpd_nfas1, this->cold_thres);

    int total_num_of_states = 0;
    for (auto nn : grpd_nfas1) {
        total_num_of_states += nn->size();
    }

    this->remap_intid_of_nodes_with_boudary(remap_node_id, grpd_nfas1, grp_hot_boundaries);


    //auto global_active_array = new Array2<bool> (total_num_of_states * 2);
    //global_active_array->fill(false);


    auto start_offset_node_list = new Array2<int> (grpd_nfas1.size() + 1);

    int tt = 0;
    for (int i = 0; i < grpd_nfas1.size(); i++) {
        start_offset_node_list->set(i, tt);
        tt += grpd_nfas1[i]->size();
    }

    start_offset_node_list->set(grpd_nfas1.size(), total_num_of_states);

    //start_offset_node_list->print();

    //for (int i = 0; i < start_offset_node_list->size(); i++) {
    //	cout << "start_offset_node_list = " << start_offset_node_list->get(i) << endl;
    //}

    auto num_hot_states = new Array2<int> (grpd_nfas1.size());
    for (int i = 0; i < grp_hot_boundaries.size(); i++) {
        num_hot_states->set(i, grp_hot_boundaries[i]);
    }


    int max_num_cold_state_in_tb = 0;
    for (int i = 0; i < grpd_nfas1.size(); i++) {
        max_num_cold_state_in_tb = std::max( max_num_cold_state_in_tb, grpd_nfas1[i]->size() - grp_hot_boundaries[i]);
    }

    cout << "max_num_cold_state_in_tb = " << max_num_cold_state_in_tb << endl;
    //num_hot_states->print();

    // --- for characterization  ---------------------------------

    int num_hot = 0;
    int num_cold = 0;
    for (auto nn1 : grpd_nfas1) {
        for (int i = 0; i < nn1->size(); i++) {
            auto node = nn1->get_node_by_int_id(i);
            //cout << "nodeid = " << node->str_id <<  " node->hot_degree = " << node->hot_degree << endl;

            assert(node->hot_degree >= 0.0 && node->hot_degree <= 1.0);

            if (node->hot_degree <= this->cold_thres) {
                // cold
                num_cold ++;
            } else {
                num_hot  ++;
            }
        }
    }

    cout << "active_queue_size = " << this->active_queue_size << endl;
    cout << "max_indegree_of_cold_states = " << this->max_indegree_of_cold_states << endl;

    cout << "num_hot = " << num_hot << endl;
    cout << "num_cold = " << num_cold << endl;

    //cout << "hot_states_min_activations_in_profile = " << this->hot_states_min_activations_in_profile << endl;

    cout << "cold_threshold = " << std::fixed << this->cold_thres << endl;



    // ----------end num hot cold characterization ----------------


    //for (int i = 0; i < num_hot_states->size(); i++) {
    //	cout << "num_hot_states " << num_hot_states->get(i) << endl;
    //}

    //for (int i = 0; i < grpd_nfas1.size(); i++ ) {
    //	grpd_nfas1[i]->print();
    //}

    auto node_info_lists = nfa_utils::create_nodelist_for_nfa_groups2(grpd_nfas1, nfa_utils::create_STE_nodeinfos_new);
    auto node_ms_lists =  nfa_utils::create_nodelist_for_nfa_groups2(grpd_nfas1,  nfa_utils::create_STE_matchset_new);

    node_info_lists->copy_to_device();
    node_ms_lists->copy_to_device();


    //auto node_lists = nfa_utils::create_nodelist_for_nfa_groups2(grpd_nfas1, nfa_utils::create_STE_dev4_compressed_edges);
    cout << "nodelist_size = " << node_info_lists->size() << endl;


    Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();


    //for (int i = 0; i < node_lists->size(); i++) {
    //	STE_dev4 it = node_lists->get(i);
    //	cout << "nodeid = " << i << " edges = " << it.edges << " attr = " << (int) it.attribute << " degree = " << it.degree << endl;
    //}



    start_offset_node_list->copy_to_device();
    num_hot_states->copy_to_device();

    input_stream->copy_to_device();

    //global_active_array->copy_to_device();

    auto remap_input_stream_array = new Array2<int> (symbol_streams.size());
    for (int i = 0; i < remap_input_stream_array->size(); i++) {
        remap_input_stream_array->set(i, i);
    }

    if (remap_input_stream) {
        std::random_shuffle(remap_input_stream_array->get_host(), remap_input_stream_array->get_host() + remap_input_stream_array->size());
    }

    remap_input_stream_array->copy_to_device();

    int queuesize = this->active_queue_size;

    //auto global_queue = new Array2<int> (grpd_nfas1.size() * symbol_streams.size() * queuesize * 2);
    //global_queue->fill(0);
    //global_queue->copy_to_device();

    dim3 blocksPerGrid1(grpd_nfas1.size(), symbol_streams.size(), 1);
    dim3 threadsPerBlock1(this->block_size, 1, 1);

    cout << "num_of_tb_x = " << grpd_nfas1.size() << endl;
    cout << "num_of_block = " << grpd_nfas1.size() * symbol_streams.size() * 1 << endl;


    int num_cell_nodup_bitset =  ceil(1.0 * max_num_cold_state_in_tb / 32.0);
    cout << "num_cell_nodup_bitset = " << num_cell_nodup_bitset << endl;

    int smemsize = sizeof(int) * queuesize * 2 + num_cell_nodup_bitset * sizeof(int);

    cout << "smemsize = " << smemsize << endl;
    cout << "shared_memory_size_KB = " << std::fixed << smemsize * 1.0 / 1024.0 << endl;

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    if (!this->hot_stage_only) {
        hotstart_ea_kernel<true>
                <<<blocksPerGrid1, threadsPerBlock1, smemsize>>>
                                                     (
                                                             this->real_output_array->get_dev(),
                                                                     this->tail_of_real_output_array->get_dev(),
                                                                     input_stream->get_dev(),
                                                                     this->symbol_streams[0].size(),
                                                                     start_offset_node_list->get_dev(),
                                                                     num_hot_states->get_dev(),
                                                                     node_info_lists->get_dev(),
                                                                     node_ms_lists->get_dev(),
                                                                     this->report_on,
                                                                     queuesize,
                                                                     num_cell_nodup_bitset,
                                                                     remap_input_stream_array->get_dev()
                                                     );
    } else {
        hotstart_ea_kernel<false>
                <<<blocksPerGrid1, threadsPerBlock1, smemsize>>>
                                                     (
                                                             this->real_output_array->get_dev(),
                                                                     this->tail_of_real_output_array->get_dev(),
                                                                     input_stream->get_dev(),
                                                                     this->symbol_streams[0].size(),
                                                                     start_offset_node_list->get_dev(),
                                                                     num_hot_states->get_dev(),
                                                                     node_info_lists->get_dev(),
                                                                     node_ms_lists->get_dev(),
                                                                     this->report_on,
                                                                     queuesize,
                                                                     num_cell_nodup_bitset,
                                                                    remap_input_stream_array->get_dev()
                                                     );

    }



    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time : %f ms\n" ,elapsedTime);

    float sec = elapsedTime / 1000.0;
    cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * symbol_streams.size()) / 1000000.0  / sec  << endl;


    auto cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess)
    {
        printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    }



    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess)
    {
        printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    }

    report_formatter rf;
    this->tail_of_real_output_array->copy_back();
    this->real_output_array->copy_back();
    cout << "test_all_outputs =  " << tail_of_real_output_array->get(0) << endl;

    if(this->report_on) {
        this->organize_reports2(this->real_output_array, this->tail_of_real_output_array->get(0), grpd_nfas1, rf);
        rf.print_to_file(this->output_file, true);
    }

    if (validation) {
        if (!this->report_on) {
            int dupnum = opt->duplicate_input_stream > 1
                                                    ? opt->duplicate_input_stream
                                                    : 1;
            if (opt->quick_validation >= 0 &&
							opt->quick_validation * dupnum <= tail_of_real_output_array->get(0)) {
						if (opt->quick_validation * dupnum == tail_of_real_output_array->get(0)) {
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
        int dupnum = opt->duplicate_input_stream > 1 ? opt->duplicate_input_stream : 1;
        if (opt->quick_validation * dupnum <= rf.size()) {
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
                gs, input_stream->get_host(),
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

void one_byte_at_a_time::hotstart_ea_without_MC2() {
    cout << "one_byte_at_a_time::hotstart_ea_without_MC2" << endl;

    vector<NFA *> tmp_nfa_vec;
    tmp_nfa_vec.push_back(this->nfa);
    cout << "indegree_of_original_nfas" << endl;
    nfa_utils::print_indegrees(tmp_nfa_vec);
    cout << "end----indegree_of_original_nfas" << endl;

    cout << "cold_thres = " << std::fixed << this->cold_thres << endl;

    this->preprocessing_enable_active(); // already limit 4 degree in this function.

    nfa_utils::print_indegrees(this->ccs);

    prepare_output_buffer();
    tail_of_real_output_array->copy_to_device();

    if (this->hot_limit_by_bfs_layer > 0) {
        nfa_utils::assign_hot_cold_by_hot_limit_by_bfs_layer(this->ccs, this->hot_limit_by_bfs_layer, this->block_size);
    }

    //int warp_size = 32;

    for (auto cc : this->ccs) {
        for (int i = 0; i < cc->size(); i++) {
            auto node = cc->get_node_by_int_id(i);
            int indeg_of_node = cc->get_indegree_of_node(node->str_id);
            if (indeg_of_node > max_indegree_of_cold_states) {
                node->hot_degree = std::max(this->cold_thres + 0.00001, node->hot_degree);
                // if hot_degree > cold _thres then the node is hot.

                //std::max(4, node->hot_degree);
            }

        }
    }

    auto grps = nfa_utils::group_nfas_by_hotcold(this->block_size, this->ccs, this->cold_thres);


    /*for (int i = 0; i < grps.size(); i++) {
        cout << "group " << i << endl;
        for (auto it : grps[i]) {
            cout << it << " ";
        }
        cout << endl;
    }*/


    // so the max group size of hot states is this->block_size - warp_size
    // let's do transition table.

    auto grpd_nfas = nfa_utils::merge_nfas_by_group(grps, this->ccs);

    auto grpd_nfas1 = nfa_utils::order_nfa_intid_by_hotcold(grpd_nfas);

    /*for (auto cc : grpd_nfas) {
        delete cc;
    }

    grpd_nfas.clear(); */

    cout << "num_nfa_group = " << grpd_nfas1.size() << endl;

    auto grp_hot_boundaries = nfa_utils::get_hot_boundaries_of_grps(grpd_nfas1, this->cold_thres);

    int total_num_of_states = 0;
    for (auto nn : grpd_nfas1) {
        total_num_of_states += nn->size();
    }

    this->remap_intid_of_nodes_with_boudary(remap_node_id, grpd_nfas1, grp_hot_boundaries);


    //auto global_active_array = new Array2<bool> (total_num_of_states * 2);
    //global_active_array->fill(false);


    auto start_offset_node_list = new Array2<int> (grpd_nfas1.size() + 1);

    int tt = 0;
    for (int i = 0; i < grpd_nfas1.size(); i++) {
        start_offset_node_list->set(i, tt);
        tt += grpd_nfas1[i]->size();
    }

    start_offset_node_list->set(grpd_nfas1.size(), total_num_of_states);

    //start_offset_node_list->print();

    //for (int i = 0; i < start_offset_node_list->size(); i++) {
    //	cout << "start_offset_node_list = " << start_offset_node_list->get(i) << endl;
    //}

    auto num_hot_states = new Array2<int> (grpd_nfas1.size());
    for (int i = 0; i < grp_hot_boundaries.size(); i++) {
        num_hot_states->set(i, grp_hot_boundaries[i]);
    }


    int max_num_cold_state_in_tb = 0;
    for (int i = 0; i < grpd_nfas1.size(); i++) {
        max_num_cold_state_in_tb = std::max( max_num_cold_state_in_tb, grpd_nfas1[i]->size() - grp_hot_boundaries[i]);
    }

    cout << "max_num_cold_state_in_tb = " << max_num_cold_state_in_tb << endl;
    //num_hot_states->print();

    // --- for characterization  ---------------------------------

    int num_hot = 0;
    int num_cold = 0;
    for (auto nn1 : grpd_nfas1) {
        for (int i = 0; i < nn1->size(); i++) {
            auto node = nn1->get_node_by_int_id(i);
            //cout << "nodeid = " << node->str_id <<  " node->hot_degree = " << node->hot_degree << endl;

            assert(node->hot_degree >= 0.0 && node->hot_degree <= 1.0);

            if (node->hot_degree <= this->cold_thres) {
                // cold
                num_cold ++;
            } else {
                num_hot  ++;
            }
        }
    }

    cout << "active_queue_size = " << this->active_queue_size << endl;
    cout << "max_indegree_of_cold_states = " << this->max_indegree_of_cold_states << endl;

    cout << "num_hot = " << num_hot << endl;
    cout << "num_cold = " << num_cold << endl;

    //cout << "hot_states_min_activations_in_profile = " << this->hot_states_min_activations_in_profile << endl;

    cout << "cold_threshold = " << std::fixed << this->cold_thres << endl;



    // ----------end num hot cold characterization ----------------


    //for (int i = 0; i < num_hot_states->size(); i++) {
    //	cout << "num_hot_states " << num_hot_states->get(i) << endl;
    //}

    //for (int i = 0; i < grpd_nfas1.size(); i++ ) {
    //	grpd_nfas1[i]->print();
    //}

    //auto node_info_lists = nfa_utils::create_nodelist_for_nfa_groups2(grpd_nfas1, nfa_utils::create_STE_dev4_compressed_edges);
    //node_info_lists->copy_to_device();


    auto node_info_lists = nfa_utils::create_nodelist_for_nfa_groups2(grpd_nfas1, nfa_utils::create_STE_nodeinfos_new2);
    auto node_ms_lists =  nfa_utils::create_nodelist_for_nfa_groups2(grpd_nfas1, nfa_utils::create_STE_matchset_new);

    node_info_lists->copy_to_device();
    node_ms_lists->copy_to_device();


    //auto node_lists = nfa_utils::create_nodelist_for_nfa_groups2(grpd_nfas1, nfa_utils::create_STE_dev4_compressed_edges);
    cout << "nodelist_size = " << node_info_lists->size() << endl;


    Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();


    //for (int i = 0; i < node_lists->size(); i++) {
    //	STE_dev4 it = node_lists->get(i);
    //	cout << "nodeid = " << i << " edges = " << it.edges << " attr = " << (int) it.attribute << " degree = " << it.degree << endl;
    //}



    start_offset_node_list->copy_to_device();
    num_hot_states->copy_to_device();

    input_stream->copy_to_device();

    //global_active_array->copy_to_device();

    auto remap_input_stream_array = new Array2<int> (symbol_streams.size());
    for (int i = 0; i < remap_input_stream_array->size(); i++) {
        remap_input_stream_array->set(i, i);
    }

    if (remap_input_stream) {
        std::random_shuffle(remap_input_stream_array->get_host(), remap_input_stream_array->get_host() + remap_input_stream_array->size());
    }

    remap_input_stream_array->copy_to_device();

    int queuesize = this->active_queue_size;
    dim3 blocksPerGrid1(grpd_nfas1.size(), symbol_streams.size(), 1);
    dim3 threadsPerBlock1(this->block_size, 1, 1);

    cout << "num_of_tb_x = " << grpd_nfas1.size() << endl;
    cout << "num_of_block = " << grpd_nfas1.size() * symbol_streams.size() * 1 << endl;


    int num_cell_nodup_bitset =  ceil(1.0 * max_num_cold_state_in_tb / 32.0);
    cout << "num_cell_nodup_bitset = " << num_cell_nodup_bitset << endl;

    int smemsize = sizeof(int) * queuesize * 2 + num_cell_nodup_bitset * sizeof(int);

    cout << "smemsize = " << smemsize << endl;
    cout << "shared_memory_size_KB = " << std::fixed << smemsize * 1.0 / 1024.0 << endl;

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    if (!this->hot_stage_only) {
        hotstart_ea_kernel_without_MC2<true>
                <<<blocksPerGrid1, threadsPerBlock1, smemsize>>>
                                                     (
                                                             this->real_output_array->get_dev(),
                                                                     this->tail_of_real_output_array->get_dev(),
                                                                     input_stream->get_dev(),
                                                                     this->symbol_streams[0].size(),
                                                                     start_offset_node_list->get_dev(),
                                                                     num_hot_states->get_dev(),
                                                                     node_info_lists->get_dev(),
                                                                     node_ms_lists->get_dev(),
                                                                     this->report_on,
                                                                     queuesize,
                                                                     num_cell_nodup_bitset,
                                                                     remap_input_stream_array->get_dev()
                                                     );
    } else {
        hotstart_ea_kernel_without_MC2<false>
                <<<blocksPerGrid1, threadsPerBlock1, smemsize>>>
                                                     (
                                                             this->real_output_array->get_dev(),
                                                                     this->tail_of_real_output_array->get_dev(),
                                                                     input_stream->get_dev(),
                                                                     this->symbol_streams[0].size(),
                                                                     start_offset_node_list->get_dev(),
                                                                     num_hot_states->get_dev(),
                                                                     node_info_lists->get_dev(),
                                                                     node_ms_lists->get_dev(),
                                                                     this->report_on,
                                                                     queuesize,
                                                                     num_cell_nodup_bitset,
                                                                     remap_input_stream_array->get_dev()
                                                     );

    }



    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time : %f ms\n" ,elapsedTime);

    float sec = elapsedTime / 1000.0;
    cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * symbol_streams.size())  / 1000000.0 / sec  << endl;


    auto cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess)
    {
        printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    }



    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess)
    {
        printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    }

    report_formatter rf;
    this->tail_of_real_output_array->copy_back();
    this->real_output_array->copy_back();
    cout << "test_all_outputs =  " << tail_of_real_output_array->get(0) << endl;

    if(this->report_on) {
        this->organize_reports2(this->real_output_array, this->tail_of_real_output_array->get(0), grpd_nfas1, rf);
        rf.print_to_file(this->output_file, true);
    }

    if (validation) {
        if (!this->report_on) {
            int dupnum = opt->duplicate_input_stream > 1
                                                    ? opt->duplicate_input_stream
                                                    : 1;
            if (opt->quick_validation >= 0 &&
							opt->quick_validation * dupnum <= tail_of_real_output_array->get(0)) {
						if (opt->quick_validation * dupnum == tail_of_real_output_array->get(0)) {
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
        int dupnum = opt->duplicate_input_stream > 1 ? opt->duplicate_input_stream : 1;
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
                gs, input_stream->get_host(),
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



void one_byte_at_a_time::test_hotcold_nodup_queue_mc_CaH() {
	cout << "one_byte_at_a_time::test_hotcold_nodup_queue_mc_CaH" << endl;

	vector<NFA *> tmp_nfa_vec;
	tmp_nfa_vec.push_back(this->nfa);
	cout << "indegree_of_original_nfas" << endl;
	nfa_utils::print_indegrees(tmp_nfa_vec);
	cout << "end----indegree_of_original_nfas" << endl;

	cout << "cold_thres = " << std::fixed << this->cold_thres << endl;

	this->preprocessing_enable_active(); // already limit 4 degree in this function. 

	nfa_utils::print_indegrees(this->ccs);
	
	prepare_output_buffer();
	tail_of_real_output_array->copy_to_device();

	if (this->hot_limit_by_bfs_layer > 0) {
		nfa_utils::assign_hot_cold_by_hot_limit_by_bfs_layer(this->ccs, this->hot_limit_by_bfs_layer, this->block_size); 
	}

	//int warp_size = 32;

	for (auto cc : this->ccs) {
		for (int i = 0; i < cc->size(); i++) {
			auto node = cc->get_node_by_int_id(i);
			int indeg_of_node = cc->get_indegree_of_node(node->str_id);
			if (indeg_of_node > max_indegree_of_cold_states) {
				node->hot_degree = std::max(this->cold_thres + 0.00001, node->hot_degree);
				// if hot_degree > cold _thres then the node is hot. 

				//std::max(4, node->hot_degree);
			}

		}
	}

	auto grps = nfa_utils::group_nfas_by_hotcold(this->block_size, this->ccs, this->cold_thres);


	/*for (int i = 0; i < grps.size(); i++) {
		cout << "group " << i << endl;
		for (auto it : grps[i]) {
			cout << it << " ";
		}
		cout << endl;
	}*/
	

	// so the max group size of hot states is this->block_size - warp_size
	// let's do transition table. 

	auto grpd_nfas = nfa_utils::merge_nfas_by_group(grps, this->ccs);

	auto grpd_nfas1 = nfa_utils::order_nfa_intid_by_hotcold(grpd_nfas);

	/*for (auto cc : grpd_nfas) {
		delete cc;
	}

	grpd_nfas.clear(); */

	cout << "num_nfa_group = " << grpd_nfas1.size() << endl;
	
	auto grp_hot_boundaries = nfa_utils::get_hot_boundaries_of_grps(grpd_nfas1, this->cold_thres);

	int total_num_of_states = 0;
	for (auto nn : grpd_nfas1) {
		total_num_of_states += nn->size();
	}

	this->remap_intid_of_nodes_with_boudary(remap_node_id, grpd_nfas1, grp_hot_boundaries);


	//auto global_active_array = new Array2<bool> (total_num_of_states * 2);
	//global_active_array->fill(false);


	auto start_offset_node_list = new Array2<int> (grpd_nfas1.size() + 1);
	
	int tt = 0;
	for (int i = 0; i < grpd_nfas1.size(); i++) {
		start_offset_node_list->set(i, tt);
		tt += grpd_nfas1[i]->size();
	}

	start_offset_node_list->set(grpd_nfas1.size(), total_num_of_states);

	//start_offset_node_list->print();

	//for (int i = 0; i < start_offset_node_list->size(); i++) {
	//	cout << "start_offset_node_list = " << start_offset_node_list->get(i) << endl;
	//}

	auto num_hot_states = new Array2<int> (grpd_nfas1.size());
	for (int i = 0; i < grp_hot_boundaries.size(); i++) {
		num_hot_states->set(i, grp_hot_boundaries[i]);
	}


	int max_num_cold_state_in_tb = 0; 
	for (int i = 0; i < grpd_nfas1.size(); i++) {
		max_num_cold_state_in_tb = std::max( max_num_cold_state_in_tb, grpd_nfas1[i]->size() - grp_hot_boundaries[i]);
	}

	cout << "max_num_cold_state_in_tb = " << max_num_cold_state_in_tb << endl;
	//num_hot_states->print();

	// --- for characterization  ---------------------------------

	int num_hot = 0;
	int num_cold = 0;
	for (auto nn1 : grpd_nfas1) {
		for (int i = 0; i < nn1->size(); i++) {
			auto node = nn1->get_node_by_int_id(i);
			//cout << "nodeid = " << node->str_id <<  " node->hot_degree = " << node->hot_degree << endl;

			assert(node->hot_degree >= 0.0 && node->hot_degree <= 1.0);

			if (node->hot_degree <= this->cold_thres) {
				// cold
				num_cold ++;
			} else {
				num_hot  ++;
			}
		}
	}

	cout << "active_queue_size = " << this->active_queue_size << endl;
	cout << "max_indegree_of_cold_states = " << this->max_indegree_of_cold_states << endl;

	cout << "num_hot = " << num_hot << endl;
	cout << "num_cold = " << num_cold << endl;

	//cout << "hot_states_min_activations_in_profile = " << this->hot_states_min_activations_in_profile << endl;

	cout << "cold_threshold = " << std::fixed << this->cold_thres << endl;



	// ----------end num hot cold characterization ----------------


	//for (int i = 0; i < num_hot_states->size(); i++) {
	//	cout << "num_hot_states " << num_hot_states->get(i) << endl;
	//}

	//for (int i = 0; i < grpd_nfas1.size(); i++ ) {
	//	grpd_nfas1[i]->print();
	//}

	auto node_info_lists = nfa_utils::create_nodelist_for_nfa_groups2(grpd_nfas1, nfa_utils::create_STE_nodeinfos_new);
	auto node_ms_lists =  nfa_utils::create_nodelist_for_nfa_groups2(grpd_nfas1,  nfa_utils::create_STE_matchset_new);

	node_info_lists->copy_to_device();
	node_ms_lists->copy_to_device();


	//auto node_lists = nfa_utils::create_nodelist_for_nfa_groups2(grpd_nfas1, nfa_utils::create_STE_dev4_compressed_edges);
	cout << "nodelist_size = " << node_info_lists->size() << endl;


	Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();


	//for (int i = 0; i < node_lists->size(); i++) {
	//	STE_dev4 it = node_lists->get(i);
	//	cout << "nodeid = " << i << " edges = " << it.edges << " attr = " << (int) it.attribute << " degree = " << it.degree << endl;	
	//}



	start_offset_node_list->copy_to_device();
	num_hot_states->copy_to_device();
	
	input_stream->copy_to_device();

	//global_active_array->copy_to_device();



  	int queuesize = this->active_queue_size;

	//auto global_queue = new Array2<int> (grpd_nfas1.size() * symbol_streams.size() * queuesize * 2);
	//global_queue->fill(0);
	//global_queue->copy_to_device();

	dim3 blocksPerGrid1(grpd_nfas1.size(), symbol_streams.size(), 1);
  	dim3 threadsPerBlock1(this->block_size, 1, 1);

  	cout << "num_of_tb_x = " << grpd_nfas1.size() << endl;
  	cout << "num_of_block = " << grpd_nfas1.size() * symbol_streams.size() * 1 << endl;


  	int num_cell_nodup_bitset =  ceil(1.0 * max_num_cold_state_in_tb / 32.0); 
  	cout << "num_cell_nodup_bitset = " << num_cell_nodup_bitset << endl;

  	int smemsize = sizeof(bool) * (block_size) * 2 + sizeof(int) * queuesize * 2 + num_cell_nodup_bitset * sizeof(int);

  	cout << "smemsize = " << smemsize << endl;
  	cout << "shared_memory_size_KB = " << std::fixed << smemsize * 1.0 / 1024.0 << endl;

	cudaDeviceSynchronize();

  	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);


    hotcold_nodup_queue_mc_cold_after_hot
    <<<blocksPerGrid1, threadsPerBlock1, smemsize>>>
    (
        this->real_output_array->get_dev(),
        this->tail_of_real_output_array->get_dev(),
        input_stream->get_dev(),
        this->symbol_streams[0].size(),
        start_offset_node_list->get_dev(),
        num_hot_states->get_dev(),
        node_info_lists->get_dev(),
        node_ms_lists->get_dev(),
        this->report_on,
        queuesize,
        num_cell_nodup_bitset
    );

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n" ,elapsedTime);

	 	
	float sec = elapsedTime / 1000.0;
	cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * symbol_streams.size())  / 1000000.0 / sec  << endl;

	
	auto cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
	{
		 printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}



	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
	{
		 printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}

	report_formatter rf;
	this->tail_of_real_output_array->copy_back();
	this->real_output_array->copy_back();
    cout << "test_all_outputs =  " << tail_of_real_output_array->get(0) << endl;

    if(this->report_on) {
        this->organize_reports2(this->real_output_array, this->tail_of_real_output_array->get(0), grpd_nfas1, rf);
	    rf.print_to_file(this->output_file, true);
    }

     if (validation) {
        if (!this->report_on) {
            int dupnum = opt->duplicate_input_stream > 1
                                                    ? opt->duplicate_input_stream
                                                    : 1;
            if (opt->quick_validation >= 0 &&
							opt->quick_validation * dupnum <= tail_of_real_output_array->get(0)) {
						if (opt->quick_validation * dupnum == tail_of_real_output_array->get(0)) {
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
        int dupnum = opt->duplicate_input_stream > 1 ? opt->duplicate_input_stream : 1;
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
                gs, input_stream->get_host(),
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


void one_byte_at_a_time::hotstart_aa() {
    cout << "one_byte_at_a_time::hotstart_aa" << endl;
    cout << "cold_thres = " << std::fixed << this->cold_thres << endl;

    prepare_output_buffer();
    tail_of_real_output_array->copy_to_device();

    preprocessing_active_active();

    cout << "finished preprocessing active active" << endl;

    if (this->hot_limit_by_bfs_layer > 0) {
        nfa_utils::assign_hot_cold_by_hot_limit_by_bfs_layer(this->ccs, this->hot_limit_by_bfs_layer, this->block_size);
    }

    for (auto cc : this->ccs) {
        for (int i = 0; i < cc->size(); i++) {
            auto node = cc->get_node_by_int_id(i);
            int indeg_of_node = cc->get_indegree_of_node(node->str_id);
            if (indeg_of_node > max_indegree_of_cold_states) {
                node->hot_degree = std::max(this->cold_thres + 0.00001, node->hot_degree);
            }
        }
    }

    auto grps = nfa_utils::group_nfas_by_hotcold(this->block_size, this->ccs, this->cold_thres);

    cout << "finished group nfas by hotcold " << endl;

    /*for (int i = 0; i < grps.size(); i++) {
        cout << "group " << i << endl;
        for (auto it : grps[i]) {
            cout << it << " ";
        }
        cout << endl;
    }*/


    auto grpd_nfas = nfa_utils::merge_nfas_by_group(grps, this->ccs);
    auto grpd_nfas1 = nfa_utils::order_nfa_intid_by_hotcold(grpd_nfas);

    int list_size = 0;
    for (int i = 0; i < grpd_nfas1.size(); i++) {
        list_size += ALPHABET_SIZE * grpd_nfas1[i]->size();
        cout << "grp_nfa_" << i << " = " << grpd_nfas1[i]->size() << endl;

       // grpd_nfas1[i]->print(); // DEBUG
    }

    auto trans_table = nfa_utils::create_nodelist(grpd_nfas1, nfa_utils::create_ull64_for_nfa, list_size);

    trans_table->copy_to_device();

    //trans_table->print();

//    cout << "-------------- transtable print ----------------------------------------" << endl;
//    trans_table->print();
//
//    for (int i = 0; i < trans_table->size(); i++) {
//        auto out_nodes = trans_table->get(i);
//        for (int to = 0; to < 4; to++) {
//            unsigned int edgeto = (out_nodes >> (16 * to)) & 65535;
//            cout << edgeto << " ";
//        }
//        cout << endl;
//    }
//
//    cout << "-------------- end transtable print ------------------------------------" << endl;


    //cout << "finish creating trans table " << endl;

    /*auto state_starting_point_in_tb = new Array2<int> (grpd_nfas1.size() + 1);
    int tmp_sum = 0;
    for (int i = 0; i < grpd_nfas1.size(); i++) {
        state_starting_point_in_tb->set(i, tmp_sum);
        tmp_sum += grpd_nfas1[i]->size() * ALPHABET_SIZE;
    }
    state_starting_point_in_tb->copy_to_device();
    */


    cout << "num_nfa_group = " << grpd_nfas1.size() << endl;
    auto grp_hot_boundaries = nfa_utils::get_hot_boundaries_of_grps(grpd_nfas1, this->cold_thres);
    for (int i = 0; i < grp_hot_boundaries.size(); i++) {
        cout << "grp_hot_boundaries_" << i << " = " << grp_hot_boundaries[i] << endl;
    }

    auto num_hot_states = new Array2<int> (grpd_nfas1.size());
    for (int i = 0; i < grp_hot_boundaries.size(); i++) {
        num_hot_states->set(i, grp_hot_boundaries[i]);
    }

    num_hot_states->copy_to_device();

    int max_num_cold_state_in_tb = 0;
    for (int i = 0; i < grpd_nfas1.size(); i++) {
        max_num_cold_state_in_tb = std::max( max_num_cold_state_in_tb, grpd_nfas1[i]->size() - grp_hot_boundaries[i]);
    }

    cout << "max_num_cold_state_in_tb = " << max_num_cold_state_in_tb << endl;

    //cout << "num_hot_states " << endl;
    //num_hot_states->print();

    // --- for characterization  ---------------------------------
    int num_hot = 0;
    int num_cold = 0;
    for (auto nn1 : grpd_nfas1) {
        for (int i = 0; i < nn1->size(); i++) {
            auto node = nn1->get_node_by_int_id(i);
            //cout << "nodeid = " << node->str_id <<  " node->hot_degree = " << node->hot_degree << endl;

            assert(node->hot_degree >= 0.0 && node->hot_degree <= 1.0);

            if (node->hot_degree <= this->cold_thres) {
                // cold
                num_cold ++;
            } else {
                num_hot  ++;
            }
        }
    }

    cout << "active_queue_size = " << this->active_queue_size << endl;
    cout << "max_indegree_of_cold_states = " << this->max_indegree_of_cold_states << endl;

    cout << "num_hot = " << num_hot << endl;
    cout << "num_cold = " << num_cold << endl;

    //cout << "hot_states_min_activations_in_profile = " << this->hot_states_min_activations_in_profile << endl;

    cout << "cold_threshold = " << std::fixed << this->cold_thres << endl;

    // ----------end num hot cold characterization ----------------

    cout << "prepare_start_offset_list" << endl;
    auto start_offset_node_list = new Array2<int> (grpd_nfas1.size() + 1, "start_offset_node_list");

    int tt = 0;
    for (int i = 0; i < grpd_nfas1.size(); i++) {
        start_offset_node_list->set(i, tt);
        tt += grpd_nfas1[i]->size();
    }

    start_offset_node_list->set(grpd_nfas1.size(), tt);
    start_offset_node_list->copy_to_device();

    assert(trans_table->size() == tt * ALPHABET_SIZE);

    auto is_report = nfa_utils::create_nodelist_for_nfa_groups2(grpd_nfas1, nfa_utils::get_attribute_array);
    is_report->copy_to_device();

    Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
    input_stream->copy_to_device();

    int queuesize = this->active_queue_size;

    dim3 blocksPerGrid1(grpd_nfas1.size(), symbol_streams.size(), 1);
    dim3 threadsPerBlock1(this->block_size, 1, 1);

    cout << "num_of_tb_x = " << grpd_nfas1.size() << endl;
    cout << "num_of_block = " << grpd_nfas1.size() * symbol_streams.size() * 1 << endl;

    int num_cell_nodup_bitset =  ceil(1.0 * max_num_cold_state_in_tb / 32.0);
    cout << "num_cell_nodup_bitset = " << num_cell_nodup_bitset << endl;

    int smemsize = sizeof(int) * queuesize * 2 + num_cell_nodup_bitset * sizeof(int);

    cout << "smemsize = " << smemsize << endl;
    cout << "shared_memory_size_KB = " << std::fixed << smemsize * 1.0 / 1024.0 << endl;

    if (this->hot_stage_only) {
        cout << "only run hot stage" << endl;
    }

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    if (!this->hot_stage_only) {
        hotstart_transition_table<true>
                <<<blocksPerGrid1, threadsPerBlock1, smemsize>>>
                                                     (
                                                             this->real_output_array->get_dev(),
                                                                     this->tail_of_real_output_array->get_dev(),
                                                                     input_stream->get_dev(),
                                                                     this->symbol_streams[0].size(),
                                                                     num_hot_states->get_dev(),
                                                                     this->report_on,
                                                                     queuesize,
                                                                     num_cell_nodup_bitset,
                                                                     trans_table->get_dev(),
                                                                     is_report->get_dev(),
                                                                     start_offset_node_list->get_dev()
                                                     );
    } else {
        hotstart_transition_table<false>
                <<<blocksPerGrid1, threadsPerBlock1, smemsize>>>
                                                     (
                                                             this->real_output_array->get_dev(),
                                                                     this->tail_of_real_output_array->get_dev(),
                                                                     input_stream->get_dev(),
                                                                     this->symbol_streams[0].size(),
                                                                     num_hot_states->get_dev(),
                                                                     this->report_on,
                                                                     queuesize,
                                                                     num_cell_nodup_bitset,
                                                                     trans_table->get_dev(),
                                                                     is_report->get_dev(),
                                                                     start_offset_node_list->get_dev()
                                                     );

    }


    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time : %f ms\n" ,elapsedTime);


    float sec = elapsedTime / 1000.0;
    cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * symbol_streams.size())  / 1000000.0 / sec  << endl;


    auto cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess)
    {
        printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    }



    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess)
    {
        printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    }

    report_formatter rf;
    this->tail_of_real_output_array->copy_back();
    this->real_output_array->copy_back();
    cout << "test_all_outputs =  " << tail_of_real_output_array->get(0) << endl;

    if(this->report_on) {
        this->organize_reports2(this->real_output_array, this->tail_of_real_output_array->get(0), grpd_nfas1, rf);
        rf.print_to_file(this->output_file, true);
    }

     if (validation) {
        if (!this->report_on) {
            int dupnum = opt->duplicate_input_stream > 1
                                                    ? opt->duplicate_input_stream
                                                    : 1;
            if (opt->quick_validation >= 0 &&
							opt->quick_validation * dupnum <= tail_of_real_output_array->get(0)) {
						if (opt->quick_validation * dupnum == tail_of_real_output_array->get(0)) {
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
        int dupnum = opt->duplicate_input_stream > 1 ? opt->duplicate_input_stream : 1;
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
                gs, input_stream->get_host(),
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


void one_byte_at_a_time::set_node_active_freq_map(map<string, double> freq_map) {
	if (freq_map.size() == 0) {
		cout << "empty freq map ; do nothing related to hot/cold classification. " << endl;
	}

	cout << "set hot degree to nodes " << endl;
	cout << "cold_thres = " << this->cold_thres << endl;
	for (auto it : freq_map) {
		auto str_id = it.first;

		auto nodes = nfa->get_nodes_by_original_id(str_id);

		double enb_rate = freq_map[str_id];

		for (auto str_id : nodes ) {
			auto node = nfa->get_node_by_str_id(str_id);
			node->hot_degree = enb_rate;
		}

	}


	this->freq_map = freq_map;
}

void one_byte_at_a_time::set_hot_limit_by_bfs_layer(int hot_limit_by_bfs_layer) {
	assert(hot_limit_by_bfs_layer > 0);

	this->hot_limit_by_bfs_layer = hot_limit_by_bfs_layer;
}


void one_byte_at_a_time::set_active_queue_size(int queuesize) {
	this->active_queue_size = queuesize;

}


void one_byte_at_a_time::print_node_matchset_complete_info(const vector<NFA*> &ccs) {
	int complete = 0;
	int complement = 0;
	int num_of_state = 0;

	for (auto cc : ccs) {
		num_of_state += cc->size();
		for (int i = 0; i < cc->size(); i++) {
			
			auto node = cc->get_node_by_int_id(i);

			if (node->complete && !node->complement) {
				complete ++;
			} else if (node->complement && node->complement) {
				complement ++;
			}

		}
	}

	cout << "num_of_complete = " << complete << endl;
	cout << "num_of_complement = " << complement << endl;
	cout << "num_of_complete_or_complement = " << complete + complement << endl;
	cout << "num_of_state = " << num_of_state << endl;
	cout << "complete_ratio = " << std::fixed << (complete + complement + 0.0) / num_of_state << endl;

}


void one_byte_at_a_time::test_data_movement_read_input_stream_only(int num_tb_x) {
	
	dim3 blocksPerGrid1(num_tb_x, this->symbol_streams.size(), 1);
  	dim3 threadsPerBlock1(this->block_size, 1, 1);

	cudaDeviceSynchronize();

  	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
	input_stream->copy_to_device();

	obat_only_read_input_stream
	<<<blocksPerGrid1, threadsPerBlock1>>>
	(input_stream->get_dev(),
	 this->symbol_streams[0].size()
	);

	cudaDeviceSynchronize();

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n" ,elapsedTime);

	float sec = elapsedTime / 1000.0;
	cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * num_tb_x )  / 1000000.0 / sec  << endl;

	delete input_stream;

}


void one_byte_at_a_time::test_data_movement_read_input_stream_only2(int num_tb_x) {
	dim3 blocksPerGrid1(num_tb_x, this->symbol_streams.size(), 1);
  	dim3 threadsPerBlock1(this->block_size, 1, 1);

	cudaDeviceSynchronize();

	int shr_mem_size = this->block_size * sizeof(uint8_t);

  	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	Array2<uint8_t> *input_stream = this->concat_input_streams_to_array2();
	input_stream->copy_to_device();

	obat_only_read_input_stream_shr
	<<<blocksPerGrid1, threadsPerBlock1, shr_mem_size>>>
	(input_stream->get_dev(),
	 this->symbol_streams[0].size()
	);

	cudaDeviceSynchronize();

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n" ,elapsedTime);

	auto cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
	{
		 printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}

	float sec = elapsedTime / 1000.0;
	cout << "throughput = " << std::fixed << (symbol_streams[0].get_length() * symbol_streams.size())  / 1000000.0 / sec  << endl;

	delete input_stream;


}



