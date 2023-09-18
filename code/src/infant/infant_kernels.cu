#include "infant_kernels.h"
#include "commons/NFA.h"
#include "gpunfautils/common.h"
#include "device_funcs.h"



__global__ void infant_kernel_one_output(
	int  *src_table, 
	int  *dst_table, 
	int  *trans_table_start_position, 
    int  *symbol_trans_len,
	int  *symbol_trans_index,

    char *states_status, // 00    is always enabled; is output;  
    int  *state_start_position,
    int  *num_of_states_per_tb,

	// input streams
	uint8_t *input_streams,
	int input_stream_length,
	
	// state vector
	int *enabled_bitvec,
	//int *active_bitvec,
	int state_bitvec_length, // num of int per block

	// output processing
	match_entry  *match_array, 
	unsigned long long int match_array_capacity,
	unsigned long long int 	*match_count,
	bool report_on
)
{
	extern __shared__ int shr_bitvecs[];

	//__shared__ int output_tb_buffer[OUTPUT_BUFFER_TB];


	//int global_tid_x = blockDim.x * blockIdx.x + threadIdx.x;
	int local_tid = threadIdx.x;

	int nfa_chunk_id = blockIdx.x;
	int input_stream_id = blockIdx.y;

	int *shr_enabled_bitvec = &shr_bitvecs[0];
	int *shr_active_bitvec  = &shr_bitvecs[state_bitvec_length];


	int state_start_position_cur_tb =  state_start_position[nfa_chunk_id] ;
	/*if (local_tid == 0) {
		printf("kernel state_bitvec_length %d \n", state_bitvec_length);
		printf("kernel output capacity %d \n", match_array_capacity);
		printf("kernel input length %d \n", input_stream_length);
	}*/

	//printf("%d\n", enabled_bitvec[0]);

	int offset = local_tid;
	while (offset < state_bitvec_length) {
		shr_active_bitvec[offset] = enabled_bitvec[ nfa_chunk_id * state_bitvec_length + offset ];
		shr_enabled_bitvec[offset] = 0;
		offset += blockDim.x;
	}

	__syncthreads();


	//if (local_tid == 0) {
	//	printf ("init nfa %d statevector %d %d     readfrom%d \n", nfa_chunk_id,  shr_enabled_bitvec[0], shr_active_bitvec[0], nfa_chunk_id * state_bitvec_length);
	//}

	//if (local_tid == 0) {
	//	printf("nfaid = %d num_of_states = %d \n", nfa_chunk_id, num_of_states_per_tb[nfa_chunk_id] ); 
	//}

	for (int input_position = 0; input_position < input_stream_length; input_position++) {
		//printf("%d / %d \n", input_position, input_stream_length);
		
		uint8_t symbol = input_streams[input_stream_id * input_stream_length + input_position];   
		
		//if (local_tid == 0) {
		//	printf("%d %c\n", input_position, symbol);
		//}

		int num_transition_current_symbol = symbol_trans_len[  256 * nfa_chunk_id +   symbol ];

		//printf("nt = %d barrd = %d\n ", num_transition_current_symbol, base_address);

		//assert(! (num_transition_current_symbol > 0 && symbol_trans_index[  256 * nfa_chunk_id + symbol   ] == -1) );
		// symbol tran len and symbol trans index have problems.  

		int current_enabled = (input_position % 2 == 0) ?  state_bitvec_length : 0;
		int current_active  = (input_position % 2 == 0) ?  0 : state_bitvec_length;  
		// swap enabled vector and active vector. 

		shr_enabled_bitvec = &shr_bitvecs[current_enabled];
	    shr_active_bitvec  = &shr_bitvecs[current_active];


		/*if (local_tid == 0) {
			printf ("nfa %d cycle %d statevector %d %d\n", nfa_chunk_id, input_position, shr_enabled_bitvec[0], shr_active_bitvec[0]);
		    printf ("nfa %d cycle %d statevector %d %d\n", nfa_chunk_id, input_position, shr_enabled_bitvec[1], shr_active_bitvec[1]);
		}*/

		__syncthreads();

	    offset = local_tid;
	    while (offset < state_bitvec_length) {
	    	shr_active_bitvec[offset] = 0;
	    	offset += blockDim.x;
	    } // fill to zero

	
	    __syncthreads();

		int base_address = trans_table_start_position[nfa_chunk_id] + symbol_trans_index[  256 * nfa_chunk_id + symbol   ];

		offset = local_tid;
		while (offset < num_transition_current_symbol) {
			int src_state_id = src_table[base_address + offset];
			
			if (get_bit(shr_enabled_bitvec, state_bitvec_length, src_state_id)) { 

				//printf("transfer on symbol (%c) %d %d\n", symbol, src_state_id, dst_state_id);
				int dst_state_id = dst_table[base_address + offset];

				set_bit(shr_active_bitvec, state_bitvec_length, dst_state_id);

			}

			offset += blockDim.x;
		} // state transition. 
		
		//if (local_tid == 0) {
		//	printf ("after nfa %d cycle %d statevector %d %d\n", nfa_chunk_id, input_position, shr_enabled_bitvec[0], shr_active_bitvec[0]);
		//}

		__syncthreads();

		// output processing
		offset = local_tid;
		while (offset < num_of_states_per_tb[nfa_chunk_id]) {
			if (get_bit(shr_enabled_bitvec, state_bitvec_length, offset) && (states_status[ state_start_position_cur_tb + offset] & 1 == 1)) { // is output
			//printf("%d %d %d\n", nfa_chunk_id, offset, input_position);
				unsigned long long int current_output_position = atomicAdd(match_count, 1);
				if(report_on) {
					assert(current_output_position < match_array_capacity);
					match_array[current_output_position].state_id = offset;
					match_array[current_output_position].symbol_offset = input_position;
					match_array[current_output_position].cc_id = nfa_chunk_id;
					match_array[current_output_position].stream_id = input_stream_id;
				}
			}

			offset += blockDim.x;
		}

		__syncthreads();


		// always enabled. 
	    offset = local_tid;
	    while (offset < num_of_states_per_tb[nfa_chunk_id]) {
	    	if ( states_status[state_start_position_cur_tb + offset]  &  (1 << 1)) {
	    		//printf("stateid = %d %d\n", offset, states_status[ state_start_position[nfa_chunk_id] + offset]);
				set_bit(shr_active_bitvec, state_bitvec_length, offset);
			}

			offset += blockDim.x;
	    } // for always enabled.


		__syncthreads();

	}


}

