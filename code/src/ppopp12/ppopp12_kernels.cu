#include "ppopp12_kernels.h"
#include "gpunfautils/common.h"

__global__ void ppopp12_kernel(
	const __restrict__  int4 *transition_table,
	const int transition_table_length,

	int *state_start_position_tb, 
	//int *num_state_tb,
	int *state_compatible_group,
	int *initial_active_state_array,
	int active_state_array_size, // currently it is the same as block size

	// for output / and start always enabled. 
	int8_t *states_status, // 00    is always enabled; is output;  
    
	// input
	uint8_t *input_streams,
	int input_stream_length,
	
	// output processing
	match_entry  *match_array, // fixed size for each thread block,
	const unsigned long long int match_array_capacity,
	unsigned long long int  	*match_count,
	bool report_on
) 

{
	extern __shared__ int active_state_array[]; // size : blockSize * 2

	int *current_active_state_array = &active_state_array[0];
	int *next_active_state_array    = &active_state_array[active_state_array_size];

	int offset = threadIdx.x;
	while (offset < active_state_array_size) {
		current_active_state_array[offset] = initial_active_state_array[active_state_array_size * blockIdx.x + offset];
		offset += blockDim.x;
	}

	uint8_t *cur_input_stream = ((uint8_t*) input_streams) + (blockIdx.y * input_stream_length);

	__syncthreads();


	int current_state_start_position_tb = state_start_position_tb[blockIdx.x];
	
	for (int input_position = 0; input_position < input_stream_length; input_position++) {

		uint8_t symbol = cur_input_stream[input_position];   

		int current_active_state_array_idx = (input_position % 2 == 0) ?  0 : active_state_array_size;
		int next_active_state_array_idx    = (input_position % 2 == 0) ?  active_state_array_size : 0;

		//assert(current_active_state_array_idx != next_active_state_array_idx);
		current_active_state_array = &active_state_array[current_active_state_array_idx]; 
		next_active_state_array    = &active_state_array[next_active_state_array_idx]; 

		__syncthreads();

		offset = threadIdx.x;
		while (offset < active_state_array_size) {
			next_active_state_array[offset] = -1;
			offset += blockDim.x;
		}

		__syncthreads();

		offset = threadIdx.x;
		while (offset < active_state_array_size) {
			int node_id = current_active_state_array[offset];

			if (node_id != -1) {
				int idx_transtable = (current_state_start_position_tb + node_id) * 256 + ((int) symbol); 
				assert(idx_transtable < transition_table_length);
				assert(idx_transtable >= 0);

				int4 out_nodes = transition_table[ idx_transtable ];

				int out_nodes_arr[4];
				out_nodes_arr[0] = out_nodes.x;
				out_nodes_arr[1] = out_nodes.y;
				out_nodes_arr[2] = out_nodes.z;
				out_nodes_arr[3] = out_nodes.w;

				for (int kk = 0; kk < 4; kk++ ) {
					if (out_nodes_arr[kk] != -1) {
						int write_to = state_compatible_group[current_state_start_position_tb + out_nodes_arr[kk] ];
						next_active_state_array[write_to] = out_nodes_arr[kk];
					}
				}

				// check if current node_id is always enabled. 
				if (states_status[current_state_start_position_tb + node_id] &  (1 << 1) ) {
					int write_to = state_compatible_group[current_state_start_position_tb + node_id ];
					next_active_state_array[write_to] = node_id;
				}
			}	

			offset += blockDim.x;
		}

		__syncthreads();


 		//output processing
 		offset = threadIdx.x;
 		while (offset < active_state_array_size) {
 			int node_id = next_active_state_array[offset];
 			if (node_id != -1) {
	 			if ((states_status[current_state_start_position_tb + node_id] & 1)) {
					unsigned long long int  current_output_position = atomicAdd(match_count, 1);
					if(report_on){
						match_entry mp;
						mp.state_id = node_id;
						mp.symbol_offset = input_position;
						mp.cc_id = blockIdx.x;
						mp.stream_id = blockIdx.y;
						match_array[current_output_position] = mp;
					}
 				} 
 			}

 			offset += blockDim.x;
 		}

		__syncthreads();
	}

}




// TODO merge ``state status" to transition table. 
__global__ void ppopp12_kernel_shrreadchunk(
	const __restrict__  int4 *transition_table,
	const int transition_table_length,
	int *state_start_position_tb, 
	
	int *state_compatible_group,
	int *initial_active_state_array,
	int active_state_array_size, // currently it is the same as block size

	// for output / and start always enabled. 
	int8_t *states_status, // 00    is always enabled; is output;  
    
	// input
	uint8_t *input_stream,
	int input_stream_length,
	
	// output processing
	match_entry  *match_array, // fixed size for each thread block,
	const unsigned long long int match_array_capacity,
	unsigned long long int  	*match_count,
	bool report_on
) 

{
	extern __shared__ int active_state_array[]; // size : blockSize * 2

	int *current_active_state_array = &active_state_array[0];
	int *next_active_state_array    = &active_state_array[active_state_array_size];

	uint8_t *shrd_chunk = (uint8_t *) (active_state_array + active_state_array_size * 2);

	int offset = threadIdx.x;
	while (offset < active_state_array_size) {
		current_active_state_array[offset] = initial_active_state_array[active_state_array_size * blockIdx.x + offset];
		next_active_state_array[offset] = -1;
		offset += blockDim.x;
	}

	uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);

	int current_state_start_position_tb = state_start_position_tb[blockIdx.x];

	__syncthreads();


	int base_idx = 0;
	while (base_idx < input_stream_length) {
		int local_id = threadIdx.x;
		
		while (local_id < blockDim.x) { // read a chunk. 
			if (base_idx + local_id < input_stream_length) {
				shrd_chunk[local_id] = cur_input_stream[base_idx + local_id];	
			}
			local_id += blockDim.x;
		}

		__syncthreads();

		int end = min(base_idx + blockDim.x, input_stream_length);

		//__syncthreads();

		for (int i = base_idx; i < end; i++) {
			uint8_t symbol = shrd_chunk[i - base_idx];

			offset = threadIdx.x;
			while (offset < active_state_array_size) {
				int node_id = current_active_state_array[offset];

				if (node_id != -1) {
					int idx_transtable = (current_state_start_position_tb + node_id) * 256 + ((int) symbol); 
					assert(idx_transtable < transition_table_length);
					assert(idx_transtable >= 0);

					int4 out_nodes = transition_table[ idx_transtable ];

					int out_nodes_arr[4];
					out_nodes_arr[0] = out_nodes.x;
					out_nodes_arr[1] = out_nodes.y;
					out_nodes_arr[2] = out_nodes.z;
					out_nodes_arr[3] = out_nodes.w;

					//printf("nodeid = %d symbol = %d\n", node_id, (int) symbol);
					//printf("%d %d %d %d\n", out_nodes_arr[0], out_nodes_arr[1], out_nodes_arr[2], out_nodes_arr[3]);
					//printf("nodeid = %d readtable = %d\n", node_id, idx_transtable);
					// extend to nextblockDim
					for (int kk = 0; kk < 4; kk++ ) {
						if (out_nodes_arr[kk] != -1) {
							int write_to = state_compatible_group[current_state_start_position_tb + out_nodes_arr[kk] ];
							next_active_state_array[write_to] = out_nodes_arr[kk];

							//printf("(%c)  edge =  %d %d %d\n", symbol, node_id, out_nodes_arr[kk], write_to);
						}
					}

					// check if current node_id is always enabled. 
					if (states_status[current_state_start_position_tb + node_id] &  (1 << 1) ) {
						int write_to = state_compatible_group[current_state_start_position_tb + node_id ];
						next_active_state_array[write_to] = node_id;
					}
				}	

				offset += blockDim.x;
			}

			__syncthreads();


	 		//output processing
	 		offset = threadIdx.x;
	 		while (offset < active_state_array_size) {
	 			int node_id = next_active_state_array[offset];
	 			if (node_id != -1) {
		 			if ((states_status[current_state_start_position_tb + node_id] & 1)) {
						unsigned long long int  current_output_position = atomicAdd(match_count, 1);
						if(report_on){
							match_entry mp;
							mp.state_id = node_id;
							mp.symbol_offset = i;
							mp.cc_id = blockIdx.x ;
							mp.stream_id = blockIdx.y;
							match_array[current_output_position] = mp;
						}
	 				} 
	 			}

	 			offset += blockDim.x;
	 		}


			__syncthreads();


			offset = threadIdx.x;
			while (offset < active_state_array_size) {
				current_active_state_array[offset] = next_active_state_array[offset];
				next_active_state_array[offset] = -1;
				offset += blockDim.x;
			}

			__syncthreads();

		} // i

		base_idx += blockDim.x;

	}
}

