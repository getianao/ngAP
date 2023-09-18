#ifndef PPOPP12_KERNELS_H_
#define PPOPP12_KERNELS_H_

#include "gpunfautils/common.h"
#include <cuda.h>


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
	unsigned long long int 	*match_count,
	bool report_on
) ;

__global__ void ppopp12_kernel_shrreadchunk(
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
	unsigned long long int 	*match_count,
	bool report_on
);



#endif



