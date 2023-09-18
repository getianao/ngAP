#ifndef INFANT_KERNELS_H_
#define INFANT_KERNELS_H_

#include "gpunfautils/common.h"
#include <cuda.h>


/**
*
*  stores to one output array
*  thread blocks have to compete on matc
*
*
*/
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
	match_entry  *match_array, // fixed size for each thread block,
	unsigned long long int match_array_capacity,
	unsigned long long int 	*match_count,
	bool report_on


);


#endif /* INFANT_KERNELS_H_ */





