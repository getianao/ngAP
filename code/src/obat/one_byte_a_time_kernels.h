#ifndef ONEBYTE_A_TIME_KERNELS_
#define ONEBYTE_A_TIME_KERNELS_

#include "gpunfautils/common.h"

const int WARP_SIZE = 32;


// default
__global__ void one_byte_at_a_time_enable_active_fetch_multiple_symbols_all_together(
        match_entry  *real_output_array,
        unsigned long long int 		*tail_of_real_output_array,
        // real report

        const  STE_dev<4>*  __restrict__  node_list,
        const  uint8_t* __restrict__  input_stream,
        const  int input_stream_length,

        const  int* __restrict__  num_of_state_array,
        const bool generate_report
)
{

    extern __shared__ bool is_active_state1[];

    //printf("%d\n", blockIdx.y);

    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);

    int num_of_state = num_of_state_array[blockIdx.x];

    if (threadIdx.x < num_of_state) {
        STE_dev<4> cur_node;
        cur_node = node_list[blockIdx.x * blockDim.x + threadIdx.x];

        is_active_state1[threadIdx.x] = (cur_node.attribute & (1 << 2)) ;

        is_active_state1[threadIdx.x + blockDim.x] = false; //(cur_node.attribute & (1 << 1)) ? true : false;


        __syncthreads();

        for (int ft = 0; ft < input_stream_length / 4; ft ++) {
            uint32_t bigsymbol = ((uint32_t *) cur_input_stream) [ft];

            for (int s = 0; s < 4; s++) {
                int symbol_pos = ft * 4 + s;
                uint8_t symbol = (uint8_t) (bigsymbol & 255 );
                bigsymbol >>= 8;

                //if (threadIdx.x == 0) {
                //	printf("%d %c\n", symbol_pos, symbol);
                //}

                if ( (cur_node.ms[symbol / 32] & (1 << (symbol % 32))) != 0  &&  is_active_state1[threadIdx.x]) {
                    for (int to = 0; to < cur_node.degree; to++) {
                        is_active_state1[ cur_node.edge_dst[to] + blockDim.x] = true;  // enable
                    }

                    //printf("sid = %d act %d\n", threadIdx.x, cur_node.edge_dst[to]);

                    if ((cur_node.attribute & 1)) {
                        unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                        if(generate_report){
                        real_output_array[write_to].symbol_offset = symbol_pos ;
                        real_output_array[write_to].state_id = threadIdx.x;
                        real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                        }
                    }
                }

                __syncthreads();

                is_active_state1[threadIdx.x] = ( (cur_node.attribute & (1 << 1)) ) || is_active_state1[threadIdx.x + blockDim.x];

                __syncthreads();
                is_active_state1[threadIdx.x + blockDim.x] = false;

                __syncthreads();
            }

        }

    }

};



// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
__global__ void OBAT_baseline_kernel2_fix_(
        match_entry  *real_output_array,
        unsigned long long int         *tail_of_real_output_array,
        // real report

        const  STE_nodeinfo_new_imp2 * __restrict__  node_list,
        const  STE_matchset_new_imp * __restrict__  node_ms,

        const  uint8_t* __restrict__  input_stream,
        const  int input_stream_length,

        const   int*  __restrict__ num_of_state_array,
        const bool generate_report
)
{
    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);

    extern __shared__ bool is_active_state1[];

    int num_of_state = num_of_state_array[blockIdx.x];

    if (threadIdx.x < num_of_state) {
        const STE_nodeinfo_new_imp2 cur_node = node_list[blockIdx.x * blockDim.x + threadIdx.x];
        const STE_matchset_new_imp ms = node_ms[blockIdx.x * blockDim.x + threadIdx.x];

        is_active_state1[threadIdx.x] = (cur_node.attribute & (1 << 2)) ;

        is_active_state1[threadIdx.x + blockDim.x] = false; //(cur_node.attribute & (1 << 1)) ? true : false;


        __syncthreads();

        for (int symbol_pos = 0; symbol_pos < input_stream_length; symbol_pos ++) {
            uint8_t symbol = cur_input_stream[symbol_pos];

            //if (threadIdx.x == 0) {
            //printf("%d %c %llu\n", symbol_pos, symbol, cur_node.edges);
            //}

            if ( (ms.ms[symbol / 32] & (1 << (symbol % 32))) != 0  &&  is_active_state1[threadIdx.x]) {
                for (int to = 0; to < cur_node.degree; to++) {
                    unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;
                    //printf("sid = %d act %d\n", threadIdx.x, edgeto);
                    is_active_state1[ edgeto + blockDim.x] = true;  // enable
                }

                //printf("sid = %d act %d\n", threadIdx.x, cur_node.edge_dst[to]);

                if ( (cur_node.attribute & 1)) {
                    unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                    if(generate_report) {
                                           real_output_array[write_to].symbol_offset = symbol_pos ;
                    real_output_array[write_to].state_id = threadIdx.x;
                    real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y; 
                    }

                }
            }

            __syncthreads();

            is_active_state1[threadIdx.x] = ( (cur_node.attribute & (1 << 1)) ) || is_active_state1[threadIdx.x + blockDim.x];

            __syncthreads();
            is_active_state1[threadIdx.x + blockDim.x] = false;

            __syncthreads();
        }
    }
}


__global__ void obat_matchset_compression_new_imp(
        match_entry  *real_output_array,
        unsigned long long int         *tail_of_real_output_array,
        // real report

        const  STE_nodeinfo_new_imp * __restrict__  node_list,
        const  STE_matchset_new_imp * __restrict__  node_ms,

        const  uint8_t* __restrict__  input_stream,
        const  int input_stream_length,

        const  int * __restrict__  num_of_state_array,
        const bool generate_report
)
{
    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);

    extern __shared__ bool is_active_state1[];

    int num_of_state = num_of_state_array[blockIdx.x];

    if (threadIdx.x < num_of_state) {

        const STE_nodeinfo_new_imp cur_node = node_list[blockIdx.x * blockDim.x + threadIdx.x];

        STE_matchset_new_imp cur_node_ms;

        const bool complement = ( cur_node.attribute & ((1 << 4) | (1 << 3))  )   ==  ((1 << 4) | (1 << 3));
        const bool complete = ((cur_node.attribute & (1 << 3)) != 0 ) && (!((cur_node.attribute & (1 << 4)) != 0 ) );

        if (!complete && !complement) {
            cur_node_ms = node_ms[blockIdx.x * blockDim.x + threadIdx.x];
        }

        is_active_state1[threadIdx.x] = (cur_node.attribute & (1 << 2)) ;

        is_active_state1[threadIdx.x + blockDim.x] = false; //(cur_node.attribute & (1 << 1)) ? true : false;

        __syncthreads();

        for (int symbol_pos = 0; symbol_pos < input_stream_length; symbol_pos ++) {
            uint8_t symbol = cur_input_stream[symbol_pos];

            bool in_range = symbol >= cur_node.start  &&   symbol <= cur_node.end;

            if (  ( (complement && !in_range) || (complete   &&  in_range) ||  (!complete && !complement && cur_node_ms.ms[symbol / 32] & (1 << (symbol % 32)))  )
                  && is_active_state1[threadIdx.x]) {
                for (int to = 0; to < cur_node.degree; to++) {
                    unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;
                    //printf("sid = %d act %d\n", threadIdx.x, edgeto);
                    is_active_state1[ edgeto + blockDim.x] = true;  // enable
                }

                if ( (cur_node.attribute & 1)) {
                    unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                    if(generate_report) {
                                            real_output_array[write_to].symbol_offset = symbol_pos ;
                    real_output_array[write_to].state_id = threadIdx.x;
                    real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                    }

                }
            }

            __syncthreads();

            is_active_state1[threadIdx.x] = ( (cur_node.attribute & (1 << 1)) ) || is_active_state1[threadIdx.x + blockDim.x];

            __syncthreads();
            is_active_state1[threadIdx.x + blockDim.x] = false;

            __syncthreads();
        }
    }
}



__device__ inline bool insert_val_to_array(int *arr, int &len, int val) {
    if (val == -1) {
        return false;
    }

    for (int i = 0; i < len; i++) {
        if (arr[i] == val) {
            return false;
        }
    }

    arr[len++] = val;

    return true;
};


// COLD AFTER HOT
__global__
void hotcold_nodup_queue_mc_cold_after_hot(
    match_entry  *real_output_array,
    unsigned long long int         *tail_of_real_output_array,
    // real report

    // input stream 
    const  uint8_t* __restrict__  input_stream,
    const  int input_stream_length, 
    // end input stream
    
    const int *__restrict__ start_offset_node_list,
    const int *__restrict__ num_hot_states,

    const  STE_nodeinfo_new_imp * __restrict__  node_list,
    const  STE_matchset_new_imp * __restrict__  node_ms,

    const bool generate_report,
    const int queuesize,
    const int dedup_bitset_length
) 
{

    extern __shared__ bool shared_pool[];
    bool *is_active_state1 = shared_pool;

    __shared__ int *cold_warp_active_queue;
    __shared__ int *cold_warp_next_active_queue;

    if (threadIdx.x == 0 ) {
        cold_warp_active_queue = (int *) (shared_pool + 2 * blockDim.x);
        cold_warp_next_active_queue = &cold_warp_active_queue[queuesize];
    }

    unsigned int *dedup_bs = ((unsigned int * ) (shared_pool + 2 * blockDim.x) ) + queuesize * 2;

    int tt = threadIdx.x;
    while (tt < dedup_bitset_length) {
        dedup_bs[tt] = 0;
        tt += blockDim.x;
    }

    int start_offset = start_offset_node_list[blockIdx.x];
    int num_of_state = num_hot_states[blockIdx.x];

    // fixed. 
    STE_nodeinfo_new_imp cur_node;
    STE_matchset_new_imp cur_node_ms;

    bool complement;
    bool complete;

    if (threadIdx.x < num_of_state) {
        cur_node = node_list[start_offset + threadIdx.x];   
        
        complement = ( cur_node.attribute & ((1 << 4) | (1 << 3))  )   ==  ((1 << 4) | (1 << 3)); 
        complete = ((cur_node.attribute & (1 << 3)) != 0 ) && (!((cur_node.attribute & (1 << 4)) != 0 ) ); 
                
        if (!complete && !complement) {
            cur_node_ms = node_ms[start_offset + threadIdx.x];
        }
        
        is_active_state1[threadIdx.x] = (cur_node.attribute & (1 << 2)) ;   // enable
        is_active_state1[threadIdx.x + blockDim.x] = false; // false; active
    }

    __shared__ int tail_of_the_cold_warp_active_queue;
    __shared__ int tail_of_the_cold_warp_next_active_queue;

    if (threadIdx.x == 0) {
        tail_of_the_cold_warp_active_queue = 0;
        tail_of_the_cold_warp_next_active_queue = 0;
    }

    __syncthreads();

    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);


    // main loop
    for (int symbol_pos = 0; symbol_pos < input_stream_length; symbol_pos ++) {
        uint8_t symbol = cur_input_stream[symbol_pos];

        //if (threadIdx.x == 0) {
        //    printf("%d %c\n", symbol_pos, symbol);
        //}

        if (threadIdx.x == 1) {
            int *tmp = cold_warp_active_queue;
            cold_warp_active_queue = cold_warp_next_active_queue;
            cold_warp_next_active_queue = tmp;
        }

        if (threadIdx.x == 0) {
            tail_of_the_cold_warp_active_queue = tail_of_the_cold_warp_next_active_queue;
            tail_of_the_cold_warp_next_active_queue = 0;
        }

        int tt = threadIdx.x;
        while (tt < dedup_bitset_length) {
            dedup_bs[tt] = 0;
            tt += blockDim.x;
        }

        __syncthreads();

        
        if (threadIdx.x < num_of_state) { // hot state
            bool in_range = symbol >= cur_node.start  &&   symbol <= cur_node.end;

            if (  ( (complement && !in_range) || (complete   &&  in_range) 
                ||  (!complete && !complement && cur_node_ms.ms[symbol / 32] & (1 << (symbol % 32)))  )  && is_active_state1[threadIdx.x]) {
                for (int to = 0; to < cur_node.degree; to++) {
                    unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;
                    //printf("active -- bitset %d\n", edgeto);

                    if (edgeto < num_of_state) {
                        is_active_state1[ edgeto + blockDim.x] = true;  // enable   
                    } else { // hot to cold
                        int n_bit = edgeto - num_of_state;
                        assert(n_bit >= 0);
                        int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                        if ((!(is_in_the_queue & (1 << (n_bit % 32))))) {
                            int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                            assert(cur_tail < queuesize);
                            cold_warp_next_active_queue[cur_tail] = edgeto;
                            //printf("push to wl %d\n", edgeto);
                        }
                    }
                }

                if ( (cur_node.attribute & 1)) {
                    unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                    if(generate_report) {
                                            real_output_array[write_to].symbol_offset = symbol_pos ;
                    real_output_array[write_to].state_id = threadIdx.x; 
                    real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                    }

                }
            }
        }

    
        //__syncthreads();

        // reusable. 
        STE_nodeinfo_new_imp cur_cold_node; 
        STE_matchset_new_imp cur_cold_node_ms;
        bool complement_cold;
        bool complete_cold;

        tt = threadIdx.x;

        //if (threadIdx.x == 0 ){
        //    printf("tail_of_the_cold_warp_active_queue = %d\n", tail_of_the_cold_warp_active_queue);
        //}

        while (tt < tail_of_the_cold_warp_active_queue) {
            int enabled_sid = cold_warp_active_queue[tt];
            //printf("enable sid = %d %d\n", enabled_sid, cold_warp_next_active_queue[tt]);

            cur_cold_node = node_list[start_offset + enabled_sid];   
    
            complement_cold = ( cur_cold_node.attribute & ((1 << 4) | (1 << 3))  )   ==  ((1 << 4) | (1 << 3)); 
            complete_cold = ((cur_cold_node.attribute & (1 << 3)) != 0 ) && (!((cur_cold_node.attribute & (1 << 4)) != 0 ) ); 
            
            if (!complement_cold && !complete_cold) {
                cur_cold_node_ms = node_ms[start_offset + enabled_sid];
            }

            bool in_range = symbol >= cur_cold_node.start  &&   symbol <= cur_cold_node.end;

            if (  ( (complement_cold && !in_range) || (complete_cold   &&  in_range) 
                   ||  (!complete_cold && !complement_cold && cur_cold_node_ms.ms[symbol / 32] & (1 << (symbol % 32)))  )  ) {

                for (int to = 0; to < cur_cold_node.degree; to++) {
                    unsigned int edgeto = ( cur_cold_node.edges >> (16 * to) )  & 65535;

                    if (edgeto < num_of_state) {
                        is_active_state1[ edgeto + blockDim.x] = true;  // enable   
                    } else { // cold to cold
                        int n_bit = edgeto - num_of_state;
                        assert(n_bit >= 0);
                        int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                        if (!(is_in_the_queue & (1 << (n_bit % 32)))) {
                            int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                            assert(cur_tail < queuesize);
                            cold_warp_next_active_queue[cur_tail] = edgeto;
                        }
                    }
                }

                if ( (cur_cold_node.attribute & 1)) {
                    unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                    if(generate_report) {
                                           real_output_array[write_to].symbol_offset = symbol_pos ;
                    real_output_array[write_to].state_id = enabled_sid; 
                    real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y; 
                    }


                }
                
            }

            tt += blockDim.x;
        }


        __syncthreads();

        if (threadIdx.x < num_of_state) {
            is_active_state1[threadIdx.x] = ( (cur_node.attribute & (1 << 1)) ) || is_active_state1[threadIdx.x + blockDim.x];
            is_active_state1[threadIdx.x + blockDim.x] = false;
        }

        __syncthreads();

    } 
};



template <bool run_cold_stage=true>
__global__
void hotstart_ea_kernel(
        match_entry  *real_output_array,
        unsigned long long int         *tail_of_real_output_array,
        // real report

        // input stream
        const  uint8_t* __restrict__  input_stream,
        const  int input_stream_length,
        // end input stream

        const int *__restrict__ start_offset_node_list,
        const int *__restrict__ num_hot_states,

        const  STE_nodeinfo_new_imp * __restrict__  node_list,
        const  STE_matchset_new_imp * __restrict__  node_ms,

        const bool generate_report,
        const int queuesize,
        const int dedup_bitset_length,

        const int *remap_input_stream_array
)
{

    extern __shared__ bool shared_pool[];

    __shared__ int *cold_warp_active_queue;
    cold_warp_active_queue  = (int *) (shared_pool);
    __shared__ int *cold_warp_next_active_queue;
    cold_warp_next_active_queue  = &cold_warp_active_queue[queuesize];

    unsigned int *dedup_bs = (unsigned int * ) (cold_warp_active_queue + 2 * queuesize);
    int start_offset = start_offset_node_list[blockIdx.x];
    int num_of_state = num_hot_states[blockIdx.x];

    // fixed.
    STE_nodeinfo_new_imp cur_node;
    STE_matchset_new_imp cur_node_ms;

    bool complement;
    bool complete;

    if (threadIdx.x < num_of_state) {
        cur_node = node_list[start_offset + threadIdx.x];

        complement = ( cur_node.attribute & ((1 << 4) | (1 << 3))  )   ==  ((1 << 4) | (1 << 3));
        complete = ((cur_node.attribute & (1 << 3)) != 0 ) && (!((cur_node.attribute & (1 << 4)) != 0 ) );

        if (!complete && !complement) {
            cur_node_ms = node_ms[start_offset + threadIdx.x];
        }
    }

    __shared__ int tail_of_the_cold_warp_active_queue;
    __shared__ int tail_of_the_cold_warp_next_active_queue;

    if (threadIdx.x == 0) {
        tail_of_the_cold_warp_active_queue = 0;
        tail_of_the_cold_warp_next_active_queue = 0;
    }

    __syncthreads();

    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (remap_input_stream_array[blockIdx.y] * input_stream_length);

    // main loop
    for (int symbol_pos = 0; symbol_pos < input_stream_length; symbol_pos ++) {
        uint8_t symbol = cur_input_stream[symbol_pos];

        if (threadIdx.x == 1) {
            int *tmp = cold_warp_active_queue;
            cold_warp_active_queue = cold_warp_next_active_queue;
            cold_warp_next_active_queue = tmp;
        }

        if (threadIdx.x == 0) {
            tail_of_the_cold_warp_active_queue = tail_of_the_cold_warp_next_active_queue;
            tail_of_the_cold_warp_next_active_queue = 0;
        }

        int tt = threadIdx.x;
        while (tt < dedup_bitset_length) {
            dedup_bs[tt] = 0;
            tt += blockDim.x;
        }

        __syncthreads();

        if (threadIdx.x < num_of_state) { // hot state
            bool in_range = symbol >= cur_node.start  &&   symbol <= cur_node.end;

            //if (cur_node.attribute)  // TODO

            // If the node is not always enabled, we need to add another condition.
            if (  ( (complement && !in_range) || (complete   &&  in_range) ||  (!complete && !complement && cur_node_ms.ms[symbol / 32] & (1 << (symbol % 32)))  )) {
                for (int to = 0; to < cur_node.degree; to++) {
                    unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;

                    if (edgeto >= num_of_state) { // hot to cold
                        int n_bit = edgeto - num_of_state;
                        //assert(n_bit >= 0);
                        int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                        if ((!(is_in_the_queue & (1 << (n_bit % 32))))) {
                            int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                            assert(cur_tail < queuesize);
                            cold_warp_next_active_queue[cur_tail] = edgeto;
                        }
                    }
                }

                if ( (cur_node.attribute & 1)) {
                    unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                    if(generate_report) {
                                           real_output_array[write_to].symbol_offset = symbol_pos ;
                    real_output_array[write_to].state_id = threadIdx.x;
                    real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y; 
                    }

                }
            }
        }

        if (run_cold_stage) {
            // reusable.
            STE_nodeinfo_new_imp cur_cold_node;
            STE_matchset_new_imp cur_cold_node_ms;
            bool complement_cold;
            bool complete_cold;

            tt = threadIdx.x;
            while (tt < tail_of_the_cold_warp_active_queue) {
                assert(tt < queuesize);
                int enabled_sid = cold_warp_active_queue[tt];

                cur_cold_node = node_list[start_offset + enabled_sid];

                complement_cold = ( cur_cold_node.attribute & ((1 << 4) | (1 << 3))  )   ==  ((1 << 4) | (1 << 3));
                complete_cold = ((cur_cold_node.attribute & (1 << 3)) != 0 ) && (!((cur_cold_node.attribute & (1 << 4)) != 0 ) );

                if (!complement_cold && !complete_cold) {
                    cur_cold_node_ms = node_ms[start_offset + enabled_sid];
                }

                bool in_range = symbol >= cur_cold_node.start  &&   symbol <= cur_cold_node.end;

                if (  ( (complement_cold && !in_range) || (complete_cold   &&  in_range) ||  (!complete_cold && !complement_cold && cur_cold_node_ms.ms[symbol / 32] & (1 << (symbol % 32)))  )  ) {

                    for (int to = 0; to < cur_cold_node.degree; to++) {
                        unsigned int edgeto = ( cur_cold_node.edges >> (16 * to) )  & 65535;

                        if (edgeto >= num_of_state) { // cold to cold
                            int n_bit = edgeto - num_of_state;
                            //assert(n_bit >= 0);
                            int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                            if (!(is_in_the_queue & (1 << (n_bit % 32)))) {
                                int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                                assert(cur_tail < queuesize);
                                cold_warp_next_active_queue[cur_tail] = edgeto;
                            }
                        }
                    }

                    if ( (cur_cold_node.attribute & 1)) {
                        unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                        if(generate_report) {
                                                    real_output_array[write_to].symbol_offset = symbol_pos ;
                        real_output_array[write_to].state_id = enabled_sid;
                        real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                        }

                    }

                }

                tt += blockDim.x;
            }
        }

        __syncthreads();
    }
};


template <bool run_cold_stage=true>
__global__
void hotstart_ea_kernel_without_MC2(
        match_entry  *real_output_array,
        unsigned long long int         *tail_of_real_output_array,
        // real report

        // input stream
        const  uint8_t* __restrict__  input_stream,
        const  int input_stream_length,
        // end input stream

        const int *__restrict__ start_offset_node_list,
        const int *__restrict__ num_hot_states,

        const  STE_nodeinfo_new_imp2 * __restrict__  node_list,
        const  STE_matchset_new_imp * __restrict__  node_ms,

        const bool generate_report,
        const int queuesize,
        const int dedup_bitset_length,
        const int *remap_input_stream_array
)
{

    extern __shared__ bool shared_pool[];

    __shared__ int *cold_warp_active_queue;
    cold_warp_active_queue  = (int *) (shared_pool);

    __shared__ int *cold_warp_next_active_queue;
    cold_warp_next_active_queue  = &cold_warp_active_queue[queuesize];
    unsigned int *dedup_bs = (unsigned int * ) (cold_warp_active_queue + 2 * queuesize);

    int start_offset = start_offset_node_list[blockIdx.x];
    int num_of_state = num_hot_states[blockIdx.x];

    STE_nodeinfo_new_imp2 cur_node;
    STE_matchset_new_imp ms;

    if (threadIdx.x < num_of_state) {
        cur_node = node_list[start_offset + threadIdx.x];
        ms = node_ms[start_offset + threadIdx.x];
    }

    __shared__ int tail_of_the_cold_warp_active_queue;
    __shared__ int tail_of_the_cold_warp_next_active_queue;

    if (threadIdx.x == 0) {
        tail_of_the_cold_warp_active_queue = 0;
        tail_of_the_cold_warp_next_active_queue = 0;
    }

    __syncthreads();

    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (remap_input_stream_array[blockIdx.y] * input_stream_length);

    // main loop
    for (int symbol_pos = 0; symbol_pos < input_stream_length; symbol_pos ++) {
        uint8_t symbol = cur_input_stream[symbol_pos];

        if (threadIdx.x == 1) {
            int *tmp = cold_warp_active_queue;
            cold_warp_active_queue = cold_warp_next_active_queue;
            cold_warp_next_active_queue = tmp;
        }

        if (threadIdx.x == 0) {
            tail_of_the_cold_warp_active_queue = tail_of_the_cold_warp_next_active_queue;
            tail_of_the_cold_warp_next_active_queue = 0;
        }

        int tt = threadIdx.x;
        while (tt < dedup_bitset_length) {
            dedup_bs[tt] = 0;
            tt += blockDim.x;
        }

        __syncthreads();

        if (threadIdx.x < num_of_state) { // hot state
            if ( (ms.ms[symbol / 32] & (1 << (symbol % 32))) ) {
                for (int to = 0; to < cur_node.degree; to++) {
                    unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;

                    if (edgeto >= num_of_state) { // hot to cold
                        int n_bit = edgeto - num_of_state;
                        //assert(n_bit >= 0);
                        int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                        if ((!(is_in_the_queue & (1 << (n_bit % 32))))) {
                            int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                            //assert(cur_tail < queuesize);
                            cold_warp_next_active_queue[cur_tail] = edgeto;
                        }
                    }
                }

                if ( (cur_node.attribute & 1)) {
                    unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                    if(generate_report) {
                                           real_output_array[write_to].symbol_offset = symbol_pos ;
                    real_output_array[write_to].state_id = threadIdx.x;
                    real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y; 
                    }

                }
            }
        }

        if (run_cold_stage) {

            STE_nodeinfo_new_imp2 cur_cold_node;
            STE_matchset_new_imp cur_cold_ms;

            tt = threadIdx.x;
            while (tt < tail_of_the_cold_warp_active_queue) {
                int enabled_sid = cold_warp_active_queue[tt];
                cur_cold_node = node_list[start_offset + enabled_sid];
                cur_cold_ms = node_ms[start_offset + enabled_sid];

                if ( (cur_cold_ms.ms[symbol / 32] & (1 << (symbol % 32)))) {

                    for (int to = 0; to < cur_cold_node.degree; to++) {
                        unsigned int edgeto = ( cur_cold_node.edges >> (16 * to) )  & 65535;

                        if (edgeto >= num_of_state) { // cold to cold
                            int n_bit = edgeto - num_of_state;
                            //assert(n_bit >= 0);
                            int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                            if (!(is_in_the_queue & (1 << (n_bit % 32)))) {
                                int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                                //assert(cur_tail < queuesize);
                                cold_warp_next_active_queue[cur_tail] = edgeto;
                            }
                        }
                    }

                    if ( (cur_cold_node.attribute & 1)) {
                        unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                        if(generate_report) {
                        real_output_array[write_to].symbol_offset = symbol_pos ;
                        real_output_array[write_to].state_id = enabled_sid;
                        real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                        }

                    }

                }

                tt += blockDim.x;
            }
        }

        __syncthreads();
    }

};


template <bool enable_cold_stage = true>
__global__
void hotstart_transition_table(
        match_entry  *real_output_array,
        unsigned long long int         *tail_of_real_output_array,
        // real report

        // input stream
        const  uint8_t* __restrict__  input_stream,
        const  int input_stream_length,
        // end input stream

        const int *__restrict__ num_hot_states,

        const bool generate_report,
        const int queuesize,
        const int dedup_bitset_length,

        const unsigned long long * __restrict__ transition_table,
        const char * __restrict is_report,
        const int * __restrict__ node_start_position_for_tb
)
{
    //if (threadIdx.x == 0) {
    //    printf("dedup bitset length = %d\n", dedup_bitset_length );
    //}

    extern __shared__ bool shared_pool[];
    int *cold_warp_active_queue = (int *) shared_pool;
    int *cold_warp_next_active_queue = &cold_warp_active_queue[queuesize];
    unsigned int *dedup_bs = (unsigned int * ) (cold_warp_active_queue + 2 * queuesize);

    int num_of_starting_state = num_hot_states[blockIdx.x];

    __shared__ int tail_of_the_cold_warp_active_queue;
    __shared__ int tail_of_the_cold_warp_next_active_queue;

    tail_of_the_cold_warp_active_queue = 0;
    tail_of_the_cold_warp_next_active_queue = 0;

    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);

    int node_start_pos = node_start_position_for_tb[blockIdx.x];

    //if (threadIdx.x == 0) {
    //    printf("node_start_pos = %d\n", node_start_pos );
    //}

    __syncthreads();


    // main loop
    for (int symbol_pos = 0; symbol_pos < input_stream_length; symbol_pos ++) {
        uint8_t symbol = cur_input_stream[symbol_pos];

        //printf("%d %c\n", symbol_pos, symbol);

        if (threadIdx.x == 0) {
            tail_of_the_cold_warp_active_queue = tail_of_the_cold_warp_next_active_queue;
            tail_of_the_cold_warp_next_active_queue = 0;
//            if (threadIdx.x == 0 && blockIdx.y == 0) {
//            	printf("tailnow = %d tailnext = %d\n", tail_of_the_cold_warp_active_queue, tail_of_the_cold_warp_next_active_queue);
//            }
        }

        int tt = threadIdx.x;
        while (tt < dedup_bitset_length) {
            dedup_bs[tt] = 0;
            tt += blockDim.x;
        }

        __syncthreads();

        if (threadIdx.x < num_of_starting_state) {
            // hot stage
            int idx_transtable = node_start_pos * 256 + threadIdx.x * 256 + (int) symbol;

            unsigned long long out_nodes = transition_table[idx_transtable];

            for (int to = 0; to < 4; to++) {
                unsigned int edgeto = ( out_nodes >> (16 * to) ) & 65535;

                if (edgeto != EMPTY_ENTRY) {
                    if ( edgeto >= num_of_starting_state) { // hot to cold
                        // push to worklist

                        int n_bit = edgeto - num_of_starting_state;
                        //printf("edgeto = %d nbit = %d\n", edgeto, n_bit);

                        int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                        if ((!(is_in_the_queue & (1 << (n_bit & 31))))) {
                            int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                            assert(cur_tail < queuesize);
                            cold_warp_next_active_queue[cur_tail] = edgeto;
                        }
                    }

                    if ((is_report[edgeto + node_start_pos] & 1)) {
                        unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                        if(generate_report) {
                                               real_output_array[write_to].symbol_offset = symbol_pos;
                        real_output_array[write_to].state_id = edgeto;
                        real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;     
                        }

                    }
                }
            }
        }

        if (enable_cold_stage) {
            tt = threadIdx.x;
            while (tt < tail_of_the_cold_warp_active_queue) {
                int active_sid = cold_warp_active_queue[tt];
                int idx_transtable = node_start_pos * 256 + active_sid * 256 + (int) symbol;
                unsigned long long out_nodes = transition_table[idx_transtable];

                for (int to = 0; to < 4; to++) {
                    unsigned int edgeto = (out_nodes >> (16 * to)) & 65535;
                    if (edgeto != EMPTY_ENTRY) {
                        if (edgeto >= num_of_starting_state) { // cold to cold
                            int n_bit = edgeto - num_of_starting_state;
                            //printf("nbit = %d\n", n_bit);
                            int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                            if ((!(is_in_the_queue & (1 << (n_bit & 31))))) {
                                int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                                assert(cur_tail < queuesize);
                                cold_warp_next_active_queue[cur_tail] = edgeto;
                            }
                        }

                        if ((is_report[edgeto + node_start_pos] & 1)) {
                            unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                            if(generate_report) {
                                                            real_output_array[write_to].symbol_offset = symbol_pos;
                            real_output_array[write_to].state_id = edgeto;
                            real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                            }

                        }
                    }
                }

                tt += blockDim.x;
            }

            __syncthreads();

            tt = threadIdx.x;
            while (tt < tail_of_the_cold_warp_next_active_queue) {
                cold_warp_active_queue[tt] = cold_warp_next_active_queue[tt];
                tt += blockDim.x;
            }

            __syncthreads();
        }

        __syncthreads();

    }

}


__global__ void obat_only_read_input_stream(
        const uint8_t * __restrict__  input_stream,
        const  int input_stream_length
)
{
    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);

    for (int i = 0; i < input_stream_length; i++) {
        cur_input_stream[i] += 1;
        __syncthreads();
    }
}



__global__ void obat_only_read_input_stream_shr(
        const uint8_t * __restrict__  input_stream,
        const  int input_stream_length
)
{
    extern __shared__ uint8_t shrd_chunk[];
    // assume chunk size == blockDim.x

    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);



    int base_idx = 0;

    while (base_idx < input_stream_length) {
        int local_id = threadIdx.x;

        while (local_id < blockDim.x) {
            if (base_idx + local_id < input_stream_length) {
                shrd_chunk[local_id] = cur_input_stream[base_idx + local_id];
            }
            local_id += blockDim.x;
        }

        int end = min(base_idx + blockDim.x, input_stream_length);

        for (int i = base_idx; i < end; i++) {
            shrd_chunk[i - base_idx] += 1;
            cur_input_stream[i] = shrd_chunk[i - base_idx];
            __syncthreads();
        }

        base_idx += blockDim.x;

    }
}




__global__ void one_byte_at_a_time_enable_active_fetch_multiple_symbols_all_together_compress_edges_hotcold_queue(
        match_entry  *real_output_array,
        unsigned long long int         *tail_of_real_output_array,
        // real report

        // input stream
        const  uint8_t* __restrict__  input_stream,
        const  int input_stream_length,
        // end input stream

        const int *__restrict__ start_offset_node_list,
        const int *__restrict__ num_hot_states,

        const  STE_dev4* __restrict__   node_list,

        const bool generate_report,
        const int queuesize
)

{

    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);

    extern __shared__ bool shared_pool[];

    bool *is_active_state1 = shared_pool;

    int *cold_warp_active_queue = (int *) (shared_pool + 2 * blockDim.x - WARP_SIZE * 2); // this is a queue;

    int *cold_warp_next_active_queue = &cold_warp_active_queue[queuesize];


    int start_offset = start_offset_node_list[blockIdx.x];

    int num_of_state = num_hot_states[blockIdx.x];

    STE_dev4 cur_node;

    if (threadIdx.x < num_of_state && threadIdx.x / WARP_SIZE != blockDim.x / WARP_SIZE - 1) { // the last warp of the block
        cur_node = node_list[start_offset + threadIdx.x];

        is_active_state1[threadIdx.x] = (cur_node.attribute & (1 << 2)) ;	// enable
        is_active_state1[threadIdx.x + blockDim.x - WARP_SIZE] = false; // false; active
    }

    __shared__ int tail_of_the_cold_warp_active_queue;
    __shared__ int tail_of_the_cold_warp_next_active_queue;

    tail_of_the_cold_warp_active_queue = 0;
    tail_of_the_cold_warp_next_active_queue = 0;

    __syncthreads();

    // main loop
    for (int symbol_pos = 0; symbol_pos < input_stream_length; symbol_pos ++) {
        uint8_t symbol = cur_input_stream[symbol_pos];

        tail_of_the_cold_warp_next_active_queue = 0;
        //printf("tail cur active queue = %d \n", tail_of_the_cold_warp_active_queue);

        __syncthreads();

        if (threadIdx.x / WARP_SIZE == blockDim.x / WARP_SIZE - 1 ) { // cold warp

            int tt = threadIdx.x % WARP_SIZE;

            while (tt < tail_of_the_cold_warp_active_queue) {
                int enabled_sid = cold_warp_active_queue[tt];
                //printf("sid in queue = %d symbol = %c tid = %d\n", enabled_sid, symbol, threadIdx.x);
                cur_node = node_list[start_offset + enabled_sid]; // ----- we can have a transition table especially for the cold states.

                if ( (cur_node.ms[symbol / 32] & (1 << (symbol % 32))) != 0) {
                    for (int to = 0; to < cur_node.degree; to++) {
                        unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;

                        if (edgeto < num_of_state) {
                            is_active_state1[ edgeto + blockDim.x - WARP_SIZE] = true;  // enable
                        } else { // cold to cold
                            int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                            assert(cur_tail < queuesize);

                            //printf("cold enables cold cur_tail = %d\n", cur_tail);
                            cold_warp_next_active_queue[cur_tail] = edgeto;
                        }
                    }

                    if ( (cur_node.attribute & 1)) {
                        unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                        if(generate_report) {
                                                    real_output_array[write_to].symbol_offset = symbol_pos ;
                        real_output_array[write_to].state_id = enabled_sid;
                        real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                        }

                    }

                }

                tt += WARP_SIZE;

            }

        }
        else { // normal warps
            if (threadIdx.x < num_of_state) {
                if ( (cur_node.ms[symbol / 32] & (1 << (symbol % 32))) != 0  &&  is_active_state1[threadIdx.x]) {
                    for (int to = 0; to < cur_node.degree; to++) {
                        unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;
                        //printf("sid = %d act %d\n", threadIdx.x, edgeto);

                        if (edgeto < num_of_state) {
                            is_active_state1[ edgeto + blockDim.x - WARP_SIZE] = true;  // enable
                        } else { // hot to cold
                            int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);

                            assert(cur_tail < queuesize);
                            cold_warp_next_active_queue[cur_tail] = edgeto;

                        }

                    }

                    //printf("sid = %d act %d\n", threadIdx.x, cur_node.edge_dst[to]);
                    //printf("state = %d is activated\n", threadIdx.x);

                    if ( (cur_node.attribute & 1)) {
                        unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                        if(generate_report) {
                                                  real_output_array[write_to].symbol_offset = symbol_pos ;
                        real_output_array[write_to].state_id = threadIdx.x;
                        real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;  
                        }


                        //printf("hot warp report %d %d %d\n", symbol_pos, threadIdx.x, blockIdx.x);
                    }
                }

            }

        }

        __syncthreads();


        if (threadIdx.x / WARP_SIZE == blockDim.x / WARP_SIZE - 1 ) { // cold warp
            // do nothing
            // copy the queue from next_enable to current_enable.
            //printf("copy tail_of_the_cold_warp_next_active_queue = %d\n", tail_of_the_cold_warp_next_active_queue);

            int tt = threadIdx.x % WARP_SIZE;
            while (tt < tail_of_the_cold_warp_next_active_queue) {
                //printf("copy from %d to %d\n", cold_warp_next_active_queue[tt], cold_warp_active_queue[tt] );
                cold_warp_active_queue[tt] = cold_warp_next_active_queue[tt];
                tt += WARP_SIZE;
            }

            // we reset the next queue tail in the beginning of the main loop.
        } else {
            if (threadIdx.x < num_of_state) {
                is_active_state1[threadIdx.x] = ( (cur_node.attribute & (1 << 1)) ) || is_active_state1[threadIdx.x + blockDim.x - WARP_SIZE];

                is_active_state1[threadIdx.x + blockDim.x - WARP_SIZE] = false;

            }
        }

        __syncthreads();

        // do we need this sync thread here?

        tail_of_the_cold_warp_active_queue = tail_of_the_cold_warp_next_active_queue;

        __syncthreads();


    } // for s
}







__global__ void hotcold_nodup_queue(
        match_entry  *real_output_array,
        unsigned long long int         *tail_of_real_output_array,
        // real report

        // input stream
        const  uint8_t* __restrict__  input_stream,
        const  int input_stream_length,
        // end input stream

        const int *__restrict__ start_offset_node_list,
        const int *__restrict__ num_hot_states,

        const  STE_dev4* __restrict__   node_list,

        const bool generate_report,
        const int queuesize,
        const int dedup_bitset_length
)

{
    extern __shared__ bool shared_pool[];
    bool *is_active_state1 = shared_pool;
    int *cold_warp_active_queue = (int *) (shared_pool + 2 * blockDim.x - WARP_SIZE * 2); // this is a queue;
    int *cold_warp_next_active_queue = &cold_warp_active_queue[queuesize];

    unsigned int *dedup_bs = (unsigned int * ) (cold_warp_active_queue + 2 * queuesize);

    int tt = threadIdx.x;
    while (tt < dedup_bitset_length) {
        dedup_bs[tt] = 0;
        tt += blockDim.x;
    }

    int start_offset = start_offset_node_list[blockIdx.x];
    int num_of_state = num_hot_states[blockIdx.x];

    STE_dev4 cur_node;

    if (threadIdx.x < num_of_state && threadIdx.x / WARP_SIZE != blockDim.x / WARP_SIZE - 1) { // the last warp of the block
        cur_node = node_list[start_offset + threadIdx.x];

        is_active_state1[threadIdx.x] = (cur_node.attribute & (1 << 2)) ;	// enable
        is_active_state1[threadIdx.x + blockDim.x - WARP_SIZE] = false; // false; active
    }

    __shared__ int tail_of_the_cold_warp_active_queue;
    __shared__ int tail_of_the_cold_warp_next_active_queue;

    tail_of_the_cold_warp_active_queue = 0;
    tail_of_the_cold_warp_next_active_queue = 0;

    __syncthreads();

    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);

    // main loop
    for (int symbol_pos = 0; symbol_pos < input_stream_length; symbol_pos ++) {
        uint8_t symbol = cur_input_stream[symbol_pos];

        tail_of_the_cold_warp_next_active_queue = 0;

        tt = threadIdx.x;
        while (tt < dedup_bitset_length) {
            dedup_bs[tt] = 0;
            tt += blockDim.x;
        }

        __syncthreads();

        if (threadIdx.x / WARP_SIZE == blockDim.x / WARP_SIZE - 1 ) { // cold warp

            tt = threadIdx.x % WARP_SIZE;

            while (tt < tail_of_the_cold_warp_active_queue) {
                int enabled_sid = cold_warp_active_queue[tt];
                cur_node = node_list[start_offset + enabled_sid]; // ----- we can have a transition table especially for the cold states.

                if ( (cur_node.ms[symbol / 32] & (1 << (symbol % 32))) != 0) {
                    for (int to = 0; to < cur_node.degree; to++) {
                        unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;

                        if (edgeto < num_of_state) {
                            is_active_state1[ edgeto + blockDim.x - WARP_SIZE] = true;  // enable
                        } else { // cold to cold
                            int n_bit = edgeto - num_of_state;
                            assert(n_bit >= 0);
                            int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                            if (!(is_in_the_queue & (1 << (n_bit % 32)))) {
                                int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                                assert(cur_tail < queuesize);
                                cold_warp_next_active_queue[cur_tail] = edgeto;
                            }
                        }
                    }

                    if ( (cur_node.attribute & 1)) {
                        unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                        if(generate_report) {
                                                    real_output_array[write_to].symbol_offset = symbol_pos ;
                        real_output_array[write_to].state_id = enabled_sid;
                        real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                        }

                    }

                }

                tt += WARP_SIZE;

            }

        }
        else { // normal warps
            if (threadIdx.x < num_of_state) {
                if ( (cur_node.ms[symbol / 32] & (1 << (symbol % 32))) != 0  &&  is_active_state1[threadIdx.x]) {
                    for (int to = 0; to < cur_node.degree; to++) {
                        unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;

                        if (edgeto < num_of_state) {
                            is_active_state1[ edgeto + blockDim.x - WARP_SIZE] = true;  // enable
                        } else { // hot to cold
                            int n_bit = edgeto - num_of_state;
                            assert(n_bit >= 0);
                            int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                            if ((!(is_in_the_queue & (1 << (n_bit % 32))))) {
                                int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                                assert(cur_tail < queuesize);
                                cold_warp_next_active_queue[cur_tail] = edgeto;
                            }
                        }
                    }

                    if ( (cur_node.attribute & 1)) {
                        unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                        if(generate_report) {
                                                    real_output_array[write_to].symbol_offset = symbol_pos ;
                        real_output_array[write_to].state_id = threadIdx.x;
                        real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                        }

                    }
                }

            }

        }

        __syncthreads();


        if (threadIdx.x / WARP_SIZE == blockDim.x / WARP_SIZE - 1 ) { // cold warp
            tt = threadIdx.x % WARP_SIZE;
            while (tt < tail_of_the_cold_warp_next_active_queue) {
                cold_warp_active_queue[tt] = cold_warp_next_active_queue[tt];
                tt += WARP_SIZE;
            }

            // we reset the next queue tail in the beginning of the main loop.
        } else {
            if (threadIdx.x < num_of_state) {
                is_active_state1[threadIdx.x] = ( (cur_node.attribute & (1 << 1)) ) || is_active_state1[threadIdx.x + blockDim.x - WARP_SIZE];

                is_active_state1[threadIdx.x + blockDim.x - WARP_SIZE] = false;

            }
        }

        __syncthreads();

        tail_of_the_cold_warp_active_queue = tail_of_the_cold_warp_next_active_queue;

        __syncthreads();


    }
}




__global__ void hotcold_nodup_queue_mc(
        match_entry  *real_output_array,
        unsigned long long int         *tail_of_real_output_array,
        // real report

        // input stream
        const  uint8_t* __restrict__  input_stream,
        const  int input_stream_length,
        // end input stream

        const int *__restrict__ start_offset_node_list,
        const int *__restrict__ num_hot_states,

        const  STE_nodeinfo_new_imp * __restrict__  node_list,
        const  STE_matchset_new_imp * __restrict__  node_ms,

        const bool generate_report,
        const int queuesize,
        const int dedup_bitset_length
)

{
    extern __shared__ bool shared_pool[];
    bool *is_active_state1 = shared_pool;
    int *cold_warp_active_queue = (int *) (shared_pool + 2 * blockDim.x - WARP_SIZE * 2); // this is a queue;
    int *cold_warp_next_active_queue = &cold_warp_active_queue[queuesize];

    unsigned int *dedup_bs = (unsigned int * ) (cold_warp_active_queue + 2 * queuesize);

    int tt = threadIdx.x;
    while (tt < dedup_bitset_length) {
        dedup_bs[tt] = 0;
        tt += blockDim.x;
    }

    int start_offset = start_offset_node_list[blockIdx.x];
    int num_of_state = num_hot_states[blockIdx.x];

    STE_nodeinfo_new_imp cur_node;
    STE_matchset_new_imp cur_node_ms;
    bool complement;
    bool complete;

    if (threadIdx.x < num_of_state && threadIdx.x / WARP_SIZE != blockDim.x / WARP_SIZE - 1) { // the last warp of the block
        cur_node = node_list[start_offset + threadIdx.x];

        complement = ( cur_node.attribute & ((1 << 4) | (1 << 3))  )   ==  ((1 << 4) | (1 << 3));
        complete = ((cur_node.attribute & (1 << 3)) != 0 ) && (!((cur_node.attribute & (1 << 4)) != 0 ) );

        if (!complete && !complement) {
            cur_node_ms = node_ms[start_offset + threadIdx.x];
        }

        is_active_state1[threadIdx.x] = (cur_node.attribute & (1 << 2)) ;	// enable
        is_active_state1[threadIdx.x + blockDim.x - WARP_SIZE] = false; // false; active
    }

    __shared__ int tail_of_the_cold_warp_active_queue;
    __shared__ int tail_of_the_cold_warp_next_active_queue;

    tail_of_the_cold_warp_active_queue = 0;
    tail_of_the_cold_warp_next_active_queue = 0;

    __syncthreads();

    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);

    // main loop
    for (int symbol_pos = 0; symbol_pos < input_stream_length; symbol_pos ++) {
        uint8_t symbol = cur_input_stream[symbol_pos];

        tail_of_the_cold_warp_next_active_queue = 0;

        tt = threadIdx.x;
        while (tt < dedup_bitset_length) {
            dedup_bs[tt] = 0;
            tt += blockDim.x;
        }

        __syncthreads();

        if (threadIdx.x / WARP_SIZE == blockDim.x / WARP_SIZE - 1 ) { // cold warp

            tt = threadIdx.x % WARP_SIZE;

            while (tt < tail_of_the_cold_warp_active_queue) {
                int enabled_sid = cold_warp_active_queue[tt];

                cur_node = node_list[start_offset + enabled_sid];

                complement = ( cur_node.attribute & ((1 << 4) | (1 << 3))  )   ==  ((1 << 4) | (1 << 3));
                complete = ((cur_node.attribute & (1 << 3)) != 0 ) && (!((cur_node.attribute & (1 << 4)) != 0 ) );

                if (!complete && !complement) {
                    cur_node_ms = node_ms[start_offset + enabled_sid];
                }

                bool in_range = symbol >= cur_node.start  &&   symbol <= cur_node.end;

                if (  ( (complement && !in_range) || (complete   &&  in_range)
                        ||  (!complete && !complement && cur_node_ms.ms[symbol / 32] & (1 << (symbol % 32)))  )  ) {

                    for (int to = 0; to < cur_node.degree; to++) {
                        unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;

                        if (edgeto < num_of_state) {
                            is_active_state1[ edgeto + blockDim.x - WARP_SIZE] = true;  // enable
                        } else { // cold to cold
                            int n_bit = edgeto - num_of_state;
                            assert(n_bit >= 0);
                            int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                            if (!(is_in_the_queue & (1 << (n_bit % 32)))) {
                                int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                                assert(cur_tail < queuesize);
                                cold_warp_next_active_queue[cur_tail] = edgeto;
                            }
                        }
                    }

                    if ( (cur_node.attribute & 1)) {
                        unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                        if(generate_report){
                                                    real_output_array[write_to].symbol_offset = symbol_pos ;
                        real_output_array[write_to].state_id = enabled_sid;
                        real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                        }

                    }

                }

                tt += WARP_SIZE;

            }

        }
        else { // hot warps
            if (threadIdx.x < num_of_state) {
                bool in_range = symbol >= cur_node.start  &&   symbol <= cur_node.end;

                if (  ( (complement && !in_range) || (complete   &&  in_range)
                        ||  (!complete && !complement && cur_node_ms.ms[symbol / 32] & (1 << (symbol % 32)))  )  && is_active_state1[threadIdx.x]) {
                    for (int to = 0; to < cur_node.degree; to++) {
                        unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;

                        if (edgeto < num_of_state) {
                            is_active_state1[ edgeto + blockDim.x - WARP_SIZE] = true;  // enable
                        } else { // hot to cold
                            int n_bit = edgeto - num_of_state;
                            assert(n_bit >= 0);
                            int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                            if ((!(is_in_the_queue & (1 << (n_bit % 32))))) {
                                int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                                assert(cur_tail < queuesize);
                                cold_warp_next_active_queue[cur_tail] = edgeto;
                            }
                        }
                    }

                    if ( (cur_node.attribute & 1)) {
                        unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                        if(generate_report){
                                                    real_output_array[write_to].symbol_offset = symbol_pos ;
                        real_output_array[write_to].state_id = threadIdx.x;
                        real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                        }

                    }
                }

            }

        }

        __syncthreads();


        if (threadIdx.x / WARP_SIZE == blockDim.x / WARP_SIZE - 1 ) { // cold warp
            tt = threadIdx.x % WARP_SIZE;
            while (tt < tail_of_the_cold_warp_next_active_queue) {
                cold_warp_active_queue[tt] = cold_warp_next_active_queue[tt];
                tt += WARP_SIZE;
            }

            // we reset the next queue tail in the beginning of the main loop.
        } else {
            if (threadIdx.x < num_of_state) {
                is_active_state1[threadIdx.x] = ( (cur_node.attribute & (1 << 1)) ) || is_active_state1[threadIdx.x + blockDim.x - WARP_SIZE];

                is_active_state1[threadIdx.x + blockDim.x - WARP_SIZE] = false;

            }
        }

        __syncthreads();

        tail_of_the_cold_warp_active_queue = tail_of_the_cold_warp_next_active_queue;

        __syncthreads();


    }
}




__global__ void one_byte_at_a_time_queue_profile(
        match_entry  *real_output_array,
        unsigned long long int         *tail_of_real_output_array,
        // real report

        // input stream
        const  uint8_t* __restrict__  input_stream,
        const  int input_stream_length,
        // end input stream

        const int *__restrict__ start_offset_node_list,
        const int *__restrict__ num_hot_states,

        const  STE_dev4* __restrict__   node_list,

        const bool generate_report,
        const int queuesize,

        unsigned long long int *hot_to_cold_activations,
        unsigned long long int *cold_to_cold_activations
        //int *cold_to_hot_activations

)

{

    uint8_t *cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);

    extern __shared__ bool shared_pool[];

    //extern __shared__ bool is_active_state1[];
    bool *is_active_state1 = shared_pool;
    // size ==  2 * blockDim.x - 64

    int *cold_warp_active_queue = (int*) (shared_pool + 2 * blockDim.x - 64); // this is a queue;

    int *cold_warp_next_active_queue = cold_warp_active_queue + queuesize;


    int start_offset = start_offset_node_list[blockIdx.x];

    int num_of_state = num_hot_states[blockIdx.x];


    STE_dev4 cur_node;

    if (threadIdx.x / 32 != blockDim.x / 32 - 1) { // the last warp of the block
        cur_node = node_list[start_offset + threadIdx.x];

        is_active_state1[threadIdx.x] = (cur_node.attribute & (1 << 2)) ;	// enable

        is_active_state1[threadIdx.x + blockDim.x - 32] = false; // false; active
    }



    // assume the size of the queue is 64. --- temporary.

    __shared__ int tail_of_the_cold_warp_active_queue;
    __shared__ int tail_of_the_cold_warp_next_active_queue;

    tail_of_the_cold_warp_active_queue = 0;
    tail_of_the_cold_warp_next_active_queue = 0;

    __syncthreads();

    // main loop
    for (int ft = 0; ft < input_stream_length / 4; ft ++) {
        uint32_t bigsymbol = ((uint32_t *) cur_input_stream) [ft];

        //__syncthreads();

        for (int s = 0; s < 4; s++) {
            int symbol_pos = ft * 4 + s;
            uint8_t symbol = (uint8_t) (bigsymbol & 255 );
            //for (int symbol_pos = 0; symbol_pos < input_stream_length ; symbol_pos ++ ) {
            bigsymbol >>= 8;

            //uint8_t symbol = cur_input_stream[symbol_pos];
            //if (threadIdx.x == 0) {
            //	printf("cursymbol = %c\n", symbol);
            //}

            //if (threadIdx.x == 0) {
            //	printf("cursymbol = %c %d\n", symbol, symbol_pos);
            //}

            //if (threadIdx.x == 224) {
            //	printf("224 sees symbol = %c %d\n", symbol, symbol_pos);
            //}

            //__syncthreads();

            //printf("threadIdx.x = %d blockDim.x = %d\n", threadIdx.x, blockDim.x );

            tail_of_the_cold_warp_next_active_queue = 0;
            //printf("tail cur active queue = %d \n", tail_of_the_cold_warp_active_queue);

            __syncthreads();

            if (threadIdx.x / 32 == blockDim.x / 32 - 1 ) { // cold warp
                //int num_cold_state = start_offset_node_list[blockIdx.x + 1] - num_of_state - start_offset;

                // do nothing
                // use the queue.

                int tt = threadIdx.x % 32;

                //printf("cold warp tail_of_the_cold_warp_active_queue = %d\n", tail_of_the_cold_warp_active_queue);

                while (tt < tail_of_the_cold_warp_active_queue) {
                    int enabled_sid = cold_warp_active_queue[tt];
                    //printf("sid in queue = %d symbol = %c tid = %d\n", enabled_sid, symbol, threadIdx.x);
                    cur_node = node_list[start_offset + enabled_sid];

                    if ( (cur_node.ms[symbol / 32] & (1 << (symbol % 32))) != 0) {
                        for (int to = 0; to < cur_node.degree; to++) {
                            unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;

                            if (edgeto < num_of_state) {
                                is_active_state1[ edgeto + blockDim.x - 32] = true;  // enable
                            } else {
                                int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);

                                atomicAdd(cold_to_cold_activations, 1);

                                if (cur_tail == queuesize) {
                                    printf("ctoc_queuefull\n");
                                    for (int k = 0; k < cur_tail; k++) {
                                        printf("%d ", cold_warp_next_active_queue[k]);
                                    }
                                    printf("\n");

                                    return;
                                    //assert(cur_tail < queuesize);

                                }

                                //printf("cold enables cold cur_tail = %d\n", cur_tail);
                                cold_warp_next_active_queue[cur_tail] = edgeto;
                            }
                        }

                        if ( (cur_node.attribute & 1)) {
                            unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                            if(generate_report) {
                                                         real_output_array[write_to].symbol_offset = symbol_pos ;
                            real_output_array[write_to].state_id = enabled_sid;
                            real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                            }
   
                        }

                    }

                    tt += 32;

                }

            }
            else { // normal warps
                if (threadIdx.x < num_of_state) {
                    if ( (cur_node.ms[symbol / 32] & (1 << (symbol % 32))) != 0  &&  is_active_state1[threadIdx.x]) {
                        for (int to = 0; to < cur_node.degree; to++) {
                            unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;
                            //printf("sid = %d act %d\n", threadIdx.x, edgeto);

                            if (edgeto < num_of_state) {
                                is_active_state1[ edgeto + blockDim.x - 32] = true;  // enable
                            } else {
                                int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);

                                //assert(cur_tail < queuesize);
                                //printf("hot enables cold -- cur_tail = %d sb = %c edgeto = %d \n", cur_tail, symbol, edgeto);
                                cold_warp_next_active_queue[cur_tail] = edgeto;

                                atomicAdd(hot_to_cold_activations, 1);

                                if (cur_tail == queuesize) {
                                    printf("htoc_queuefull\n");
                                    for (int k = 0; k < cur_tail; k++) {
                                        printf("%d ", cold_warp_next_active_queue[k]);
                                    }
                                    printf("\n");

                                    return;
                                    //assert(cur_tail < queuesize);

                                }

                                //printf("enable a cold state %d set %d\n", edgeto, (start_offset + edgeto) * 2 + 1);

                                //cold_warp_is_active[ edgeto % 32 + 32] = true; // enable

                                // state_active_global[(start_offset  + edgeto) * 2 + 1] = true;

                                //printf("set state_active_global[%d] = %d\n",  (start_offset  + edgeto) * 2 + 1,  (int) state_active_global[(start_offset  + edgeto) * 2 + 1]);

                            }

                        }

                        //printf("sid = %d act %d\n", threadIdx.x, cur_node.edge_dst[to]);
                        //printf("state = %d is activated\n", threadIdx.x);

                        if ((cur_node.attribute & 1)) {
                            unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                            if(generate_report) {
                                                            real_output_array[write_to].symbol_offset = symbol_pos ;
                            real_output_array[write_to].state_id = threadIdx.x;
                            real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                            }


                            //printf("hot warp report %d %d %d\n", symbol_pos, threadIdx.x, blockIdx.x);
                        }
                    }

                }

            }

            __syncthreads();


            if (threadIdx.x / 32 == blockDim.x / 32 - 1 ) { // cold warp
                // do nothing
                // copy the queue from next_enable to current_enable.
                //printf("copy tail_of_the_cold_warp_next_active_queue = %d\n", tail_of_the_cold_warp_next_active_queue);

                int tt = threadIdx.x % 32;
                while (tt < tail_of_the_cold_warp_next_active_queue) {
                    //printf("copy from %d to %d\n", cold_warp_next_active_queue[tt], cold_warp_active_queue[tt] );
                    cold_warp_active_queue[tt] = cold_warp_next_active_queue[tt];
                    tt += 32;
                }



                // we reset the next queue tail in the beginning of the main loop.
            }

            __syncthreads();

            // do we need this sync thread here?


            if (threadIdx.x / 32 != blockDim.x / 32 - 1 ) { // hot warps
                if (threadIdx.x < num_of_state) {
                    //__syncthreads();
                    is_active_state1[threadIdx.x] = ( (cur_node.attribute & (1 << 1)) ) || is_active_state1[threadIdx.x + blockDim.x - 32];

                    //__syncthreads();
                    is_active_state1[threadIdx.x + blockDim.x - 32] = false;

                    //__syncthreads();
                }

            }

            tail_of_the_cold_warp_active_queue = tail_of_the_cold_warp_next_active_queue;

            __syncthreads();


        } // for s

        //__syncthreads();
    }



}





template <bool run_cold_stage=true>
__global__
void hotstart_ea_kernel_20190606_halfhalf(
        match_entry  *real_output_array,
        unsigned long long int         *tail_of_real_output_array,
        // real report

        // input stream
        const  uint8_t* __restrict__  input_stream,
        const  int input_stream_length,
        // end input stream

        const int *__restrict__ start_offset_node_list,
        const int *__restrict__ num_hot_states,

        const  STE_nodeinfo_new_imp * __restrict__  node_list,
        const  STE_matchset_new_imp * __restrict__  node_ms,

        const bool generate_report,
        const int queuesize,
        const int dedup_bitset_length

        //const int *remap_input_stream_array
)
{

    extern __shared__ bool shared_pool[];

    __shared__ int *cold_warp_active_queue;
    cold_warp_active_queue  = (int *) (shared_pool);

    __shared__ int *cold_warp_next_active_queue;
    cold_warp_next_active_queue  = &cold_warp_active_queue[queuesize];
    unsigned int *dedup_bs = (unsigned int * ) (cold_warp_active_queue + 2 * queuesize);

    int start_offset = start_offset_node_list[blockIdx.x];
    int num_of_state = num_hot_states[blockIdx.x];

    // fixed.
    STE_nodeinfo_new_imp cur_node;
    STE_matchset_new_imp cur_node_ms;

    bool complement;
    bool complete;

    if (threadIdx.x < num_of_state) {
        cur_node = node_list[start_offset + threadIdx.x];

        complement = ( cur_node.attribute & ((1 << 4) | (1 << 3))  )   ==  ((1 << 4) | (1 << 3));
        complete = ((cur_node.attribute & (1 << 3)) != 0 ) && (!((cur_node.attribute & (1 << 4)) != 0 ) );

        if (!complete && !complement) {
            cur_node_ms = node_ms[start_offset + threadIdx.x];
        }
    }

    __shared__ int tail_of_the_cold_warp_active_queue;
    __shared__ int tail_of_the_cold_warp_next_active_queue;

    if (threadIdx.x == 0) {
        tail_of_the_cold_warp_active_queue = 0;
        tail_of_the_cold_warp_next_active_queue = 0;
    }

    __syncthreads();

    uint8_t *cur_input_stream;

    if (blockIdx.x < blockDim.x / 2) {
        cur_input_stream = ((uint8_t*) input_stream) + (blockIdx.y * input_stream_length);
    } else {
        cur_input_stream = ((uint8_t*) input_stream) + ( (blockIdx.y + blockDim.y ) * input_stream_length);
    }

    // main loop
    for (int symbol_pos = 0; symbol_pos < input_stream_length; symbol_pos ++) {
        uint8_t symbol = cur_input_stream[symbol_pos];

        if (threadIdx.x == 1) {
            int *tmp = cold_warp_active_queue;
            cold_warp_active_queue = cold_warp_next_active_queue;
            cold_warp_next_active_queue = tmp;
        }

        if (threadIdx.x == 0) {
            tail_of_the_cold_warp_active_queue = tail_of_the_cold_warp_next_active_queue;
            tail_of_the_cold_warp_next_active_queue = 0;
        }

        int tt = threadIdx.x;
        while (tt < dedup_bitset_length) {
            dedup_bs[tt] = 0;
            tt += blockDim.x;
        }

        __syncthreads();

        if (threadIdx.x < num_of_state) { // hot state
            bool in_range = symbol >= cur_node.start  &&   symbol <= cur_node.end;

            if (  ( (complement && !in_range) || (complete   &&  in_range) ||  (!complete && !complement && cur_node_ms.ms[symbol / 32] & (1 << (symbol % 32)))  )) {
                for (int to = 0; to < cur_node.degree; to++) {
                    unsigned int edgeto = ( cur_node.edges >> (16 * to) )  & 65535;

                    if (edgeto >= num_of_state) { // hot to cold
                        int n_bit = edgeto - num_of_state;
                        //assert(n_bit >= 0);
                        int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                        if ((!(is_in_the_queue & (1 << (n_bit % 32))))) {
                            int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                            //assert(cur_tail < queuesize);
                            cold_warp_next_active_queue[cur_tail] = edgeto;
                        }
                    }
                }

                if ( (cur_node.attribute & 1)) {
                    unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                    if(generate_report) {
                                           real_output_array[write_to].symbol_offset = symbol_pos ;
                    real_output_array[write_to].state_id = threadIdx.x;
                    real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y; 
                    }

                }
            }
        }

        if (run_cold_stage) {
            // reusable.
            STE_nodeinfo_new_imp cur_cold_node;
            STE_matchset_new_imp cur_cold_node_ms;
            bool complement_cold;
            bool complete_cold;

            tt = threadIdx.x;
            while (tt < tail_of_the_cold_warp_active_queue) {
                int enabled_sid = cold_warp_active_queue[tt];

                cur_cold_node = node_list[start_offset + enabled_sid];

                complement_cold = ( cur_cold_node.attribute & ((1 << 4) | (1 << 3))  )   ==  ((1 << 4) | (1 << 3));
                complete_cold = ((cur_cold_node.attribute & (1 << 3)) != 0 ) && (!((cur_cold_node.attribute & (1 << 4)) != 0 ) );

                if (!complement_cold && !complete_cold) {
                    cur_cold_node_ms = node_ms[start_offset + enabled_sid];
                }

                bool in_range = symbol >= cur_cold_node.start  &&   symbol <= cur_cold_node.end;

                if (  ( (complement_cold && !in_range) || (complete_cold   &&  in_range) ||  (!complete_cold && !complement_cold && cur_cold_node_ms.ms[symbol / 32] & (1 << (symbol % 32)))  )  ) {

                    for (int to = 0; to < cur_cold_node.degree; to++) {
                        unsigned int edgeto = ( cur_cold_node.edges >> (16 * to) )  & 65535;

                        if (edgeto >= num_of_state) { // cold to cold
                            int n_bit = edgeto - num_of_state;
                            //assert(n_bit >= 0);
                            int is_in_the_queue = atomicOr(&dedup_bs[n_bit / 32], (1 << (n_bit % 32)));
                            if (!(is_in_the_queue & (1 << (n_bit % 32)))) {
                                int cur_tail = atomicAdd(&tail_of_the_cold_warp_next_active_queue, 1);
                                //assert(cur_tail < queuesize);
                                cold_warp_next_active_queue[cur_tail] = edgeto;
                            }
                        }
                    }

                    if ( (cur_cold_node.attribute & 1)) {
                        unsigned long long int write_to = atomicAdd(tail_of_real_output_array, 1);
                        if(generate_report) {
                        real_output_array[write_to].symbol_offset = symbol_pos ;
                        real_output_array[write_to].state_id = enabled_sid;
                        real_output_array[write_to].cc_id = blockIdx.x;
                    real_output_array[write_to].stream_id = blockIdx.y;
                        }

                    }

                }

                tt += blockDim.x;
            }
        }

        __syncthreads();
    }
};









#endif





