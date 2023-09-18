#ifndef INFANT_KERNELS_DEV_FUNC_H_
#define INFANT_KERNELS_DEV_FUNC_H_



#define  OUTPUT_BUFFER_TB  256

__device__ inline bool get_bit(int *arr, int len, int n_bit) {
    int n_cell = n_bit / (sizeof(int) * 8);
    int offset = n_bit % (sizeof(int) * 8);

    return arr[n_cell] & (1 << offset);
}



__device__ inline void set_bit(int *arr, int len, int n_bit) {
    int n_cell = n_bit / (sizeof(int) * 8);
    int offset = n_bit % (sizeof(int) * 8);

    atomicOr(&arr[n_cell], (1 << offset));

}


template <class T>
__device__ inline bool get_bit_single(T ele, int n_bit) {
	return ele & (1 << n_bit);
}


#endif