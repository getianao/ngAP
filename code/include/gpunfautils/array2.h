
#ifndef ARRAY2_H_
#define ARRAY2_H_


#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>
#include <iostream>
#include <cstring>

using std::cout;
using std::cerr;
using std::endl;


template <class T>
class Array2 {
public:
	Array2(int arr_length):  Array2(arr_length, "noname_array") {
		
	}

	Array2(int arr_length, std::string arr_id) : h_arr(NULL), d_arr(NULL) {
		this->arr_id = arr_id;

		// cout << "arr_length = " << arr_length << endl;

		this->arr_length = arr_length;
		this->element_bytes = sizeof(T);

		if (arr_length <= 0) {
			// cout << "arr_length = " << arr_length << " arr_id = " << arr_id << endl;
			assert(arr_length >= 0);
			return;
		}

		h_arr = new T[arr_length];

		//assert(element_bytes * arr_length > 0);

		auto errcode = cudaMalloc(&d_arr, 1ULL * element_bytes * arr_length);
		
		if (errcode != cudaSuccess) {
			cerr << "try to allocate " << 1ULL * arr_length * element_bytes << " byte of memory failed" << " arrid = " << arr_id << endl;
			cerr << "cannot allocate cuda memory " << errcode << endl; 
			exit(-1);
		}

	}

	
	virtual ~Array2() {
		assert(h_arr != NULL);
		delete [] h_arr;

		assert(d_arr != NULL);
		cudaFree(d_arr);

	}


	int size() const {
		return arr_length;
	}

	int size_of_T() const {
		return element_bytes;
	}

	unsigned long long num_of_byte() const {
		return 1ULL * element_bytes * arr_length;
	}

	T *get_dev() const {
		return d_arr;
	} 

	T get(int idx) const {
		assert(idx >= 0 && idx < size());
		return h_arr[idx];
	}

	T *get_host() const {
		return h_arr;
	}

	void clear_to_zero() {
		memset(h_arr, 0, num_of_byte());
	}


	void fill(T val) {
		for (int i = 0; i < arr_length; i++) {
			h_arr[i] = val;
		}
	}


	T operator[] (int idx) {
		assert(idx >= 0 && idx < size());
#ifdef __CUDA_ARCH__
    return d_arr[idx];
#else
    return h_arr[idx];
#endif
	}

	void set(int idx, T v) {
		if (!(idx >= 0 && idx < size())) {
			cout << "assert(idx >= 0 && idx < size());  " << idx << endl;
			assert(idx >= 0 && idx < size());
		}

		h_arr[idx] = v;

	}

	void copy_to_device() {

		//cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
		if (num_of_byte() == 0) {
			// std::cout<< "FAIL: copy_to_device !!!!!\n";
			// printf("%d, %d", element_bytes ,arr_length);
			return;
		}


		auto errcode = cudaMemcpy(d_arr, h_arr, num_of_byte(), cudaMemcpyHostToDevice);

		if (errcode != cudaSuccess) {
			cout << "trying to copy " << num_of_byte() << " byte to device " << endl;
			cout << "cannot copy to device error code = " << errcode << endl; 
			exit(-1);
		}
	}

	void copy_back() {
		auto errcode = cudaMemcpy(h_arr, d_arr, num_of_byte(), cudaMemcpyDeviceToHost);
		if (errcode != cudaSuccess) {
			cerr << "cannot copy back" << "  " << errcode << endl; 
			exit(-1);
		}
	}

	T *copy_to_host(int num_of_element) {
		assert(num_of_element <= arr_length);
		
		T *arr = new T[num_of_element];
		auto errcode = cudaMemcpy(arr, d_arr, 1ULL * sizeof(T) * num_of_element, cudaMemcpyDeviceToHost);
		if (errcode != cudaSuccess) {
			cerr << "cannot copy_to_host " << "  " << errcode << endl; 
			exit(-1);
		}

		return arr;
		
	}

	void print() const {
		cout << "print for debug array2 length = " << arr_length <<  endl;
		for (int i = 0; i < arr_length; i++) {
			cout << h_arr[i] << " " ;
		}

		cout << endl;
	}

	T *copy_to_host_async(int num_of_element) {
		assert(num_of_element <= arr_length);
		
		T *arr = new T[num_of_element];
		auto errcode = cudaMemcpyAsync(arr, d_arr, 1ULL * sizeof(T) * num_of_element, cudaMemcpyDeviceToHost);
		if (errcode != cudaSuccess) {
			cerr << "cannot copy_to_host " << "  " << errcode << endl; 
			exit(-1);
		}

		return arr;	
	}

// private:
	int arr_length;
	int element_bytes;

	T *h_arr; 
	T *d_arr;

	std::string arr_id;

};


#endif