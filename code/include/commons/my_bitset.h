
#pragma once

#include <bitset>
#include <iostream>
#include <math.h>

typedef struct My_bitset256 {
  uint32_t data[8];

  __host__ __device__ My_bitset256() { memset(data, 0, sizeof(data)); }

  __host__ __device__ My_bitset256(const My_bitset256 &other) {
    memcpy(data, other.data, sizeof(data));
  }

  __host__ __device__ ~My_bitset256() {}

  __host__ __device__ cudaError_t reset() {
    memset(data, 0, sizeof(data));
    return cudaSuccess;
  }

  __host__ __device__ cudaError_t set(uint8_t offset, int value) {
    int pos = (offset / 32);
    data[pos] = data[pos] | ((value & 1) << (offset % 32));
    return cudaSuccess;
  }

  __host__ __device__ bool test(uint8_t offset) {
    int pos = (offset / 32);
    return data[pos] & (1 << (offset % 32));
  }

  __host__ __device__ My_bitset256 &operator=(const My_bitset256 &other) {
    memcpy(data, other.data, sizeof(data));
    return *this;
  }

  void fromBitset(std::bitset<256> column) {
    for (int i = 0; i < 256; i++) {
      if (column.test(i)) {
        set(i, 1);
      } else {
        set(i, 0);
      }
    }
  }

} My_bitset256;

struct My_bitsetN {
  uint32_t N;
  uint32_t size;
  uint32_t *data;

  __host__ __device__ My_bitsetN(int N = 256) : N(N) {
    // this->N = N;
    this->size = (N - 1) / 32 + 1;
    data = new uint32_t[size];
    memset(data, 0, sizeof(uint32_t) * size);
  }

  __host__ __device__ My_bitsetN(const My_bitsetN &other) {
    this->N = N;
    this->size = (N - 1) / 32 + 1;
    data = new uint32_t[size];
    memcpy(data, other.data, sizeof(uint32_t) * size);
  }

  __host__ __device__ ~My_bitsetN() { delete[] data; }

  __host__ __device__ cudaError_t reset() {
    memset(data, 0, sizeof(uint32_t) * size);
    return cudaSuccess;
  }

  __host__ __device__ cudaError_t set(int offset, int value) {
    int pos = (offset / 32);
    data[pos] = data[pos] | ((value & 1) << (offset % 32));
    return cudaSuccess;
  }

  __host__ __device__ bool test(int offset) {
    int pos = (offset / 32);
    return data[pos] & (1 << (offset % 32));
  }

  __host__ __device__ My_bitsetN &operator=(const My_bitsetN &other) {
    this->N = N;
    this->size = (N - 1) / 32 + 1;
    data = new uint32_t[size];
    memcpy(data, other.data, sizeof(uint32_t) * size);
    return *this;
  }

  // void fromBitset(std::bitset<N> column) {
  //   for (int i = 0; i < N; i++) {
  //     if (column.test(i)) {
  //       set(i, 1);
  //     } else {
  //       set(i, 0);
  //     }
  //   }
  // }
};



