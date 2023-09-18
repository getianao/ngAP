#ifndef PRECOMPUTE_TABLE_H
#define PRECOMPUTE_TABLE_H
#include <vector>
#include <cstdint>
#include <assert.h>
#include <stdio.h>

class PrecTable {
public:
  uint64_t size = 0;
  int depth = 0;
  int cutoff;
  uint64_t nonzeroVerticesNum = 0;
  uint64_t nonzeroResultsNum = 0;
  bool isCompress;
  int maxkey = 0;

  std::vector<int *> vertices;
  std::vector<int> vertices_length;
  std::vector<int *> results;
  std::vector<int> results_length;

  std::vector<uint32_t> nonzeroVerticesMap; // from index to vertix
  std::vector<uint32_t> nonzeroResultsMap;  // from index to result

  int *d_vertices;
  int *d_vertices_offsets;
  int *d_results;
  int *d_results_offsets;

  uint32_t* d_nonzeroVerticesMap; // from index to vertix
  uint32_t* d_nonzeroResultsMap; // from index to result

  PrecTable(){}

  PrecTable(uint64_t size);

  void allocate(uint64_t size, int depth, bool isCompress = true);

  void setVertices(uint32_t index, std::vector<int> &v);

  void setResults(uint32_t index, std::vector<int> &v);


  int printHistogram();

  void calcCutoff();

  void calcCutoffMedian();

  void toDevice(bool use_uvm = false);

  void releaseHost();

  void releaseDevice();


  // template <typename T>
  __device__ __forceinline__ int getVertexSymbolIndex(uint32_t symbol) {
    if(isCompress)
      return binary_search(d_nonzeroVerticesMap, nonzeroVerticesNum, symbol);
    else
      return (int)symbol;
  }

  // template <typename T>
  __device__ __forceinline__ int getResultSymbolIndex(uint32_t symbol) {
    if(isCompress)
      return binary_search(d_nonzeroResultsMap, nonzeroResultsNum, symbol);
    else
      return (int)symbol;
  }

  // template <typename T>
  __host__ __forceinline__ int getVertexSymbolIndexHost(uint32_t symbol) {
    return binary_search(&nonzeroVerticesMap[0], nonzeroVerticesNum, symbol);
  }

  // template <typename T>
  __host__ __forceinline__ int getResultSymbolIndexHost(uint32_t symbol) {
    return binary_search(&nonzeroResultsMap[0], nonzeroResultsNum, symbol);
  }

  // template <typename T>
  __device__ __host__ __forceinline__ int binary_search(uint32_t *arr,
                                                             int n,
                                                             uint32_t x) {
    int start = 0;
    int end = n - 1;
    while (start <= end) {
      int mid = (start + end) / 2;
      if (arr[mid] == x)
        return (int)mid;
      else if (arr[mid] < x)
        start = mid + 1;
      else
        end = mid - 1;
    }
    return -1;
  }
};

#endif