
#include "precompute_table.h"
#include <cstring>
#include <map>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <algorithm>
#include "gpunfautils/utils.h"
#include <inttypes.h>

// PrecTable::PrecTable(uint64_t size) : size(size) {
//   vertices = new int *[size];
//   vertices_length = new int[size];
//   results = new int *[size];
//   results_length = new int[size];
//   nonzeroVerticesMap = new uint64_t[size];
//   nonzeroResultsMap = new uint64_t[size];
//   memset(vertices, 0, size);
//   memset(results, 0, size);
//   memset(nonzeroVerticesMap, 0, size * sizeof(uint64_t));
//   memset(nonzeroResultsMap, 0, size * sizeof(uint64_t));
// }

void PrecTable::allocate(uint64_t size, int depth, bool isCompress) {
  this->size = size;
  this->depth = depth;
  this->isCompress = isCompress;
  if(!isCompress) {
    vertices.reserve(size);
    vertices_length.reserve(size);
    results.reserve(size);
    results_length.reserve(size);
  }
}

void PrecTable::setVertices(uint32_t symbol, std::vector<int> &v) {
  if (!isCompress) {
    if (v.empty()) {
      vertices_length[symbol] = 0;
      vertices[symbol] = NULL;
      return;
    }
    vertices_length[symbol] = v.size();
    vertices[symbol] = new int[v.size()];
    std::copy(v.begin(), v.end(), vertices[symbol]);
    nonzeroVerticesNum++;
  } else {
    if (!v.empty()) {
      nonzeroVerticesMap.push_back(symbol);
      vertices_length.push_back(v.size());
      int *vertices_value = new int[v.size()];
      std::copy(v.begin(), v.end(), vertices_value);
      vertices.push_back(vertices_value);
      nonzeroVerticesNum++;
      assert(nonzeroVerticesNum == nonzeroVerticesMap.size());
      assert(nonzeroVerticesNum == vertices_length.size());
      assert(nonzeroVerticesNum == vertices.size());
    }
  }
}

void PrecTable::setResults(uint32_t symbol, std::vector<int> &v) {
  if(!isCompress) {
    if (v.empty()) {
      results_length[symbol] = 0;
      results[symbol] = NULL;
      return;
    }
    results_length[symbol] = v.size();
    results[symbol] = new int[v.size()];
    std::copy(v.begin(), v.end(), results[symbol]);
    nonzeroResultsNum++;
  } else {
    if (!v.empty()) {
      nonzeroResultsMap.push_back(symbol);
      results_length.push_back(v.size());
      int *results_value = new int[v.size()];
      std::copy(v.begin(), v.end(), results_value);
      results.push_back(results_value);
      nonzeroResultsNum++;
      assert(nonzeroResultsNum == nonzeroResultsMap.size());
      assert(nonzeroResultsNum == results_length.size());
      assert(nonzeroResultsNum == results.size());
    }
  }
}

int PrecTable::printHistogram() {
  printf("Precompute table %d info:\n", depth);
  printf("    Precompute cutoff = %d\n", cutoff);
  printf("    Precompute depth = %d\n", depth);
  printf("    Precompute ratio of zero (vertices) = %f\n",
         1.0 * (size - nonzeroVerticesNum) / size);
  printf("    Precompute ratio of zero (results) = %f\n",
         1.0 * (size - nonzeroResultsNum) / size);

  std::map<uint32_t, uint32_t> prec_histogram;
  if (isCompress) {
    for (uint32_t i = 0; i < nonzeroVerticesNum; i++) {
      ++prec_histogram[vertices_length[i]];
    }
    prec_histogram[0] = size - nonzeroVerticesNum;
  } else {
    for (uint64_t i = 0; i < size; i++)
      ++prec_histogram[vertices_length[i]];
  }

  uint32_t idx = 1;
  std::cout << "    vertices histogram: ";
  for (auto it = prec_histogram.begin(); it != prec_histogram.end(); it++) {
    idx++;
    std::cout << it->first << "(" << it->second << "), ";
    if (idx % 15 == 1)
      std::cout << "\n    ";
  }
  std::cout << "\n";

  std::map<uint32_t, uint32_t> prec_result_histogram;
  if (isCompress) {
    for (uint32_t i = 0; i < nonzeroResultsNum; i++) {
      ++prec_result_histogram[results_length[i]];
    }
    prec_result_histogram[0] = size - nonzeroResultsNum;
  } else {
    for (uint64_t i = 0; i < size; i++)
      ++prec_histogram[results_length[i]];
  }
  maxkey = prec_histogram.rbegin()->first;

  idx = 1;
  std::cout << "    results histogram: ";
  for (auto it = prec_result_histogram.begin();
       it != prec_result_histogram.end(); it++) {
    idx++;
    std::cout << it->first << "(" << it->second << "), ";
    if (idx % 15 == 1)
      std::cout << "\n    ";
  }
  std::cout << "\n";

  uint32_t vlt = 0;
  uint32_t rlt = 0;
  double tablem;
  if (isCompress) {
    for (uint32_t i = 0; i < nonzeroVerticesNum; i++) {
      vlt += vertices_length[i];
    }
    for (uint32_t i = 0; i < nonzeroResultsNum; i++) {
      rlt += results_length[i];
    }
    tablem = (nonzeroVerticesNum + nonzeroResultsNum) * (4) / 1000000.0;
  } else {
    for (uint64_t i = 0; i < size; i++) {
      vlt += vertices_length[i];
      rlt += results_length[i];
    }
    tablem = 2.0 * size * 4 / 1000000;
  }

  double vltm = vlt * 4.0 / 1000000;
  double rltm = rlt * 4.0 / 1000000;
  printf("    vertices number=%" PRIu32 ", size=%f MB\n", vlt, vltm);
  printf("    results number=%" PRIu32 ", size=%f MB\n", rlt, rltm);
  printf("    table_%d_index = %f MB\n", depth, tablem);
  printf("    table_%d_content = %f MB\n", depth, vltm + rltm);
  printf("    total_%d_size = %f MB\n", depth, vltm + rltm + tablem);
  return vltm + rltm + tablem;
}

void PrecTable::calcCutoff() {
  uint64_t vertices_total_length = 0;
  if (isCompress) {
    for (uint32_t i = 0; i < nonzeroVerticesNum; i++) {
      vertices_total_length += vertices_length[i];
    }
  } else {
    for (uint64_t i = 0; i < size; i++) {
      vertices_total_length += vertices_length[i];
    }
  }
  if (nonzeroVerticesNum > 0)
    cutoff = vertices_total_length / nonzeroVerticesNum;
  else
    cutoff = 0;
}

void PrecTable::calcCutoffMedian() {
  std::vector<int> v;
  for (uint32_t i = 0; i < size; i++) {
    // if (vertices_length[i] > 0)
    v.push_back(vertices_length[i]);
  }
  std::sort(v.begin(), v.end());
  int length = v.size();
  if (length % 2 != 0)
    cutoff = v[length / 2];
  else
    cutoff = (v[length / 2] + v[(length - 1) / 2]) / 2;
  printf("Precompute cutoff = %d\n", cutoff);
}

void PrecTable::toDevice(bool pc_use_uvm) {
  uint64_t vertices_total_length = 0;
  uint64_t results_total_length = 0;
  if (isCompress) {
    for (uint32_t i = 0; i < nonzeroVerticesNum; i++) {
      vertices_total_length += vertices_length[i];
    }
    for (uint32_t i = 0; i < nonzeroResultsNum; i++) {
      results_total_length += results_length[i];
    }
    assert(nonzeroVerticesMap.size() == nonzeroVerticesNum);
    assert(nonzeroResultsMap.size() == nonzeroResultsNum);

    int *h_vertices;
    int *h_results;
    int *h_vertices_offsets;
    int *h_results_offsets;

  // printf("~~~~~~ %d %d %d %d\n", vertices_total_length, results_total_length, nonzeroVerticesNum, nonzeroResultsNum);
    
    if (pc_use_uvm) {
      CHECK_ERROR(cudaMallocManaged((void **)&h_vertices,
                                    sizeof(int) * (vertices_total_length + 1)));
      CHECK_ERROR(cudaMallocManaged((void **)&h_results,
                                    sizeof(int) * (results_total_length + 1)));
      CHECK_ERROR(cudaMallocManaged((void **)&h_vertices_offsets,
                                    sizeof(int) * (nonzeroVerticesNum + 1)));
      CHECK_ERROR(cudaMallocManaged((void **)&h_results_offsets,
                                    sizeof(int) * (nonzeroResultsNum + 1)));
    } else {
      h_vertices = new int[vertices_total_length + 1];
      h_results = new int[results_total_length + 1];
      h_vertices_offsets = new int[nonzeroVerticesNum + 1];
      h_results_offsets = new int[nonzeroResultsNum + 1];
    }

    int h_vertices_end = 0;
    int h_results_end = 0;
    for (uint32_t i = 0; i <= nonzeroVerticesNum; i++) {
      if (i == nonzeroVerticesNum) {
        h_vertices_offsets[i] = h_vertices_end;
      } else {
        assert(vertices_length[i] > 0);
        memcpy(h_vertices + h_vertices_end, vertices[i],
               sizeof(int) * vertices_length[i]);
        h_vertices_offsets[i] = h_vertices_end;
        h_vertices_end += vertices_length[i];
      }
    }
    for (uint32_t i = 0; i <= nonzeroResultsNum; i++) {
      if (i == nonzeroResultsNum) {
        h_results_offsets[i] = h_results_end;
      } else {
        assert(results_length[i] > 0);
        memcpy(h_results + h_results_end, results[i],
               sizeof(int) * results_length[i]);
        h_results_offsets[i] = h_results_end;
        h_results_end += results_length[i];
      }
    }
    assert(h_vertices_end == vertices_total_length);
    assert(h_results_end == results_total_length);

#define cudaMallocAndCpy(buffer, data, length)                                 \
  CHECK_ERROR(cudaMalloc((void **)&buffer, sizeof(int) * length));             \
  CHECK_ERROR(cudaMemcpy((void *)buffer, (int *)data, sizeof(int) * length,    \
                         cudaMemcpyHostToDevice));
#define cudaMallocAndCpyU32(buffer, data, length)                              \
  CHECK_ERROR(cudaMalloc((void **)&buffer, sizeof(uint32_t) * length));        \
  if (length > 1)                                                              \
    CHECK_ERROR(cudaMemcpy((void *)buffer, (uint32_t *)data,                   \
                           sizeof(uint32_t) * length,                          \
                           cudaMemcpyHostToDevice));

#define cudaMallocAndCpyUVM(buffer, data, length)                              \
  CHECK_ERROR(cudaMallocManaged((void **)&buffer, sizeof(int) * length));      \
  memcpy((void *)buffer, (int *)data, sizeof(int) * length);
#define cudaMallocAndCpyU32UVM(buffer, data, length)                           \
  CHECK_ERROR(cudaMallocManaged((void **)&buffer, sizeof(uint32_t) * length)); \
  if (length > 1)                                                              \
    memcpy((void *)buffer, (uint32_t *)data, sizeof(uint32_t) * length);

    if (pc_use_uvm) {
      printf("Use pc uvm\n");
      // cudaMallocAndCpyUVM(d_vertices, h_vertices, (vertices_total_length + 1));
      // cudaMallocAndCpyUVM(d_results, h_results, (results_total_length + 1));
      // cudaMallocAndCpyUVM(d_vertices_offsets, h_vertices_offsets,
      //                     (nonzeroVerticesNum + 1));
      // cudaMallocAndCpyUVM(d_results_offsets, h_results_offsets,
      //                     (nonzeroResultsNum + 1));
      d_vertices = h_vertices;
      d_results = h_results;
      d_vertices_offsets = h_vertices_offsets;
      d_results_offsets = h_results_offsets;
      cudaMallocAndCpyU32UVM(d_nonzeroVerticesMap, (nonzeroVerticesMap.data()),
                             (nonzeroVerticesNum + 1));
      cudaMallocAndCpyU32UVM(d_nonzeroResultsMap, (nonzeroResultsMap.data()),
                             (nonzeroResultsNum + 1));
    } else {
      cudaMallocAndCpy(d_vertices, h_vertices, (vertices_total_length + 1));
      cudaMallocAndCpy(d_results, h_results, (results_total_length + 1));
      cudaMallocAndCpy(d_vertices_offsets, h_vertices_offsets,
                       (nonzeroVerticesNum + 1));
      cudaMallocAndCpy(d_results_offsets, h_results_offsets,
                       (nonzeroResultsNum + 1));
      cudaMallocAndCpyU32(d_nonzeroVerticesMap, (nonzeroVerticesMap.data()),
                          (nonzeroVerticesNum + 1));
      cudaMallocAndCpyU32(d_nonzeroResultsMap, (nonzeroResultsMap.data()),
                          (nonzeroResultsNum + 1));
      delete[] h_vertices;
      delete[] h_vertices_offsets;
      delete[] h_results;
      delete[] h_results_offsets;
    }
  } else {
    for (uint32_t i = 0; i < size; i++) {
      vertices_total_length += vertices_length[i];
      results_total_length += results_length[i];
    }

    int *h_vertices;
    int *h_results;
    int *h_vertices_offsets;
    int *h_results_offsets;

    // printf("~~~~~~ %d %d %d %d\n", vertices_total_length, results_total_length, size, size);
    if (pc_use_uvm) {
      CHECK_ERROR(cudaMallocManaged((void **)&h_vertices,
                                    sizeof(int) * (vertices_total_length)));
      CHECK_ERROR(cudaMallocManaged((void **)&h_results,
                                    sizeof(int) * (results_total_length)));
      CHECK_ERROR(cudaMallocManaged((void **)&h_vertices_offsets,
                                    sizeof(int) * (size + 1)));
      CHECK_ERROR(cudaMallocManaged((void **)&h_results_offsets,
                                    sizeof(int) * (size + 1)));
    } else {
      h_vertices = new int[vertices_total_length];
      h_results = new int[results_total_length];
      h_vertices_offsets = new int[size + 1];
      h_results_offsets = new int[size + 1];
    }

    int h_vertices_end = 0;
    int h_results_end = 0;
    for (uint64_t i = 0; i < size; i++) {
      if (vertices_length[i] > 0)
        memcpy(h_vertices + h_vertices_end, vertices[i],
               sizeof(int) * vertices_length[i]);
      if (results_length[i] > 0)
        memcpy(h_results + h_results_end, results[i],
               sizeof(int) * results_length[i]);

      h_vertices_offsets[i] = h_vertices_end;
      h_results_offsets[i] = h_results_end;
      h_vertices_end += vertices_length[i];
      h_results_end += results_length[i];
    }
    h_vertices_offsets[size] = h_vertices_end;
    h_results_offsets[size] = h_results_end;
    assert(h_vertices_end == vertices_total_length);
    assert(h_results_end == results_total_length);

#define cudaMallocAndCpy(buffer, data, length)                                 \
  CHECK_ERROR(cudaMalloc((void **)&buffer, sizeof(int) * length));             \
  CHECK_ERROR(cudaMemcpy((void *)buffer, (int *)data, sizeof(int) * length,    \
                         cudaMemcpyHostToDevice));

#define cudaMallocAndCpyUVM2(buffer, data, length)                              \
  CHECK_ERROR(cudaMallocManaged((void **)&buffer, sizeof(int) * length));      \
  memcpy((void *)buffer, (int *)data, sizeof(int) * length);

    if (pc_use_uvm) {
      printf("Use pc uvm\n");
      // cudaMallocAndCpyUVM2(d_vertices, h_vertices, vertices_total_length);
      // cudaMallocAndCpyUVM2(d_results, h_results, results_total_length);
      // cudaMallocAndCpyUVM2(d_vertices_offsets, h_vertices_offsets, (size + 1));
      // cudaMallocAndCpyUVM2(d_results_offsets, h_results_offsets, (size + 1));
      d_vertices = h_vertices;
      d_results = h_results;
      d_vertices_offsets = h_vertices_offsets;
      d_results_offsets = h_results_offsets;
    } else {
      cudaMallocAndCpy(d_vertices, h_vertices, vertices_total_length);
      cudaMallocAndCpy(d_results, h_results, results_total_length);
      cudaMallocAndCpy(d_vertices_offsets, h_vertices_offsets, (size + 1));
      cudaMallocAndCpy(d_results_offsets, h_results_offsets, (size + 1));
      delete[] h_vertices;
      delete[] h_vertices_offsets;
      delete[] h_results;
      delete[] h_results_offsets;
    }
  }
}

void PrecTable::releaseHost() {
  if (!isCompress) {
    for (uint64_t i = 0; i < size; i++) {
      if (vertices_length[i] != 0)
        delete[] vertices[i];
      if (results_length[i] != 0)
        delete[] results[i];
    }
  } else {
    for (uint32_t i = 0; i < nonzeroVerticesNum; i++) {
      delete[] vertices[i];
    }
    for (uint32_t i = 0; i < nonzeroResultsNum; i++) {
      delete[] results[i];
    }
  }
  vertices = std::vector<int *>();
  results = std::vector<int *>();
  vertices_length = std::vector<int>();
  results_length = std::vector<int>();
  nonzeroVerticesMap = std::vector<uint32_t>();
  nonzeroResultsMap = std::vector<uint32_t>();
}

void PrecTable::releaseDevice() {
  CHECK_ERROR(cudaFree((void *)d_vertices));
  CHECK_ERROR(cudaFree((void *)d_results));
  CHECK_ERROR(cudaFree((void *)d_vertices_offsets));
  CHECK_ERROR(cudaFree((void *)d_results_offsets));
  CHECK_ERROR(cudaFree((void *)d_nonzeroVerticesMap));
  CHECK_ERROR(cudaFree((void *)d_nonzeroResultsMap));
}