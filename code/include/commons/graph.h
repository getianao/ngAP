#ifndef GRAPH_H_
#define GRAPH_H_

//  borrow from gunrock

#include "gpunfautils/array2.h"
#include "device_intrinsics.h"
#include "my_bitset.h"
#include "precompute_table.h"
#include "gpunfautils/utils.h"
#include <cmath>

struct Matchset {
  bool use_soa = false;
  uint32_t *d_data = NULL;
  int sizeofdata = 0;
  int size = 0;

  void release() { cudaFree((void *)d_data); }

  __device__ inline bool test(int vertex, int symbol) {
    if (use_soa) {
      return *(d_data + symbol * sizeofdata + (vertex / 32)) &
             (1 << (vertex % 32));
    } else {
      return *(d_data + vertex * sizeofdata + (symbol / 32)) &
             (1 << (symbol % 32));
    }
  }
};

struct MatchsetUnique {
  bool use_soa = false;
  uint32_t *d_data = NULL;
  int sizeofdata = 0; // 8 bytes for 256
  int size = 0;

  uint16_t *matchsetidx;

  void release() {
    cudaFree((void *)d_data);
    cudaFree((void *)matchsetidx);
  }

  __device__ inline bool test(int vertex, int symbol) {
    
      // printf("vertex: %d, symbolset_id: %d\n", vertex, matchsetidx[vertex]);
      vertex = matchsetidx[vertex]; // get the real index
      return *(d_data + vertex * sizeofdata + (symbol / 32)) &
             (1 << (symbol % 32));
    }
};

template <typename T1, typename ArrayT, typename SizeT, typename LessOp>
__forceinline__ SizeT BinarySearch_LeftMost(const T1 &element_to_find,
                                            const ArrayT &elements,
                                            SizeT left_index, SizeT right_index,
                                            LessOp less,
                                            bool check_boundaries = true) {
  return BinarySearch_LeftMost(
      element_to_find, elements, left_index, right_index, less,
      [](const T1 &a, const T1 &b) { return (a == b); }, check_boundaries);
}

template <typename T1, typename ArrayT, typename SizeT>
__forceinline__ SizeT BinarySearch_LeftMost(const T1 &element_to_find,
                                            const ArrayT &elements,
                                            SizeT left_index, SizeT right_index,
                                            bool check_boundaries = true) {
  return BinarySearch_LeftMost(
      element_to_find, elements, left_index, right_index,
      [](const T1 &a, const T1 &b) { return (a < b); },
      [](const T1 &a, const T1 &b) { return (a == b); }, check_boundaries);
}

template <typename T1, typename ArrayT, typename SizeT, typename LessOp,
          typename EqualOp>
SizeT BinarySearch_LeftMost(
    const T1 &element_to_find,
    const ArrayT &elements, // the array
    SizeT left_index,       // left index of range, inclusive
    SizeT right_index,      // right index of range, inclusive
    LessOp less,            // strictly less
    EqualOp equal, bool check_boundaries = true) {
  SizeT org_right_index = right_index;
  SizeT center_index = 0;

  if (check_boundaries) {
    if ((!(less(elements[left_index], element_to_find))) &&
        (!(equal(elements[left_index], element_to_find))))
      return left_index - 1;
    if (less(elements[right_index], element_to_find))
      return right_index;
  }

  while (right_index - left_index > 1) {
    center_index = ((long long)left_index + (long long)right_index) >> 1;
    if (less(elements[center_index], element_to_find))
      left_index = center_index;
    else
      right_index = center_index;
  }

  if (center_index < org_right_index &&
      less(elements[center_index], element_to_find) &&
      equal(elements[center_index + 1], element_to_find)) {
    center_index++;
  } else if (center_index > 0 &&
             !less(elements[center_index], element_to_find) &&
             !equal(elements[center_index], element_to_find)) {
    center_index--;
  }

  while (center_index > 0 && equal(elements[center_index - 1], element_to_find))
    center_index--;
  return center_index;
}
class Edge {
public:
  int x, y;
};

class Graph {
public:
  Graph();
  Graph(Graph &c);

  Array2<Edge> *edge_pairs;
  Array2<My_bitset256> *symbol_sets;
  Array2<uint8_t> *node_attrs;
  Array2<int> *always_active_nodes;
  Array2<int> *start_active_nodes;

  int nodesNum = 0;
  int edgesNum = 0;
  int alwaysActiveNum = 0;
  int startActiveNum = 0;
  int reportingStateNum = 0;
  int input_length = 0;

  Array2<My_bitset256> *symbol_sets_unique;
  Array2<uint16_t> *node2matchsetidx;

  cudaError_t ReadANML(std::string filename);
  cudaError_t ReadNFA(NFA *nfa);
  cudaError_t allocate(int nodesNum, int edgesNum, int alwaysActiveNum,
                       int startActiveNum);
  cudaError_t release();
  cudaError_t copyToDevice();
  Matchset get_matchset_device(bool is_soa);
  MatchsetUnique get_matchset_unique_device(int matchset_unique_num, bool use_soa);
};

class Csr {
public:
  int *d_column_indices;
  int *d_row_offsets;

  int *h_column_indices;
  int *h_row_offsets;

  int *d_row_offsets_compressed_type; // node -> type
  int *d_row_offsets_compressed;   // type -> offset
  int *d_column_indices_compressed;  // data

  int *h_row_offsets_compressed_type;
  int *h_row_offsets_compressed;
  int *h_column_indices_compressed;

  int nodesNum = 0;
  int edgesNum = 0;
  int alwaysActiveNum = 0;
  int startActiveNum = 0;
  int reportingStateNum = 0;
  int input_length = 0;

// O3 precompute
  PrecTable *h_pts;
  PrecTable *d_pts;
  int precompute_depth = 0;
  int precompute_cutoff;

  Csr() {}

  Csr(Graph &g) {
    nodesNum = g.nodesNum;
    edgesNum = g.edgesNum;
    alwaysActiveNum = g.alwaysActiveNum;
    startActiveNum = g.startActiveNum;
    reportingStateNum = g.reportingStateNum;
    input_length = g.input_length;
  }

  // ~Csr(){
    
  // }

  void release() {
    delete[] h_row_offsets;
    delete[] h_column_indices;
  }
  void releaseDevice() {
    cudaFree((void *)d_column_indices);
    cudaFree((void *)d_row_offsets);
  }

  void moveToDevice() {
    if (edgesNum > 0) {
      CHECK_ERROR(
          cudaMalloc((void **)&d_column_indices, sizeof(int) * edgesNum));
      CHECK_ERROR(cudaMemcpy((void *)d_column_indices, h_column_indices,
                             sizeof(int) * edgesNum, cudaMemcpyHostToDevice));
    }
    CHECK_ERROR(
        cudaMalloc((void **)&d_row_offsets, sizeof(int) * (nodesNum + 1)));
    CHECK_ERROR(cudaMemcpy((void *)d_row_offsets, h_row_offsets,
                           sizeof(int) * (nodesNum + 1),
                           cudaMemcpyHostToDevice));
  }

  void fromCoo(Edge *eps) {
    if (edgesNum <= 0) {
      h_row_offsets = new int[nodesNum + 1];
      memset(h_row_offsets, 0, sizeof(int) * (nodesNum + 1));
      return;
    }
    auto compareEdge = [](Edge a, Edge b) -> bool {
      if (a.x == b.x)
        return a.y < b.y;
      else
        return a.x < b.x;
    };
    std::sort(eps, eps + edgesNum, compareEdge);
    // assign column_indices
    h_column_indices = new int[edgesNum];
    for (int i = 0; i < edgesNum; i++) {
      h_column_indices[i] = eps[i].y;
    }
    // assign row_offsets
    h_row_offsets = new int[nodesNum + 1];
    auto row_edge_compare = [](const Edge &edge_pair, const int &row) {
      return edge_pair.x < row;
    };

    for (int row = 0; row < nodesNum + 1; row++) {
      if (row <= eps[0].x)
        h_row_offsets[row] = 0;
      else if (row < nodesNum) {
        auto pos = BinarySearch_LeftMost(
            row, eps, 0, edgesNum - 1, row_edge_compare,
            [](const Edge &pair, const int &row) { return (pair.x == row); });
        while (pos < edgesNum && row > eps[pos].x)
          pos++;

        h_row_offsets[row] = pos;
      } else
        h_row_offsets[row] = edgesNum;
    }
  }

  // struct SetHash {
  //   template <typename T> std::size_t operator()(const std::set<T> &s) const {
  //     std::size_t hash = 0;
  //     for (const T &elem : s) {
  //       hash ^= std::hash<T>()(elem) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  //     }
  //     return hash;
  //   }
  // };

  void compressCsr() {
    // cagegorize the nodes by the neighbor
    std::map<std::set<int>, int> unique_set_map; // unique neigbor set -> type
    std::vector<std::set<int>> unique_sets; // type -> unique neigbor set
    std::map<int, int> unique_set_num_map;       // type -> type number
    
    std::vector<int> node_to_set; // node -> type

    int num_nodes = nodesNum;
    for (int node = 0; node < num_nodes; ++node) {
      // Extract the neighbors for the current node
      int start = GetNeighborListOffset(node);
      int end = start + GetNeighborListLength(node);

      std::set<int> neighbors(h_column_indices + start, h_column_indices + end);

      int type;
      if (unique_set_map.find(neighbors) == unique_set_map.end()) {
        type = unique_sets.size();
        unique_set_map[neighbors] = type;
        unique_set_num_map[type] = 1;
        unique_sets.push_back(neighbors);
      } else {
        type = unique_set_map[neighbors];
        unique_set_num_map[type] += 1;
      }
      node_to_set.push_back(type);
    }

    printf("node num: %d\n", num_nodes);
    printf("unique_sets.size(): %d\n", unique_sets.size());

    // Node to type
    h_row_offsets_compressed_type = new int[num_nodes];
    memcpy(h_row_offsets_compressed_type, node_to_set.data(),
           sizeof(int) * num_nodes);
    // type to offset
    // offset to data
    h_row_offsets_compressed = new int[unique_sets.size() + 1];
    std::vector<int> column_indices_compressed;
    for (int i = 0; i <= unique_sets.size(); i++) {
      if (i == 0) {
        h_row_offsets_compressed[i] = 0;
      } else {
        h_row_offsets_compressed[i] =
            h_row_offsets_compressed[i - 1] + unique_sets[i - 1].size();
        column_indices_compressed.insert(column_indices_compressed.end(),
                                         unique_sets[i - 1].begin(),
                                         unique_sets[i - 1].end());
      }
    }

    memcpy(h_column_indices_compressed, column_indices_compressed.data(),
           sizeof(int) * column_indices_compressed.size());
    CHECK_ERROR(cudaMalloc((void **)&d_row_offsets_compressed_type,
                           sizeof(int) * num_nodes));
    CHECK_ERROR(cudaMemcpy((void *)d_row_offsets_compressed_type,
                           h_row_offsets_compressed_type,
                           sizeof(int) * num_nodes, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMalloc((void **)&d_row_offsets_compressed,
                           sizeof(int) * (unique_sets.size() + 1)));
    CHECK_ERROR(cudaMemcpy(
        (void *)d_row_offsets_compressed, h_row_offsets_compressed,
        sizeof(int) * (unique_sets.size() + 1), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMalloc((void **)&d_column_indices_compressed,
                           sizeof(int) * column_indices_compressed.size()));
    CHECK_ERROR(cudaMemcpy((void *)d_column_indices_compressed,
                           column_indices_compressed.data(),
                           sizeof(int) * column_indices_compressed.size(),
                           cudaMemcpyHostToDevice));
  }

  __device__ __forceinline__ int GetNeighborListOffset2(const int &v) const {
    int type = _ldg(d_row_offsets_compressed_type + v);
    return _ldg(d_row_offsets_compressed + type);
  }

  __device__ __forceinline__ int GetNeighborListLength2(const int &v) const {
    int type = _ldg(d_row_offsets_compressed_type + v);

    return _ldg(d_row_offsets_compressed + (type + 1)) -
           _ldg(d_row_offsets_compressed + type);
  }

  __device__ __host__ __forceinline__ int
  GetNeighborListOffset(const int &v) const {
#ifdef __CUDA_ARCH__
    return _ldg(d_row_offsets + v);
#else
    return _ldg(h_row_offsets + v);
#endif
  }

  __device__ __host__ __forceinline__ int
  GetNeighborListLength(const int &v) const {
    if (v < 0 || v >= this->nodesNum)
      return 0;
#ifdef __CUDA_ARCH__
    return _ldg(d_row_offsets + (v + 1)) - _ldg(d_row_offsets + v);
#else
    return _ldg(h_row_offsets + (v + 1)) - _ldg(h_row_offsets + v);
#endif
  }

  __device__ __host__ __forceinline__ int GetEdgeDest(const int &e) const {
#ifdef __CUDA_ARCH__
    return d_column_indices[e];
#else
    return h_column_indices[e];
#endif
  }
};

int *GetHistogram(Csr &graph);

void PrintHistogram(Csr &graph);

#endif