#ifndef GROUP_GRAPH_H_
#define GROUP_GRAPH_H_

#include "graph.h"

class GroupCsr {
public:
  int size;
  Csr *groups_csr;
  Csr *h_groups_csr;

  void init(std::vector<Graph *> &gs) {
    this->size = gs.size();
    h_groups_csr = new Csr[size];
    CHECK_ERROR(cudaMalloc(&groups_csr, sizeof(Csr) * size));
    for (int i = 0; i < size; i++) {
      Graph *graph = gs[i];
      Csr csr(*graph);
      csr.fromCoo(graph->edge_pairs->get_host());
      csr.moveToDevice();
      h_groups_csr[i] = csr;
      CHECK_ERROR(cudaMemcpy((void *)(groups_csr + i), (Csr *)&csr, sizeof(Csr),
                             cudaMemcpyHostToDevice));
    }
  }

  void release() {
    if (size > 0) {
      CHECK_ERROR(cudaFree((void *)groups_csr));
      delete[] h_groups_csr;
    }
  }
};

class GroupMatchset {
public:
  int size;
  MatchsetUnique *groups_ms;

  void init(std::vector<Graph *> &gs, bool use_soa,
            Array2<uint8_t> *input_stream, int input_total_size) {
    this->size = gs.size();
    CHECK_ERROR(cudaMalloc(&groups_ms, sizeof(MatchsetUnique) * size));
    for (int i = 0; i < size; i++) {
      Graph *graph = gs[i];
      MatchsetUnique ms = graph->get_matchset_unique_device(
          graph->symbol_sets_unique->size(), use_soa);

      // CC stream
      
      uint32_t cc_string_size = (input_total_size + 31) / 32;
      CHECK_ERROR(cudaMalloc(&ms.cc_stream,
                             sizeof(uint32_t *) * ms.size * cc_string_size));
      ms.cc_stream_size = cc_string_size;
      printf("cc stream size: %f KB\n", ms.size * cc_string_size * 4 / 1024.0);
      for (int cc_id = 0; cc_id < ms.size; cc_id++) {
        uint32_t *cc_stream = new uint32_t[cc_string_size];
        memset(cc_stream, 0, sizeof(uint32_t) * cc_string_size);
        printf("cc_id: %d\n", cc_id);
        printf("input_total_size: %d\n", input_total_size);
        My_bitset256 match_set_cc = graph->symbol_sets_unique->get_host()[cc_id];
        for (uint32_t k = 0; k < input_total_size; k++) {
          uint8_t c = input_stream->get_host()[k];
          if (match_set_cc.test(c)) {
            // printf("k: %d, c: %d\n", k, c);
            cc_stream[k / 32] |= 1 << (31 - k % 32);
          }
        }

        CHECK_ERROR(cudaMemcpy(ms.cc_stream + cc_id * cc_string_size, cc_stream,
                               sizeof(uint32_t) * cc_string_size,
                               cudaMemcpyHostToDevice));
        delete[] cc_stream;
      }
      printf("cc stream size: %f KB\n", ms.size * cc_string_size * 4 / 1000.0);
      CHECK_ERROR(cudaMemcpy((void *)(groups_ms + i), (MatchsetUnique *)&ms,
                             sizeof(MatchsetUnique), cudaMemcpyHostToDevice));
    }
  }

  void release() {
    if (size > 0) {
      CHECK_ERROR(cudaFree((void *)groups_ms));
    }
  }
};

class GroupNodeAttrs {
public:
  int size;
  uint8_t **groups_node_attrs;

  void init(std::vector<Graph *> &gs) {
    this->size = gs.size();
    CHECK_ERROR(cudaMalloc(&groups_node_attrs, sizeof(uint8_t *) * size));
    for (int i = 0; i < size; i++) {
      Graph *graph = gs[i];
      uint8_t *pointer = graph->node_attrs->get_dev();
      CHECK_ERROR(cudaMemcpy((void *)(groups_node_attrs + i), (void *)&pointer,
                             sizeof(uint8_t *), cudaMemcpyHostToDevice));
    }
  }

  void release() {
    if (size > 0) {
      // for (int i = 0; i < size; i++) {
      //   CHECK_ERROR(cudaFree((void *)groups_node_attrs[i]));
      // }
      CHECK_ERROR(cudaFree((void *)groups_node_attrs));
    }
  }
};

class GroupAAS {
public:
  int size;
  int **groups_always_active_states;

  void init(std::vector<Graph *> &gs) {
    this->size = gs.size();
    CHECK_ERROR(cudaMalloc(&groups_always_active_states, sizeof(int *) * size));
    for (int i = 0; i < size; i++) {
      Graph *graph = gs[i];
      int *pointer = graph->always_active_nodes->get_dev();
      CHECK_ERROR(cudaMemcpy((void *)(groups_always_active_states + i),
                             (void *)&pointer, sizeof(int *),
                             cudaMemcpyHostToDevice));
    }
  }

  void release() {
    if (size > 0) {
      // for (int i = 0; i < size; i++) {
      //   CHECK_ERROR(cudaFree((void *)groups_always_active_states[i]));
      // }
      CHECK_ERROR(cudaFree((void *)groups_always_active_states));
    }
  }
};

#endif