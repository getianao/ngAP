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
  Matchset *groups_ms;

  void init(std::vector<Graph *> &gs, bool use_soa) {
    this->size = gs.size();
    CHECK_ERROR(cudaMalloc(&groups_ms, sizeof(Matchset) * size));
    for (int i = 0; i < size; i++) {
      Graph *graph = gs[i];
      Matchset ms = graph->get_matchset_device(use_soa);
      CHECK_ERROR(cudaMemcpy((void *)(groups_ms + i), (Matchset *)&ms,
                             sizeof(Matchset), cudaMemcpyHostToDevice));
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