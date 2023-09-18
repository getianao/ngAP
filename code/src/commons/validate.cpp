#include "validate.h"

#include "omp.h"
#include <chrono>
#include <cmath>

void automata_utils::automataGroupsReference(std::vector<Graph *> &gs,
                                             uint8_t *input_str, int num_seg,
                                             int input_length,
                                             std::vector<uint64_t> *results,
                                             std::vector<uint64_t> *db_results,
                                             int debug_iter, GroupCsr gcsr) {
  printf("use automata_utils automataGroupsReference\n");

  auto addResult = [](uint32_t node, uint32_t index, uint32_t nfa_index,
                      std::vector<uint64_t> *results) {
    uint64_t r = 0;
    // if(node > 0xffffff)
    //  printf("!! %p\n", node);
    node = (node | (nfa_index << 22));
    r = (uint32_t)node;
    r = r << 32;
    // index = (index | (nfa_index << 20));
    r = r | (uint32_t)index;
#pragma omp critical
    results->push_back(r);
  };

  auto start_time = std::chrono::high_resolution_clock::now();
  omp_set_num_threads(num_seg * gs.size());

#pragma omp parallel
  {
    std::vector<int> frontiers[2];
    int ns = omp_get_thread_num();
    int nfa_index = ns / num_seg;
    int input_index = ns % num_seg;
    Graph *g = gs[nfa_index];
    Csr csr = gcsr.h_groups_csr[nfa_index];

    int iter = input_index * input_length;
    frontiers[0].clear();
    frontiers[1].clear();
    if (input_index == 0)
      for (int i = 0; i < g->startActiveNum; i++) {
        frontiers[0].push_back(g->start_active_nodes->get_host()[i]);
      }
    for (int i = 0; i < g->alwaysActiveNum; i++) {
      frontiers[0].push_back(g->always_active_nodes->get_host()[i]);
    }

    while (!frontiers[iter % 2].empty() &&
           iter < input_length * (input_index + 1)) {
      // auto printFrontier = [iter](std::vector<int> frontier,
      //                             const char *message) {
      //   printf("[CPU automata]:%d %s frontier:", iter, message);
      //   for (auto v : frontier) {
      //     printf("%d, ", v);
      //   }
      //   printf("\n");
      // };
      char current_symbol = input_str[iter];
      auto &curr_frontier = frontiers[iter % 2];
      auto &next_frontier = frontiers[(iter + 1) % 2];
      next_frontier.clear();
      for (auto v : curr_frontier) {
#ifdef DEBUG_PL_FILTER_ITER
        // debug per iteration
        if (iter == debug_iter) {
          addResult(v, iter, db_results);
        }
#endif
        if (g->symbol_sets->get_host()[v].test(current_symbol)) {
          // report
          if (g->node_attrs->get_host()[v] & 0b10) {
            addResult(v, iter, nfa_index, results);
          }
          int e_start = csr.GetNeighborListOffset(v);
          int e_end = e_start + csr.GetNeighborListLength(v);
          for (int e = e_start; e < e_end; e++) {
            int u = csr.GetEdgeDest(e);
            next_frontier.push_back(u);
          }
        }
      }
      if (false) {
        std::sort(next_frontier.begin(), next_frontier.end());
        next_frontier.erase(
            std::unique(next_frontier.begin(), next_frontier.end()),
            next_frontier.end());
      }
      // #ifdef DEBUG_PL_ADVANCE_CONCAT
      for (int i = 0; i < g->alwaysActiveNum; i++) {
        next_frontier.push_back(g->always_active_nodes->get_host()[i]);
      }
      // #endif
      iter++;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  double throughput =
      (double)input_length * num_seg / (duration.count() * 1000);
  std::cout << "Reference CPU elapsed time: " << duration.count() << " ms,"
            << " " << throughput << " MB/s" << std::endl;
}

void automata_utils::automataReference(Graph &g, uint8_t *input_str,
                                       int num_seg, int input_length,
                                       std::vector<uint64_t> *results,
                                       std::vector<uint64_t> *db_results,
                                       int debug_iter, Csr csr) {
  printf("use automata_utils automataReference\n");

  auto addResult = [](uint32_t node, uint32_t index,
                      std::vector<uint64_t> *results) {
    uint64_t r = 0;
    r = (uint32_t)node;
    r = r << 32;
    r = r | (uint32_t)index;
#pragma omp critical
    results->push_back(r);
  };

  auto start_time = std::chrono::high_resolution_clock::now();
  omp_set_num_threads(num_seg);

#pragma omp parallel
  {
    std::vector<int> frontiers[2];
    int ns = omp_get_thread_num();
    int iter = ns * input_length;
    frontiers[0].clear();
    frontiers[1].clear();
    if (ns == 0)
      for (int i = 0; i < g.startActiveNum; i++) {
        frontiers[0].push_back(g.start_active_nodes->get_host()[i]);
      }
    for (int i = 0; i < g.alwaysActiveNum; i++) {
      frontiers[0].push_back(g.always_active_nodes->get_host()[i]);
    }

    while (!frontiers[iter % 2].empty() && iter < input_length * (ns + 1)) {
      // auto printFrontier = [iter](std::vector<int> frontier,
      //                             const char *message) {
      //   printf("[CPU automata]:%d %s frontier:", iter, message);
      //   for (auto v : frontier) {
      //     printf("%d, ", v);
      //   }
      //   printf("\n");
      // };
      char current_symbol = input_str[iter];
      auto &curr_frontier = frontiers[iter % 2];
      auto &next_frontier = frontiers[(iter + 1) % 2];
      next_frontier.clear();
      for (auto v : curr_frontier) {
#ifdef DEBUG_PL_FILTER_ITER
        // debug per iteration
        if (iter == debug_iter) {
          addResult(v, iter, db_results);
        }
#endif
        if (g.symbol_sets->get_host()[v].test(current_symbol)) {
          // report
          if (g.node_attrs->get_host()[v] & 0b10) {
            addResult(v, iter, results);
          }
          int e_start = csr.GetNeighborListOffset(v);
          int e_end = e_start + csr.GetNeighborListLength(v);
          for (int e = e_start; e < e_end; e++) {
            int u = csr.GetEdgeDest(e);
            next_frontier.push_back(u);
          }
        }
      }
      if (true) {
        std::sort(next_frontier.begin(), next_frontier.end());
        next_frontier.erase(
            std::unique(next_frontier.begin(), next_frontier.end()),
            next_frontier.end());
      }
      // #ifdef DEBUG_PL_ADVANCE_CONCAT
      for (int i = 0; i < g.alwaysActiveNum; i++) {
        next_frontier.push_back(g.always_active_nodes->get_host()[i]);
      }
      // #endif
      iter++;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  double throughput =
      (double)input_length * num_seg / (duration.count() * 1000);
  std::cout << "Reference CPU elapsed time: " << duration.count() << " ms,"
            << " " << throughput << " MB/s" << std::endl;
}


bool automata_utils::automataValidation(std::vector<uint64_t> *results,
                                  std::vector<uint64_t> *ref_results,
                                  bool ifPrintBoth) {
  auto printBoth = [](std::vector<uint64_t> *results,
                      std::vector<uint64_t> *ref_results) {
    auto printVector = [](std::vector<uint64_t> *v) {
      if (v->size() > 40) {
        // if (false) {
        for (int i = 0; i < 40; i++)
          printf("0x%lx, ", (*v)[i]);
        printf("...\n");
      } else {
        for (int i = 0; i < v->size(); i++)
          printf("0x%lx, ", (*v)[i]);
        printf("\n");
      }
    };
    printf("Result(%zu): \n", results->size());
    printVector(results);
    printf("Reference result(%zu): \n", ref_results->size());
    printVector(ref_results);
    printf("\n");
  };

  auto compareResult = [](uint64_t r1, uint64_t r2) -> bool {
    uint32_t input_1, state_1, input_2, state_2;
    input_1 = (uint32_t)(0xffffffff & r1);
    state_1 = (uint32_t)(r1 >> 32);
    input_2 = (uint32_t)(0xffffffff & r2);
    state_2 = (uint32_t)(r2 >> 32);
    // printf("{%u, %u}, ", state, input);
    if (input_1 == input_2)
      return state_1 < state_2;
    else
      return input_1 < input_2;
  };
  std::sort(results->begin(), results->end(), compareResult);
  results->erase(unique(results->begin(), results->end()), results->end());
  std::sort(ref_results->begin(), ref_results->end(), compareResult);
  ref_results->erase(unique(ref_results->begin(), ref_results->end()),
                     ref_results->end());

  int size = results->size();
  int ref_size = ref_results->size();
  if (size != ref_size) {
    tge_log("Validation FAILED!", BOLDRED);
    std::cerr << BOLDRED << "Validation FAILED!" << RESET << "\n";
    int bound = std::min(size, ref_size);
    for (int i = 0; i < bound; i++) {
      if ((*results)[i] != (*ref_results)[i]) {
        printf("0x%lx != 0x%lx\n", (*results)[i], (*ref_results)[i]);
        break;
      }
    }
    printBoth(results, ref_results);
    return false;
  }
  for (int i = 0; i < size; i++) {
    if ((*results)[i] != (*ref_results)[i]) {
      tge_log("Validation FAILED!", BOLDRED);
      std::cerr << BOLDRED << "Validation FAILED!" << RESET << "\n";
      printBoth(results, ref_results);
      return false;
    }
  }
  tge_log("Validation PASS!", BOLDGREEN);
  // std::cerr<<"Validation PASS!\n";
  if (ifPrintBoth)
    printBoth(results, ref_results);
  return true;
}
