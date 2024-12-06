#include "gpunfautils/array2.h"
#include "device_intrinsics.h"
#include "graph.h"
#include "pugixml/pugixml.hpp"
#include "commons/vasim_helper.h"

#include <omp.h>
#include <string>
#include <unordered_map>

Graph::Graph(){

}
Graph::Graph(Graph &c) {
  edge_pairs = c.edge_pairs;
  symbol_sets = c.symbol_sets;
  node_attrs = c.node_attrs;
  always_active_nodes = c.always_active_nodes;
  start_active_nodes = c.start_active_nodes;

  nodesNum = c.nodesNum;
  edgesNum = c.edgesNum;
  alwaysActiveNum = c.alwaysActiveNum;
  startActiveNum = c.startActiveNum;
  reportingStateNum = c.reportingStateNum;
  input_length = c.input_length;
}

cudaError_t Graph::ReadANML(std::string filename) {

  cudaError_t retval = cudaSuccess;

  nodesNum = 0;
  edgesNum = 0;
  alwaysActiveNum = 0;
  startActiveNum = 0;
  reportingStateNum = 0;

  // Calculate the size of automata
  pugi::xml_document doc;
  if (!doc.load_file(filename.c_str(),
                     pugi::parse_default | pugi::parse_declaration)) {
    std::cout << "Could not load .xml file: " << filename << std::endl;
    return cudaErrorUnknown;
  }
  // can handle finding automata-network at one or two layers under root
  pugi::xml_node nodes = doc.child("anml").child("automata-network");
  if (nodes.name() == "") {
    nodes = doc.child("automata-network");
  }
  std::string id_tmp = nodes.attribute("id").value();
  std::unordered_map<std::string, int> steNames;
  for (pugi::xml_node node = nodes.first_child(); node;
       node = node.next_sibling()) {
    std::string str = node.name();
    if (str.compare("state-transition-element") == 0) {
      std::string steName = node.attribute("id").value();
      std::string stdStart = node.attribute("start").value();
      if (!stdStart.empty()) {
        if (stdStart.compare("all-input") == 0)
          alwaysActiveNum++;
        else if (stdStart.compare("start-of-data") == 0)
          startActiveNum++;
      }
      if (steNames.find(steName) != steNames.end()) {
        std::cout << "Error parsing ANML graph: redundant ste" << std::endl;
        return cudaErrorUnknown;
      }
      steNames[steName] = nodesNum;
      nodesNum++;
      for (pugi::xml_node aom : node.children("activate-on-match")) {
        edgesNum++;
      }
      for (pugi::xml_node aom : node.children("report-on-match")) {
        reportingStateNum++;
        break;
      }
    }
  }

  
  // Allocate coo graph
  allocate(nodesNum, edgesNum, alwaysActiveNum, startActiveNum);

  int edgeIndex = 0;
  int alwaysActiveIndex = 0;
  int startActiveIndex = 0;

  
  for (pugi::xml_node node = nodes.first_child(); node;
       node = node.next_sibling()) {
    std::string str = node.name();
    if (str.compare("state-transition-element") == 0) {
      std::string startSteName = node.attribute("id").value();
      if (steNames.find(startSteName) == steNames.end()) {
        std::cout << "Error parsing ANML graph: can't find start vertice "
                  << startSteName << std::endl;
        return cudaErrorUnknown;
      }
      std::string stdSymbolSet = node.attribute("symbol-set").value();
      std::string stdStart = node.attribute("start").value();
      if (!stdSymbolSet.empty()) {
        std::bitset<256> column;
        VASim::parseSymbolSet(column, stdSymbolSet);
        symbol_sets->get_host()[steNames[startSteName]].fromBitset(column);
      }
      if (!stdStart.empty()) {
        if (stdStart.compare("all-input") == 0) {
          // always active
          node_attrs->get_host()[steNames[startSteName]] =
              (node_attrs->get_host()[steNames[startSteName]] | 0x1);
          always_active_nodes->get_host()[alwaysActiveIndex] = steNames[startSteName];
          alwaysActiveIndex++;
        } else if (stdStart.compare("start-of-data") == 0) {
          start_active_nodes->get_host()[startActiveIndex] = steNames[startSteName];
          startActiveIndex++;
        }
      }
      if (!stdStart.empty()) {
        
      }
      // children to move
      for (pugi::xml_node aom : node.children("activate-on-match")) {
        std::string endSteName = aom.attribute("element").value();
        if (steNames.find(endSteName) != steNames.end()) {
          edge_pairs->get_host()[edgeIndex].x =
              steNames[startSteName];                        // zero-based array
          edge_pairs->get_host()[edgeIndex].y = steNames[endSteName]; // zero-based array
          edgeIndex++;
        } else {
          std::cout << "Error parsing ANML graph: can't find end vertice "
                    << endSteName << std::endl;
          return cudaErrorUnknown;
        }
      }
      // reporting
      for (pugi::xml_node aom : node.children("report-on-match")) {
        // TODO(tge): ignore report code.
        node_attrs->get_host()[steNames[startSteName]] =
            node_attrs->get_host()[steNames[startSteName]] | (0x1 << 1);
      }
    } else if (str.compare("and") == 0) {
      std::cout << "Error parsing ANML graph: gate \"and\" is not implemented"
                << std::endl;
      return cudaErrorUnknown;
    } else if (str.compare("or") == 0) {
      std::cout << "Error parsing ANML graph: gate \"or\" is not implemented"
                << std::endl;
      return cudaErrorUnknown;
    } else if (str.compare("counter") == 0) {
      std::cout
          << "Error parsing ANML graph: gate \"counter\" is not implemented"
          << std::endl;
      return cudaErrorUnknown;
    } else if (str.compare("inverter") == 0) {
      std::cout
          << "Error parsing ANML graph: gate \"inverter\" is not implemented"
          << std::endl;
      return cudaErrorUnknown;
    } else if (str.compare("description") == 0) {
      // do nothing
      ;
    }
  }
  
  return retval;
}

cudaError_t Graph::ReadNFA(NFA *nfa) {
  cudaError_t retval = cudaSuccess;
  // Allocate coo graph
  nodesNum = nfa->size();
  edgesNum =  nfa->edge_size();
  alwaysActiveNum = nfa->always_active_nodes_num;
  startActiveNum = nfa->start_active_nodes_num;
  // reportingStateNum = 0;

  allocate(nodesNum, edgesNum, alwaysActiveNum, startActiveNum);
  int edgeIndex = 0;
  int alwaysActiveIndex = 0;
  int startActiveIndex = 0;

  for (int i = 0; i < nfa->size(); i++) {
    Node *node = nfa->get_node_by_int_id(i);
    symbol_sets->get_host()[i].fromBitset(node->symbol_set);
    // TODO(tge):useless
    if (node->is_start_always_enabled()) {
      node_attrs->get_host()[i] = (node_attrs->get_host()[i] | 0x1);
      always_active_nodes->get_host()[alwaysActiveIndex] = i;
      alwaysActiveIndex++;
    }
    if (node->is_start() && !node->is_start_always_enabled()) {
      start_active_nodes->get_host()[startActiveIndex] = i;
      startActiveIndex++;
    }
    if (node->is_report())
      node_attrs->get_host()[i] = (node_attrs->get_host()[i] | (0x1 << 1));
    for (int j = 0; j < nfa->adj[node->str_id].size(); j++) {
      edge_pairs->get_host()[edgeIndex].x = i;
      edge_pairs->get_host()[edgeIndex].y =
          nfa->get_node_by_str_id(nfa->adj[node->str_id][j])->sid;
      edgeIndex++;
    }
  }
  return retval;
}

cudaError_t Graph::allocate(int nodesNum, int edgesNum, int alwaysActiveNum,
                            int startActiveNum) {

  symbol_sets = new Array2<My_bitset256>(nodesNum);
  node_attrs = new Array2<uint8_t>(nodesNum);
  memset(node_attrs->get_host(), 0, sizeof(uint8_t) * nodesNum);
  edge_pairs = new Array2<Edge>(edgesNum);
  always_active_nodes = new Array2<int>(alwaysActiveNum);
  start_active_nodes = new Array2<int>(startActiveNum);

  // printf("graph info: nodesNum:%d "
  //        "edgesNum:%d  alwaysActiveNum:%d  startActiveNum:%d \n",
  //        nodesNum, edgesNum, alwaysActiveNum, startActiveNum);
  return cudaSuccess;
}

Matchset Graph::get_matchset_device(bool is_soa) {
  Matchset ms;
  if (is_soa) {
    My_bitsetN **h_symbol_sets_soa = new My_bitsetN *[256];
    for (int i = 0; i < 256; i++) {
      h_symbol_sets_soa[i] = new My_bitsetN(nodesNum);
    }
    for (int i = 0; i < 256; i++) {
      for (int j = 0; j < nodesNum; j++) {
        int value = symbol_sets->get_host()[j].test(i);
        h_symbol_sets_soa[i]->set(j, value);
      }
    }
    ms.use_soa = true;
    ms.sizeofdata = (nodesNum - 1) / 32 + 1;
    ms.size = 256;
    cudaMalloc((void **)&ms.d_data, 256 * sizeof(uint32_t) * ms.sizeofdata);
    for (int i = 0; i < 256; i++) {
      cudaMemcpy(ms.d_data + i * ms.sizeofdata, h_symbol_sets_soa[i]->data,
                 sizeof(uint32_t) * ms.sizeofdata, cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < 256; i++) {
      delete h_symbol_sets_soa[i];
    }
    delete[] h_symbol_sets_soa;
  } else {
    ms.use_soa = false;
    ms.sizeofdata = 8;
    ms.size = nodesNum;
    cudaMalloc((void **)&ms.d_data,
               nodesNum * sizeof(uint32_t) * ms.sizeofdata);
    for (int i = 0; i < nodesNum; i++) {
      cudaMemcpy(ms.d_data + i * ms.sizeofdata, symbol_sets->get_host()[i].data,
                 sizeof(uint32_t) * ms.sizeofdata, cudaMemcpyHostToDevice);
    }
  }
  return ms;
}

cudaError_t Graph::release() {
  delete symbol_sets;
  delete node_attrs;
  delete edge_pairs;
  return cudaSuccess;
}

cudaError_t Graph::copyToDevice(){
  symbol_sets->copy_to_device();
  node_attrs->copy_to_device();
  edge_pairs->copy_to_device();
  always_active_nodes->copy_to_device();
  start_active_nodes->copy_to_device();
  return cudaSuccess;
}

/**
 * @brief Find log-scale degree histogram of the graph.
 */

int *GetHistogram(Csr &graph) {
  // cudaError_t retval = cudaSuccess;
  int *histogram = new int[16];
  memset(histogram, 0, sizeof(int) * 16);

  // Count
#pragma omp parallel for
  for (int v = 0; v < graph.nodesNum; v++) {
    auto num_neighbors = graph.GetNeighborListLength(v);
    int log_length = 0;
    while (num_neighbors >= (1 << log_length)) {
      log_length++;
    }
    assert(log_length < 16);
#pragma omp atomic
    *(histogram + log_length) += 1;
  }
  return histogram;
}

void PrintHistogram(Csr &graph) {
  std::cout << "Degree Histogram (" + std::to_string(graph.nodesNum) +
                   " vertices, " + std::to_string(graph.edgesNum) + " edges):"
            << std::endl;
  int *histogram = GetHistogram(graph);
  int max_log_length = 0;
  for (int i = 16 - 1; i >= 0; i--) {
    if (histogram[i] > 0) {
      max_log_length = i;
      break;
    }
  }
  for (int i = 0; i <= max_log_length; i++) {
    std::cout << "    Degree " +
                     (i == 0 ? "0" : ("2^" + std::to_string(i - 1))) + ": " +
                     std::to_string(histogram[i]) + " (" +
                     std::to_string(histogram[i] * 100.0 / graph.nodesNum) +
                     " %)"
              << std::endl;
  }
  delete[] histogram;
}