#include "graph_helper.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <cassert>
#include <cstring>

#include <queue>

using std::queue;
using std::map;
using std::vector;
using std::fill;
using std::cout;
using std::endl;


DAG::DAG(int V) {
	this->V = V;

	gg = new bool*[V];
	for (int i = 0; i < V; i++) {
		gg[i] = new bool[V];
	}

	for (int i = 0; i < V; i++) {
		for (int j = 0; j < V; j++) {
			gg[i][j] = false;
		}
	}

	for (int i = 0; i < V; i++) {
		in_degree[i] = 0;
	}

	flag = new bool[V];
	std::fill(flag, flag+V, false);
}


void DAG::addEdge(int v, int w) {
	if (v >= 0 && v < V && w >=0 && w < V) {
		if (!gg[v][w]) {
			adj[v].push_back(w);
			gg[v][w] = true;
			in_degree[w] += 1;
		}
	} else {
		cout << "out of range DAGGGGG" << endl;
	}
}


void DAG::topological_sort() {
	int n = 0;
	topo_order.clear();
	//cout << "topo" << endl;
	while (true) {
		vector<int> zeroind;
		for (int i = 0; i < V; i++) {
			if (in_degree[i] == 0 && !flag[i]) {
				zeroind.push_back(i);
				flag[i] = true;
				n++;
			}
		}

		if (zeroind.size() == 0) {
			if (n != V) {
				for (int j = 0; j < V; j++) {
					if (!flag[j]) {
						cout << j << "  ";
					}
				}
				cout << "not DAG" << endl;
			}

			break;
		} else {
			/*for (auto it : zeroind) {
				cout << it << "  ";
			}*/
			//cout << endl;
			topo_order.push_back(zeroind);
		}

		for (auto node : zeroind) {
			for (auto to : adj[node]) {
				if (!flag[to]) {
					in_degree[to] -= 1;
					assert(in_degree[to] >= 0);
				}
			}
		}
	}
	//cout << "gege" << endl;
}


DAG::~DAG() {
	for (int i = 0; i < V; i++) {
		delete[] gg[i];
	}

	delete[] gg;
	delete[] flag;
}


std::vector<std::vector<int> > &DAG::get_topo_order() {
	return topo_order;
}





MyGraph::MyGraph(int V)
{
    this->V = V;
    pre = new int[V];
    low = new int[V];
    stk = new int[V];

    visited = new bool[V];
    bfs_layer = new int[V];

    time_ = -1;
    tops = -1;
    sccN = -1;

    adj.clear();

    

}

MyGraph::~MyGraph() {
	delete[] pre;
	delete[] low;
	delete[] stk;
	delete[] visited;
	delete[] bfs_layer;
}


void MyGraph::bfs() {
	memset(visited, false, sizeof(bool) * V);
	memset(bfs_layer, 0, sizeof(int) * V);

	queue< std::pair<int, int>  > q;
	for (int i = 0; i < V; i++) {
		if (!visited[i]) {
			q = queue< std::pair<int, int>  > ();
			q.push(std::make_pair(i, 0)  );

			while (!q.empty()) {
				auto head = q.front();
				q.pop();
				auto adj0 = adj[head.first];
				bfs_layer[head.first] = head.second;

				for (auto it : adj0) {
					if (!visited[it]) {
						visited[it] = true;
						q.push(std::make_pair(it, head.second+1));
					}
				} 
			}
		}
	}
}
    



void MyGraph::addEdge(int v, int w)
{
	if (v >= 0 && v < V && w >=0 && w < V) {
		adj[v].push_back(w);
	} else {
		cout << "out of range GGGGG" << endl;
	}

}

void MyGraph::calc_SCC() {
	sccN = -1;
	tops = 0;
	time_ = 100;

	for (int i = 0; i < V; i++) {
		scc[i] = -3;
		pre[i] = 0;
	}


	for (int i = 0; i < V; i++) {
		//std::cout << i << std::endl;
		if (pre[i] == 0) {
			tops = 0;
			tarjan(i);
		}
	}
}

void MyGraph::tarjan(int s) {
	//cout << "s = " << s << endl;
	pre[s] = low[s] = ++time_;
	int minlow = low[s];
	stk[tops++] = s;
	auto adj0 = adj[s];
	if (adj0.size() > 0) {
		for (auto it : adj0) {
			if (pre[it] == 0) {
				tarjan(it);
			} 
			minlow = std::min(minlow, low[it]);
		}
	}

	if (minlow < low[s]) {
		low[s] = minlow;
		return;
	}

	++sccN;
	do {
		scc[stk[--tops]] = sccN;
		low[stk[tops]] = 0x7fffffff;
	} while (stk[tops] != s);
}

std::map<int, int> MyGraph::get_scc() {
	return scc;
}

int MyGraph::get_n_scc() {
	return sccN + 1;
}

void MyGraph::print_SCC() {
	for (auto it : scc) {
		cout << it.first << " in scc " << it.second << endl;
	}
}





