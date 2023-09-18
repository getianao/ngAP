#ifndef GRAPH_HELPER_H
#define GRAPH_HELPER_H


#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <list>

using std::map;
using std::vector;
using std::fill;
using std::cout;
using std::endl;



class DAG { // for topological sort.
private:
	int V;
	std::map<int, std::list<int> > adj;
	std::map<int, int> in_degree;
	bool **gg;

	std::vector<std::vector<int> > topo_order;
	bool *flag;

public:
	DAG (int V);
	~DAG();
	void addEdge(int v, int w);
	void topological_sort();
	std::vector<std::vector<int> > &get_topo_order();

};


// A class that represents an directed graph
class MyGraph
{
    int V;    // No. of vertices
    std::map<int, std::list<int> > adj;    // A dynamic array of adjacency lists
    std::map<int, int> scc;
    int time_, *pre, *low, *stk, tops, sccN;
    void tarjan(int s);

    //bool dag;


    bool *visited;
    int *bfs_layer;

public:
    MyGraph(int V);   // Constructor
    ~MyGraph();
    void addEdge(int v, int w);   // function to add an edge to graph

    void calc_SCC();
    int get_n_scc();

    void bfs();
    
    //bool is_dag() const {
     //   return dag;
    //}

    std::map<int, int> get_scc();

    const int *get_bfs_layers() {
        return bfs_layer;
    } 

    void print_SCC();

};


#endif