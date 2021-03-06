#define VERSION 1
#define SEED_NUM 123
/*
Common cpp libraries
*/
#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <random>
#include <bits/stdc++.h>

/*
Vector operations
*/
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

/*
GPU functions
*/


using namespace std;
/*==============================================================================
CSR graph class
==============================================================================*/

// class definition
template <typename ValueT, typename SizeT>
class CSR
{
public:
  CSR(const char *);
  ~CSR() {
    cudaFree(csr);
    cudaFree(offset);
    cudaFree(colors);
    cudaFree(rand);
  };
//  SizeT &operator[] (SizeT); DEPRICATED
//  void print_adj();
  void print_tup();
  void print_arrays();
  void check_conflict();
  unsigned int edges;
  unsigned int vertices;
  SizeT* csr;
  SizeT* offset;
  SizeT* colors;
  ValueT* rand;
private:
  vector <tuple<SizeT,SizeT>> coo;
//  SizeT* adj_matrix; DEPRICATED
};

// class outline (some functions taken from EEC289Q)

// constructor
template <typename ValueT, typename SizeT>
CSR<ValueT,SizeT>::CSR(const char filename[]) {
  string line;
  ifstream infile(filename);
  if (infile.fail()) {
    cout << "ERROR:  failed to open file" << endl;
    return;
  }

  cout << "Making adj matrix ..." << endl;

  while (getline(infile, line)) {
    istringstream iss(line);
    if (line.find("%") == string::npos)
      break;
  }

  istringstream iss(line);
  SizeT num_rows, num_cols, num_edges;
  iss >> num_rows >> num_cols >> num_edges;

  this->vertices = num_rows;
  this->edges = num_edges;

  cout << "Number of vertices: " << num_rows << endl;
  cout << "Number of directed edges: " << num_edges << endl;


  /* vvvvvvvvv DEPRICATED CODE, ALLOC ERROR WHEN GRAPH IS LARGE vvvvvvvvvvvvvvvv
  // this->adj_matrix = new SizeT[num_rows * num_rows];
  // memset(this->adj_matrix, 0, num_rows * num_rows * sizeof(bool));
  //
  // while (getline(infile, line)) {
  //   istringstream iss(line);
  //   SizeT node1, node2, weight;
  //   iss >> node1 >> node2 >> weight;
  //
  //   this->adj_matrix[(node1 - 1) * num_rows + (node2 - 1)] = true;
  //   this->adj_matrix[(node2 - 1) * num_rows + (node1 - 1)] = true;
  // }
  // infile.close();
  //
  // cout << "Making csr and offset ..." << endl;
  //
  // // declare csr and offset
  // int csr_length = thrust::reduce(thrust::host,
  //   this->adj_matrix, this->adj_matrix + this->vertices * this->vertices);

  cudaMallocManaged(&(this->csr), this->edges * sizeof(SizeT));
  cudaMallocManaged(&(this->offset), this->vertices * sizeof(SizeT));
  // this->csr = new SizeT[this->edges ];
  // this->offset = new SizeT[this->vertices];

  // populate csr and offset
  int count = 0;
  for (SizeT v = 0 ; v < this->vertices; v++) {
    this->offset[v] = thrust::reduce(thrust::host,
      this->adj_matrix + (v * this->vertices),
      this->adj_matrix + ((v + 1) * this->vertices) );
    for (SizeT adj = 0; adj < this->vertices; adj++) {
      if (this->adj_matrix[v * this->vertices + adj]) {
        this->csr[count] = adj;
        count++;
      }
    }
  }
  thrust::exclusive_scan(thrust::host, this->offset,
    this->offset + this->vertices, this->offset);
 ^^^^^^^^^^^^ DEPRICATED CODE, ALLOC ERROR WHEN GRAPH IS LARGE ^^^^^^^^^^^^^^^*/

 cout << "Making coo ..." << endl;

 while (getline(infile, line)) {
   istringstream iss(line);
   SizeT node1, node2, weight;
   iss >> node1 >> node2 >> weight;
   this->coo.push_back(make_tuple(node1 - 1, node2 - 1));
   this->coo.push_back(make_tuple(node2 - 1, node1 - 1));
 }
 infile.close();

 // erase redundant nodes
 sort(this->coo.begin(), this->coo.end());
 this->coo.erase(unique(this->coo.begin(), this->coo.end()), this->coo.end());

 cout << "Making csr ..." << endl;

 cudaMallocManaged(&(this->csr), this->coo.size() * sizeof(SizeT));
 cudaMallocManaged(&(this->offset), this->vertices * sizeof(SizeT));

 for (int i = 0; i < this->coo.size(); i++) {
   this->csr[i] = get<1>(this->coo[i]);
 }

 for (SizeT v = 0; v < this->vertices; v++) {
   int count = 0;
   for (int i = 0; i < this->coo.size(); i++) {
     if (get<0>(this->coo[i]) == v)
      count ++;
   }
   this->offset[v] = count;
 }

  cout << "Making rand ..." << endl;
  // create rand array for IS
  cudaMallocManaged(&(this->rand), this->vertices * sizeof(ValueT));
  // this->rand = new ValueT[this->vertices];
  random_device rd;
  mt19937 e2(rd());
  e2.seed(SEED_NUM);
  uniform_real_distribution<> dist(0,100);
  for (int v = 0; v < this->vertices; v++) {
    this->rand[v] = dist(e2);
  }
  thrust::exclusive_scan(thrust::host, this->offset,
    this->offset + this->vertices, this->offset);

  cout << "Making colors ..." << endl;
  // allocate memory for colors
  cudaMallocManaged(&(this->colors), this->vertices * sizeof(SizeT));
  // this->colors = new SizeT[this->vertices];
  memset(this->colors, -1, this->vertices * sizeof(SizeT));
};


// index overload DEPRICATED
// template <typename ValueT, typename SizeT>
// SizeT & CSR<ValueT,SizeT>::operator[](SizeT idx) {
//    return this->adj_matrix[idx];
// };

// print first 20 x 20 entries for adj matrix  DEPRICATED
// template <typename ValueT, typename SizeT>
// void CSR<ValueT, SizeT>::print_adj() {
//   SizeT max_idx = 20;
//   if(this->vertices < 20)
//     max_idx = this->vertices;
//   for (int i = 0; i < max_idx; i++) {
//     cout << i << " : [";
//     for (int j = 0; j < max_idx; j++) {
//       cout << this->adj_matrix[i * this->vertices + j] << ", ";
//     }
//     cout << "]" << endl;
//   }
// };

// print first 20 tuples
template <typename ValueT, typename SizeT>
void CSR<ValueT, SizeT>::print_tup() {
    SizeT max_idx = 20;
    if(this->vertices < 20)
      max_idx = this->vertices;
    for (int i = 0; i < max_idx; i++) {
      cout << "(" << get<0>(this->coo[i]) << ", " << get<1>(this->coo[i]) << ")" <<endl;
    }
};

// print first 20 entries for offset and csr
template <typename ValueT, typename SizeT>
void CSR<ValueT, SizeT>::print_arrays() {
  SizeT max_idx = 20;
  if(this->vertices < 20)
    max_idx = this->vertices;
    cout << "CSR: [";
    for (int i = 0; i < max_idx; i++) {
      cout << this->csr[i] << ", ";
    }
    cout << "]" << endl;

    cout << "OFFSET: [";
    for (int i = 0; i < max_idx; i++) {
      cout << this->offset[i] << ", ";
    }
    cout << "]" << endl;

    cout << "COLORS: [";
    for (int i = 0; i < max_idx; i++) {
      cout << this->colors[i] << ", ";
    }
    cout << "]" << endl;

    cout << "RAND: [";
    for (int i = 0; i < max_idx; i++) {
      cout << this->rand[i] << ", ";
    }
    cout << "]" << endl;
};

/*==============================================================================
Check for color conflict
==============================================================================*/
template <typename ValueT, typename SizeT>
void CSR<ValueT, SizeT>::check_conflict() {
  for (SizeT v = 0; v < this->vertices; v++) {
    SizeT start_edge = offset[v];
    SizeT num_neighbors = offset[v + 1] -  offset[v];
    for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
      SizeT u = csr[e];
      if ((this->colors[v] == this->colors[u]) && (u != v)) {
        cout << "ERROR: Conflict at node " << v << "and node " << u
        << " at color" << colors[v] << endl;
      }
    }
  }
}

/*==============================================================================
IS color operation - outline taken from Gunrock jpl_color_op
==============================================================================*/
template <typename ValueT, typename SizeT>
#if defined(VERSION) && VERSION == 1
__global__
#else
__device__
#endif
void color_op(SizeT* csr, SizeT* offset, ValueT* rand,
              SizeT* colors, int num_vertices, int iteration) {
  unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < num_vertices) {
    if (colors[v] != -1) return;

    SizeT start_edge = offset[v];
    SizeT num_neighbors = offset[v + 1] -  offset[v];

    bool colormax = true;
    bool colormin = true;
    int color = iteration * 2;

    for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
      SizeT u = csr[e];

      if ((colors[u] != -1) && (colors[u] != color + 1) &&
              (colors[u] != color + 2) ||
          (v == u))
        continue;
      if (rand[v] <= rand[u]) colormax = false;
      if (rand[v] >= rand[u]) colormin = false;
    }

    if (colormax) colors[v] = color + 1;
    if (colormin) colors[v] = color + 2;
  }
};

/*==============================================================================
IS color stop condition
==============================================================================*/
template <typename ValueT, typename SizeT>
__host__ __device__
bool stop_condition(SizeT* colors, unsigned int num_vertices) {
#if defined(VERSION) && VERSION == 1
  for (int v = 0; v < num_vertices; v++) {
    if (colors[v] == -1)
      return true;
  }
  return false;
#else
  return false;
#endif
}

/*==============================================================================
IS Kernel function
==============================================================================*/
// template <typename ValueT, typename SizeT>
// __global__
// void ISKernel(SizeT csr, SizeT offset, ValueT rand, SizeT colors, int num_vertices) {
//   int iteration = 0;
//   while (stop_condition(colors)) {
//     color_op(csr, offset, rand, colors, num_vertices, iteration);
//     // TODO: grid wise synchronization
//   }
// };

/*==============================================================================
IS Kernel Driver
==============================================================================*/
// template <typename ValueT, typename SizeT>
// void ISKernelDriver(CSR<ValueT, SizeT> graph) {
//   unsigned int num_threads = 32;
//   unsigned int num_blocks = graph.vertices / num_threads + 1;
//   ISKernel<ValueT, SizeT><<<num_blocks, num_threads>>>
//   (graph.csr,
//    graph.offset,
//    graph.rand,
//    graph.colors,
//    graph.vertices);
// }

/*==============================================================================
Tester - version 1
==============================================================================*/
template <typename ValueT, typename SizeT>
void test_1(CSR <float, int> graph) {

  int iteration = 0;
  unsigned int num_threads = 32;
  unsigned int num_blocks = graph.vertices / num_threads + 1;

  cout << "Kernel loop start" << endl;

  while (stop_condition<float, int>(graph.colors, graph.vertices)) {
      color_op<float, int><<<num_blocks, num_threads>>>
      (graph.csr,
       graph.offset,
       graph.rand,
       graph.colors,
       graph.vertices,
       iteration);
       cudaDeviceSynchronize();
       iteration ++;
  }

  // graph.print_adj(); DEPRICATED
  cout << "==== Graph Samples: ====" <<endl;
  graph.print_tup();
  graph.print_arrays();
  graph.check_conflict();
};

/*==============================================================================
Main function
==============================================================================*/

int main(int argc, char const *argv[]) {
#if defined(VERSION) && VERSION == 1
  // cout << "Test small graph" << endl;
  // CSR <float, int> * graph = new CSR<float, int>("../gunrock/dataset/small/test_cc.mtx");
  // test_1 <float, int> (*graph);

  cout << "Test large graph" << endl;
  CSR <float, int> * graph = new CSR<float, int>("/data-2/topc-datasets/gc-data/offshore/offshore.mtx");
  test_1 <float, int> (*graph);

#endif
  return 0;
}

