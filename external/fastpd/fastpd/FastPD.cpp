#include <cstdio>
#include <iostream>
#include "FastPD.h"

namespace fastpd {

FastPD::FastPD( int num_nodes, int num_labels, int num_pairs, int* vec_pairs, float* unaryCosts, int max_iterations, bool copy_unary) :
    m_initial_energy(0),
    m_display(NULL),
    m_displayinst(NULL),
    m_num_nodes(num_nodes),
    m_num_labels(num_labels),
    m_num_pairs(num_pairs),
    m_max_iterations(max_iterations),
    pairs(vec_pairs)
  {
    m_time                = -1;
    m_energy_change_time  = -1;

    if ( m_num_labels >= pow(256.0f,float(sizeof(Graph::Label))) )
    {
      std::cout << "Change Graph::Label type (it is too small to hold all labels)" << std::endl;
      assert(0);
    }

    children          = new Graph::node *[m_num_nodes];
    source_nodes_tmp1 = new int[m_num_nodes];
    source_nodes_tmp2 = new int[m_num_nodes];
    for (int i = 0; i < m_num_nodes; i++)
    {
      source_nodes_tmp1[i] = -2;
      source_nodes_tmp2[i] = -2;
    }

    // for each pair we will be adding two edges
    const int num_edges = 2*m_num_pairs;

    // initialization of graph topology
    graph_nodes = new Graph::node[m_num_nodes*m_num_labels];
    graph_edges  = new Graph::arc[  num_edges*m_num_labels];

    // create for each label a single graph
    graphs = new Graph *[m_num_labels];
    for (int i = 0; i < m_num_labels; ++i)
    {
      Graph::node* current_graph_nodes = &graph_nodes[m_num_nodes*i];
      Graph::arc*  current_graph_edges = &graph_edges[ num_edges*i];
      graphs[i] = new Graph(current_graph_nodes, current_graph_edges, m_num_nodes, err_fun);
      fillGraph( graphs[i] );
    }

    // initialize node, edge, and pair info structures
    m_active_list = -1;
    node_info = new Node_info[m_num_nodes];
    createNeighbors();

    // initialize height variables
    if(copy_unary)
    {
      height = new Real[m_num_labels*m_num_nodes];
      m_delete_height = true;
    }
    else
    {
      height = unaryCosts;
      m_delete_height = false;
    }

    // initialize balance variables
    balance = new Real[m_num_pairs*m_num_labels];
  }


FastPD::~FastPD()
{
  delete[] graph_nodes;
  delete[] graph_edges;

  Graph::Label i;
  for( i = 0; i < m_num_labels; i++ )
    delete graphs[i];
  delete[] graphs;

  delete[] node_info;
  delete[] edge_info;
  delete[] pair_info;
  delete[] pairs_arr;

  delete[] balance;
  if(m_delete_height) delete[] height;

  delete[] source_nodes_tmp1;
  delete[] source_nodes_tmp2;
  delete[] children;
}


void FastPD::getLabeling(int *labeling)
{
  for(int p = 0; p < m_num_nodes; ++p)
    labeling[p] = node_info[p].label;
}

void FastPD::fillGraph( Graph *_graph )
{
  // does not really add the nodes but sets for
  // each node the outgoing arc/edges to NULL
  // and its capacity to zero
  _graph->add_nodes();

  // adds for each pair two directed edges/arcs
  // between the participating nodes
  _graph->add_edges( pairs, m_num_pairs );
}


void FastPD::createNeighbors()
{
  // Fill auxiliary structures related to neighbors
  pairs_arr = new int[m_num_pairs*2];

  for(int i = 0; i < m_num_nodes; i++ )
    node_info[i].numpairs = 0;

  for(int i = 0; i < m_num_pairs; i++ )
  {
    int i0 = pairs[i<<1];
    int i1 = pairs[(i<<1)+1];
    node_info[i0].numpairs++;
    node_info[i1].numpairs++;
  }

  int offset = 0;
  for(int i = 0; i < m_num_nodes; i++ )
  {
    node_info[i].pairs = &pairs_arr[offset];
    offset += node_info[i].numpairs;
    node_info[i].numpairs = 0;
  }

  pair_info = new Pair_info[m_num_pairs];
  edge_info = new Arc_info[m_num_pairs];

  for(int i = 0; i < m_num_pairs; i++ )
  {
    int i0 = pairs[i<<1];
    int i1 = pairs[(i<<1)+1];
    node_info[i0].pairs[node_info[i0].numpairs++] =  i;
    node_info[i1].pairs[node_info[i1].numpairs++] = -i;

    edge_info[i].tail = i0;
    edge_info[i].head = i1;

    pair_info[i].i0 = i0;
    pair_info[i].i1 = i1;
    pair_info[i].time = -1;
  }
}

}
