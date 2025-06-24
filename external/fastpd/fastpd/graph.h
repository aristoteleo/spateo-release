//#############################################################################
//#
//# FastPD Optimization (c) 2008.
//#
//# Max-flow Computation.
//#
//# Original Author: Vladimir Kolmogorov
//# Modification: Nikos Komodakis
//#
//# This is an implementation of the modified max-flow computation described in:
//# N. Komodakis, G. Tziritas, N. Paragios,
//# Fast, Approximately Optimal Solutions for Single and Dynamic MRFs,
//# IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2007.
//#
//# THE WORK IS ONLY FOR RESEARCH AND NON-COMMERCIAL PURPOSES. THE OPTIMIZATION
//# CODE IS PROTECTED FROM SEVERAL US/EU/CHINA PENDING PATENT APPLICATIONS.
//# IF YOU WOULD LIKE TO USE THIS SOFTWARE FOR COMMERCIAL PURPOSES OR LICENSING
//# THE TECHNOLOGY, PLEASE CONTACT:
//# ECOLE CENTRALE DE PARIS, PROF. NIKOS PARAGIOS (nikos.paragios@ecp.fr).
//#
//# If you intend to use this code or results obtained with it, the above
//# mentioned paper should be cited within your publication.
//#
//#############################################################################

//#############################################################################
//# Header for the original max-flow computation by Yuri Boykov and
//# Vladimir Kolmogorov.
//#############################################################################
/* graph.h */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001. */

/*
   This software library is a modification of the maxflow algorithm
   described in

   An Experimental Comparison of Min-Cut/Max-Flow Algorithms
   for Energy Minimization in Computer Vision.
   Yuri Boykov and Vladimir Kolmogorov.
   In Third International Workshop on Energy Minimization
   Methods in Computer Vision and Pattern Recognition, September 2001

   This algorithm was originally developed at Siemens.
   The main modification is that two trees are used for finding
   augmenting paths - one grows from the source and the other
   from the sink. (The original algorithm used only the former one).
   Details will be described in my PhD thesis.

   This implementation uses an adjacency list graph representation.
   Memory allocation:
Nodes: 22 bytes + one field to hold a residual capacity
of t-links (by default it is 'short' - 2 bytes)
Arcs: 12 bytes + one field to hold a residual capacity
(by default it is 'short' - 2 bytes)
(Note that arcs are always added in pairs - in forward and reverse directions)

Example usage (computes a maxflow on the following graph):

SOURCE
/       \
1/         \2
/      3    \
node0 -----> node1
|   <-----   |
|      4     |
\            /
5\          /6
\        /
SINK

///////////////////////////////////////////////////

#include <stdio.h>
#include "graph.h"

void main()
{
Graph::node_id nodes[2];
Graph *g = new Graph();

nodes[0] = g -> add_node();
nodes[1] = g -> add_node();
g -> set_tweights(nodes[0], 1, 5);
g -> set_tweights(nodes[1], 2, 6);
g -> add_edge(nodes[0], nodes[1], 3, 4);

Graph::flowtype flow = g -> maxflow();

printf("Flow = %d\n", flow);
printf("Minimum cut:\n");
if (g->what_segment(nodes[0]) == Graph::SOURCE)
printf("node0 is in the SOURCE set\n");
else
printf("node0 is in the SINK set\n");
if (g->what_segment(nodes[1]) == Graph::SOURCE)
printf("node1 is in the SOURCE set\n");
else
printf("node1 is in the SINK set\n");

delete g;
}

///////////////////////////////////////////////////
*/

#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "block.h"

/*
   Nodes, arcs and pointers to nodes are
   added in blocks for memory and time efficiency.
   Below are numbers of items in blocks
   */
#define NODE_BLOCK_SIZE 512
#define ARC_BLOCK_SIZE 1024
#define NODEPTR_BLOCK_SIZE 128

/*
   special constants for node->parent
   */
#define TERMINAL ( (Graph::arc *) 1 )		/* to terminal */
#define ORPHAN   ( (Graph::arc *) 2 )		/* orphan */

#define INFINITE_D 1000000000		/* infinite distance to the terminal */

class Graph
{
 public:

#define _MANY_LABELS_
#ifdef _MANY_LABELS_
  typedef int Label;
#else
  typedef unsigned char Label;
#endif

  typedef enum
  {
    SOURCE	= 0,
    SINK	= 1
  } termtype; /* terminals */

  /* Type of edge weights.
     Can be changed to char, int, float, double, ... */
  typedef float captype;  
  typedef captype Real;

  /* Type of total flow */
  typedef float flowtype;

  typedef void * node_id;

  /* interface functions */

  /* Destructor */
  ~Graph();

  /* Adds a node to the graph */
  void add_nodes();

  /* Adds a bidirectional edge between 'from' and 'to'
     with the weights 'cap' and 'rev_cap' */
  void add_edges( int *pairs, int numpairs );

  /* Sets the weights of the edges 'SOURCE->i' and 'i->SINK'
     Can be called at most once for each node before any call to 'add_tweights'.
     Weights can be negative */
  void set_tweights(node_id i, captype cap_source, captype cap_sink);

  /* Adds new edges 'SOURCE->i' and 'i->SINK' with corresponding weights
     Can be called multiple times for each node.
     Weights can be negative */
  void add_tweights(node_id i, captype cap_source, captype cap_sink);

  /* After the maxflow is computed, this function returns to which
     segment the node 'i' belongs (Graph::SOURCE or Graph::SINK) */
  termtype what_segment(node_id i);

  /* Computes the maxflow. Can be called only once. */
  flowtype apply_maxflow( int track_source_nodes );

  /***********************************************************************/
  /***********************************************************************/
  /***********************************************************************/

  /* internal variables and functions */

  struct arc_st;

  /* node structure */
  typedef struct node_st
  {
    arc_st			*first;		/* first outcoming arc */

    arc_st			*parent;	/* node's parent */
    node_st			*next;		/* pointer to the next active node
                             (or to itself if it is the last node in the list) */
    int				TS;			/* timestamp showing when DIST was computed */
    int				DIST;		/* distance to the terminal */
    short			is_sink;	/* flag showing whether the node is in the source or in the sink tree */

    captype			tr_cap;		/* if tr_cap > 0 then tr_cap is residual capacity of the arc SOURCE->node
                             otherwise         -tr_cap is residual capacity of the arc node->SINK */
#ifndef _METRIC_DISTANCE_
    short            conflict_time;
#endif
  } node;

  /* arc structure */
  typedef struct arc_st           // arc pq
  {
    node_st			*head;		/* node q, i.e. node the arc points to */
    arc_st			*next;		/* next arc with the same originating node */
    arc_st			*sister;	/* arc qp, i.e. reverse arc */

    captype			r_cap;		/* residual capacity */

    Real            cap;        // cap_{pq}   
  } arc;

  /* 'pointer to node' structure */
  typedef struct nodeptr_st
  {
    node_st			*ptr;
    nodeptr_st		*next;
  } nodeptr;

  Block<node>			*node_block;
  Block<arc>			*arc_block;
  DBlock<nodeptr>		*nodeptr_block;

  void	(*error_function)(char *);	/* this function is called if a error occurs,
                                       with a corresponding error message
                                       (or exit(1) is called if it's NULL) */

  flowtype			flow;		    /* total flow */

  /* Constructor. Optional argument is the pointer to the
     function which will be called if an error occurs;
     an error message is passed to this function. If this
     argument is omitted, exit(1) will be called. */
  Graph(node *nodes, arc *arcs, int num_nodes, void (*err_function)(char *) = NULL);

  void reset_flow( void )
  {
    flow = 0;
  }

  arc                *_arcs;
  node               *_nodes;
  int                 _num_nodes;

  //private:

  /***********************************************************************/

  node				*queue_first[2], *queue_last[2];	/* list of active nodes */
  nodeptr				*orphan_first, *orphan_last;		/* list of pointers to orphans */
  int					TIME;								/* monotonically increasing global counter */

  /***********************************************************************/

  /*
     Functions for processing active list.
     i->next points to the next node in the list
     (or to i, if i is the last node in the list).
     If i->next is NULL iff i is not in the list.

     There are two queues. Active nodes are added
     to the end of the second queue and read from
     the front of the first queue. If the first queue
     is empty, it is replaced by the second queue
     (and the second queue becomes empty).
     */
  inline void set_active(node *i)
  {
    if (!i->next)
    {
      /* it's not in the list yet */
      if (queue_last[1]) queue_last[1] -> next = i;
      else               queue_first[1]        = i;
      queue_last[1] = i;
      i -> next = i;
    }
  }
  node *next_active();

  void maxflow_init();
  void augment(arc *middle_arc);
  void process_source_orphan(node *i);
  void process_sink_orphan(node *i);
};

#endif
