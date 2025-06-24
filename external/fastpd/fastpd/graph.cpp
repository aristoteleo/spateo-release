/* graph.cpp */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001. */

#include <stdio.h>
#include <iostream>
#include "graph.h"

Graph::Graph(node *nodes, arc *arcs, int num_nodes, void (*err_function)(char *))
{
  error_function = err_function;
  _nodes = nodes;
  _arcs  = arcs ;
  _num_nodes = num_nodes;
  flow = 0;
}

Graph::~Graph()
{
}

void Graph::add_nodes()
{
  for( int i = 0; i < _num_nodes; i++ )
  {
    _nodes[i].first = NULL;
    _nodes[i].tr_cap = 0;
#ifndef _METRIC_DISTANCE_
    _nodes[i].conflict_time = -1;
#endif
  }
}

void Graph::add_edges(int *pairs, int numpairs )
{
  captype cap = 0;
  captype rev_cap = 0;

  // numpairs iterations (note i+=2)
  for( int i = 0; i < 2*numpairs; i+=2 )
  {
    // retrieval of the node_id's (node*)
    node_id from = &_nodes[pairs[i]];
    node_id to   = &_nodes[pairs[i+1]];

    // there are always two arcs inserted into the graph
    // 'from -> to' as well as 'to -> from'
    arc *a, *a_rev;

    // retrieve the two arcs from the _arcs array
    // the array is indexed in the same way as pairs
    // arcs[i]   = node[pairs[i]] -> node[pairs[i+1]]
    // arcs[i+1] = node[pairs[i+1]] -> node[pairs[i]]
    a = &_arcs[i];
    a_rev = a + 1;

    // sister holds the reverse arcs
    a -> sister = a_rev;
    a_rev -> sister = a;

    // next points to the next outgoing arc/edge in from
    // since there is exactly one outgoing edge, it points
    // to itself!
    a -> next = ((node*)from) -> first; // i.e. a->next = a
    ((node*)from) -> first = a;

    // here holds the same as above
    a_rev -> next = ((node*)to) -> first;
    ((node*)to) -> first = a_rev;

    // head is pointing to the destination node
    a -> head = (node*)to;
    a_rev -> head = (node*)from;

    // and finally, we configure the capacities
    a -> r_cap = cap;
    a_rev -> r_cap = rev_cap;
  }
}

void Graph::set_tweights(node_id i, captype cap_source, captype cap_sink)
{
  flow += (cap_source < cap_sink) ? cap_source : cap_sink;
  ((node*)i) -> tr_cap = cap_source - cap_sink;
}

void Graph::add_tweights(node_id i, captype cap_source, captype cap_sink)
{
  register captype delta = ((node*)i) -> tr_cap;
  if (delta > 0) cap_source += delta;
  else           cap_sink   -= delta;
  flow += (cap_source < cap_sink) ? cap_source : cap_sink;
  ((node*)i) -> tr_cap = cap_source - cap_sink;
}

/***********************************************************************/
/* maxflow.cpp */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001. */
/***********************************************************************/

/*
   Returns the next active node.
   If it is connected to the sink, it stays in the list,
   otherwise it is removed from the list
   */
inline Graph::node * Graph::next_active()
{
  node *i;

  while ( 1 )
  {
    if (!(i=queue_first[0]))
    {
      queue_first[0] = i = queue_first[1];
      queue_last[0]  = queue_last[1];
      queue_first[1] = NULL;
      queue_last[1]  = NULL;
      if (!i) return NULL;
    }

    /* remove it from the active list */
    if (i->next == i) queue_first[0] = queue_last[0] = NULL;
    else              queue_first[0] = i -> next;
    i -> next = NULL;

    /* a node in the list is active iff it has a parent */
    if (i->parent) return i;
  }
}

/***********************************************************************/

void Graph::maxflow_init()
{
  node *i;

  queue_first[0] = queue_last[0] = NULL;
  queue_first[1] = queue_last[1] = NULL;
  orphan_first = NULL;

  int k;
  for( k = 0, i = _nodes; k < _num_nodes; k++, i++ )
  {
    i -> next = NULL;
    i -> TS = 0;
    if (i->tr_cap > 0)
    {
      /* i is connected to the source */
      i -> is_sink = 0;
      i -> parent = TERMINAL;
      set_active(i);
      i -> TS = 0;
      i -> DIST = 1;
    }
    else if (i->tr_cap < 0)
    {
      /* i is connected to the sink */
      i -> is_sink = 1;
      i -> parent = TERMINAL;
      i -> TS = 0;
      i -> DIST = 1;
    }
    else
    {
      i -> parent = NULL;
    }
  }
  TIME = 0;
}

/***********************************************************************/

void Graph::augment(arc *middle_arc)
{
  node *i;
  arc *a;
  captype bottleneck;
  nodeptr *np;


  /* 1. Finding bottleneck capacity */
  /* 1a - the source tree */
  bottleneck = middle_arc -> r_cap;
  for (i=middle_arc->sister->head; ; i=a->head)
  {
    a = i -> parent;
    if (a == TERMINAL) break;
    if (bottleneck > a->sister->r_cap) bottleneck = a -> sister -> r_cap;
  }
  if (bottleneck > i->tr_cap) bottleneck = i -> tr_cap;
  /* 1b - the sink tree */
  for (i=middle_arc->head; ; i=a->head)
  {
    a = i -> parent;
    if (a == TERMINAL) break;
    if (bottleneck > a->r_cap) bottleneck = a -> r_cap;
  }
  if (bottleneck > - i->tr_cap) bottleneck = - i -> tr_cap;


  /* 2. Augmenting */
  /* 2a - the source tree */
  middle_arc -> sister -> r_cap += bottleneck;
  middle_arc -> r_cap -= bottleneck;
  for (i=middle_arc->sister->head; ; i=a->head)
  {
    a = i -> parent;
    if (a == TERMINAL) break;
    a -> r_cap += bottleneck;
    a -> sister -> r_cap -= bottleneck;
    if (!a->sister->r_cap)
    {
      /* add i to the adoption list */
      i -> parent = ORPHAN;
      np = nodeptr_block -> New();
      np -> ptr = i;
      np -> next = orphan_first;
      orphan_first = np;
    }
  }
  i -> tr_cap -= bottleneck;
  if (!i->tr_cap)
  {
    /* add i to the adoption list */
    i -> parent = ORPHAN;
    np = nodeptr_block -> New();
    np -> ptr = i;
    np -> next = orphan_first;
    orphan_first = np;
  }
  /* 2b - the sink tree */
  for (i=middle_arc->head; ; i=a->head)
  {
    a = i -> parent;
    if (a == TERMINAL) break;
    a -> sister -> r_cap += bottleneck;
    a -> r_cap -= bottleneck;
    if (!a->r_cap)
    {
      /* add i to the adoption list */
      i -> parent = ORPHAN;
      np = nodeptr_block -> New();
      np -> ptr = i;
      np -> next = orphan_first;
      orphan_first = np;
    }
  }
  i -> tr_cap += bottleneck;
  if (!i->tr_cap)
  {
    i->parent = NULL;
  }

  flow += bottleneck;
}

/***********************************************************************/

void Graph::process_source_orphan(node *i)
{
  node *j;
  arc *a0, *a0_min = NULL, *a;
  nodeptr *np;
  int d, d_min = INFINITE_D;

  /* trying to find a new parent */
  for (a0=i->first; a0; a0=a0->next)
    if (a0->sister->r_cap)
    {
      j = a0 -> head;
      if (!j->is_sink && (a=j->parent))
      {
        /* checking the origin of j */
        d = 0;
        while ( 1 )
        {
          if (j->TS == TIME)
          {
            d += j -> DIST;
            break;
          }
          a = j -> parent;
          d ++;
          if (a==TERMINAL)
          {
            j -> TS = TIME;
            j -> DIST = 1;
            break;
          }
          if (a==ORPHAN) { d = INFINITE_D; break; }
          j = a -> head;
        }
        if (d<INFINITE_D) /* j originates from the source - done */
        {
          if (d<d_min)
          {
            a0_min = a0;
            d_min = d;
          }
          /* set marks along the path */
          for (j=a0->head; j->TS!=TIME; j=j->parent->head)
          {
            j -> TS = TIME;
            j -> DIST = d --;
          }
        }
      }
    }

  if ((i->parent = a0_min))
  {
    i -> TS = TIME;
    i -> DIST = d_min + 1;
  }
  else
  {
    /* no parent is found */
    i -> TS = 0;

    /* process neighbors */
    for (a0=i->first; a0; a0=a0->next)
    {
      j = a0 -> head;
      if (!j->is_sink && (a=j->parent))
      {
        if (a0->sister->r_cap) set_active(j);
        if (a!=TERMINAL && a!=ORPHAN && a->head==i)
        {
          /* add j to the adoption list */
          j -> parent = ORPHAN;
          np = nodeptr_block -> New();
          np -> ptr = j;
          if (orphan_last) orphan_last -> next = np;
          else             orphan_first        = np;
          orphan_last = np;
          np -> next = NULL;
        }
      }
    }
  }
}

void Graph::process_sink_orphan(node *i)
{
  node *j;
  arc *a0, *a0_min = NULL, *a;
  nodeptr *np;
  int d, d_min = INFINITE_D;

  /* trying to find a new parent */
  for (a0=i->first; a0; a0=a0->next)
    if (a0->r_cap)
    {
      j = a0 -> head;
      if (j->is_sink && (a=j->parent))
      {
        /* checking the origin of j */
        d = 0;
        while ( 1 )
        {
          if (j->TS == TIME)
          {
            d += j -> DIST;
            break;
          }
          a = j -> parent;
          d ++;
          if (a==TERMINAL)
          {
            j -> TS = TIME;
            j -> DIST = 1;
            break;
          }
          if (a==ORPHAN) { d = INFINITE_D; break; }
          j = a -> head;
        }
        if (d<INFINITE_D) /* j originates from the sink - done */
        {
          if (d<d_min)
          {
            a0_min = a0;
            d_min = d;
          }
          /* set marks along the path */
          for (j=a0->head; j->TS!=TIME; j=j->parent->head)
          {
            j -> TS = TIME;
            j -> DIST = d --;
          }
        }
      }
    }

  if ((i->parent = a0_min))
  {
    i -> TS = TIME;
    i -> DIST = d_min + 1;
  }
  else
  {
    /* no parent is found */
    i -> TS = 0;

    /* process neighbors */
    for (a0=i->first; a0; a0=a0->next)
    {
      j = a0 -> head;
      if (j->is_sink && (a=j->parent))
      {
        if (a0->r_cap) set_active(j);
        if (a!=TERMINAL && a!=ORPHAN && a->head==i)
        {
          /* add j to the adoption list */
          j -> parent = ORPHAN;
          np = nodeptr_block -> New();
          np -> ptr = j;
          if (orphan_last) orphan_last -> next = np;
          else             orphan_first        = np;
          orphan_last = np;
          np -> next = NULL;
        }
      }
    }
  }
}

/***********************************************************************/

Graph::flowtype Graph::apply_maxflow( int init_on )
{
  node *i, *j, *current_node = NULL;
  arc *a;
  nodeptr *np, *np_next;

  if ( init_on )
    maxflow_init();
  nodeptr_block = new DBlock<nodeptr>(NODEPTR_BLOCK_SIZE, error_function);

  //int num_iter = 0;					//ADDED BY BEN
  //while ( num_iter < 10000 )		//ADDED BY BEN

  while ( 1 )
  {
    //num_iter++;					//ADDED BY BEN

    if ((i=current_node))
    {
      i -> next = NULL; /* remove active flag */
      if (!i->parent) i = NULL;
    }
    if (!i)
    {
      if (!(i = next_active())) break;
    }

    /* growth */
    if (!i->is_sink)
    {
      /* grow source tree */
      for (a=i->first; a; a=a->next)
        if (a->r_cap)
        {
          j = a -> head;
          if (!j->parent)
          {
            j -> is_sink = 0;
            j -> parent = a -> sister;
            j -> TS = i -> TS;
            j -> DIST = i -> DIST + 1;
            set_active(j);
          }
          else if (j->is_sink) break;
          else if (j->TS <= i->TS &&
                   j->DIST > i->DIST)
          {
            /* heuristic - trying to make the distance from j to the source shorter */
            j -> parent = a -> sister;
            j -> TS = i -> TS;
            j -> DIST = i -> DIST + 1;
          }
        }
    }
    else a = NULL;

    TIME ++;

    if (a)
    {
      i -> next = i; /* set active flag */
      current_node = i;

      /* augmentation */
      augment(a);
      /* augmentation end */

      /* adoption */
      while ((np=orphan_first))
      {
        np_next = np -> next;
        np -> next = NULL;

        while ((np=orphan_first))
        {
          orphan_first = np -> next;
          i = np -> ptr;
          nodeptr_block -> Delete(np);
          if (!orphan_first) orphan_last = NULL;
          if (i->is_sink) process_sink_orphan(i);
          else            process_source_orphan(i);
        }

        orphan_first = np_next;
      }
      /* adoption end */
    }
    else current_node = NULL;
  }

  delete nodeptr_block;

  return flow;
}

/***********************************************************************/

Graph::termtype Graph::what_segment(node_id i)
{
  if (((node*)i)->parent && !((node*)i)->is_sink) return SOURCE;
  return SINK;
}
