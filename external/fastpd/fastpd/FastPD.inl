
//
//#define _FASTPD_METRIC_DISTANCE_
#define FASTPD_PAIR(i,l0,l1)   (pw(i,l0,l1))
#define FASTPD_UPDATE_BALANCE_VAR0(balance,d,h0,h1) { (balance)+=(d); (h0)+=(d); (h1)-=d; }
#define FASTPD_NEW_LABEL(n) ((n)->parent && !((n)->is_sink))
#define FASTPD_REV(a) ((a)+1)


namespace fastpd {

template<class PairwiseCostFunctor>
double FastPD::run(float *unaryCosts, PairwiseCostFunctor & pw)
{
  init_duals_primals(unaryCosts, pw);

  int iter = 0;
  while ( iter < m_max_iterations )
  {
    double prevm_energy = m_energy;
    m_iterations = iter;

    if ( !iter )
    {
      for( Graph::Label l = 0; l < m_num_labels; l++ )
      {
        inner_iteration( l, pw);
      }
    }
    else 
    {
      for( Graph::Label l = 0; l < m_num_labels; l++ )
      {
        inner_iteration_adapted( l, pw );
      }
    }

    if ( prevm_energy <= m_energy )
      break;

    iter++;
  }

  return m_energy;
}

template<class PairwiseCostFunctor>
void FastPD::init_duals_primals(float *unaryCosts, PairwiseCostFunctor & pw)
{
  if(m_delete_height) 
    memcpy(height, unaryCosts, sizeof(Real)*m_num_labels*m_num_nodes);

  // Set initial values for primal and dual variables
  for(int i = 0; i < m_num_nodes; i++ )
  {
    node_info[i].label =  0;
    node_info[i].time  = -1;
    node_info[i].next  = -1;
    node_info[i].prev  = -2;
  }

  for(Graph::Label l = 0; l < m_num_labels; l++ )
  {
    Real *cy = &balance[m_num_pairs*l];
    Real *ch = &height[m_num_nodes*l];
    for(int i = 0; i < m_num_pairs; i++ )
    {
      ch[pairs[2*i]] += ( cy[i] = FASTPD_PAIR(i,l,l) );
    }
  }

  for(int i = 0; i < m_num_pairs; i++ )
  {
    int id0 = edge_info[i  ].tail;
    int id1 = edge_info[i  ].head;
    int l0  = node_info[id0].label;
    int l1  = node_info[id1].label;

    if ( l0 != l1 )
    {
      Real d  = FASTPD_PAIR(i,l0,l1) - (balance[l0*m_num_pairs+i] - balance[l1*m_num_pairs+i] + FASTPD_PAIR(i,l1,l1));
      FASTPD_UPDATE_BALANCE_VAR0( balance[l0*m_num_pairs+i], d, height[l0*m_num_nodes+id0], height[l0*m_num_nodes+id1] );
    }

    edge_info[i].balance = -balance[l1*m_num_pairs+i]+FASTPD_PAIR(i,l1,l1);			
  }

  // Get initial primal function
  m_energy = 0;
  for(int i = 0; i < m_num_nodes; i++ )
  {
    node_info[i].height = height[node_info[i].label*m_num_nodes+i];
    m_energy += node_info[i].height;
  }
  m_initial_energy = m_energy;
}

template <class PairwiseCostFunctor>
void FastPD::inner_iteration( Graph::Label label, PairwiseCostFunctor & pw )
{
  int i;

  Graph       *_graph =  graphs[label];
  Graph::node *_nodes = &graph_nodes [m_num_nodes*label];
  Graph::arc  *_arcs  = &graph_edges  [(m_num_pairs*label)<<1];
  Real        *_curbalance = &balance         [m_num_pairs*label];

  m_time++;
  _graph->flow = 0;

  if ( m_energy_change_time < m_time - m_num_labels )
    return;

  // Update balance and height variables
  //
  Arc_info  *einfo = edge_info;
  Graph::arc *arcs = _arcs;
  Real *curbalance = &balance[m_num_pairs*label];
  Real *curheight = &height[m_num_nodes*label];
  for( i = 0; i < m_num_pairs; i++, einfo++, arcs+=2, curbalance++ )
  {
    int l0,l1;
    if ( (l1=node_info[einfo->head].label) != label && (l0=node_info[einfo->tail].label) != label ) 
    {
      Real delta  = FASTPD_PAIR(i,label,l1)+FASTPD_PAIR(i,l0,label)-FASTPD_PAIR(i,l0,l1)-FASTPD_PAIR(i,label,label);
      Real delta1 = FASTPD_PAIR(i,label,l1)-( (*curbalance)+einfo->balance);
      Real delta2;
      if ( delta1 < 0 || (delta2=delta-delta1) < 0 ) 
      {
        FASTPD_UPDATE_BALANCE_VAR0( *curbalance, delta1, curheight[einfo->tail], curheight[einfo->head] )
            arcs->cap = arcs->r_cap = 0;

#ifndef _FASTPD_METRIC_DISTANCE_
        if ( delta < 0 ) // This may happen only for non-metric distances
        {
          delta = 0;
          _nodes[einfo->head].conflict_time = m_time;
        }
#endif

        FASTPD_REV(arcs)->r_cap = delta;
      }
      else
      {
        arcs->cap = arcs->r_cap = delta1;
        FASTPD_REV(arcs)->r_cap = delta2;
      }
    }
    else
    {
      arcs->cap = arcs->r_cap = 0;
      FASTPD_REV(arcs)->r_cap = 0;
    }
  }

  Real total_cap = 0;
  Node_info *pinfo = node_info;
  Graph::node *nodes = _nodes;
  for( i = 0; i < m_num_nodes; i++, pinfo++, nodes++, curheight++ )
  {
    Real delta = pinfo->height - (*curheight);
    nodes->tr_cap = delta;
    if (delta > 0) total_cap += delta;
  }

  // Run max-flow and update the primal variables
  //
  Graph::flowtype max_flow = _graph -> apply_maxflow(1);
  m_energy -= (total_cap - max_flow);
  if ( total_cap > max_flow )
    m_energy_change_time = m_time;

  curbalance = &balance[m_num_pairs*label];
  einfo =  edge_info;
  arcs  =  _arcs;
  for( i = 0; i < m_num_pairs; i++, einfo++, arcs+=2, curbalance++ )
    if ( node_info[einfo->head].label != label && node_info[einfo->tail].label != label )
    {
      if ( FASTPD_NEW_LABEL(&_nodes[einfo->head]) )
        einfo->balance = -(*curbalance + arcs->cap - arcs->r_cap) + FASTPD_PAIR(i,label,label);
    }
    else if ( node_info[einfo->head].label != label )
    {
      if ( FASTPD_NEW_LABEL(&_nodes[einfo->head]) )
        einfo->balance = -(*curbalance) + FASTPD_PAIR(i,label,label);
    }

  curheight = &height[m_num_nodes*label];
  pinfo =  node_info;
  nodes =  _nodes;
  for( i = 0; i < m_num_nodes; i++, pinfo++, nodes++, curheight++ )
  {
    if ( pinfo->label != label )
    {
      if ( FASTPD_NEW_LABEL(nodes) )
      {
#ifndef _FASTPD_METRIC_DISTANCE_
        if ( nodes->conflict_time > pinfo->time )
        {
          int k;
          for( k = 0; k < pinfo->numpairs; k++)
          {
            int pid = pinfo->pairs[k];
            if ( pid <= 0 )
            {
              Pair_info *pair = &pair_info[-pid];
              if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
              {
                Graph::Label l0 = node_info[pair->i0].label;
                Graph::Label l1 =  pinfo->label;
                Real delta = (FASTPD_PAIR(-pid,l0,label)+FASTPD_PAIR(-pid,label,l1)-
                              FASTPD_PAIR(-pid,l0,l1)-FASTPD_PAIR(-pid,label,label));
                if ( delta < 0 )
                {
                  _curbalance[-pid]  -= delta;
                  edge_info[-pid].balance = -_curbalance[-pid] + FASTPD_PAIR(-pid,label,label);
                  pinfo->height += delta; 
                  m_energy          += delta;
                  _nodes[pair->i0].tr_cap += delta;
                }
              }
            }
          }
        }
#endif

        pinfo->label = label;
        pinfo->height -= nodes->tr_cap;
        nodes->tr_cap = 0;
        pinfo->time = m_time;
      }
    }
    *curheight = pinfo->height;
  }

  if (m_display)
  {
    std::vector<int> temp_labeling(m_num_nodes);
    getLabeling(&temp_labeling[0]);
    //model->applyLabeling(&temp_labeling[0]);
    m_display(m_displayinst);
  }
}

template<class PairwiseCostFunctor>
void FastPD::inner_iteration_adapted( Graph::Label label, PairwiseCostFunctor & pw )
{
  if ( m_iterations > 1 )
    return track_source_linked_nodes( label, pw );

  int i;
  Graph       *_graph =  graphs[label];
  Graph::node *_nodes = &graph_nodes [m_num_nodes*label];
  Graph::arc  *_arcs  = &graph_edges  [(m_num_pairs*label)<<1];
  Real        *_curbalance = &balance         [m_num_pairs*label];
  Real        *_curheight = &height         [m_num_nodes*label];

  m_time++;
  _graph->flow = 0;

  if ( m_energy_change_time < m_time - m_num_labels )
    return;

  // Update dual vars (i.e. balance and height variables)
  //
  int dt = m_time - m_num_labels;
  for( i = 0; i < m_num_pairs; i++ )
  {
    int i0 = pairs[ i<<1   ];
    int i1 = pairs[(i<<1)+1];
    if ( node_info[i0].time >= dt || node_info[i1].time >= dt )
    {
      Graph::arc *arc0 = &_arcs[i<<1];

      if ( _curheight[i0] != node_info[i0].height )
      {
        Real h = _curheight[i0] - _nodes[i0].tr_cap;
        _nodes[i0].tr_cap = node_info[i0].height - h;
        _curheight[i0] = node_info[i0].height;
      }

      if ( _curheight[i1] != node_info[i1].height )
      {
        Real h = _curheight[i1] - _nodes[i1].tr_cap;
        _nodes[i1].tr_cap = node_info[i1].height - h;
        _curheight[i1] = node_info[i1].height;
      }

      int l0,l1;
      if ( (l0=node_info[i0].label) != label && (l1=node_info[i1].label) != label )
      {
        Graph::arc *arc1 = &graph_edges[(m_num_pairs*l1+i)<<1];
        Real y_pq =   _curbalance[i] + arc0->cap - arc0->r_cap ;
        Real y_qp = -(balance[m_num_pairs*l1+i] + arc1->cap - arc1->r_cap) + FASTPD_PAIR(i,l1,l1);
        Real delta  = (FASTPD_PAIR(i,label,l1)+FASTPD_PAIR(i,l0,label)-FASTPD_PAIR(i,l0,l1)-FASTPD_PAIR(i,label,label));
        Real delta1 = FASTPD_PAIR(i,label,l1)-(y_pq+y_qp);
        Real delta2;
        if ( delta1 < 0 || (delta2=delta-delta1) < 0 ) 
        {
          _curbalance[i] = y_pq+delta1;
          arc0->r_cap = arc0->cap = 0;

#ifndef _FASTPD_METRIC_DISTANCE_
          if ( delta < 0 ) // This may happen only for non-metric distances
          {
            delta = 0;
            _nodes[i1].conflict_time = m_time;
          }
#endif

          FASTPD_REV(arc0)->r_cap = delta;

          _nodes[i0].tr_cap -= delta1;
          _nodes[i1].tr_cap += delta1;
        }
        else
        {
          _curbalance[i] = y_pq;
          arc0->r_cap = arc0->cap = delta1;
          FASTPD_REV(arc0)->r_cap = delta2;
        }
      }
      else
      {
        _curbalance[i] += arc0->cap - arc0->r_cap;
        FASTPD_REV(arc0)->r_cap = arc0->r_cap = arc0->cap = 0;	
      }
    }
  }

  // Run max-flow and update the primal variables
  //
  assert( m_iterations <= 1 );
  Graph::flowtype max_flow = _graph -> apply_maxflow(1);

  double prevm_energy = m_energy;
  for( i = 0; i < m_num_nodes; i++ )
  {
    Node_info *pinfo = &node_info[i];
    if ( FASTPD_NEW_LABEL(&_nodes[i]) )
    {
#ifndef _FASTPD_METRIC_DISTANCE_
      if ( _nodes[i].conflict_time > pinfo->time )
      {
        Real total_delta = 0;
        int k;
        for( k = 0; k < pinfo->numpairs; k++)
        {
          int pid = pinfo->pairs[k];
          if ( pid <= 0 )
          {
            Pair_info *pair = &pair_info[-pid];
            if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
            {
              Graph::Label l0 = node_info[pair->i0].label;
              Graph::Label l1 =  pinfo->label;
              Real delta = (FASTPD_PAIR(-pid,l0,label)+FASTPD_PAIR(-pid,label,l1)-
                            FASTPD_PAIR(-pid,l0,l1)-FASTPD_PAIR(-pid,label,label));

              if ( delta < 0 )
              {
                _curbalance[-pid] -= delta;
                total_delta  += delta;
                _nodes[pair->i0].tr_cap += delta;
              }
            }
          }
        }
        if ( total_delta )
          _nodes[i].tr_cap -= total_delta;
      }
#endif

      pinfo->height -= _nodes[i].tr_cap; 
      m_energy          -= _nodes[i].tr_cap;
      pinfo->time    = m_time;
      pinfo->label   = label;

      if ( pinfo->prev == -2 ) // add to active list
      {
        pinfo->next = m_active_list;
        pinfo->prev = -1;
        if (m_active_list >= 0)
          node_info[m_active_list].prev = i;
        m_active_list = i;
      }
    }
  }
  if ( m_energy < prevm_energy )
    m_energy_change_time = m_time;

  if (m_display)
  {
    std::vector<int> temp_labeling(m_num_nodes);
    getLabeling(&temp_labeling[0]);
    //model->applyLabeling(&temp_labeling[0]);
    m_display(m_displayinst);
  }
}

template<class PairwiseCostFunctor>
void FastPD::track_source_linked_nodes( Graph::Label label, PairwiseCostFunctor & pw )
{
  int i;
  assert( m_iterations > 1 );

  Graph       *_graph =  graphs[label];
  Graph::node *_nodes = &graph_nodes [m_num_nodes*label];
  Graph::arc  *_arcs  = &graph_edges  [(m_num_pairs*label)<<1];
  Real        *_curbalance = &balance         [m_num_pairs*label];
  Real        *_curheight = &height         [m_num_nodes*label];

  m_time++;
  _graph->flow = 0;

  if ( m_energy_change_time < m_time - m_num_labels )
    return;

  int source_nodes_start1 = -1;
  int source_nodes_start2 = -1;

  int dt = m_time - m_num_labels;
  i = m_active_list;
  while ( i >= 0 )
  {
    Node_info *n = &node_info[i];
    int i_next = n->next;

    if ( n->time >= dt )
    {
      if ( _curheight[i] != n->height )
      {
        Real h = _curheight[i] - _nodes[i].tr_cap;
        _nodes[i].tr_cap = n->height - h;
        _curheight[i] = n->height;
      }

      if ( _nodes[i].tr_cap )
      {
        //assert(  _nodes[i].tr_cap < 0 );
        _nodes[i].parent  = TERMINAL;
        _nodes[i].is_sink = 1;
        _nodes[i].DIST = 1;
      }
      else _nodes[i].parent = NULL;
    }
    else 
    {
      int prev = n->prev;
      if ( prev >= 0 )
      {
        node_info[prev].next = n->next;
        if (n->next >= 0)
          node_info[n->next].prev = prev;
      }
      else 
      {
        m_active_list = n->next;
        if ( m_active_list >= 0 )
          node_info[m_active_list].prev = -1;
      }
      n->prev = -2;
    }

    i = i_next;
  }

  // Update balance and height variables.
  //
  i = m_active_list;
  while ( i >= 0 )
  {
    Node_info *n = &node_info[i];
    int i_next = n->next;

    int k;
    Node_info *n0,*n1;
    for( k = 0; k < n->numpairs; k++)
    {
      int i0,i1,ii;
      Pair_info *pair;
      int pid = n->pairs[k];
      if ( pid >= 0 )
      {
        pair = &pair_info[pid];
        if ( pair->time == m_time )
          continue;

        i0 = i; i1 = pair->i1;
        n0 = n; n1 = &node_info[i1];
        ii = i1;
      }
      else
      {
        pid = -pid;
        pair = &pair_info[pid];
        if ( pair->time == m_time )
          continue;

        i1 = i; i0 = pair->i0;
        n1 = n; n0 = &node_info[i0];
        ii = i0;
      }
      pair->time = m_time;

      int l0,l1;
      Graph::arc *arc0 = &_arcs[pid<<1];
      if ( (l0=n0->label) != label && (l1=n1->label) != label )
      {
        Graph::arc *arc1 = &graph_edges[(m_num_pairs*l1+pid)<<1];
        Real y_pq =   _curbalance[pid] + arc0->cap - arc0->r_cap ;
        Real y_qp = -(balance[m_num_pairs*l1+pid] + arc1->cap - arc1->r_cap) + FASTPD_PAIR(pid,l1,l1);
        Real delta  = (FASTPD_PAIR(pid,label,l1)+FASTPD_PAIR(pid,l0,label)-FASTPD_PAIR(pid,l0,l1)-FASTPD_PAIR(pid,label,label));
        Real delta1 = FASTPD_PAIR(pid,label,l1)-(y_pq+y_qp);
        Real delta2;
        if ( delta1 < 0 || (delta2=delta-delta1) < 0 )
        {
          _curbalance[pid] = y_pq+delta1;
          arc0->r_cap = arc0->cap = 0;

#ifndef _FASTPD_METRIC_DISTANCE_
          if ( delta < 0 ) // This may happen only for non-metric distances
          {
            delta = 0;
            _nodes[i1].conflict_time = m_time;
          }
#endif

          FASTPD_REV(arc0)->r_cap = delta;

          _nodes[i0].tr_cap -= delta1;
          _nodes[i1].tr_cap += delta1;

          if ( node_info[ii].prev == -2 && source_nodes_tmp2[ii] == -2 )
          {
            source_nodes_tmp2[ii] = source_nodes_start2;
            source_nodes_start2 = ii;
          }
        }
        else
        {
          _curbalance[pid] = y_pq;
          arc0->r_cap = arc0->cap = delta1;
          FASTPD_REV(arc0)->r_cap = delta2;
        }
      }
      else
      {
        _curbalance[pid] += arc0->cap - arc0->r_cap;
        FASTPD_REV(arc0)->r_cap = arc0->r_cap = arc0->cap = 0;	
      }
    }

    Graph::node *nd = &_nodes[i];
    if ( nd->tr_cap > 0 )
    {
      nd -> is_sink = 0;
      nd -> parent = TERMINAL;
      nd -> DIST = 1;

      _graph->set_active(nd);

      source_nodes_tmp1[i] = source_nodes_start1;
      source_nodes_start1 = i;
    }
    else if (nd->tr_cap < 0)
    {
      nd -> is_sink = 1;
      nd -> parent = TERMINAL;
      nd -> DIST = 1;
    }
    else nd -> parent = NULL;
    //n -> TS = 0;

    i = i_next;
  }

  for( i = source_nodes_start2; i >= 0; )
  {
    Graph::node *nd = &_nodes[i];
    if ( nd->tr_cap > 0 )
    {
      nd -> is_sink = 0;
      nd -> parent = TERMINAL;
      nd -> DIST = 1;

      _graph->set_active(nd);

      source_nodes_tmp1[i] = source_nodes_start1;
      source_nodes_start1 = i;
    }
    else if (nd->tr_cap < 0)
    {
      nd -> is_sink = 1;
      nd -> parent = TERMINAL;
      nd -> DIST = 1;
    }
    else nd -> parent = NULL;
    //n -> TS = 0;

    int tmp = i;
    i = source_nodes_tmp2[i];
    source_nodes_tmp2[tmp] = -2;
  }

  // Run max-flow and update primal variables
  //
  Graph::flowtype max_flow = _graph -> apply_maxflow(0);

  double prevm_energy = m_energy;
  int numchildren = 0;
  for( i = source_nodes_start1; i >= 0; )
  {
    Graph::node *n = &_nodes[i];
    if ( n->parent == TERMINAL )
    {
      Node_info *pinfo = &node_info[i];

#ifndef _FASTPD_METRIC_DISTANCE_
      if ( n->conflict_time > pinfo->time )
      {
        Real total_delta = 0;
        int k;
        for( k = 0; k < pinfo->numpairs; k++)
        {
          int pid = pinfo->pairs[k];
          if ( pid <= 0 )
          {
            Pair_info *pair = &pair_info[-pid];
            if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
            {
              Graph::Label l0 = node_info[pair->i0].label;
              Graph::Label l1 =  pinfo->label;
              Real delta = (FASTPD_PAIR(-pid,l0,label)+FASTPD_PAIR(-pid,label,l1)-
                            FASTPD_PAIR(-pid,l0,l1)-FASTPD_PAIR(-pid,label,label));

              if ( delta < 0 )
              {
                _curbalance[-pid] -= delta;
                total_delta  += delta;
                _nodes[pair->i0].tr_cap += delta;
              }
            }
          }
        }
        if ( total_delta )
          n->tr_cap -= total_delta;
      }
#endif

      pinfo->height   -= n->tr_cap; 
      m_energy            -= n->tr_cap;
      pinfo->label     = label;
      pinfo->time      =m_time;

      if ( pinfo->prev == -2 ) // add to active list
      {
        pinfo->next = m_active_list;
        pinfo->prev = -1;
        if (m_active_list >= 0)
          node_info[m_active_list].prev = i;
        m_active_list = i;
      }

      Graph::arc *a;
      for ( a=n->first; a; a=a->next )
      {
        Graph::node *ch = a->head;
        if ( ch->parent == a->sister )
          children[numchildren++] = ch;
      }
    }

    int tmp = i;
    i = source_nodes_tmp1[i];
    source_nodes_tmp1[tmp] = -2;
  }

  for( i = 0; i < numchildren; i++ )
  {
    Graph::node *n = children[i];
    unsigned int id  = ((unsigned int)(n - _nodes)) ;
    Node_info *pinfo = &node_info[id];

#ifndef _FASTPD_METRIC_DISTANCE_
    if ( n->conflict_time > pinfo->time )
    {
      Real total_delta = 0;
      int k;
      for( k = 0; k < pinfo->numpairs; k++)
      {
        int pid = pinfo->pairs[k];
        if ( pid <= 0 )
        {
          Pair_info *pair = &pair_info[-pid];
          if ( !(_nodes[pair->i0].parent) || _nodes[pair->i0].is_sink)
          {
            Graph::Label l0 = node_info[pair->i0].label;
            Graph::Label l1 =  pinfo->label;
            Real delta = (FASTPD_PAIR(-pid,l0,label)+FASTPD_PAIR(-pid,label,l1)-
                          FASTPD_PAIR(-pid,l0,l1)-FASTPD_PAIR(-pid,label,label));

            if ( delta < 0 )
            {
              _curbalance[-pid] -= delta;
              total_delta  += delta;
              _nodes[pair->i0].tr_cap += delta;
            }
          }
        }
      }
      if ( total_delta )
        n->tr_cap -= total_delta;
    }
#endif

    pinfo->height   -= n->tr_cap; 
    m_energy            -= n->tr_cap;
    pinfo->label     = label;
    pinfo->time      =m_time;

    if ( pinfo->prev == -2 ) // add to active list
    {
      pinfo->next = m_active_list;
      pinfo->prev = -1;
      if (m_active_list >= 0)
        node_info[m_active_list].prev = id;
      m_active_list = id;
    }

    Graph::arc *a;
    for ( a=n->first; a; a=a->next )
    {
      Graph::node *ch = a->head;
      if ( ch->parent == a->sister )
        children[numchildren++] = ch;
    }
  }

  if ( m_energy < prevm_energy )
    m_energy_change_time = m_time;
}

}
