ADD_DEFINITIONS(-fPIC)
SET(FASTPD_SRC
  FastPD.cpp
  graph.cpp
  block.h
  FastPD.h
  FastPD.inl
  graph.h
  )
ADD_LIBRARY(fastpd ${FASTPD_SRC})
