#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "fastpd/FastPD.h"
#include <memory>
#include <cmath>
#include <vector>

typedef std::vector<int> vi;
namespace py = pybind11;
typedef py::buffer buf;


class CostFunctor {
 public:

  CostFunctor(int _n, float **_cost): n(_n), cost(_cost) {
  };

  inline float operator()(int pair, int labelA, int labelB) {
    return cost[pair][labelA*n + labelB];
  }

 private:
  int n;
  float **cost;
};

// extracts python buffers (numpy arrays)
template<typename T>
void extractBuffer(buf *b, int *size, T** ptr){
    py::buffer_info info = b->request();
    *size = info.size;
    *ptr = (T*) info.ptr;
}

static vi py_fastpd(buf bu, std::vector<buf> vbb, buf bpairs, int numIterations) {
    float *unary;
    int *pairs;
    int numUnary, numBinary, numPairsDoubled;

    // unary and pairs
    extractBuffer(&bu, &numUnary, &unary);
    extractBuffer(&bpairs, &numPairsDoubled, &pairs);
    int numPairs = numPairsDoubled/2;

    // extract binary energies
    std::vector<float*> vecBinary(numPairs);
    for (int i = 0; i < numPairs; i++) {
        extractBuffer(&vbb[i], &numBinary, &vecBinary[i]);
    }

    // deducting the size of graph
    int numLabels = round(sqrt(numBinary));
    int numNodes = numUnary/numLabels;

    vi labeling(numNodes);
    std::shared_ptr<fastpd::FastPD> opt(new fastpd::FastPD(
        numNodes,
        numLabels,
        numPairs,
        pairs,
        unary,
        numIterations));

    CostFunctor costFunctor(numLabels, &vecBinary[0]);
    double energy = opt->run<CostFunctor>(unary, costFunctor);

    //Get labeling result
    opt->getLabeling(labeling.data());

    return labeling;
}

PYBIND11_PLUGIN(libfastpd) {
    py::module m("libfastpd", "FastPD Wrapper (ECP CVN property)");
    m.def("fastpd", py_fastpd);
    return m.ptr();
}

