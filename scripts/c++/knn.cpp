#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
using namespace mlpack;
using namespace mlpack::neighbor; // NeighborSearch and NearestNeighborSort
using namespace mlpack::metric; // ManhattanDistance
int main(){
  arma::mat data;
  data::Load("../../train_proc.csv", data, true);
  NeighborSearch<NearestNeighborSort, ManhattanDistance> nn(data);
  // Create the object we will store the nearest neighbors in.
  arma::Mat<size_t> neighbors;
  arma::mat distances; // We need to store the distance too.
  // Compute the neighbors.
  nn.Search(1, neighbors, distances);
  // Write each neighbor and distance using Log.
  for (size_t i = 0; i < neighbors.n_elem; ++i)
  {
    std::cout << "Nearest neighbor of point " << i << " is point "
        << neighbors[i] << " and the distance is " << distances[i] << ".\n";
  }
}