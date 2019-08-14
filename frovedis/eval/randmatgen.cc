#include <frovedis/matrix/crs_matrix.hpp>
#include <stdlib.h>
#include <set>

using namespace std;
using namespace frovedis;

int main(int argc, char* argv[]){
  if(argc != 4) {
    cerr << argv[0] << " outfile matrix_size nnz_per_row" << endl;
    exit(1);
  }

  std::string outfile(argv[1]);
  size_t mat_size = atoi(argv[2]);
  size_t nnz_per_row = atoi(argv[3]);
  cout << "outfile = " << outfile << endl;
  cout << "mat_size = " << mat_size << endl;
  cout << "nnz_per_row = " << nnz_per_row << endl;
  

  crs_matrix_local<float,int> mat(mat_size,mat_size);
  mat.val.resize(nnz_per_row * mat_size);
  mat.idx.resize(nnz_per_row * mat_size);
  mat.off.resize(mat_size + 1);
  for(size_t i = 0; i < mat.val.size(); i++) mat.val[i] = 1;
  for(size_t i = 0; i < mat_size + 1; i++) mat.off[i] = nnz_per_row * i;
  srand48(0);
  for(size_t i = 0; i < mat_size; i++) {
    std::set<size_t> idxtmp;
    while(idxtmp.size() < nnz_per_row) {
      idxtmp.insert(drand48() * mat_size);
    }
    size_t j = 0;
    for(auto it = idxtmp.begin(); it != idxtmp.end(); ++it) {
      mat.idx[i * nnz_per_row + j++] = *it;
    }
  }
  std::ofstream str("./tmp3");
  str << mat;
  mat.savebinary(outfile);
}
