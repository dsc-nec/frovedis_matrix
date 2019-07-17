#include <frovedis/matrix/crs_matrix.hpp>
#include <frovedis/matrix/jds_crs_hybrid.hpp>

using namespace frovedis;
using namespace std;

int main(int argc, char* argv[]){
  time_spent t;
  if(argc != 3) {
    cerr << argv[0] << " input num_threads" << endl;
    exit(1);
  }

  auto crs = make_crs_matrix_local_loadbinary<float,int>(argv[1]);
  t.show("load: ");
  int num_threads = atoi(argv[2]);
  auto crss = get_scattered_crs_matrices(crs, num_threads);
  t.show("separate matrix: ");
  auto num_col = crs.local_num_col;
  auto num_row = crs.local_num_row;
  vector<float> v(num_col);
  vector<float> r(num_row);
  for(size_t i = 0; i < num_col; i++) v[i] = 1;
  t.show("create vector: ");
  std::vector<size_t> num_rows(num_threads);
  std::vector<size_t> pos(num_threads);
  for(size_t i = 0; i < num_threads; i++) {
    num_rows[i] = crss[i].local_num_row;
  }
  for(size_t i = 1; i < num_threads; i++) {
    pos[i] = pos[i-1] + num_rows[i-1];
  }
  vector<float*> rps(num_threads);
  for(size_t i = 0; i < num_threads; i++) {
    rps[i] = r.data() + pos[i];
  }
  t.show("calculate return vector position: ");
#pragma omp parallel for num_threads(num_threads)
  for(size_t i = 0; i < num_threads; i++) {
    crs_matrix_spmv_impl(crss[i], rps[i], v.data());
  }
  t.show("crs: ");
  for(size_t i = 0; i < num_row; i++) r[i] = 0;
  std::vector<jds_crs_hybrid_local<float,int>> hybs(num_threads);
  for(size_t i = 0; i < num_threads; i++) {
    hybs[i] = jds_crs_hybrid_local<float,int>(crss[i]);
  }
  t.show("create hyb: ");
#pragma omp parallel for num_threads(num_threads)
  for(size_t i = 0; i < num_threads; i++) {
    jds_crs_hybrid_spmv_impl(hybs[i], rps[i], v.data());
  }
  t.show("hyb: ");
}
