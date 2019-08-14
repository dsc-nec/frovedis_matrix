#include <frovedis/matrix/spgemm.hpp>
#include <malloc.h>

using namespace frovedis;
using namespace std;

int main(int argc, char* argv[]){

  time_spent t;
  if(argc != 4) {
    cerr << argv[0] << " input num_threads num_merge" << endl;
    exit(1);
  }

  auto crs = make_crs_matrix_local_loadbinary<float,int>(argv[1]);
  t.show("load: ");
  int num_threads = atoi(argv[2]);
  auto crss = get_scattered_crs_matrices(crs, num_threads);
  int num_merge = atoi(argv[3]);
  t.show("separate matrix: ");
#pragma omp parallel num_threads(num_threads)
  {
    mallopt(M_MMAP_MAX,0);
    mallopt(M_TRIM_THRESHOLD,-1);
    //set_loglevel(DEBUG);
#pragma omp for 
    for(size_t i = 0; i < num_threads; i++) {
      spgemm(crss[i], crs, spgemm_type::block_esc, num_merge);
    }
#pragma omp single
    t.show("block_esc: ");
#pragma omp for
    for(size_t i = 0; i < num_threads; i++) {
      spgemm(crss[i], crs, spgemm_type::hash, num_merge);
    }
#pragma omp single
    t.show("hash: ");
#pragma omp for
    for(size_t i = 0; i < num_threads; i++) {
      spgemm(crss[i], crs, spgemm_type::hash_sort, num_merge);
    }
#pragma omp single
    t.show("hash_sort: ");
#pragma omp for
    for(size_t i = 0; i < num_threads; i++) {
      spgemm(crss[i], crs, spgemm_type::spa, num_merge);
    }
#pragma omp single
    t.show("spa: ");
#pragma omp for
    for(size_t i = 0; i < num_threads; i++) {
      spgemm(crss[i], crs, spgemm_type::spa_sort, num_merge);
    }
#pragma omp single
    t.show("spa_sort: ");
#pragma omp for
    for(size_t i = 0; i < num_threads; i++) {
      spgemm_hybrid(crss[i], crs,
                    spgemm_type::spa_sort,
                    spgemm_type::block_esc,
                    num_merge,
                    num_merge,
                    128);
    }
#pragma omp single
    t.show("hybrid: ");
  }
}
