#include <mpi.h>
#include <frovedis/matrix/spgemm.hpp>
#include <malloc.h>

using namespace frovedis;
using namespace std;

int main(int argc, char* argv[]){
  int required = MPI_THREAD_SERIALIZED;
  int provided;
  MPI_Init_thread(&argc, &argv, required, &provided);

  mallopt(M_MMAP_MAX,0);
  mallopt(M_TRIM_THRESHOLD,-1);

  time_spent t;
  if(argc != 4) {
    cerr << argv[0] << " input num_merge num_split" << endl;
    exit(1);
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  //if(rank == 0) set_loglevel(DEBUG);
  auto crs = make_crs_matrix_local_loadbinary<float,int>(argv[1]);
  if(rank == 0) t.show("load: ");
  crs_matrix_local<float,int> mypart;
  {
    auto crss = get_scattered_crs_matrices(crs, size);
    mypart = crss[rank];
  }
  int num_merge = atoi(argv[2]);
  int num_split = atoi(argv[3]);
  if(rank == 0) t.show("separate matrix: ");
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("start: ");
  spgemm(mypart, crs, spgemm_type::block_esc, num_merge);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("block_esc: ");
  spgemm(mypart, crs, spgemm_type::hash, num_merge);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("hash: ");
  spgemm(mypart, crs, spgemm_type::hash_sort, num_merge);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("hash_sort: ");
  spgemm(mypart, crs, spgemm_type::spa, num_merge);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("spa: ");
  spgemm(mypart, crs, spgemm_type::spa_sort, num_merge);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("spa_sort: ");
  spgemm_hybrid(mypart, crs,
                spgemm_type::spa_sort,
                spgemm_type::block_esc,
                num_merge,
                num_merge,
                num_split);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("hybrid w/ block_esc: ");
  spgemm_hybrid(mypart, crs,
                spgemm_type::spa_sort,
                spgemm_type::hash_sort,
                num_merge,
                num_merge,
                num_split);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("hybrid w/ hash_sort: ");
  MPI_Finalize();
}
