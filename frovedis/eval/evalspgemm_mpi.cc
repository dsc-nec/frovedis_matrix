#include <mpi.h>
#include <frovedis/matrix/spgemm.hpp>
#include <malloc.h>

using namespace frovedis;
using namespace std;

// here we assume that val/idx/off is less than 2GB!
template <class T, class I, class O>
void broadcast_crs_matrix(crs_matrix_local<T,I,O>& crs) {
  time_spent t;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Bcast(&crs.local_num_row, sizeof(size_t), MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&crs.local_num_col, sizeof(size_t), MPI_CHAR, 0, MPI_COMM_WORLD);
  size_t nnz = crs.val.size();
  MPI_Bcast(&nnz, sizeof(size_t), MPI_CHAR, 0, MPI_COMM_WORLD);
  if(rank != 0) {
    crs.val.resize(nnz);
    crs.idx.resize(nnz);
    crs.off.resize(crs.local_num_row+1);
  }
  MPI_Bcast(crs.val.data(), nnz*sizeof(T), MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(crs.idx.data(), nnz*sizeof(I), MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(crs.off.data(), (crs.local_num_row+1)*sizeof(O), MPI_CHAR, 0,
            MPI_COMM_WORLD);
}

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
  crs_matrix_local<float,int> crs;
  crs = make_crs_matrix_local_loadbinary<float,int>(argv[1]);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("load: ");
  auto mypart = separate_crs_matrix_for_spgemm_mpi(crs,crs);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("separate matrix: ");
  int num_merge = atoi(argv[2]);
  int num_split = atoi(argv[3]);
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
