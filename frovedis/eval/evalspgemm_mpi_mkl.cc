// /usr/bin/mpirun --bind-to socket --mca btl_base_warn_component_unused 0 -np 12 ./evalspgemm_mpi_mkl inputfile
#include <mpi.h>
#include <frovedis/matrix/spgemm.hpp>
#include <malloc.h>
#include <mkl_spblas.h>

using namespace frovedis;
using namespace std;

int main(int argc, char* argv[]){
  int required = MPI_THREAD_SERIALIZED;
  int provided;
  MPI_Init_thread(&argc, &argv, required, &provided);

  mallopt(M_MMAP_MAX,0);
  mallopt(M_TRIM_THRESHOLD,-1);

  time_spent t;
  if(argc != 2) {
    cerr << argv[0] << " input" << endl;
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
  if(rank == 0) t.show("separate matrix: ");
  MPI_Barrier(MPI_COMM_WORLD);

  sparse_matrix_t A;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  MKL_INT rows = mypart.local_num_row;
  MKL_INT cols = mypart.local_num_col;
  std::vector<int> newoff(mypart.off.size());
  for(size_t i = 0; i < newoff.size(); i++) newoff[i] = mypart.off[i];
  MKL_INT *rows_start = newoff.data();
  MKL_INT *rows_end = newoff.data() + 1;
  MKL_INT *col_indx = mypart.idx.data();
  float *values = mypart.val.data();

  sparse_status_t status = mkl_sparse_s_create_csr 
    (&A, indexing, rows, cols, rows_start, rows_end, col_indx, values);
  if(status != SPARSE_STATUS_SUCCESS) {
    cerr << "mkl_sparse_s_create_csr: " << status << endl;
    exit(1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("create MKL csr left: ");

  sparse_matrix_t B;
  MKL_INT trows = crs.local_num_row;
  MKL_INT tcols = crs.local_num_col;
  std::vector<int> newoff2(crs.off.size());
  for(size_t i = 0; i < newoff2.size(); i++) newoff2[i] = crs.off[i];
  MKL_INT *trows_start = newoff2.data();
  MKL_INT *trows_end = newoff2.data() + 1;
  MKL_INT *tcol_indx = crs.idx.data();
  float *tvalues = crs.val.data();

  status = mkl_sparse_s_create_csr 
    (&B, indexing, trows, tcols, trows_start, trows_end, tcol_indx, tvalues);
  if(status != SPARSE_STATUS_SUCCESS) {
    cerr << "mkl_sparse_s_create_csr: " << status << endl;
    exit(1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("create MKL csr right: ");
  
  sparse_matrix_t C;
  sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
  status = mkl_sparse_spmm (operation, A, B, &C);
  if(status != SPARSE_STATUS_SUCCESS) {
    cerr << "mkl_sparse_spmm: " << status << endl;
    exit(1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("MKL spggemm: ");
  status = mkl_sparse_order (C);
  if(status != SPARSE_STATUS_SUCCESS) {
    cerr << "mkl_sparse_order: " << status << endl;
    exit(1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t.show("MKL index sort: ");
  MPI_Finalize();
}
