//numactl --cpunodebind=0 --membind=0  ./evalspgemm_mkl input
#include <frovedis/matrix/spgemm.hpp>
#include <malloc.h>
#include <mkl_spblas.h>

using namespace frovedis;
using namespace std;

int main(int argc, char* argv[]){
  mallopt(M_MMAP_MAX,0);
  mallopt(M_TRIM_THRESHOLD,-1);

  time_spent t;
  if(argc != 2) {
    cerr << argv[0] << " input" << endl;
    exit(1);
  }

  auto crs = make_crs_matrix_local_loadbinary<float,int>(argv[1]);
  t.show("load: ");  
  sparse_matrix_t A;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  MKL_INT rows = crs.local_num_row;
  MKL_INT cols = crs.local_num_col;
  std::vector<int> newoff(crs.off.size());
  for(size_t i = 0; i < newoff.size(); i++) newoff[i] = crs.off[i];
  MKL_INT *rows_start = newoff.data();
  MKL_INT *rows_end = newoff.data() + 1;
  MKL_INT *col_indx = crs.idx.data();
  float *values = crs.val.data();

  sparse_status_t status = mkl_sparse_s_create_csr 
    (&A, indexing, rows, cols, rows_start, rows_end, col_indx, values);
  if(status != SPARSE_STATUS_SUCCESS) {
    cerr << "mkl_sparse_s_create_csr: " << status << endl;
    exit(1);
  }
  t.show("create MKL csr: ");
  sparse_matrix_t C;
  sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
  status = mkl_sparse_spmm (operation, A, A, &C);
  if(status != SPARSE_STATUS_SUCCESS) {
    cerr << "mkl_sparse_spmm: " << status << endl;
    exit(1);
  }
  t.show("MKL spggemm: ");
  status = mkl_sparse_order (C);
  if(status != SPARSE_STATUS_SUCCESS) {
    cerr << "mkl_sparse_order: " << status << endl;
    exit(1);
  }
  t.show("MKL index sort: ");
}
