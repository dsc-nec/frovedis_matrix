#include <frovedis/matrix/spgemm.hpp>
#include <malloc.h>

using namespace frovedis;
using namespace std;

int main(int argc, char* argv[]){

  time_spent t;
  if(argc != 2) {
    cerr << argv[0] << " input" << endl;
    exit(1);
  }

  auto crs = make_crs_matrix_local_loadbinary<float,int>(argv[1]);
  t.show("load: ");
  cout << "nnz of A: " << crs.val.size() << endl;
  set_loglevel(DEBUG);
  auto r = spgemm(crs, crs, spgemm_type::esc);
  cout << "nnz of A^2: " << r.val.size() << endl;
}
