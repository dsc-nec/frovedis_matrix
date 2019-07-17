#include <frovedis/matrix/crs_matrix.hpp>
#include <frovedis/matrix/ccs_matrix.hpp>
#include <frovedis/matrix/jds_matrix.hpp>
#include <frovedis/matrix/jds_crs_hybrid.hpp>

using namespace std;
using namespace frovedis;

int main() {
  auto crs = make_crs_matrix_local_load<float,int>("crs.txt");
  cout << "loaded from text: " << endl;
  cout << crs << endl;
  cout << "debug print: " << endl;
  crs.debug_print();
  cout << endl;

  auto num_col = crs.local_num_col;
  vector<float> v(num_col);
  for(size_t i = 0; i < num_col; i++) v[i] = 1;
  rowmajor_matrix_local<float> rm(num_col, 2);
  for(size_t i = 0; i < num_col * 2; i++) rm.val[i] = 1;

  ccs_matrix_local<float,int> ccs(crs);
  cout << "ccs format, debug print: " << endl;
  ccs.debug_print();
  cout << endl;
  auto r = ccs * v; // SpMV
  cout << "SpMV: " << endl;
  for(size_t i = 0; i < r.size(); i++) cout << r[i] << " ";
  cout << endl << endl;;
  auto r2 = ccs * rm; // SpMM
  cout << "SpMM: " << endl;
  cout << r2 << endl;

  jds_matrix_local<float,int> jds(crs);
  cout << "jds format, debug print: " << endl;
  jds.debug_print();
  cout << endl;
  r = jds * v; // SpMV
  cout << "SpMV: " << endl;
  for(size_t i = 0; i < r.size(); i++) cout << r[i] << " ";
  cout << endl << endl;;
  r2 = jds * rm; // SpMM
  cout << "SpMM: " << endl;
  cout << r2 << endl;

  jds_crs_hybrid_local<float,int> hyb(crs,2); // usually hyb(crs) is OK
  cout << "hybrid format, debug print: " << endl;
  hyb.debug_print();
  cout << endl;
  r = hyb * v; // SpMV
  cout << "SpMV: " << endl;
  for(size_t i = 0; i < r.size(); i++) cout << r[i] << " ";
  cout << endl << endl;;
  r2 = hyb * rm; // SpMM
  cout << "SpMM: " << endl;
  cout << r2 << endl;
}
