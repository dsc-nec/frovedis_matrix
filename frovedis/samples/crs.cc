#include <frovedis/matrix/crs_matrix.hpp>
#include <frovedis/matrix/spgemm.hpp>

using namespace std;
using namespace frovedis;

int main() {
  auto crs = make_crs_matrix_local_load<float,int>("crs.txt");
  cout << "loaded from text: " << endl;
  cout << crs << endl;
  cout << "debug print: " << endl;
  crs.debug_print();
  cout << endl;
  cout << "debug pretty print: " << endl;
  crs.debug_pretty_print();
  cout << endl;
  crs.savebinary("./saved_crs");
  auto crs2 = make_crs_matrix_local_loadbinary<float,int>("./saved_crs");
  cout << "loaded from binary: " << endl;
  cout << crs2 << endl;
  auto crss = get_scattered_crs_matrices(crs, 2);
  cout << "separated into two parts.\nfirst one" << endl;
  cout << crss[0] << endl;
  cout << "second one" << endl;
  cout << crss[1] << endl;
  // make second argument true in case of 0-based index
  auto from_coo = make_crs_matrix_local_loadcoo<float,int>("coo.txt", true);
  cout << "loaded from coo: " << endl;
  cout << from_coo << endl;

  auto num_col = crs.local_num_col;
  vector<float> v(num_col);
  for(size_t i = 0; i < num_col; i++) v[i] = 1;
  auto r = crs * v; // SpMV
  cout << "SpMV: " << endl;
  for(size_t i = 0; i < r.size(); i++) cout << r[i] << " ";
  cout << endl << endl;;

  rowmajor_matrix_local<float> rm(num_col, 2);
  for(size_t i = 0; i < num_col * 2; i++) rm.val[i] = 1;
  auto r2 = crs * rm; // SpMM
  cout << "SpMM: " << endl;
  cout << r2 << endl;

  auto r3 = spgemm(crs,crs);
  cout << "SpGEMM" << endl;
  cout << r3 << endl;

  set_diag_zero(crs);
  cout << "set zero to diagonal" << endl;
  cout << crs << endl;

  crs_matrix_local<float,int> u, l;
  separate_upper_lower(crs, u, l);
  cout << "separate upper and lower" << endl;
  u.debug_pretty_print();
  cout << endl;
  l.debug_pretty_print();
  cout << endl;

  auto ep = elementwise_product(crs,crs);
  cout << "elementwise product (diag is set zero)" << endl;
  cout << ep << endl;
}
