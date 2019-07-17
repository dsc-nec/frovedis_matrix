#include <frovedis/matrix/rowmajor_matrix.hpp>
#include <frovedis/matrix/colmajor_matrix.hpp>

using namespace std;
using namespace frovedis;

int main() {
  auto rm = make_rowmajor_matrix_local_load<float>("rowmajor.txt");
  cout << "loaded from text: " << endl;
  cout << rm << endl;
  cout << "debug print: " << endl;
  rm.debug_print();
  cout << endl;
  rm.savebinary("./saved_rowmajor");
  auto rm2 = make_rowmajor_matrix_local_loadbinary<float>("./saved_rowmajor");
  cout << "loaded from binary: " << endl;
  cout << rm2 << endl;
  auto rms = get_scattered_rowmajor_matrices(rm, 2);
  cout << "separated into two parts.\nfirst one" << endl;
  cout << rms[0] << endl;
  cout << "second one" << endl;
  cout << rms[1] << endl;
  colmajor_matrix_local<float> cm(rm);
  cout << "converted into colmajor format, debug print: " << endl;
  cm.debug_print();
}
