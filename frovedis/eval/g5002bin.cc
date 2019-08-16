#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <frovedis/matrix/crs_matrix.hpp>

using namespace frovedis;
using namespace std;

int main(int argc, char* argv[]){
  time_spent t;
  if(argc != 3) {
    cerr << argv[0] << " input output" << endl;
    exit(1);
  }

  std::ifstream ifs(argv[1]);
  std::ofstream ofs("./tmp");
  std::string line;
  while(std::getline(ifs,line)) {
    ofs << line << " " << 1 << endl;
  }
  auto crs = make_crs_matrix_local_loadcoo<float,int>("./tmp");
  std::ofstream ofs2("./tmp2");
  ofs2 << crs;  
  crs.savebinary(argv[2]);
}
