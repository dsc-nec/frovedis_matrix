#include <fstream>
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
    if(line[0] == '#' || line.size() == 0) continue;
    // remove trailing \r and add 1
    ofs << line.substr(0, line.size() - 1) << "\t1" << endl;
  }
  auto crs = make_crs_matrix_local_loadcoo<float,int>("./tmp",true);
  crs.savebinary(argv[2]);
}
