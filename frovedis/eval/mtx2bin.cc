// very simple converter; does not work all the mtx files!
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
  bool is_symmetric = false;
  bool is_pattern = false;
  std::getline(ifs,line);
  if(!(line.find("symmetric") == string::npos)) is_symmetric = true;
  if(!(line.find("pattern") == string::npos)) is_pattern = true;
  if(is_symmetric && is_pattern)
    throw runtime_error("symmetric & pattern is not supported yet");
  while(std::getline(ifs,line)) {
    if(line[0] == '%') continue;
    else break;
  }
  size_t num_row, num_col, nnz;
  istringstream iss(line);
  iss >> num_row >> num_col >> nnz;
  while(std::getline(ifs,line)) {
    if(is_pattern) {
      ofs << line << " " << 1 << endl;
    } else {
      ofs << line << endl;
      if(is_symmetric) {
        istringstream iss2(line);
        size_t row, col;
        string val; // for arbitrary precision
        iss2 >> row >> col >> val;
        if(row != col) ofs << col << " " << row << " " << val << endl;
      }
    }
  }
  auto crs = make_crs_matrix_local_loadcoo<float,int>("./tmp");
  crs.local_num_col = num_col;
  //std::ofstream ofs2("./tmp2");
  //ofs2 << crs;  
  crs.savebinary(argv[2]);
}
