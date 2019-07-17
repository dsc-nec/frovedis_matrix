#include <frovedis/core/utility.hpp>
#include <iostream>

using namespace std;
using namespace frovedis;

int main() {
  vector<float> a = {0,1,2,3,4,5,6,7,8,9};
  savebinary(a, "./saved_binary");
  auto r = loadbinary<float>("./saved_binary");
  for(size_t i = 0; i < r.size(); i++) {
    std::cout << r[i] << " ";
  }
  std::cout << std::endl;
}
