#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "log.hpp"
#include <cmath>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace frovedis {

double get_dtime();

template <class T>
T ceil_div(T a, T b){
  if(a == 0) return 0;
  else return (a - 1) / b + 1;
}

template <class T>
T add(T a, T b){return a + b;}

class time_spent {
public:
  time_spent() : t0(get_dtime()), t1(0), lap_sum(0), loglevel(INFO) {}
  time_spent(log_level l) : t0(get_dtime()), t1(0), lap_sum(0), loglevel(l) {}
  void show(const std::string& mes) {
    t1 = get_dtime();
    LOG(loglevel) << mes << t1 - t0 << " sec" << std::endl;
    /* Since printing itself takes some time on VE, changed to exclude it. */
    // t0 = t1; 
    t0 = get_dtime();
  }
  void reset(){t0 = get_dtime(); lap_sum = 0;}
  void lap_start(){t0 = get_dtime();}
  void lap_stop(){lap_sum += get_dtime() - t0;}
  double get_lap(){return lap_sum;}
  void show_lap(const std::string& mes){
    LOG(loglevel) << mes << lap_sum << " sec" << std::endl;
  }
private:
  double t0, t1;
  double lap_sum;
  log_level loglevel;
};

void make_directory(const std::string&);
bool directory_exists(const std::string&);

template <class T>
void savebinary(std::vector<T> v, const std::string& path) {
  int fd = ::open(path.c_str(), O_CREAT|O_RDWR, 0666);
  if(fd == -1) {
    perror("open failed:");
    throw std::runtime_error("open failed");
  } 
  auto written = ::write(fd, v.data(), v.size() * sizeof(T));
  if(written != v.size() * sizeof(T)) {
    ::close(fd);
    throw std::runtime_error("write failed");
  }
  ::close(fd);
}

template <class T>
std::vector<T> loadbinary(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDWR, 0666);
  if(fd == -1) {
    perror("open failed:");
    throw std::runtime_error("open failed");
  }
  struct stat sb;
  if (stat(path.c_str(), &sb) != 0) {
    perror("stat failed:");
    ::close(fd);
    throw std::runtime_error("stat failed");
  }
  auto fsize = sb.st_size;
  auto size = fsize / sizeof(T);
  if(size * sizeof(T) != fsize) 
    throw std::runtime_error("file size is not multiple of data size");
  std::vector<T> ret(size);
  auto read_size = ::read(fd, ret.data(), fsize);
  if(read_size != fsize) {
    ::close(fd);
    throw std::runtime_error("read failed");
  }
  ::close(fd);
  return ret;
}

// temporary; to improve vectorization
#ifdef __ve__
inline double myexp(double _Left) {
  return (__builtin_exp(_Left));
}
inline float myexp(float _Left) {
  return (__builtin_expf(_Left));
}
#else
inline double myexp(double _Left) {
  return std::exp(_Left);
}
inline float myexp(float _Left) {
  return std::exp(_Left);
}
#endif


}

#endif
