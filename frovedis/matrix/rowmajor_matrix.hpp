#ifndef ROWMAJOR_MATRIX_HPP
#define ROWMAJOR_MATRIX_HPP

#include <fstream>
#include <dirent.h>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <../core/utility.hpp>

#define MAT_VLEN 256

namespace frovedis {

template <class T, class I, class O>
struct crs_matrix_local;

template <class T>
struct rowmajor_matrix_local {
  rowmajor_matrix_local(){}
  rowmajor_matrix_local(size_t r, size_t c) :
    local_num_row(r), local_num_col(c) {
    val.resize(r*c);
  }
  rowmajor_matrix_local(size_t r, size_t c,
                        T* raw_data) :
    local_num_row(r), local_num_col(c) {
    val.resize(r*c);
    T* valp = &val[0];
    for(size_t i=0; i<val.size(); ++i) valp[i] = raw_data[i];
  }
  // move operations are defined because SX C++ compiler's STL
  // does not know move currently
  rowmajor_matrix_local(rowmajor_matrix_local<T>&& m) {
    val.swap(m.val);
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
  }
  rowmajor_matrix_local<T>& operator=(rowmajor_matrix_local<T>&& m) {
    val.swap(m.val);
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
    return *this;
  }
  // copy ctor is explicitly defined because SX C++ compiler
  // produce very inefficient code otherwise...
  rowmajor_matrix_local(const rowmajor_matrix_local<T>& m) {
    val = m.val;
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
  }
  rowmajor_matrix_local<T>&
  operator=(const rowmajor_matrix_local<T>& m) {
    val = m.val;
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
    return *this;
  }
  // implicit conversion: (lvalue) vector -> rowmajor_matrix_local
  rowmajor_matrix_local(const std::vector<T>& vec) {
    set_local_num(vec.size(),1);
    val = vec; // copying since (lvalue)
  }
  // implicit conversion: (rvalue) vector -> rowmajor_matrix_local
  rowmajor_matrix_local(std::vector<T>&& vec) {
    set_local_num(vec.size(),1); // this setting must be done before the below swap
    val.swap(vec); // swapping since (rvalue)
  }
  void set_local_num(size_t r, size_t c) {
    local_num_row = r; local_num_col = c;
  }
  template <class I = size_t, class O = size_t>
  crs_matrix_local<T,I,O> to_crs();
  void debug_print();
  std::vector<T> get_row(size_t r) const;
  void save(const std::string& file) {
    std::ofstream str(file.c_str());
    str << *this;
  }
  void clear() {
    std::vector<T> tmpval; tmpval.swap(val);
    local_num_row = 0;
    local_num_col = 0;
  }
  void savebinary(const std::string&);
  rowmajor_matrix_local<T> transpose() const;
  rowmajor_matrix_local<T> pow_val(T exponent) const;
  std::vector<T> val;
  size_t local_num_row;
  size_t local_num_col;
};

template <class T>
std::vector<T> rowmajor_matrix_local<T>::get_row(size_t k) const {
  std::vector<T> r(local_num_col);
  if(k > local_num_row) throw std::runtime_error("get_row: invalid position");
  const T* valp_off = val.data() + local_num_col * k;
  T* rp = r.data();
  for(size_t i = 0; i < local_num_col; i++) rp[i] = valp_off[i];
  return r;
}

template <class T>
std::ostream& operator<<(std::ostream& str,
                         const rowmajor_matrix_local<T>& mat) {
  for(size_t r = 0; r < mat.local_num_row; r++) {
    size_t c;
    for(c = 0; c < mat.local_num_col - 1; c++) {
      str << mat.val[mat.local_num_col * r + c] << " ";
    }
    str << mat.val[mat.local_num_col * r + c] << "\n";
  }
  return str;
}

template <class T>
void rowmajor_matrix_local<T>::debug_print() {
  std::cout << "local_num_row = " << local_num_row
            << ", local_num_col = " << local_num_col
            << ", val = ";
  for(auto i: val){ std::cout << i << " "; }
  std::cout << std::endl;
}

template <class T>
void make_rowmajor_matrix_local_parseline(std::string&, std::vector<T>&,
                                          size_t&);
template <>
void make_rowmajor_matrix_local_parseline(std::string&, std::vector<double>&,
                                          size_t&);
template <>
void make_rowmajor_matrix_local_parseline(std::string&, std::vector<float>&,
                                          size_t&);
template <>
void make_rowmajor_matrix_local_parseline(std::string&, std::vector<int>&,
                                          size_t&);

template <class T>
rowmajor_matrix_local<T>
make_rowmajor_matrix_local_readstream(std::istream& str) {
  rowmajor_matrix_local<T> ret;
  std::string line;
  size_t width = 0;
  size_t lines = 0;
  while(std::getline(str,line)) {
    // width should be initializes as 0; otherwise not changed
    make_rowmajor_matrix_local_parseline<T>(line, ret.val, width);
    lines++;
  }
  ret.local_num_row = lines;
  ret.local_num_col = width;
  return ret;
}

template <class T>
std::istream& operator>>(std::istream& str,
                         rowmajor_matrix_local<T>& mat) {
  mat = make_rowmajor_matrix_local_readstream<T>(str);
  return str;
}

template <class T>
rowmajor_matrix_local<T>
make_rowmajor_matrix_local_load(const std::string& file) {
  std::ifstream str(file.c_str());
  return make_rowmajor_matrix_local_readstream<T>(str);
}

// used for making (distributed) rowmajor_matrix
template <class T>
rowmajor_matrix_local<T>
make_rowmajor_matrix_local_vectorstring(std::vector<std::string>& vs) {
  rowmajor_matrix_local<T> ret;
  size_t width = 0;
  for(size_t i = 0; i < vs.size(); i++) {
    make_rowmajor_matrix_local_parseline(vs[i], ret.val, width);
  }
  ret.local_num_row = vs.size();
  ret.local_num_col = width;
  return ret;
}


/* // use BLAS instead
template <class T>
rowmajor_matrix_local<T> operator*(const rowmajor_matrix_local<T>& a,
                                   const rowmajor_matrix_local<T>& b) {
  if(a.local_num_col != b.local_num_row)
    throw std::runtime_error("invalid size for matrix multiplication");
  size_t imax = a.local_num_row;
  size_t jmax = b.local_num_col;
  size_t kmax = a.local_num_col; // == b.local_num_row
  rowmajor_matrix_local<T> c(imax, jmax);
  const T* ap = &a.val[0];
  const T* bp = &b.val[0];
  T* cp = &c.val[0];
  // let the SX compiler detect matmul
  for(size_t i = 0; i < imax; i++) {
    for(size_t j = 0; j < jmax; j++) {
      for(size_t k = 0; k < kmax; k++) {
        //cp[i][j] += ap[i][k] * bp[k][j];
        cp[i * jmax + j] += ap[i * kmax + k] * bp[k * jmax + j];
      }
    }
  }
  return c;
}
*/

template <class T>
std::vector<T> operator*(const rowmajor_matrix_local<T>& a,
                         const std::vector<T>& b) {
  if(a.local_num_col != b.size())
    throw std::runtime_error("invalid size for matrix vector multiplication");
  size_t imax = a.local_num_row;
  size_t jmax = a.local_num_col; // == b.local_num_row
  std::vector<T> c(imax, 0);
  const T* ap = a.val.data();
  const T* bp = b.data();
  T* cp = c.data();
  for(size_t j = 0; j < jmax; j++) {
    for(size_t i = 0; i < imax; i++) {
        cp[i] += ap[i * jmax + j] * bp[j];
    }
  }
  return c;
}

template <class T>
std::vector<T> sum_of_cols(const rowmajor_matrix_local<T>& m) {
  auto nrow = m.local_num_row;
  auto ncol = m.local_num_col;
  std::vector<T> ret(nrow,0);
  T* retp = &ret[0];
  const T* matp = &m.val[0];
#if defined(_SX) || defined(__ve__)
  if (nrow > ncol) {
    for(size_t i = 0; i<nrow; i += MAT_VLEN) {
      auto range = (i + MAT_VLEN <= nrow) ? (i + MAT_VLEN) : nrow;
      for(size_t j = 0; j<ncol; ++j) {
        for (size_t k=i; k<range; ++k) {
          retp[k] += matp[k * ncol + j];
        }
      }
    }
  }
  else {
    for (size_t i=0; i<nrow; ++i) {
      for(size_t j = 0; j<ncol; ++j) {
        retp[i] += matp[i * ncol + j];
      }
    }
  }
#else
  for (size_t i=0; i<nrow; ++i) {
    for(size_t j = 0; j<ncol; ++j) {
      retp[i] += matp[i * ncol + j];
    }
  }
#endif
  return ret;
}

template <class T>
std::vector<T> squared_sum_of_cols(const rowmajor_matrix_local<T>& m) {
  auto nrow = m.local_num_row;
  auto ncol = m.local_num_col;
  std::vector<T> ret(nrow,0);
  T* retp = &ret[0];
  const T* matp = &m.val[0];
#if defined(_SX) || defined(__ve__)
  if (nrow > ncol) {
    for(size_t i = 0; i<nrow; i += MAT_VLEN) {
      auto range = (i + MAT_VLEN <= nrow) ? (i + MAT_VLEN) : nrow;
      for(size_t j = 0; j<ncol; ++j) {
        for (size_t k=i; k<range; ++k) {
          retp[k] += (matp[k * ncol + j] * matp[k * ncol + j]);
        }
      }
    }
  }
  else {
    for (size_t i=0; i<nrow; ++i) {
      for(size_t j = 0; j<ncol; ++j) {
        retp[i] += (matp[i * ncol + j] * matp[i * ncol + j]); 
      }
    }
  }
#else
  for (size_t i=0; i<nrow; ++i) {
    for(size_t j = 0; j<ncol; ++j) {
      retp[i] += (matp[i * ncol + j] * matp[i * ncol + j]);
    }
  }
#endif
  return ret;
}

template <class T>
std::vector<T> sum_of_rows(const rowmajor_matrix_local<T>& m) {
  auto nrow = m.local_num_row;
  auto ncol = m.local_num_col;
  std::vector<T> ret(ncol,0);
  T* retp = &ret[0];
  const T* matp = &m.val[0];
#if defined(_SX) || defined(__ve__)
  if (nrow > ncol) {
    for(size_t i = 0; i<nrow; i += MAT_VLEN) {
      auto range = (i + MAT_VLEN <= nrow) ? (i + MAT_VLEN) : nrow;
      for(size_t j =0; j<ncol; ++j) {
        for (size_t k=i; k<range; ++k) {
          retp[j] += matp[k * ncol + j];
        }
      }
    }
  }
  else {
    for (size_t i=0; i<nrow; ++i) {
      for(size_t j =0; j<ncol; ++j) {
        retp[j] += matp[i * ncol + j];
      }
    }
  }
#else
  for (size_t i=0; i<nrow; ++i) {
    for(size_t j =0; j<ncol; ++j) {
      retp[j] += matp[i * ncol + j];
    }
  }
#endif
  return ret;
}

template <class T>
std::vector<T> squared_sum_of_rows(const rowmajor_matrix_local<T>& m) {
  auto nrow = m.local_num_row;
  auto ncol = m.local_num_col;
  std::vector<T> ret(ncol,0);
  T* retp = &ret[0];
  const T* matp = &m.val[0];
#if defined(_SX) || defined(__ve__)
  if (nrow > ncol) {
    for(size_t i = 0; i<nrow; i += MAT_VLEN) {
      auto range = (i + MAT_VLEN <= nrow) ? (i + MAT_VLEN) : nrow;
      for(size_t j =0; j<ncol; ++j) {
        for (size_t k=i; k<range; ++k) {
          retp[j] += (matp[k * ncol + j] * matp[k * ncol + j]);
        }
      }
    }
  }
  else {
    for (size_t i=0; i<nrow; ++i) {
      for(size_t j =0; j<ncol; ++j) {
        retp[j] += (matp[i * ncol + j] * matp[i * ncol + j]);
      }
    }
  }
#else
  for (size_t i=0; i<nrow; ++i) {
    for(size_t j =0; j<ncol; ++j) {
      retp[j] += (matp[i * ncol + j] * matp[i * ncol + j]);
    }
  }
#endif
  return ret;
}

template <class T>
rowmajor_matrix_local<T> operator+(const rowmajor_matrix_local<T>& a,
                                   const rowmajor_matrix_local<T>& b) {
  if(a.local_num_row != b.local_num_row || a.local_num_col != b.local_num_col)
    throw std::runtime_error("invalid size for matrix addition");
    
  size_t imax = a.local_num_row;
  size_t jmax = a.local_num_col;
  rowmajor_matrix_local<T> c(imax, jmax);
  auto* ap = a.val.data();
  auto* bp = b.val.data();
  auto* cp = c.val.data();
  for(size_t i = 0; i < imax; i++) {
    for(size_t j = 0; j < jmax; j++) {
      cp[i * jmax + j] = ap[i * jmax + j] + bp[i * jmax + j];
    }
  } 
  return c; 
}

template <class T>
rowmajor_matrix_local<T> operator-(const rowmajor_matrix_local<T>& a,
                                   const rowmajor_matrix_local<T>& b) {
  if(a.local_num_row != b.local_num_row || a.local_num_col != b.local_num_col)
    throw std::runtime_error("invalid size for matrix addition");
    
  size_t imax = a.local_num_row;
  size_t jmax = a.local_num_col;
  rowmajor_matrix_local<T> c(imax, jmax);
  auto* ap = a.val.data();
  auto* bp = b.val.data();
  auto* cp = c.val.data();
  for(size_t i = 0; i < imax; i++) {
    for(size_t j = 0; j < jmax; j++) {
      cp[i * jmax + j] = ap[i * jmax + j] - bp[i * jmax + j];
    }
  } 
  return c; 
}

template <class T>
rowmajor_matrix_local<T> rowmajor_matrix_local<T>::transpose() const {
  rowmajor_matrix_local<T> ret(local_num_col, local_num_row);
  T* retp = &ret.val[0];
  const T* valp = &val[0];
  for(size_t i = 0; i < local_num_row; i++) {
    for(size_t j = 0; j < local_num_col; j++) {
      retp[j * local_num_row + i] = valp[i * local_num_col + j];
    }
  }
  return ret;
}

template <class T>
rowmajor_matrix_local<T> rowmajor_matrix_local<T>::pow_val(T exponent) const {
  rowmajor_matrix_local<T> ret(*this);
  auto* valp = ret.val.data();
  auto valsize = ret.val.size();
  for (size_t i = 0; i < valsize; i++) {
    valp[i] = std::pow(valp[i], exponent);
  }
  return ret;
}

/*
  The directory should contain following files:
  - val: big endian binary data file that contains values of the matrix 
         in row major order
  - nums: text file that contains num_row in the first line and
          num_col in the second line
 */
template <class T>
rowmajor_matrix_local<T>
make_rowmajor_matrix_local_loadbinary(const std::string& input) {
  std::string valfile = input + "/val";
  std::string numsfile = input + "/nums";
  std::ifstream numstr;
  numstr.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  numstr.open(numsfile.c_str());
  size_t num_row, num_col;
  numstr >> num_row >> num_col;
  rowmajor_matrix_local<T> ret;
  ret.set_local_num(num_row, num_col);
  auto tmp = loadbinary<T>(valfile);
  ret.val.swap(tmp);
  return ret;
}

template <class T>
void rowmajor_matrix_local<T>::savebinary(const std::string& dir) {
  struct stat sb;
  if(stat(dir.c_str(), &sb) != 0) { // no file/directory
    mode_t mode = S_IRWXU | S_IRWXG | S_IRWXO; // man 2 stat
    if(mkdir(dir.c_str(), mode) != 0) {
      perror("mkdir failed:");
      throw std::runtime_error("mkdir failed");
    }
  } else if(!S_ISDIR(sb.st_mode)) {
    throw std::runtime_error(dir + " is not a directory");
  }
  std::string valfile = dir + "/val";
  std::string numsfile = dir + "/nums";
  std::ofstream numstr;
  numstr.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  numstr.open(numsfile.c_str());
  numstr << local_num_row << "\n" << local_num_col << std::endl;
  frovedis::savebinary(val, valfile);
}

std::vector<size_t> get_block_sizes(size_t num_row, size_t wsize);

// scattering a vector into a number of requested pieces
template <class T>
std::vector<std::vector<T>>
get_scattered_vectors(std::vector<T>& vec, 
                      size_t nrow, size_t ncol, size_t wsize) {
  auto rows = get_block_sizes(nrow, wsize);
  std::vector<size_t> sizevec(wsize);
  auto sizevecp = sizevec.data();
  auto rowsp = rows.data();
  for(size_t i = 0; i < wsize; i++) {
    sizevecp[i] = rowsp[i] * ncol;
  }
  std::vector<std::vector<T>> src2(wsize);
  const T* srcp = &vec[0];
  for(size_t i = 0; i < wsize; i++) {
    src2[i].resize(sizevecp[i]);
    for(size_t j = 0; j < sizevecp[i]; j++) {
      src2[i][j] = srcp[j];
    }
    srcp += sizevecp[i];
  }
  return src2;
}

// scattering the local matrix into given number of chunks
template <class T>
std::vector<rowmajor_matrix_local<T>>
get_scattered_rowmajor_matrices(rowmajor_matrix_local<T>& m,
                                size_t wsize) {
  auto nrow = m.local_num_row;
  auto ncol = m.local_num_col;
  auto rows = get_block_sizes(nrow, wsize);
  auto src2 = get_scattered_vectors(m.val,nrow,ncol,wsize);
  std::vector<rowmajor_matrix_local<T>> ret(wsize);
  for(size_t i=0; i < wsize; ++i) {
    rowmajor_matrix_local<T> tmp;
    tmp.val.swap(src2[i]);
    tmp.set_local_num(rows[i],ncol);
    ret[i] = tmp;
  }
  return ret;
}

}
#endif
