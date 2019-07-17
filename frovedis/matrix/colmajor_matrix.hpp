#ifndef COLMAJOR_MATRIX_HPP
#define COLMAJOR_MATRIX_HPP

#include "rowmajor_matrix.hpp"

namespace frovedis {

template <class T>
struct colmajor_matrix_local {
  colmajor_matrix_local(){}
  colmajor_matrix_local(size_t r, size_t c)
    : local_num_row(r), local_num_col(c) {
    val.resize(r*c);
  }
  colmajor_matrix_local(colmajor_matrix_local<T>&& m) {
    val.swap(m.val);
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
  }
  colmajor_matrix_local(const colmajor_matrix_local<T>& m) {
    val = m.val;
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
  }
  colmajor_matrix_local(const rowmajor_matrix_local<T>& m) {
    if(m.local_num_col > 1) {
      auto tmp = m.transpose();
      val.swap(tmp.val);
    } else {
      val = m.val; // 'm' is lvalue vector (so just copied)
    }
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
  }
  colmajor_matrix_local(rowmajor_matrix_local<T>&& m) {
    if(m.local_num_col > 1) {
      auto tmp = m.transpose();
      val.swap(tmp.val);
    } else {
      val.swap(m.val); // 'm' is rvalue vector (so just moved)
    }
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
  }
  colmajor_matrix_local<T>&
  operator=(const colmajor_matrix_local<T>& m) {
    val = m.val;
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
    return *this;
  }
  colmajor_matrix_local<T>&
  operator=(colmajor_matrix_local<T>&& m) {
    val.swap(m.val);
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
    return *this;
  }
  rowmajor_matrix_local<T> to_rowmajor() {
    rowmajor_matrix_local<T> ret;
    if(local_num_col > 1) {
      auto tmp = transpose();
      ret.val.swap(tmp.val);
    } else {
      ret.val = val;
    }
    ret.local_num_row = local_num_row;
    ret.local_num_col = local_num_col;
    return ret;
  }

  void clear() {
    std::vector<T> tmpval; tmpval.swap(val);
    local_num_row = 0;
    local_num_col = 0;
  }
  void debug_print();
  size_t get_nrows() { return local_num_row; }
  size_t get_ncols() { return local_num_col; }
  colmajor_matrix_local<T> transpose() const;
  std::vector<T> val;
  size_t local_num_row;
  size_t local_num_col;
};

template <class T>
void colmajor_matrix_local<T>::debug_print() {
  std::cout << "local_num_row = " << local_num_row
            << ", local_num_col = " << local_num_col
            << ", val = ";
  for(auto i: val){ std::cout << i << " "; }
  std::cout << std::endl;
}

/*
template <class T>
colmajor_matrix_local<T> operator*(const colmajor_matrix_local<T>& a,
                                   const colmajor_matrix_local<T>& b) {
  if(a.local_num_col != b.local_num_row)
    throw std::runtime_error("invalid size for matrix multiplication");
  size_t imax = a.local_num_row;
  size_t jmax = b.local_num_col;
  size_t kmax = a.local_num_col; // == b.local_num_row
  colmajor_matrix_local<T> c(imax, jmax);
  const T* ap = &a.val[0];
  const T* bp = &b.val[0];
  T* cp = &c.val[0];
  // let the SX compiler detect matmul
  for(size_t i = 0; i < imax; i++) {
    for(size_t j = 0; j < jmax; j++) {
      for(size_t k = 0; k < kmax; k++) {
        //cp[i][j] += ap[i][k] * bp[k][j];
        cp[i + imax * j] += ap[i + imax * k] * bp[k + kmax * j];
      }
    }
  }
  return c;
}
*/

template <class T>
colmajor_matrix_local<T> colmajor_matrix_local<T>::transpose() const {
  colmajor_matrix_local<T> ret(local_num_col, local_num_row);
  T* retp = &ret.val[0];
  const T* valp = &val[0];
  for(size_t i = 0; i < local_num_row; i++) {
    for(size_t j = 0; j < local_num_col; j++) {
      retp[j + local_num_col * i] = valp[i + local_num_row * j];
    }
  }
  return ret;
}


}
#endif
