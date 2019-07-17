#ifndef CCS_MATRIX_HPP
#define CCS_MATRIX_HPP

#include "crs_matrix.hpp"

#define CCS_VLEN 256

namespace frovedis {

template <class T, class I = size_t, class O = size_t>
struct ccs_matrix_local {
  ccs_matrix_local() {}
  ccs_matrix_local(ccs_matrix_local<T,I,O>&& ccs) {
    val.swap(ccs.val);
    idx.swap(ccs.idx);
    off.swap(ccs.off);
    local_num_row = ccs.local_num_row;
    local_num_col = ccs.local_num_col;
  }
  ccs_matrix_local(crs_matrix_local<T,I,O>& crs) {
    auto tmp = crs.transpose();
    val.swap(tmp.val);
    idx.swap(tmp.idx);
    off.swap(tmp.off);
    local_num_row = crs.local_num_row;
    local_num_col = crs.local_num_col;
  }
  ccs_matrix_local(const ccs_matrix_local<T,I,O>& m) {
    val = m.val;
    idx = m.idx;
    off = m.off;
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
  }
  ccs_matrix_local<T,I,O>& operator=(const ccs_matrix_local<T,I,O>& m) {
    val = m.val;
    idx = m.idx;
    off = m.off;
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
    return *this;
  }
  ccs_matrix_local<T,I,O>& operator=(ccs_matrix_local<T,I,O>&& m) {
    val.swap(m.val);
    idx.swap(m.idx);
    off.swap(m.off);
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
    return *this;
  }
  std::vector<T> val;
  std::vector<I> idx;
  std::vector<O> off;
  size_t local_num_row;
  size_t local_num_col;
  ccs_matrix_local<T,I,O> transpose();
  crs_matrix_local<T,I,O> to_crs() {
    auto tmp = transpose();
    crs_matrix_local<T,I,O> ret;
    ret.val.swap(tmp.val);
    ret.idx.swap(tmp.idx);
    ret.off.swap(tmp.off);
    ret.local_num_row = local_num_row;
    ret.local_num_col = local_num_col;
    return ret;
  }
  void set_local_num(size_t r){
    local_num_row = r;
    local_num_col = off.size() - 1;
  }
  void debug_print() const {
    std::cout << "local_num_row = " << local_num_row
              << ", local_num_col = " << local_num_col
              << std::endl;
    std::cout << "val : ";
    for(auto i: val) std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "idx : ";
    for(auto i: idx) std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "off : ";
    for(auto i: off) std::cout << i << " ";
    std::cout << std::endl;
  }
};

template <class T, class I, class O>
ccs_matrix_local<T,I,O> ccs_matrix_local<T,I,O>::transpose() {
  ccs_matrix_local<T,I,O> ret;
  transpose_compressed_matrix(val, idx, off, ret.val, ret.idx, ret.off,
                              local_num_col, local_num_row);
  ret.local_num_col = local_num_row;
  ret.local_num_row = local_num_col;
  return ret;
}

template <class T, class I, class O>
ccs_matrix_local<T,I,O> crs2ccs(crs_matrix_local<T,I,O>& crs) {
  return ccs_matrix_local<T,I,O>(crs);
}

template <class T, class I, class O>
crs_matrix_local<T,I,O> ccs2crs(ccs_matrix_local<T,I,O>& ccs) {
  return ccs.to_crs();
}

template <class T, class I, class O>
std::vector<T> operator*(const ccs_matrix_local<T,I,O>& mat,
                         const std::vector<T>& v) {
  std::vector<T> ret(mat.local_num_row);
  T* retp = &ret[0];
  const T* vp = &v[0];
  const T* valp = &mat.val[0];
  const I* idxp = &mat.idx[0];
  const O* offp = &mat.off[0];
  for(size_t c = 0; c < mat.local_num_col; c++) {
#pragma cdir on_adb(retp)
#pragma cdir nodep
#pragma _NEC ivdep
#pragma _NEC vovertake
#pragma _NEC vob
    for(O r = offp[c]; r < offp[c+1]; r++) {
      retp[idxp[r]] = retp[idxp[r]] + valp[r] * vp[c];
    }
  }
  return ret;
}

template <class T, class I, class O>
rowmajor_matrix_local<T> operator*(ccs_matrix_local<T,I,O>& mat,
                                   rowmajor_matrix_local<T>& v) {
  rowmajor_matrix_local<T> ret(mat.local_num_row, v.local_num_col);
  T* retvalp = &ret.val[0];
  T* vvalp = &v.val[0];
  T* valp = &mat.val[0];
  I* idxp = &mat.idx[0];
  O* offp = &mat.off[0];
  size_t num_col = v.local_num_col;
  for(size_t c = 0; c < mat.local_num_col; c++) {
    O r = offp[c];
    for(; r + CCS_VLEN < offp[c+1]; r += CCS_VLEN) {
      for(size_t mc = 0; mc < num_col; mc++) {
#pragma cdir nodep
#pragma _NEC ivdep
#pragma cdir on_adb(retvalp)
        for(O i = 0; i < CCS_VLEN; i++) {
          retvalp[idxp[r+i] * num_col + mc] +=
            valp[r+i] * vvalp[c * num_col + mc];
        }
      }
    }
    for(size_t mc = 0; mc < num_col; mc++) {
#pragma cdir nodep
#pragma _NEC ivdep
#pragma cdir on_adb(retvalp)
      for(O i = 0; r + i < offp[c+1]; i++) {
        retvalp[idxp[r+i] * num_col + mc] +=
          valp[r+i] * vvalp[c * num_col + mc];
      }
    }
    r = offp[c+1];
  }
  return ret;
}

template <class T, class I, class O>
std::vector<T> call_ccs_mv(const ccs_matrix_local<T,I,O>& mat,
                           const std::vector<T>& v) {
  return mat * v;
}

}
#endif
