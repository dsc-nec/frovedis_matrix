#ifndef JDS_CRS_HYBRID_HPP
#define JDS_CRS_HYBRID_HPP

#include "jds_matrix.hpp"

namespace frovedis {

template <class T, class I = size_t, class O = size_t, class P = size_t>
struct jds_crs_hybrid_local {
  jds_crs_hybrid_local() {}
  jds_crs_hybrid_local(const jds_crs_hybrid_local<T,I,O,P>&) = default;
  jds_crs_hybrid_local<T,I,O,P>&
  operator=(const jds_crs_hybrid_local<T,I,O,P>&) = default;
  jds_crs_hybrid_local(jds_crs_hybrid_local<T,I,O,P>&& m) = default;
  jds_crs_hybrid_local<T,I,O,P>&
  operator=(jds_crs_hybrid_local<T,I,O,P>&& m) = default;
  jds_crs_hybrid_local(const crs_matrix_local<T,I,O>&, size_t t = 256);
  jds_matrix_local<T,I,O,P> jds;
  crs_matrix_local<T,I,O> crs;
  size_t local_num_col;
  size_t local_num_row;
  void debug_print() const {
    std::cout << "jds part:" << std::endl;
    jds.debug_print();
    std::cout << "crs part:" << std::endl;
    crs.debug_print();
  }
  void clear() {
    jds.clear();
    crs.clear();
    local_num_row = 0;
    local_num_col = 0;
  }
};

template <class T, class I, class O, class P>
jds_crs_hybrid_local<T,I,O,P>::
jds_crs_hybrid_local(const crs_matrix_local<T,I,O>& m, size_t threashold) {
  local_num_col = m.local_num_col;
  local_num_row = m.local_num_row;
  jds.local_num_col = m.local_num_col;
  jds.local_num_row = m.local_num_row;
  if(m.val.size() == 0) {
    crs = m;
    return;
  } else if(local_num_row <= threashold) {
    crs = m;
    jds.perm.resize(m.local_num_row);
    auto jdspermp = jds.perm.data();
    for(size_t i = 0; i < m.local_num_row; i++) {
      jdspermp[i] = i;
    }
    return;
  } else if (threashold == 0) {
    jds = jds_matrix_local<T,I,O,P>(m);
    crs.local_num_col = m.local_num_col;
    crs.local_num_row = 0;
    return;
  } else {
#if defined(_SX) || defined(__ve__)
    std::vector<O> perm_tmp_key(local_num_row);
    std::vector<P> perm_tmp_val(local_num_row);
    std::vector<O> perm_tmp_first(local_num_row);
    auto perm_tmp_keyp = perm_tmp_key.data();
    auto perm_tmp_valp = perm_tmp_val.data();
    auto perm_tmp_firstp = perm_tmp_first.data();
    auto moffpp = m.off.data();
    for(size_t i = 0; i < local_num_row; i++) {
      perm_tmp_keyp[i] = moffpp[i+1] - moffpp[i];
      perm_tmp_valp[i] = i;
    }
    radix_sort(&perm_tmp_key[0], &perm_tmp_val[0], local_num_row);
    jds.perm.resize(local_num_row);
    auto jdspermp = jds.perm.data();
    for(size_t i = 0; i < local_num_row; i++) {
      perm_tmp_firstp[i] = perm_tmp_keyp[local_num_row - i - 1];
      jdspermp[i] = perm_tmp_valp[local_num_row - i - 1];
    }
    size_t crs_start_off = perm_tmp_first[threashold];
    size_t crs_num_row = threashold;
    for(;crs_num_row != 0; crs_num_row--)
      if(perm_tmp_firstp[crs_num_row - 1] != crs_start_off) break;
    size_t crs_size = 0;
    for(size_t i = 0; i < crs_num_row; i++) {
      crs_size += (perm_tmp_firstp[i] - crs_start_off);
    }
#else 
    std::vector<std::pair<O, P>> perm_tmp(local_num_row);
    for(size_t i = 0; i < local_num_row; i++) {
      perm_tmp[i].first = m.off[i+1] - m.off[i];
      perm_tmp[i].second = i;
    }
    std::sort(perm_tmp.begin(), perm_tmp.end(),
              std::greater<std::pair<O, P>>());
    jds.perm.resize(local_num_row);
    for(size_t i = 0; i < local_num_row; i++) {
      jds.perm[i] = perm_tmp[i].second;
    }
    size_t crs_start_off = perm_tmp[threashold].first;
    size_t crs_num_row = threashold;
    for(;crs_num_row != 0; crs_num_row--)
      if(perm_tmp[crs_num_row - 1].first != crs_start_off) break;
    size_t crs_size = 0;
    for(size_t i = 0; i < crs_num_row; i++) {
      crs_size += (perm_tmp[i].first - crs_start_off);
    }
#endif
    jds.off.reserve(crs_start_off+1); // jds.off[0] is already 0 by ctor
    jds.val.resize(m.val.size() - crs_size);
    jds.idx.resize(m.idx.size() - crs_size);
    crs.val.resize(crs_size);
    crs.idx.resize(crs_size);
    crs.off.reserve(crs_num_row + 1);
    crs.local_num_col = m.local_num_col;
    crs.local_num_row = crs_num_row;

    P* permp = &jds.perm[0];
    const T* mvalp = &m.val[0];
    T* jdsvalp = &jds.val[0];
    const I* midxp = &m.idx[0];
    I* jdsidxp = &jds.idx[0];
    const O* moffp = &m.off[0];
    size_t jds_col = 0;
    size_t to_store = 0;
    for(size_t row_max = local_num_row; row_max != crs_num_row; row_max--) {
#if defined(_SX) || defined(__ve__)
      O num_iter = perm_tmp_first[row_max-1] - jds_col;
#else
      O num_iter = perm_tmp[row_max-1].first - jds_col;
#endif
      for(size_t i = 0; i < num_iter; i++, jds_col++) {
        for(size_t r = 0; r < row_max; r++, to_store++) {
          jdsvalp[to_store] = mvalp[moffp[permp[r]] + jds_col];
          jdsidxp[to_store] = midxp[moffp[permp[r]] + jds_col];
        }
        jds.off.push_back(to_store);
      }
    }
    to_store = 0;
    T* crsvalp = &crs.val[0];
    I* crsidxp = &crs.idx[0];
    for(size_t r = 0; r < crs_num_row; r++) {
      for(size_t crs_col = moffp[permp[r]] + crs_start_off; 
          crs_col < moffp[permp[r]+1]; crs_col++, to_store++) {
        crsvalp[to_store] = mvalp[crs_col];
        crsidxp[to_store] = midxp[crs_col];
      }
      crs.off.push_back(to_store);
    }
  }
}

template <class T, class I, class O, class P>
void jds_crs_hybrid_spmv_impl(const jds_crs_hybrid_local<T,I,O,P>& mat,
                              T* retp, const T* vp) {
  std::vector<T> crspart(mat.crs.local_num_row);
  T* crspartp = crspart.data();
  jds_matrix_spmv_impl(mat.jds, retp, vp);
  crs_matrix_spmv_impl(mat.crs, crspartp, vp);
  const P* permp = &mat.jds.perm[0];
#pragma cdir nodep
#pragma _NEC ivdep
  for(size_t i = 0; i < crspart.size(); i++) {
    retp[permp[i]] += crspartp[i];
  }
}

template <class T, class I, class O, class P>
std::vector<T> operator*(const jds_crs_hybrid_local<T,I,O,P>& mat,
                         const std::vector<T>& v) {
  std::vector<T> ret(mat.local_num_row);
  if(mat.local_num_col != v.size())
    throw std::runtime_error("operator*: size of vector does not match");
  jds_crs_hybrid_spmv_impl(mat, ret.data(), v.data());
  return ret;
}

template <class T, class I, class O, class P>
void jds_crs_hybrid_spmm_impl(const jds_crs_hybrid_local<T,I,O,P>& mat,
                              T* retvalp, const T* vvalp,
                              size_t v_local_num_col) {
  rowmajor_matrix_local<T> crspart(mat.local_num_row, v_local_num_col);
  T* crspartvalp = crspart.val.data();
  jds_matrix_spmm_impl(mat.jds, retvalp, vvalp, v_local_num_col);
  crs_matrix_spmm_impl(mat.crs, crspartvalp, vvalp, v_local_num_col);
  const P* permp = &mat.jds.perm[0];
  auto num_row = crspart.local_num_row;
  auto num_col = crspart.local_num_col;
  // in jds_crs_hybrid, crspart.local_num_row < threshold (e.g. 256)
  for(size_t r = 0; r < num_row; r++) {
#pragma cdir nodep
#pragma _NEC ivdep
    for(size_t c = 0; c < num_col; c++) {
      retvalp[permp[r] * num_col + c] += crspartvalp[r * num_col + c];
    }
  }
}

template <class T, class I, class O, class P>
rowmajor_matrix_local<T> operator*(const jds_crs_hybrid_local<T,I,O,P>& mat,
                                   const rowmajor_matrix_local<T>& v) {
  rowmajor_matrix_local<T> ret(mat.local_num_row, v.local_num_col);
  T* retvalp = &ret.val[0];
  const T* vvalp = &v.val[0];
  jds_crs_hybrid_spmm_impl(mat, retvalp, vvalp, v.local_num_col);
  return ret;
}


}

#endif

