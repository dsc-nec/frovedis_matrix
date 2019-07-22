#ifndef SPGEMM_HPP
#define SPGEMM_HPP

#include "ccs_matrix.hpp"
#include "spmspv.hpp"

#define SPGEMM_VLEN 256

namespace frovedis {

enum spgemm_type {
  esc,
  block_esc,
  block_esc_dense,
};

template <class T>
size_t max_bits(T a) {
  if(a < 0) throw std::runtime_error("max_bits is only for positive value");
  size_t i = 0;
  size_t max = sizeof(T) * 8;
  for(; i < max; i++) {
    if(a == 0) break;
    else a >>= 1;
  }
  return i;
}

// avoid loop raking because num_merged_vec is upto 256...
// (it is possible to do loop raking for each vec, but there might be 
// load balancing problem...)
template <class I, class O>
void add_column_idx(I* mulout_idx, O* merged_total_nnzp,
                    size_t merged_count, size_t max_idx_bits) {
  for(size_t i = 0; i < merged_count; i++) {
    for(size_t j = 0; j < merged_total_nnzp[i]; j++) {
      mulout_idx[j] += (i << max_idx_bits);
    }
    mulout_idx += merged_total_nnzp[i];
  }
}

template <class I, class O>
void extract_column_idx(size_t* idxtmp, int* extracted, I* idx,
                        O size, size_t bits) {
  size_t mask = (static_cast<size_t>(1) << bits) - 1;
  for(size_t i = 0; i < size; i++) {
    extracted[i] = idxtmp[i] >> bits;
    idx[i] = idxtmp[i] & mask;
  }
}

/* same as set_separate at dataframe/set_operations.hpp
   copied to avoid dependency... */
template <class T>
std::vector<size_t> my_set_separate(const std::vector<T>& key) {
  size_t size = key.size();
  if(size == 0) {return std::vector<size_t>(1);} 
  int valid[SPGEMM_VLEN];
  for(int i = 0; i < SPGEMM_VLEN; i++) valid[i] = true;
  size_t each = ceil_div(size, size_t(SPGEMM_VLEN));
  if(each % 2 == 0) each++;
  size_t key_idx[SPGEMM_VLEN];
  size_t key_idx_stop[SPGEMM_VLEN];
  size_t out_idx[SPGEMM_VLEN];
  size_t out_idx_save[SPGEMM_VLEN];
  T current_key[SPGEMM_VLEN];
  std::vector<size_t> out;
  out.resize(size);
  size_t* outp = &out[0];
  const T* keyp = &key[0];
  if(size > 0) {
    key_idx[0] = 1;
    outp[0] = 0;
    out_idx[0] = 1;
    out_idx_save[0] = 0;
    current_key[0] = keyp[0];
  } else {
    valid[0] = false;
    key_idx[0] = size;
    out_idx[0] = size;
    out_idx_save[0] = size;
  }
  for(int i = 1; i < SPGEMM_VLEN; i++) {
    size_t pos = each * i;
    if(pos < size) {
      key_idx[i] = pos;
      out_idx[i] = pos;
      out_idx_save[i] = pos;
      current_key[i] = keyp[pos-1];
    } else {
      valid[i] = false;
      key_idx[i] = size;
      out_idx[i] = size;
      out_idx_save[i] = size;
    }
  }
  for(int i = 0; i < SPGEMM_VLEN - 1; i++) {
    key_idx_stop[i] = key_idx[i + 1];
  }
  key_idx_stop[SPGEMM_VLEN-1] = size;
  // idx 0 is manually advanced; need to be checked
  if(key_idx[0] == key_idx_stop[0]) valid[0] = false;
  size_t max_size = 0;
  for(int i = 0; i < SPGEMM_VLEN; i++) {
    auto current = key_idx_stop[i] - key_idx[i];
    if(max_size < current) max_size = current;
  }
  for(size_t j = 0; j < max_size; j++) {
#pragma cdir nodep
#pragma _NEC ivdep
    for(int i = 0; i < SPGEMM_VLEN; i++) {
      if(valid[i]) {
        auto keyval = keyp[key_idx[i]];
        if(keyval != current_key[i]) {
          outp[out_idx[i]++] = key_idx[i];
          current_key[i] = keyval;
        }
        key_idx[i]++;
        if(key_idx[i] == key_idx_stop[i]) {valid[i] = false;}
      }
    }
  }
  size_t total = 0;
  for(size_t i = 0; i < SPGEMM_VLEN; i++) {
    total += out_idx[i] - out_idx_save[i];
  }
  std::vector<size_t> ret(total+1);
  size_t* retp = &ret[0];
  size_t current = 0;
  for(size_t i = 0; i < SPGEMM_VLEN; i++) {
    for(size_t j = 0; j < out_idx[i] - out_idx_save[i]; j++) {
      retp[current + j] = out[out_idx_save[i] + j];
    }
    current += out_idx[i] - out_idx_save[i];
  }
  retp[current] = size;
  return ret;
}

template <class T, class I, class O>
void spgemm_esc_helper(T* mat_valp, I* mat_idxp, O* mat_offp,
                       T* sv_valp, I* sv_idxp, size_t sv_size,
                       O total_nnz, O* nnzp, O* pfx_sum_nnz,
                       O* merged_total_nnzp, size_t merged_count,
                       size_t max_idx_bits, size_t lnum_row,
                       std::vector<T>& retval, std::vector<I>& retidx,
                       std::vector<O>& retoff) {
  time_spent t(DEBUG);
  if(sv_size == 0) return;
  std::vector<T> mulout_val(total_nnz);
  std::vector<size_t> mulout_idx(total_nnz); // to hold enough bits
  auto mulout_valp = mulout_val.data();
  auto mulout_idxp = mulout_idx.data();
  spmspv_mul(mat_valp, mat_idxp, mat_offp, sv_valp, sv_idxp, sv_size,
             nnzp, pfx_sum_nnz, total_nnz,
             mulout_valp, mulout_idxp);
  t.show("spmspv_mul: ");
  add_column_idx(mulout_idxp, merged_total_nnzp, 
                 merged_count, max_idx_bits);
  t.show("add_column_idx: ");
  auto used_bits = max_bits(merged_count-1);
  auto max_key_size = ceil_div(max_idx_bits + used_bits, size_t(8));
  radix_sort_impl(mulout_idxp, mulout_valp, total_nnz, max_key_size);
  t.show("radix_sort: ");
  std::vector<size_t> retidxtmp;
  groupby_sum(mulout_idxp, mulout_valp, total_nnz, retidxtmp, retval);
  t.show("groupby_sum: ");
  std::vector<int> extracted_idx(retidxtmp.size());
  retidx.resize(retidxtmp.size());
  auto extracted_idxp = extracted_idx.data();
  extract_column_idx(retidxtmp.data(), extracted_idxp, retidx.data(),
                     retidxtmp.size(), max_idx_bits);
  t.show("extract_column_idx: ");
  auto separated = my_set_separate(extracted_idx);
  t.show("set_separate: ");
  // might be smaller than merged_count if sparse vector is zero
  auto separated_size = separated.size();
  retoff.resize(merged_count+1);
  std::vector<O> retofftmp(merged_count);
  auto retofftmpp = retofftmp.data();
  auto separatedp = separated.data();
  auto retoffp = retoff.data();
#pragma _NEC ivdep
#pragma _NEC vovertake
#pragma _NEC vob
  for(size_t i = 0; i < separated_size - 1; i++) {
    retofftmpp[extracted_idxp[separatedp[i]]] =
      separatedp[i+1] - separatedp[i];
  }
  prefix_sum(retofftmpp, retoffp+1, merged_count);
  t.show("prefix_sum: ");
}

template <class T, class I, class O>
void spgemm_esc(T* lval, I* lidx, O* loff,
                size_t lnnz, size_t lnum_row, size_t lnum_col,
                T* rval, I* ridx, O* roff,
                size_t rnnz, size_t rnum_col,
                std::vector<T>& retval, std::vector<I>& retidx,
                std::vector<O>& retoff) {
  if(rnnz == 0) return;
  size_t max_idx_bits = max_bits(lnum_row);

  time_spent t(DEBUG);
  std::vector<O> merged_interim_nnz(rnnz);
  auto merged_interim_nnzp = merged_interim_nnz.data();
  std::vector<O> each_column_nnz(rnum_col);
  auto each_column_nnzp =  each_column_nnz.data();
  for(size_t rc = 0; rc < rnum_col; rc++) { // rc: right column
    auto crnt_ridx = ridx + roff[rc];
    auto crnt_nnz = roff[rc+1] - roff[rc];
    for(size_t i = 0; i < crnt_nnz; i++) {
      merged_interim_nnzp[i] = loff[crnt_ridx[i]+1] - loff[crnt_ridx[i]];
    }
    for(size_t i = 0; i < crnt_nnz; i++) {
      each_column_nnzp[rc] += merged_interim_nnzp[i];
    }
    merged_interim_nnzp += crnt_nnz;
  }
  t.show("merged_interim_nnz & each_column_nnz: ");

  auto pfx_sum_merged_interim_nnz = prefix_sum(merged_interim_nnz);
  auto total_interim_nnz = pfx_sum_merged_interim_nnz[rnnz - 1];

  t.show(std::string("prefix sum, total_interim_nnz = ")
         + std::to_string(total_interim_nnz) + ", time = ");
  
  spgemm_esc_helper(lval, lidx, loff, rval, ridx, rnnz, total_interim_nnz,
                    merged_interim_nnz.data(),
                    pfx_sum_merged_interim_nnz.data(),
                    each_column_nnzp, rnum_col,
                    max_idx_bits, lnum_row, retval, retidx, retoff);
}

template <class T, class I, class O>
sparse_vector<T,I>
spgemm_helper_merged(T* mat_valp, I* mat_idxp, O* mat_offp,
                     T* sv_valp, I* sv_idxp, O sv_size,
                     O total_nnz, O* nnzp, O* pfx_sum_nnz,
                     O* merged_total_nnzp, size_t merged_count,
                     size_t max_idx_bits, O* merged_offp) {
  sparse_vector<T,I> ret;
  if(sv_size == 0) return ret;
  std::vector<T> mulout_val(total_nnz);
  std::vector<size_t> mulout_idx(total_nnz);
  auto mulout_valp = mulout_val.data();
  auto mulout_idxp = mulout_idx.data();
  spmspv_mul(mat_valp, mat_idxp, mat_offp, sv_valp, sv_idxp, sv_size,
             nnzp, pfx_sum_nnz, total_nnz,
             mulout_valp, mulout_idxp);
  add_column_idx(mulout_idxp, merged_total_nnzp,
                 merged_count, max_idx_bits);
  auto used_bits = max_bits(merged_count-1);
  auto max_key_size = ceil_div(max_idx_bits + used_bits, size_t(8));
  radix_sort_impl(mulout_idxp, mulout_valp, total_nnz, max_key_size);
  std::vector<size_t> retidxtmp;
  groupby_sum(mulout_idxp, mulout_valp, total_nnz, retidxtmp, ret.val);
  std::vector<int> extracted_idx(retidxtmp.size());
  ret.idx.resize(retidxtmp.size());
  auto extracted_idxp = extracted_idx.data();
  extract_column_idx(retidxtmp.data(), extracted_idxp, ret.idx.data(),
                     retidxtmp.size(), max_idx_bits);
  auto separated = my_set_separate(extracted_idx);
  // might be smaller than merged_count if sparse vector is zero
  auto separated_size = separated.size();
  auto separatedp = separated.data();
#pragma _NEC ivdep
#pragma _NEC vovertake
#pragma _NEC vob
  for(size_t i = 0; i < separated_size - 1; i++) {
    merged_offp[extracted_idxp[separatedp[i]]] =
      separatedp[i+1] - separatedp[i];
  }
  return ret;
}

template <class T, class I, class O>
sparse_vector<T,I>
spgemm_helper_dense(T* mat_valp, I* mat_idxp, O* mat_offp,
                    T* sv_valp, I* sv_idxp,
                    size_t sv_size, O* nnzp, size_t lnum_row) {
  std::vector<T> tmpret(lnum_row);
  std::vector<T> new_sv_val(sv_size);
  std::vector<I> new_sv_idx(sv_size);
  std::vector<O> new_nnz(sv_size);
  auto new_sv_valp = new_sv_val.data();
  auto new_sv_idxp = new_sv_idx.data();
  auto new_nnzp = new_nnz.data();
  // if nnz of column of lhs is larger than VLEN, do it like SpMV
  // other part is separated as new_sv_val/idx
  auto new_sv_size = 
    spmspv_impl_helper(mat_valp, mat_idxp, mat_offp,
                       sv_valp, sv_idxp, new_sv_valp, new_sv_idxp,
                       nnzp, new_nnzp, sv_size, tmpret.data());
  if(new_sv_size == 0) {
    return make_sparse_vector<T,I>(tmpret);
  } else {
    std::vector<O> pfx_sum_new_nnz(new_sv_size);
    prefix_sum(new_nnzp, pfx_sum_new_nnz.data(), new_sv_size);
    size_t total_new_nnz = pfx_sum_new_nnz[new_sv_size - 1];
    std::vector<T> mulout_val(total_new_nnz);
    std::vector<I> mulout_idx(total_new_nnz);
    std::vector<T> mulout_val_tmp(total_new_nnz);
    std::vector<I> mulout_idx_tmp(total_new_nnz);
    spmspv_impl(mat_valp, mat_idxp, mat_offp, 
                new_sv_valp, new_sv_idxp, new_sv_size,
                mulout_val.data(), mulout_idx.data(),
                mulout_val_tmp.data(), mulout_idx_tmp.data(),
                new_nnzp, pfx_sum_new_nnz.data(), total_new_nnz,
                tmpret.data(), lnum_row);
    return make_sparse_vector<T,I>(tmpret);
  }
}

template <class T, class I, class O>
void create_sparse_matrix(std::vector<sparse_vector<T,I>>& sv,
                          std::vector<T>& retval,
                          std::vector<I>& retidx,
                          std::vector<O>& retoff,
                          std::vector<O>& merged_off) {
  
  size_t total_nnz = 0;
  auto svsize = sv.size();
  for(size_t i = 0; i < svsize; i++) total_nnz += sv[i].val.size();
  retval.resize(total_nnz);
  retidx.resize(total_nnz);
  retoff.resize(merged_off.size()+1);
  auto crnt_retvalp = retval.data();
  auto crnt_retidxp = retidx.data();
  for(size_t i = 0; i < svsize; i++) {
    auto crnt_size = sv[i].val.size();
    auto svvalp = sv[i].val.data();
    auto svidxp = sv[i].idx.data();
    for(size_t j = 0; j < crnt_size; j++) {
      crnt_retvalp[j] = svvalp[j];
      crnt_retidxp[j] = svidxp[j];
    }
    crnt_retvalp += crnt_size;
    crnt_retidxp += crnt_size;
  }
  prefix_sum(merged_off.data(), retoff.data()+1, merged_off.size());
}


/*
  expressed as ccs matrix, but can be applied to crs matrix
  (A * B  = C <-> B^T * A^T = C^T and  CRS = CCS^T)
  return value is std::vector, because the size is not known beforehand
 */
template <class T, class I, class O>
void spgemm(T* lval, I* lidx, O* loff,
            size_t lnnz, size_t lnum_row, size_t lnum_col,
            T* rval, I* ridx, O* roff,
            size_t rnnz, size_t rnum_col,
            std::vector<T>& retval, std::vector<I>& retidx,
            std::vector<O>& retoff,
            spgemm_type type = spgemm_type::block_esc_dense,
            size_t max_merge_column = 4095,
            double dense_mode_ratio = 0.1) {
  time_spent sparse_merge(DEBUG), dense(DEBUG);
  size_t sparse_merged_columns = 0, dense_columns = 0;
  if(rnnz == 0) return;
  bool is_dense = false;
  if(type == spgemm_type::block_esc_dense) is_dense = true;
  auto dense_thr = static_cast<O>(lnum_row * dense_mode_ratio);
  size_t max_idx_bits = max_bits(lnum_row);

  std::vector<O> merged_interim_nnz(rnnz);
  auto merged_interim_nnzp = merged_interim_nnz.data();
  std::vector<O> each_column_nnz(rnum_col);
  auto each_column_nnzp =  each_column_nnz.data();
  for(size_t rc = 0; rc < rnum_col; rc++) { // rc: right column
    auto crnt_ridx = ridx + roff[rc];
    auto crnt_nnz = roff[rc+1] - roff[rc];
    for(size_t i = 0; i < crnt_nnz; i++) {
      merged_interim_nnzp[i] = loff[crnt_ridx[i]+1] - loff[crnt_ridx[i]];
    }
    for(size_t i = 0; i < crnt_nnz; i++) {
      each_column_nnzp[rc] += merged_interim_nnzp[i];
    }
    merged_interim_nnzp += crnt_nnz;
  }

  std::vector<sparse_vector<T,I>> out_sparse_vector;
  out_sparse_vector.reserve(rnum_col); // TODO: save memory?
  
  std::vector<O> retofftmp(rnum_col);
  auto retofftmpp = retofftmp.data();
  size_t crnt_rc = 0;
  merged_interim_nnzp = merged_interim_nnz.data();
  while(true) {
    size_t merge_size = 0;
    bool dense_found = false;
    size_t crnt_max_merge_column;
    if(crnt_rc + max_merge_column < rnum_col)
      crnt_max_merge_column = max_merge_column;
    else
      crnt_max_merge_column = rnum_col - crnt_rc;
    if(is_dense) {
      size_t idx = 0;
      for(; idx < crnt_max_merge_column; idx++) {
        if(each_column_nnzp[idx] > dense_thr) break;
      }
      if(idx < crnt_max_merge_column) dense_found = true;
      merge_size = idx;
    } else merge_size = crnt_max_merge_column;
    // if merge_size == 0, dense column continues, skip sparse version
    if(merge_size > 0) {
      sparse_merge.lap_start();
      sparse_merged_columns += merge_size;
      auto crnt_rval = rval + roff[crnt_rc];
      auto crnt_ridx = ridx + roff[crnt_rc];
      auto crnt_size = roff[crnt_rc + merge_size] - roff[crnt_rc];
      if(crnt_size == 0) {
        out_sparse_vector.push_back(sparse_vector<T,I>());
        crnt_rc += merge_size;
        if(crnt_rc == rnum_col) break;
        merged_interim_nnzp += crnt_size;
        each_column_nnzp += merge_size;
        retofftmpp += merge_size;
      } else {
        std::vector<O> pfx_sum_merged_interim_nnz(crnt_size);
        auto pfx_sum_merged_interim_nnzp = pfx_sum_merged_interim_nnz.data();
        prefix_sum(merged_interim_nnzp, pfx_sum_merged_interim_nnzp, crnt_size);
        auto total_interim_nnz = pfx_sum_merged_interim_nnzp[crnt_size - 1];
        out_sparse_vector.push_back
          (spgemm_helper_merged(lval, lidx, loff,
                                crnt_rval, crnt_ridx, crnt_size,
                                total_interim_nnz, merged_interim_nnzp,
                                pfx_sum_merged_interim_nnzp,
                                each_column_nnzp, merge_size,
                                max_idx_bits, retofftmpp));
        crnt_rc += merge_size;
        if(crnt_rc == rnum_col) break;
        merged_interim_nnzp += crnt_size;
        each_column_nnzp += merge_size;
        retofftmpp += merge_size;
      }
      sparse_merge.lap_stop();
    }
    if(dense_found) {
      dense.lap_start();
      dense_columns++;
      auto crnt_rval = rval + roff[crnt_rc];
      auto crnt_ridx = ridx + roff[crnt_rc];
      auto crnt_size = roff[crnt_rc + 1] - roff[crnt_rc];
      out_sparse_vector.push_back
        (spgemm_helper_dense(lval, lidx, loff, crnt_rval, crnt_ridx, crnt_size,
                             merged_interim_nnzp, lnum_row));
      retofftmpp[0] =
        out_sparse_vector[out_sparse_vector.size() - 1].val.size();
      crnt_rc += 1;
      if(crnt_rc == rnum_col) break;
      merged_interim_nnzp += crnt_size;
      each_column_nnzp += 1;
      retofftmpp += 1;
      dense.lap_stop();
    }
  }
  sparse_merge.show_lap("time for merged sparse vectors: ");
  dense.show_lap("time for dense vectors: ");
  auto out_sparse_vector_size = out_sparse_vector.size();
  time_spent t(DEBUG);
  create_sparse_matrix(out_sparse_vector, retval, retidx, retoff, retofftmp);
  t.show("time for creating sparse matrix: ");
  LOG(DEBUG) << "count of merged sparse vectors: "
             << sparse_merged_columns << std::endl;
  LOG(DEBUG) << "count of dense vectors: "
             << dense_columns << std::endl;
  LOG(DEBUG) << "count of out_sparse_vector size: "
             << out_sparse_vector_size << std::endl;
}

template <class T, class I, class O>
ccs_matrix_local<T,I,O>
spgemm(ccs_matrix_local<T,I,O>& left, 
       ccs_matrix_local<T,I,O>& right,
       spgemm_type type = spgemm_type::block_esc,
       size_t max_merge_column = 4095,
       double dense_mode_ratio = 0.1) {
  if(left.local_num_col != right.local_num_row) 
    throw std::runtime_error("spgemm: matrix size mismatch");
  ccs_matrix_local<T,I,O> ret;
  if(type == spgemm_type::esc) {
    spgemm_esc(left.val.data(), left.idx.data(), left.off.data(),
               left.val.size(), left.local_num_row, left.local_num_col,
               right.val.data(), right.idx.data(), right.off.data(),
               right.val.size(), right.local_num_col,
               ret.val, ret.idx, ret.off);
  } else {
    spgemm(left.val.data(), left.idx.data(), left.off.data(),
           left.val.size(), left.local_num_row, left.local_num_col,
           right.val.data(), right.idx.data(), right.off.data(),
           right.val.size(), right.local_num_col,
           ret.val, ret.idx, ret.off,
           type, max_merge_column, dense_mode_ratio);
  }
  ret.set_local_num(left.local_num_row);
  return ret;
}

template <class T, class I, class O>
crs_matrix_local<T,I,O>
spgemm(crs_matrix_local<T,I,O>& a,
       crs_matrix_local<T,I,O>& b, 
       spgemm_type type = spgemm_type::block_esc,
       size_t max_merge_column = 4095,
       double dense_mode_ratio = 0.1) {
  if(a.local_num_col != b.local_num_row) 
    throw std::runtime_error("spgemm: matrix size mismatch");
  crs_matrix_local<T,I,O> ret;
  if(type == spgemm_type::esc) {
    spgemm_esc(b.val.data(), b.idx.data(), b.off.data(),
               b.val.size(), b.local_num_col, b.local_num_row,
               a.val.data(), a.idx.data(), a.off.data(),
               a.val.size(), a.local_num_row,
               ret.val, ret.idx, ret.off);
  } else {
    spgemm(b.val.data(), b.idx.data(), b.off.data(),
           b.val.size(), b.local_num_col, b.local_num_row,
           a.val.data(), a.idx.data(), a.off.data(),
           a.val.size(), a.local_num_row,
           ret.val, ret.idx, ret.off,
           type, max_merge_column, dense_mode_ratio);
  }
  ret.set_local_num(b.local_num_col);
  return ret;
}

}
#endif
