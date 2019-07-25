#ifndef SPGEMM_HPP
#define SPGEMM_HPP

#include "ccs_matrix.hpp"
#include "spmspv.hpp"
#include <limits>

#ifdef __ve__
#define SPGEMM_VLEN 256
#else
#define SPGEMM_VLEN 1
#endif
#define SPGEMM_HASH_TABLE_SIZE_MULT 3
#define SPGEMM_HASH_SCALAR_THR 1024

namespace frovedis {

enum spgemm_type {
  esc,
  block_esc,
  hash,
  hash_sort
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

// avoid loop raking because number of _merged vec not so large...
// (it is possible, but there might be load balancing problem...)
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

/* modified version of  set_separate at dataframe/set_operations.hpp
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
  int valid_vreg[SPGEMM_VLEN];
  size_t key_idx_vreg[SPGEMM_VLEN];
  size_t key_idx_stop_vreg[SPGEMM_VLEN];
  size_t out_idx_vreg[SPGEMM_VLEN];
  T current_key_vreg[SPGEMM_VLEN];
#pragma _NEC vreg(valid_vreg)
#pragma _NEC vreg(key_idx_vreg)
#pragma _NEC vreg(key_idx_stop_vreg)
#pragma _NEC vreg(out_idx_vreg)
#pragma _NEC vreg(current_key_vreg)
  for(int i = 0; i < SPGEMM_VLEN; i++) {
    valid_vreg[i] = valid[i];
    key_idx_vreg[i] = key_idx[i];
    key_idx_stop_vreg[i] = key_idx_stop[i];
    out_idx_vreg[i] = out_idx[i];
    current_key_vreg[i] = current_key[i];
  }
  
  for(size_t j = 0; j < max_size; j++) {
#pragma cdir nodep
#pragma _NEC ivdep
    for(int i = 0; i < SPGEMM_VLEN; i++) {
      if(valid_vreg[i]) {
        auto keyval = keyp[key_idx_vreg[i]];
        if(keyval != current_key_vreg[i]) {
          outp[out_idx_vreg[i]++] = key_idx_vreg[i];
          current_key_vreg[i] = keyval;
        }
        key_idx_vreg[i]++;
        if(key_idx_vreg[i] == key_idx_stop_vreg[i]) {valid_vreg[i] = false;}
      }
    }
  }
  for(int i = 0; i < SPGEMM_VLEN; i++) {
    // valid[i] = valid_vreg[i]; // not used
    // key_idx[i] = key_idx_vreg[i]; // not used
    // key_idx_stop[i] = key_idx_stop_vreg[i]; // not used
    out_idx[i] = out_idx_vreg[i];
    // current_key[i] = current_key_vreg[i]; // not used
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
  size_t max_idx_bits = max_bits(lnum_row-1);

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
spgemm_block_esc_helper(T* mat_valp, I* mat_idxp, O* mat_offp,
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
void spgemm_block_esc(T* lval, I* lidx, O* loff,
                      size_t lnnz, size_t lnum_row, size_t lnum_col,
                      T* rval, I* ridx, O* roff,
                      size_t rnnz, size_t rnum_col,
                      std::vector<T>& retval, std::vector<I>& retidx,
                      std::vector<O>& retoff,
                      size_t merge_column_size) {
  time_spent t(DEBUG);
  if(rnnz == 0) return;
  size_t max_idx_bits = max_bits(lnum_row-1);

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
  t.show("count nnz: ");
  auto num_chunks = ceil_div(rnum_col,merge_column_size);
  std::vector<sparse_vector<T,I>> out_sparse_vector;
  out_sparse_vector.reserve(num_chunks);
  
  std::vector<O> retofftmp(rnum_col);
  auto retofftmpp = retofftmp.data();
  size_t crnt_rc = 0;
  merged_interim_nnzp = merged_interim_nnz.data();
  while(true) {
    size_t merge_size = 0;
    if(crnt_rc + merge_column_size < rnum_col)
      merge_size = merge_column_size;
    else
      merge_size = rnum_col - crnt_rc;
    auto crnt_rval = rval + roff[crnt_rc];
    auto crnt_ridx = ridx + roff[crnt_rc];
    auto crnt_size = roff[crnt_rc + merge_size] - roff[crnt_rc];
    if(crnt_size == 0) { // (contiguous) zero sized columns
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
        (spgemm_block_esc_helper(lval, lidx, loff,
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
  }
  t.show("time for creating sparse vectors: ");
  create_sparse_matrix(out_sparse_vector, retval, retidx, retoff, retofftmp);
  t.show("time for creating sparse matrix: ");
}

template <class T>
void spgemm_make_sparse_vector(const size_t* keyp, const T* valp, size_t size,
                               size_t unused, std::vector<size_t>& retidx,
                               std::vector<T>& retval) {
  if(size == 0) return;
  std::vector<size_t> idxtmp(size);
  std::vector<T> valtmp(size);
  size_t* idxtmpp = idxtmp.data();
  T* valtmpp = valtmp.data();
  size_t each = size / SPGEMM_VLEN; // maybe 0
  if(each % 2 == 0 && each > 1) each--;
  size_t rest = size - each * SPGEMM_VLEN;
  size_t out_ridx[SPGEMM_VLEN];
// never remove this vreg! this is needed folowing vovertake
// though this prevents ftrace...
#pragma _NEC vreg(out_ridx)
  for(size_t i = 0; i < SPGEMM_VLEN; i++) {
    out_ridx[i] = each * i;
  }
  if(each == 0) {
    size_t current = 0;
    for(size_t i = 0; i < size; i++) {
      if(keyp[i] != unused) {
        idxtmpp[current] = keyp[i];
        valtmpp[current] = valp[i];
        current++;
      }
    }
    retidx.resize(current);
    retval.resize(current);
    auto retidxp = retidx.data();
    auto retvalp = retval.data();
    for(size_t i = 0; i < current; i++) {
      retidxp[i] = idxtmpp[i];
      retvalp[i] = valtmpp[i];
    }
  } else {
#pragma _NEC vob
    for(size_t j = 0; j < each; j++) {
#pragma cdir nodep
#pragma _NEC ivdep
#pragma _NEC vovertake
      for(size_t i = 0; i < SPGEMM_VLEN; i++) {
        auto loaded_key = keyp[j + each * i];
        if(loaded_key != unused) {
          idxtmpp[out_ridx[i]] = loaded_key;
          valtmpp[out_ridx[i]] = valp[j + each * i];
          out_ridx[i]++;
        }
      }
    }
    size_t rest_idx_start = each * SPGEMM_VLEN;
    size_t rest_idx = rest_idx_start;
    if(rest != 0) {
      for(size_t j = 0; j < rest; j++) {
        auto loaded_key = keyp[j + rest_idx_start]; 
        if(loaded_key != unused) {
          idxtmpp[rest_idx] = loaded_key;
          valtmpp[rest_idx] = valp[j + rest_idx_start]; 
          rest_idx++;
        }
      }
    }
    size_t sizes[SPGEMM_VLEN];
    for(size_t i = 0; i < SPGEMM_VLEN; i++) {
      sizes[i] = out_ridx[i] - each * i;
    }
    size_t total = 0;
    for(size_t i = 0; i < SPGEMM_VLEN; i++) {
      total += sizes[i];
    }
    size_t rest_size = rest_idx - each * SPGEMM_VLEN;
    total += rest_size;
    retval.resize(total);
    retidx.resize(total);
    auto retvalp = retval.data();
    auto retidxp = retidx.data();
    size_t current = 0;
    for(size_t i = 0; i < SPGEMM_VLEN; i++) {
      for(size_t j = 0; j < sizes[i]; j++) {
        retidxp[current + j] = idxtmpp[each * i + j];
        retvalp[current + j] = valtmpp[each * i + j];
      }
      current += sizes[i];
    }
    for(size_t j = 0; j < rest_size; j++) {
      retidxp[current + j] = idxtmpp[rest_idx_start + j];
      retvalp[current + j] = valtmpp[rest_idx_start + j];
    }
  }
}

template <class T, class O>
void spgemm_hash_accumulator_unroll(size_t valid_vlen,
                                    O unroll4, O* pos_ridx, size_t shift,
                                    size_t* work_keyp, T* work_valp, 
                                    size_t max_idx_bits, size_t unused,
                                    O* table_sizep, O* table_startp,
                                    size_t* table_keyp, T* table_valp,
                                    size_t* work_collision_keyp,
                                    T* work_collision_valp,
                                    O* collision_ridx) {
  O pos_ridx_vreg[SPGEMM_VLEN];
  O collision_ridx_vreg[SPGEMM_VLEN];
#pragma _NEC vreg(pos_ridx_vreg)
#pragma _NEC vreg(collision_ridx_vreg)
  for(size_t i = 0; i < SPGEMM_VLEN; i++) {
    pos_ridx_vreg[i] = pos_ridx[i];
    collision_ridx_vreg[i] = collision_ridx[i];
  }

  for(O b = 0; b < unroll4; b++) {
#pragma _NEC ivdep
#pragma _NEC shortloop
    for(size_t i = 0; i < valid_vlen; i++) {
      auto loaded_key0 = work_keyp[pos_ridx_vreg[i]];
      auto loaded_val0 = work_valp[pos_ridx_vreg[i]];
      auto loaded_key1 = work_keyp[pos_ridx_vreg[i]+1];
      auto loaded_val1 = work_valp[pos_ridx_vreg[i]+1];
      auto loaded_key2 = work_keyp[pos_ridx_vreg[i]+2];
      auto loaded_val2 = work_valp[pos_ridx_vreg[i]+2];
      auto loaded_key3 = work_keyp[pos_ridx_vreg[i]+3];
      auto loaded_val3 = work_valp[pos_ridx_vreg[i]+3];

      auto col0 = loaded_key0 >> max_idx_bits;
      auto col1 = loaded_key1 >> max_idx_bits;
      auto col2 = loaded_key2 >> max_idx_bits;
      auto col3 = loaded_key3 >> max_idx_bits;

      auto hashval0 =
        (loaded_key0 + shift) % table_sizep[col0] + table_startp[col0];
      auto hashval1 =
        (loaded_key1 + shift) % table_sizep[col1] + table_startp[col1];
      auto hashval2 =
        (loaded_key2 + shift) % table_sizep[col2] + table_startp[col2];
      auto hashval3 =
        (loaded_key3 + shift) % table_sizep[col3] + table_startp[col3];

      auto table_key0 = table_keyp[hashval0];
      if(table_key0 == loaded_key0) { // same key already stored
        table_valp[hashval0] += loaded_val0;
      } else if (table_key0 == unused) { // first time
        table_keyp[hashval0] = loaded_key0;
        table_valp[hashval0] = loaded_val0;
      } else { // collision
        work_collision_keyp[collision_ridx_vreg[i]] = loaded_key0;
        work_collision_valp[collision_ridx_vreg[i]] = loaded_val0;
        collision_ridx_vreg[i]++;
      }

      auto table_key1 = table_keyp[hashval1];
      if(table_key1 == loaded_key1) { // same key already stored
        table_valp[hashval1] += loaded_val1;
      } else if (table_key1 == unused) { // first time
        table_keyp[hashval1] = loaded_key1;
        table_valp[hashval1] = loaded_val1;
      } else { // collision
        work_collision_keyp[collision_ridx_vreg[i]] = loaded_key1;
        work_collision_valp[collision_ridx_vreg[i]] = loaded_val1;
        collision_ridx_vreg[i]++;
      }

      auto table_key2 = table_keyp[hashval2];
      if(table_key2 == loaded_key2) { // same key already stored
        table_valp[hashval2] += loaded_val2;
      } else if (table_key2 == unused) { // first time
        table_keyp[hashval2] = loaded_key2;
        table_valp[hashval2] = loaded_val2;
      } else { // collision
        work_collision_keyp[collision_ridx_vreg[i]] = loaded_key2;
        work_collision_valp[collision_ridx_vreg[i]] = loaded_val2;
        collision_ridx_vreg[i]++;
      }

      auto table_key3 = table_keyp[hashval3];
      if(table_key3 == loaded_key3) { // same key already stored
        table_valp[hashval3] += loaded_val3;
      } else if (table_key3 == unused) { // first time
        table_keyp[hashval3] = loaded_key3;
        table_valp[hashval3] = loaded_val3;
      } else { // collision
        work_collision_keyp[collision_ridx_vreg[i]] = loaded_key3;
        work_collision_valp[collision_ridx_vreg[i]] = loaded_val3;
        collision_ridx_vreg[i]++;
      }

      pos_ridx_vreg[i]+=4;
    }
  }
  for(size_t i = 0; i < SPGEMM_VLEN; i++) {
    pos_ridx[i] = pos_ridx_vreg[i];
    collision_ridx[i] = collision_ridx_vreg[i];
  }
}

template <class T, class O>
void spgemm_hash_accumulator_scalar(O* pos_ridx, O* size_ridx, size_t shift,
                                    size_t* work_keyp, T* work_valp, 
                                    size_t max_idx_bits, size_t unused,
                                    O* table_sizep, O* table_startp,
                                    size_t* table_keyp, T* table_valp) {
  for(size_t i = 0; i < SPGEMM_VLEN; i++) {
    auto pos = pos_ridx[i];
    for(size_t j = 0; j < size_ridx[i]; j++) {
      auto loaded_key = work_keyp[pos + j];
      auto loaded_val = work_valp[pos + j];
      auto col = loaded_key >> max_idx_bits;
      size_t local_shift = shift;
      while(true) {
        auto hashval =
          (loaded_key + local_shift) % table_sizep[col] + table_startp[col];
        auto table_key = table_keyp[hashval];
        if(table_key == loaded_key) { // same key already stored
          table_valp[hashval] += loaded_val;
          break;
        } else if (table_key == unused) { // first time
          table_keyp[hashval] = loaded_key;
          table_valp[hashval] = loaded_val;
          break;
        } else { // collision
          local_shift++;
        }
      }
    }
  }
}

// keyp, valp will be destructed
template <class T, class O>
void spgemm_hash_accumulator(size_t* keyp, T* valp, O total_size,
                             O* nnz_per_columnp, size_t column_size,
                             size_t max_idx_bits,
                             std::vector<size_t>& retidx,
                             std::vector<T>& retval) {
  size_t total_table_size = total_size * SPGEMM_HASH_TABLE_SIZE_MULT;
  size_t unused = std::numeric_limits<size_t>::max();
  std::vector<size_t> table_key(total_table_size, unused);
  std::vector<T> table_val(total_table_size);
  std::vector<size_t> collision_key(total_size);
  std::vector<T> collision_val(total_size);
  std::vector<O> nnz_per_column_pfxsum(column_size);
  std::vector<O> table_size(column_size);
  std::vector<O> table_start(column_size);
  auto nnz_per_column_pfxsump = nnz_per_column_pfxsum.data();
  auto table_startp = table_start.data();
  auto table_sizep = table_size.data();
  prefix_sum(nnz_per_columnp, nnz_per_column_pfxsump+1, column_size-1);
  for(size_t i = 0; i < column_size; i++) {
    table_startp[i] = nnz_per_column_pfxsump[i] * SPGEMM_HASH_TABLE_SIZE_MULT;
    table_sizep[i] = nnz_per_columnp[i] * SPGEMM_HASH_TABLE_SIZE_MULT;
  }

  auto each = ceil_div(total_size, O(SPGEMM_VLEN));
  if(each % 2 == 0) each++;
  each = total_size / O(SPGEMM_VLEN); 
  O pos_ridx[SPGEMM_VLEN]; // ridx: idx for raking
  O pos_stop_ridx[SPGEMM_VLEN];
  auto begin_it = nnz_per_column_pfxsump;
  auto end_it = nnz_per_column_pfxsump + column_size;
  auto current_it = begin_it;
  pos_ridx[0] = 0;
  size_t ii = 1;
  for(size_t i = 1; i < SPGEMM_VLEN; i++) {
    auto it = std::lower_bound(current_it, end_it, each * i);
    if(it == current_it) continue;
    else if(it == end_it) break;
    else {
      pos_ridx[ii++] = *it;
      current_it = it;
    }
  }
  for(size_t i = ii; i < SPGEMM_VLEN; i++) {
    pos_ridx[i] = total_size;
  }
  for(size_t i = 0; i < SPGEMM_VLEN-1; i++) {
    pos_stop_ridx[i] = pos_ridx[i+1];
  }
  pos_stop_ridx[SPGEMM_VLEN-1] = total_size;
  O collision_ridx[SPGEMM_VLEN];
  O original_pos_ridx[SPGEMM_VLEN];
  for(size_t i = 0; i < SPGEMM_VLEN; i++) {
    original_pos_ridx[i] = pos_ridx[i];
    collision_ridx[i] = pos_ridx[i];
  }

  O size_ridx[SPGEMM_VLEN];
  for(size_t i = 0; i < SPGEMM_VLEN; i++)
    size_ridx[i] = pos_stop_ridx[i] - pos_ridx[i];
  O max = 0;
  for(size_t i = 0; i < SPGEMM_VLEN; i++) 
    if(size_ridx[i] > max) max = size_ridx[i];

  size_t valid_vlen = SPGEMM_VLEN;
  for(; valid_vlen > 0; valid_vlen--)
    if(size_ridx[valid_vlen-1] != 0) break;

  O min = std::numeric_limits<O>::max();
  for(size_t i = 0; i < valid_vlen; i++) 
    if(size_ridx[i] < min) min = size_ridx[i];

  auto table_keyp = table_key.data();
  auto table_valp = table_val.data();
  size_t* work_keyp = keyp;
  T* work_valp = valp;
  size_t* work_collision_keyp = collision_key.data();
  T* work_collision_valp = collision_val.data();
  size_t shift = 0;
  while(true) {
    auto unroll4 = min / 4;
    spgemm_hash_accumulator_unroll(valid_vlen, unroll4, pos_ridx, shift,
                                   work_keyp, work_valp,
                                   max_idx_bits, unused,
                                   table_sizep, table_startp,
                                   table_keyp, table_valp,
                                   work_collision_keyp,
                                   work_collision_valp,
                                   collision_ridx);

    O pos_ridx_vreg[SPGEMM_VLEN];
    O pos_stop_ridx_vreg[SPGEMM_VLEN];
    O collision_ridx_vreg[SPGEMM_VLEN];
#pragma _NEC vreg(pos_ridx_vreg)
#pragma _NEC vreg(pos_stop_ridx_vreg)
    for(size_t i = 0; i < SPGEMM_VLEN; i++) {
      pos_ridx_vreg[i] = pos_ridx[i];
      pos_stop_ridx_vreg[i] = pos_stop_ridx[i];
      collision_ridx_vreg[i] = collision_ridx[i];
    }

    auto rest = max - unroll4 * 4;
    for(O b = 0; b < rest; b++) {
#pragma _NEC ivdep
      for(size_t i = 0; i < SPGEMM_VLEN; i++) {
        if(pos_ridx_vreg[i] != pos_stop_ridx_vreg[i]) {
          auto loaded_key = work_keyp[pos_ridx_vreg[i]];
          auto loaded_val = work_valp[pos_ridx_vreg[i]];
          auto col = loaded_key >> max_idx_bits;
          auto hashval =
            (loaded_key + shift) % table_sizep[col] + table_startp[col];
          auto table_key = table_keyp[hashval];
          if(table_key == loaded_key) { // same key already stored
            table_valp[hashval] += loaded_val;
          } else if (table_key == unused) { // first time
            table_keyp[hashval] = loaded_key;
            table_valp[hashval] = loaded_val;
          } else { // collision
            work_collision_keyp[collision_ridx_vreg[i]] = loaded_key;
            work_collision_valp[collision_ridx_vreg[i]] = loaded_val;
            collision_ridx_vreg[i]++;
          }
          pos_ridx_vreg[i]++;
        }
      }
    }
    for(size_t i = 0; i < SPGEMM_VLEN; i++) {
      // pos_ridx[i] = pos_ridx_vreg[i]; // not used
      collision_ridx[i] = collision_ridx_vreg[i];
    }
    int anycollision = false;
    for(size_t i = 0; i < SPGEMM_VLEN; i++) {
      if(collision_ridx[i] != original_pos_ridx[i]) anycollision = true;
    }
    if(anycollision) {
      auto work_keyp_tmp = work_keyp;
      auto work_valp_tmp = work_valp;
      work_keyp = work_collision_keyp;
      work_valp = work_collision_valp;
      work_collision_keyp = work_keyp_tmp;
      work_collision_valp = work_valp_tmp;
      for(size_t i = 0; i < SPGEMM_VLEN; i++) {
        pos_ridx[i] = original_pos_ridx[i];
        pos_stop_ridx[i] = collision_ridx[i];
        collision_ridx[i] = pos_ridx[i];
      }
      for(size_t i = 0; i < SPGEMM_VLEN; i++)
        size_ridx[i] = pos_stop_ridx[i] - pos_ridx[i];
      max = 0;
      for(size_t i = 0; i < SPGEMM_VLEN; i++) 
        if(size_ridx[i] > max) max = size_ridx[i];
      valid_vlen = SPGEMM_VLEN;
      for(; valid_vlen > 0; valid_vlen--)
        if(size_ridx[valid_vlen-1] != 0) break;
      min = std::numeric_limits<O>::max();
      for(size_t i = 0; i < valid_vlen; i++) 
        if(size_ridx[i] < min) min = size_ridx[i];
      shift++;
      size_t remain = 0;
      for(size_t i = 0; i < SPGEMM_VLEN; i++) {
        remain += size_ridx[i];
      }
      if(remain < SPGEMM_HASH_SCALAR_THR) {
        spgemm_hash_accumulator_scalar(pos_ridx, size_ridx, shift,
                                       work_keyp, work_valp, 
                                       max_idx_bits, unused,
                                       table_sizep, table_startp,
                                       table_keyp, table_valp);
        break;
      }
    } else {
      break;
    }
  }
#ifdef __ve__
  spgemm_make_sparse_vector(table_keyp, table_valp, total_table_size, unused,
                            retidx, retval);
#else
  size_t retsize = 0;
  for(size_t i = 0; i < total_table_size; i++) 
    if(table_keyp[i] != unused) retsize++;
  retidx.resize(retsize);
  retval.resize(retsize);
  auto retidxp = retidx.data();
  auto retvalp = retval.data();
  size_t pos = 0;
  for(size_t i = 0; i < total_table_size; i++) {
    if(table_keyp[i] != unused) {
      retidxp[pos] = table_keyp[i];
      retvalp[pos] = table_valp[i];
      pos++;
    }
  }
#endif
}

template <class T, class I, class O>
sparse_vector<T,I>
spgemm_hash_helper(T* mat_valp, I* mat_idxp, O* mat_offp,
                   T* sv_valp, I* sv_idxp, O sv_size,
                   O total_nnz, O* nnzp, O* pfx_sum_nnz,
                   O* merged_total_nnzp, size_t merged_count,
                   size_t max_idx_bits, O* merged_offp,
                   bool sort) {
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
  std::vector<size_t> retidxtmp;
  spgemm_hash_accumulator(mulout_idxp, mulout_valp, total_nnz,
                          merged_total_nnzp, merged_count,
                          max_idx_bits,
                          retidxtmp, ret.val);
  if(sort) {
    auto used_bits = max_bits(merged_count-1);
    auto max_key_size = ceil_div(max_idx_bits + used_bits, size_t(8));
    radix_sort_impl(retidxtmp.data(), ret.val.data(), retidxtmp.size(),
                    max_key_size);
  }
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
void spgemm_hash(T* lval, I* lidx, O* loff,
                 size_t lnnz, size_t lnum_row, size_t lnum_col,
                 T* rval, I* ridx, O* roff,
                 size_t rnnz, size_t rnum_col,
                 std::vector<T>& retval, std::vector<I>& retidx,
                 std::vector<O>& retoff,
                 size_t merge_column_size,
                 bool sort) {
  time_spent t(DEBUG);
  if(rnnz == 0) return;
  size_t max_idx_bits = max_bits(lnum_row-1);

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
  t.show("count nnz: ");
  auto num_chunks = ceil_div(rnum_col,merge_column_size);
  std::vector<sparse_vector<T,I>> out_sparse_vector;
  out_sparse_vector.reserve(num_chunks);
  
  std::vector<O> retofftmp(rnum_col);
  auto retofftmpp = retofftmp.data();
  size_t crnt_rc = 0;
  merged_interim_nnzp = merged_interim_nnz.data();
  while(true) {
    size_t merge_size = 0;
    if(crnt_rc + merge_column_size < rnum_col)
      merge_size = merge_column_size;
    else
      merge_size = rnum_col - crnt_rc;
    auto crnt_rval = rval + roff[crnt_rc];
    auto crnt_ridx = ridx + roff[crnt_rc];
    auto crnt_size = roff[crnt_rc + merge_size] - roff[crnt_rc];
    if(crnt_size == 0) { // (contiguous) zero sized columns
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
        (spgemm_hash_helper(lval, lidx, loff,
                            crnt_rval, crnt_ridx, crnt_size,
                            total_interim_nnz, merged_interim_nnzp,
                            pfx_sum_merged_interim_nnzp,
                            each_column_nnzp, merge_size,
                            max_idx_bits, retofftmpp, sort));
      crnt_rc += merge_size;
      if(crnt_rc == rnum_col) break;
      merged_interim_nnzp += crnt_size;
      each_column_nnzp += merge_size;
      retofftmpp += merge_size;
    }
  }
  t.show("time for creating sparse vectors: ");
  create_sparse_matrix(out_sparse_vector, retval, retidx, retoff, retofftmp);
  t.show("time for creating sparse matrix: ");
}

template <class T, class I, class O>
ccs_matrix_local<T,I,O>
spgemm(ccs_matrix_local<T,I,O>& left, 
       ccs_matrix_local<T,I,O>& right,
       spgemm_type type = spgemm_type::block_esc,
       size_t merge_column_size = 4096) {
  if(left.local_num_col != right.local_num_row) 
    throw std::runtime_error("spgemm: matrix size mismatch");
  ccs_matrix_local<T,I,O> ret;
  if(type == spgemm_type::esc) {
    spgemm_esc(left.val.data(), left.idx.data(), left.off.data(),
               left.val.size(), left.local_num_row, left.local_num_col,
               right.val.data(), right.idx.data(), right.off.data(),
               right.val.size(), right.local_num_col,
               ret.val, ret.idx, ret.off);
  } else if(type == spgemm_type::block_esc) {
    spgemm_block_esc(left.val.data(), left.idx.data(), left.off.data(),
                     left.val.size(), left.local_num_row, left.local_num_col,
                     right.val.data(), right.idx.data(), right.off.data(),
                     right.val.size(), right.local_num_col,
                     ret.val, ret.idx, ret.off,
                     merge_column_size);
  } else if(type == spgemm_type::hash) {
    spgemm_hash(left.val.data(), left.idx.data(), left.off.data(),
                left.val.size(), left.local_num_row, left.local_num_col,
                right.val.data(), right.idx.data(), right.off.data(),
                right.val.size(), right.local_num_col,
                ret.val, ret.idx, ret.off,
                merge_column_size, false);
  } else if(type == spgemm_type::hash_sort) {
    spgemm_hash(left.val.data(), left.idx.data(), left.off.data(),
                left.val.size(), left.local_num_row, left.local_num_col,
                right.val.data(), right.idx.data(), right.off.data(),
                right.val.size(), right.local_num_col,
                ret.val, ret.idx, ret.off,
                merge_column_size, true);
  } else {
    throw std::runtime_error("unknown spgemm_type");
  }
  ret.set_local_num(left.local_num_row);
  return ret;
}

template <class T, class I, class O>
crs_matrix_local<T,I,O>
spgemm(crs_matrix_local<T,I,O>& a,
       crs_matrix_local<T,I,O>& b, 
       spgemm_type type = spgemm_type::block_esc,
       size_t merge_column_size = 4096) {
  if(a.local_num_col != b.local_num_row) 
    throw std::runtime_error("spgemm: matrix size mismatch");
  crs_matrix_local<T,I,O> ret;
  if(type == spgemm_type::esc) {
    spgemm_esc(b.val.data(), b.idx.data(), b.off.data(),
               b.val.size(), b.local_num_col, b.local_num_row,
               a.val.data(), a.idx.data(), a.off.data(),
               a.val.size(), a.local_num_row,
               ret.val, ret.idx, ret.off);
  } else if(type == spgemm_type::block_esc) {
    spgemm_block_esc(b.val.data(), b.idx.data(), b.off.data(),
                     b.val.size(), b.local_num_col, b.local_num_row,
                     a.val.data(), a.idx.data(), a.off.data(),
                     a.val.size(), a.local_num_row,
                     ret.val, ret.idx, ret.off,
                     merge_column_size);
  } else if(type == spgemm_type::hash) {
    spgemm_hash(b.val.data(), b.idx.data(), b.off.data(),
                b.val.size(), b.local_num_col, b.local_num_row,
                a.val.data(), a.idx.data(), a.off.data(),
                a.val.size(), a.local_num_row,
                ret.val, ret.idx, ret.off,
                merge_column_size, false);
  } else if(type == spgemm_type::hash_sort) {
    spgemm_hash(b.val.data(), b.idx.data(), b.off.data(),
                b.val.size(), b.local_num_col, b.local_num_row,
                a.val.data(), a.idx.data(), a.off.data(),
                a.val.size(), a.local_num_row,
                ret.val, ret.idx, ret.off,
                merge_column_size, true);
  } else {
    throw std::runtime_error("unknown spgemm_type");
  }
  ret.set_local_num(b.local_num_col);
  return ret;
}

}
#endif
