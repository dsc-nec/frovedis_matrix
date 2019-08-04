#ifndef CRS_MATRIX_HPP
#define CRS_MATRIX_HPP

#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <climits>
#include <algorithm>

#include "../core/prefix_sum.hpp"

#include "rowmajor_matrix.hpp"

#define CRS_VLEN 256

#define CRS_SPMM_THR 32
#define CRS_SPMM_VLEN 256
#define SPARSE_VECTOR_VLEN 256
#define TO_SKIP_REBALANCE 256

namespace frovedis {

template <class T, class I = size_t>
struct sparse_vector {
  sparse_vector() : size(0) {}
  sparse_vector(const sparse_vector<T,I>& s) {
    val = s.val;
    idx = s.idx;
    size = s.size;
  }
  sparse_vector<T,I>& operator=(const sparse_vector<T,I>& s) {
    val = s.val;
    idx = s.idx;
    size = s.size;
    return *this;
  }
  sparse_vector(sparse_vector<T,I>&& s) {
    val.swap(s.val);
    idx.swap(s.idx);
    size = s.size;
  }
  sparse_vector<T,I>& operator=(sparse_vector<T,I>&& s) {
    val.swap(s.val);
    idx.swap(s.idx);
    size = s.size;
    return *this;
  }
  sparse_vector(I vsize){
    val.resize(vsize);
    idx.resize(vsize);
  }
  sparse_vector(I vsize, T initv){
    val.assign(vsize, initv);
    idx.assign(vsize, initv);
  }   
  void debug_print() const {
    std::cout << "val : ";
    for(auto i: val) std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "idx : ";
    for(auto i: idx) std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "size : " << size << std::endl;
  }
  std::vector<T> to_vector();
  
  std::vector<T> val;
  std::vector<I> idx;
  size_t size; // logical length; might not be the same as the last value of idx
};

template <class T, class I>
void sparse_to_dense(sparse_vector<T,I>& sv, T* retp) {
  size_t valsize = sv.val.size();
  T* valp = sv.val.data();
  I* idxp = sv.idx.data();
#pragma cdir nodep
#pragma _NEC ivdep
#pragma _NEC vovertake
#pragma _NEC vob
  for(size_t i = 0; i < valsize; i++) {
    retp[idxp[i]] = valp[i];
  }
}

template <class T, class I>
std::vector<T> sparse_vector<T,I>::to_vector() {
  std::vector<T> ret(size);
  sparse_to_dense(*this, ret.data());
  return ret;
}

#if defined(_SX) || defined(__ve__)
// loop raking version
template <class T, class I = size_t>
sparse_vector<T,I> make_sparse_vector(const T* vp, size_t size) {
  if(size == 0) {
    return sparse_vector<T,I>();
  }
  std::vector<T> valtmp(size);
  std::vector<I> idxtmp(size);
  T* valtmpp = valtmp.data();
  I* idxtmpp = idxtmp.data();
  size_t each = size / SPARSE_VECTOR_VLEN; // maybe 0
  if(each % 2 == 0 && each > 1) each--;
  size_t rest = size - each * SPARSE_VECTOR_VLEN;
  size_t out_ridx[SPARSE_VECTOR_VLEN];
// never remove this vreg! this is needed folowing vovertake
// though this prevents ftrace...
// #pragma _NEC vreg(out_ridx)
  for(size_t i = 0; i < SPARSE_VECTOR_VLEN; i++) {
    out_ridx[i] = each * i;
  }
  if(each == 0) {
    size_t current = 0;
    for(size_t i = 0; i < size; i++) {
      if(vp[i] != 0) {
        valtmpp[current] = vp[i];
        idxtmpp[current] = i;
        current++;
      }
    }
    sparse_vector<T,I> ret;
    ret.size = size;
    ret.val.resize(current);
    ret.idx.resize(current);
    T* retvalp = ret.val.data();
    I* retidxp = ret.idx.data();
    for(size_t i = 0; i < current; i++) {
      retvalp[i] = valtmpp[i];
      retidxp[i] = idxtmpp[i];
    }
    return ret;
  } else {
//#pragma _NEC vob
    for(size_t j = 0; j < each; j++) {
#pragma cdir nodep
#pragma _NEC ivdep
//#pragma _NEC vovertake
      for(size_t i = 0; i < SPARSE_VECTOR_VLEN; i++) {
        auto loaded_v = vp[j + each * i];
        if(loaded_v != 0) {
          valtmpp[out_ridx[i]] = loaded_v;
          idxtmpp[out_ridx[i]] = j + each * i;
          out_ridx[i]++;
        }
      }
    }
    size_t rest_idx_start = each * SPARSE_VECTOR_VLEN;
    size_t rest_idx = rest_idx_start;
    if(rest != 0) {
      for(size_t j = 0; j < rest; j++) {
        auto loaded_v = vp[j + rest_idx_start]; 
        if(loaded_v != 0) {
          valtmpp[rest_idx] = loaded_v;
          idxtmpp[rest_idx] = j + rest_idx_start;
          rest_idx++;
        }
      }
    }
    sparse_vector<T,I> ret;
    ret.size = size;
    size_t sizes[SPARSE_VECTOR_VLEN];
    for(size_t i = 0; i < SPARSE_VECTOR_VLEN; i++) {
      sizes[i] = out_ridx[i] - each * i;
    }
    size_t total = 0;
    for(size_t i = 0; i < SPARSE_VECTOR_VLEN; i++) {
      total += sizes[i];
    }
    size_t rest_size = rest_idx - each * SPARSE_VECTOR_VLEN;
    total += rest_size;
    ret.val.resize(total);
    ret.idx.resize(total);
    T* retvalp = ret.val.data();
    I* retidxp = ret.idx.data();
    size_t current = 0;
    for(size_t i = 0; i < SPARSE_VECTOR_VLEN; i++) {
      for(size_t j = 0; j < sizes[i]; j++) {
        retvalp[current + j] = valtmpp[each * i + j];
        retidxp[current + j] = idxtmpp[each * i + j];
      }
      current += sizes[i];
    }
    for(size_t j = 0; j < rest_size; j++) {
      retvalp[current + j] = valtmpp[rest_idx_start + j];
      retidxp[current + j] = idxtmpp[rest_idx_start + j];
    }
    return ret;
  }
}
#else
// original version
template <class T, class I = size_t>
sparse_vector<T,I> make_sparse_vector(const T* vp, size_t size) {
  if(size == 0) {
    return sparse_vector<T,I>();
  }
  std::vector<T> valtmp(size);
  std::vector<I> idxtmp(size);
  T* valtmpp = valtmp.data();
  I* idxtmpp = idxtmp.data();
  size_t current = 0;
  for(size_t i = 0; i < size; i++) {
    if(vp[i] != 0) {
      valtmpp[current] = vp[i];
      idxtmpp[current] = i;
      current++;
    }
  }
  sparse_vector<T,I> ret;
  ret.size = size;
  ret.val.resize(current);
  ret.idx.resize(current);
  T* retvalp = ret.val.data();
  I* retidxp = ret.idx.data();
  for(size_t i = 0; i < current; i++) {
    retvalp[i] = valtmpp[i];
    retidxp[i] = idxtmpp[i];
  }
  return ret;
}
#endif
template <class T, class I = size_t>
sparse_vector<T,I> make_sparse_vector(const std::vector<T>& v) {
  return make_sparse_vector<T,I>(v.data(), v.size());
}

template <class T, class I = size_t, class O = size_t>
struct crs_matrix_local {
  crs_matrix_local() {off.push_back(0);}
  crs_matrix_local(size_t nrows, size_t ncols) :
    local_num_row(nrows), local_num_col(ncols) {off.push_back(0);}
  crs_matrix_local(const crs_matrix_local<T,I,O>& m) {
    val = m.val;
    idx = m.idx;
    off = m.off;
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
  }
  crs_matrix_local<T,I,O>& operator=(const crs_matrix_local<T,I,O>& m) {
    val = m.val;
    idx = m.idx;
    off = m.off;
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
    return *this;
  }
  crs_matrix_local(crs_matrix_local<T,I,O>&& m) {
    val.swap(m.val);
    idx.swap(m.idx);
    off.swap(m.off);
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
  }
  crs_matrix_local<T,I,O>& operator=(crs_matrix_local<T,I,O>&& m) {
    val.swap(m.val);
    idx.swap(m.idx);
    off.swap(m.off);
    local_num_row = m.local_num_row;
    local_num_col = m.local_num_col;
    return *this;
  }
  std::vector<T> val;
  std::vector<I> idx;
  std::vector<O> off; // size is local_num_row + 1 ("0" is added)
  size_t local_num_row;
  size_t local_num_col;
  void copy_from_jarray(int* off_, int* idx_, T* val_, size_t n) {
    val.resize(n); T* valp = &val[0];
    idx.resize(n); I* idxp = &idx[0];
    for(int i=0; i<n; ++i) {
      valp[i] = val_[i];  
      idxp[i] = idx_[i];
    }
    for(int i=0; i<local_num_row; ++i) off.push_back(off_[i+1]);
  } 
  crs_matrix_local<T,I,O> transpose();
  crs_matrix_local<T,I,O> pow_val(T exponent);
  void set_local_num(size_t c) {
    local_num_col = c;
    local_num_row = off.size() - 1;
  }
  rowmajor_matrix_local<T> to_rowmajor();
  void debug_pretty_print() const {
    for(size_t row = 0; row < local_num_row; row++) {
      std::vector<T> tmp(local_num_col);
      for(O pos = off[row]; pos < off[row+1]; pos++) {
        tmp[idx[pos]] = val[pos];
      }
      for(size_t col = 0; col < local_num_col; col++) {
        std::cout << tmp[col] << " ";
      }
      std::cout << std::endl;
    }
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
  sparse_vector<T,I> get_row(size_t r) const;
  void savebinary(const std::string&);
  void clear() {
    std::vector<T> tmpval; tmpval.swap(val);
    std::vector<I> tmpidx; tmpidx.swap(idx);
    std::vector<O> tmpoff; tmpoff.swap(off); off.push_back(0);
    local_num_row = 0;
    local_num_col = 0;
  }
};

template <class T, class I, class O>
sparse_vector<T,I> crs_matrix_local<T,I,O>::get_row(size_t k) const {
  sparse_vector<T,I> r;
  if(k > local_num_row) throw std::runtime_error("get_row: invalid position");
  size_t size = off[k+1] - off[k];
  r.val.resize(size);
  r.idx.resize(size);
  auto offp = off.data();
  auto valp = val.data();
  auto idxp = idx.data();
  auto rvalp = r.val.data();
  auto ridxp = r.idx.data();
  for(size_t i = 0; i < size; i++) rvalp[i] = valp[offp[k]+i];
  for(size_t i = 0; i < size; i++) ridxp[i] = idxp[offp[k]+i];
  r.size = local_num_col;
  return r;
}

/*
  see http://financelab.nctu.edu.tw/DataStructure/lec05.pdf
  shared with ccs_matrix_local, but row/col is for crs_matrix_local
  exchange local_num_row and local_num_col in the case of ccs_matrix_local
 */
template <class T, class I, class O>
void transpose_compressed_matrix(std::vector<T>& val,
                                 std::vector<I>& idx,
                                 std::vector<O>& off,
                                 std::vector<T>& ret_val,
                                 std::vector<I>& ret_idx,
                                 std::vector<O>& ret_off,
                                 size_t local_num_row,
                                 size_t local_num_col) {
  ret_val.resize(val.size());
  ret_idx.resize(idx.size());
  ret_off.resize(local_num_col + 1);
  std::vector<O> num_item(local_num_col);
  std::vector<O> current_item(local_num_col);
  O* num_itemp = &num_item[0];
  O* current_itemp = &current_item[0];
  T* ret_valp = &ret_val[0];
  I* ret_idxp = &ret_idx[0];
  O* ret_offp = &ret_off[0];

  T* valp = &val[0];
  I* idxp = &idx[0];
  O* offp = &off[0];

  for(size_t src_row = 0; src_row < local_num_row; src_row++) {
#pragma cdir nodep
#pragma _NEC ivdep
    for(O src_pos = offp[src_row]; src_pos < offp[src_row + 1];
        src_pos++) {
      num_itemp[idxp[src_pos]]++;
    }
  }
  prefix_sum(num_itemp, ret_offp+1, local_num_col);
  for(size_t src_row = 0; src_row < local_num_row; src_row++) {
#pragma cdir nodep
#pragma _NEC ivdep
    for(O src_pos = offp[src_row]; src_pos < offp[src_row + 1];
        src_pos++) {
      auto src_col = idxp[src_pos];
      T src_val = valp[src_pos];
      auto dst_pos = ret_offp[src_col] + current_itemp[src_col];
      ret_valp[dst_pos] = src_val;
      ret_idxp[dst_pos] = src_row;
      current_itemp[src_col]++;
    }
  }
}

template <class T, class I, class O>
crs_matrix_local<T,I,O> crs_matrix_local<T,I,O>::transpose() {
  crs_matrix_local<T,I,O> ret;
  transpose_compressed_matrix(val, idx, off, ret.val, ret.idx, ret.off,
                              local_num_row, local_num_col);
  ret.local_num_col = local_num_row;
  ret.local_num_row = local_num_col;
  return ret;
}

template <class T, class I, class O>
crs_matrix_local<T,I,O> crs_matrix_local<T,I,O>::pow_val(T exponent) {
  crs_matrix_local<T,I,O> ret(*this);
  auto* valp = ret.val.data();
  auto valsize = ret.val.size();
  for (size_t j = 0; j < valsize; j++) {
    valp[j] = std::pow(valp[j], exponent);
  }
  return ret;
}

template <class T, class I, class O>
rowmajor_matrix_local<T> crs_matrix_local<T,I,O>::to_rowmajor() {
  rowmajor_matrix_local<T> ret(local_num_row, local_num_col);
  T* retvalp = ret.val.data();
  T* valp = val.data();
  I* idxp = idx.data();
  O* offp = off.data();
  for(size_t row = 0; row < local_num_row; row++) {
    for(O pos = offp[row]; pos < offp[row+1]; pos++) {
      retvalp[local_num_col * row + idxp[pos]] = valp[pos];
    }
  }
  return ret;
}

inline std::string remove_schema(const std::string& path) {
  auto idx = path.find(':', 0);
  if(idx == std::string::npos) return path;
  else return path.substr(idx + 1);
}

/*
  The directory should contain following files:
  - val: big endian binary data file that contains values of the matrix 
         whose type is T in row major order
  - idx: big endian binary data whose type is I that contains column index of
         the value
  - off: big endian binary data whose type is O that contains offset of
         the row
  - nums: text file that contains num_row in the first line and
          num_col in the second line
 */
template <class T, class I = size_t, class O = size_t>
crs_matrix_local<T,I,O>
make_crs_matrix_local_loadbinary(const std::string& input) {
  std::string valfile = input + "/val";
  std::string idxfile = input + "/idx";
  std::string offfile = input + "/off";
#if defined(_SX) || defined(__ve__)
  std::string numsfile = remove_schema(input + "/nums");
#else  
  std::string numsfile = input + "/nums";
#endif
  std::ifstream numstr;
  numstr.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  numstr.open(numsfile.c_str());
  size_t num_row, num_col;
  numstr >> num_row >> num_col;
  crs_matrix_local<T,I,O> ret;
  auto loadval = loadbinary<T>(valfile);
  ret.val.swap(loadval);
  auto loadidx = loadbinary<I>(idxfile);
  ret.idx.swap(loadidx);
  auto loadoff = loadbinary<O>(offfile);
  ret.off.swap(loadoff);
  ret.set_local_num(num_col);
  return ret;
}

template <class T, class I, class O>
void crs_matrix_local<T,I,O>::savebinary(const std::string& dir) {
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
  std::string idxfile = dir + "/idx";
  std::string offfile = dir + "/off";
  std::string numsfile = dir + "/nums";
  std::ofstream numstr;
  numstr.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  numstr.open(numsfile.c_str());
  numstr << local_num_row << "\n" << local_num_col << std::endl;
  frovedis::savebinary(val,valfile);
  frovedis::savebinary(idx,idxfile);
  frovedis::savebinary(off,offfile);
}

template <class T>
T strtox(char* s, char** next);

template <>
float strtox<float>(char* s, char** next);

template <>
double strtox<double>(char* s, char** next);

template <>
int strtox<int>(char* s, char** next);

template <class T, class I, class O>
void make_crs_matrix_local_parseline(const std::string& line,
                                     crs_matrix_local<T,I,O>& ret) {
  char* s = const_cast<char*>(line.c_str());
  while(*s != '\0') {
    char* del;
    long long pos = strtoll(s, &del, 10);
    if(del == s) break;
    ret.idx.push_back(pos);
    s = del + 1;
    char* next;
    T val = strtox<T>(s, &next);
    ret.val.push_back(val);
    s = next;
  }
}

template <class T, class I, class O>
crs_matrix_local<T,I,O>
make_crs_matrix_local_readstream(std::istream& str) {
  crs_matrix_local<T,I,O> ret;
  std::string line;
  while(std::getline(str,line)) {
    make_crs_matrix_local_parseline<T,I,O>(line, ret);
    ret.off.push_back(ret.val.size());
  }
  return ret;
}

template <class T, class I = size_t, class O = size_t>
crs_matrix_local<T,I,O>
make_crs_matrix_local_load(const std::string& file, size_t num_col) {
  std::ifstream str(file.c_str());
  auto ret =  make_crs_matrix_local_readstream<T,I,O>(str);
  ret.local_num_row = ret.off.size() - 1;
  ret.local_num_col = num_col;
  return ret;
}

template <class T, class I = size_t, class O = size_t>
crs_matrix_local<T,I,O>
make_crs_matrix_local_load(const std::string& file) {
  std::ifstream str(file.c_str());
  auto ret =  make_crs_matrix_local_readstream<T,I,O>(str);
  ret.local_num_row = ret.off.size() - 1;
  auto it = std::max_element(ret.idx.begin(), ret.idx.end());
  if(it != ret.idx.end()) ret.local_num_col = *it + 1;
  else ret.local_num_col = 0;
  return ret;
}

// used for making (distributed) crs_matrix; size is set outside
template <class T, class I, class O>
crs_matrix_local<T,I,O>
make_crs_matrix_local_vectorstring(std::vector<std::string>& vs) {
  crs_matrix_local<T,I,O> ret;
  for(size_t i = 0; i < vs.size(); i++) {
    make_crs_matrix_local_parseline(vs[i], ret);
    ret.off.push_back(ret.val.size());
  }
  return ret;
}

template <class T, class I>
struct coo_triplet {
  I i;
  I j;
  T v;
  bool operator<(const coo_triplet<T,I>& r) const {
    if(i != r.i) return i < r.i;
    else if(j != r.j) return j < r.j;
    else return false;
  }
};

template <class T, class I>
std::vector<coo_triplet<T,I>> 
parse_coo_triplet(std::vector<std::string>& s, bool zero_origin) {
  std::vector<coo_triplet<T,I>> ret;
  for(size_t i = 0; i < s.size(); i++) {
    if(s[i][0] == '#' || s[i].size() == 0) continue;
    else {
      std::istringstream is(s[i]);
      coo_triplet<T,I> r;
      double tmp_i, tmp_j; // index might be expressed as floating point
      is >> tmp_i >> tmp_j >> r.v;
      r.i = static_cast<I>(tmp_i);
      r.j = static_cast<I>(tmp_j);
      if(!zero_origin) {
        r.i -= 1;
        r.j -= 1;
      }
      ret.push_back(r);
    }
  }
  return ret;
}

template <class T, class I, class O>
crs_matrix_local<T,I,O>
coo_to_crs(const std::vector<coo_triplet<T,I>>& coo, I current_line) {
  crs_matrix_local<T,I,O> ret;
  for(size_t i = 0; i < coo.size(); i++) {
    if(current_line == coo[i].i) {
      ret.idx.push_back(coo[i].j);
      ret.val.push_back(coo[i].v);
    } else {
      for(; current_line < coo[i].i; current_line++)
        ret.off.push_back(ret.val.size());
      ret.idx.push_back(coo[i].j);
      ret.val.push_back(coo[i].v);
    }
  }
  if(coo.size() != 0) ret.off.push_back(ret.val.size());
  return ret;
}

template <class T, class I = size_t, class O = size_t>
crs_matrix_local<T,I,O>
make_crs_matrix_local_loadcoo(const std::string& file, 
                              bool zero_origin = false) {
  std::ifstream ifs(file.c_str());
  std::vector<std::string> vecstr;
  std::string line;
  while(std::getline(ifs,line)) {vecstr.push_back(line);}
  auto parsed = parse_coo_triplet<T,I>(vecstr, zero_origin);
  std::sort(parsed.begin(), parsed.end());
  I current_line = 0;
  crs_matrix_local<T,I,O> ret = coo_to_crs<T,I,O>(parsed, current_line);
  ret.local_num_row = ret.off.size() - 1;
  auto it = std::max_element(ret.idx.begin(), ret.idx.end());
  ret.local_num_col = *it + 1;
  return ret;
}

template <class T>
std::vector<T> loadlibsvm_extract_label(std::vector<std::string>& vecstr) {
  std::vector<T> ret;
  for(size_t i = 0; i < vecstr.size(); i++) {
    char* next;
    ret.push_back(strtox<T>(const_cast<char*>(vecstr[i].c_str()), &next));
    vecstr[i].erase(0, next - vecstr[i].c_str());
  }
  return ret;
}

template <class T, class I = size_t, class O = size_t>
crs_matrix_local<T,I,O>
make_crs_matrix_local_loadlibsvm(const std::string& file, 
                                 std::vector<T>& label) {
  std::ifstream ifs(file.c_str());
  std::vector<std::string> vecstr;
  std::string line;
  while(std::getline(ifs,line)) {vecstr.push_back(line);}
  label = loadlibsvm_extract_label<T>(vecstr);
  auto ret = make_crs_matrix_local_vectorstring<T,I,O>(vecstr);
  ret.local_num_row = ret.off.size() - 1;
  auto it = std::max_element(ret.idx.begin(), ret.idx.end());
  ret.local_num_col = *it + 1;
  for(size_t i = 0; i < ret.idx.size(); i++) {
    ret.idx[i]--; // one origin to zero origin
  }
  return ret;
}

template <class T, class I, class O>
void crs_matrix_spmv_impl(const crs_matrix_local<T,I,O>& mat, T* retp,
                          const T* vp) {
  const T* valp = &mat.val[0];
  const I* idxp = &mat.idx[0];
  const O* offp = &mat.off[0];
  for(size_t r = 0; r < mat.local_num_row; r++) {
#pragma cdir on_adb(vp)
    for(O c = offp[r]; c < offp[r+1]; c++) {
      retp[r] = retp[r] + valp[c] * vp[idxp[c]];
    }
  }
}

template <class T, class I, class O>
std::vector<T> operator*(const crs_matrix_local<T,I,O>& mat,
                         const std::vector<T>& v) {
  std::vector<T> ret(mat.local_num_row);
  if(mat.local_num_col != v.size())
    throw std::runtime_error("operator*: size of vector does not match");
  crs_matrix_spmv_impl(mat, ret.data(), v.data());
  return ret;
}

template <class T, class I, class O>
std::ostream& operator<<(std::ostream& str, crs_matrix_local<T,I,O>& mat) {
  for(size_t row = 0; row < mat.local_num_row; row++) {
    for(O col = mat.off[row]; col < mat.off[row + 1]; col++) {
      str << mat.idx[col] << ":" << mat.val[col];
      if(col != mat.off[row + 1] - 1) str << " ";
    }
    str << "\n";
  }
  return str;
}

template <class T, class I, class O>
std::vector<crs_matrix_local<T,I,O>> 
get_scattered_crs_matrices(crs_matrix_local<T,I,O>& data,
                           size_t node_size) {
  size_t total = data.off[data.off.size() - 1];
  size_t each_size = frovedis::ceil_div(total, node_size);
  std::vector<size_t> divide_row(node_size+1);
  for(size_t i = 0; i < node_size + 1; i++) {
    auto it = std::lower_bound(data.off.begin(), data.off.end(),
                               each_size * i);
    if(it != data.off.end()) {
      divide_row[i] = it - data.off.begin();
    } else {
      divide_row[i] = data.local_num_row;
    }
  }
  std::vector<crs_matrix_local<T,I,O>> vret(node_size);
  T* datavalp = &data.val[0];
  I* dataidxp = &data.idx[0];
  O* dataoffp = &data.off[0];
  for(size_t i = 0; i < node_size; i++) {
    vret[i].local_num_col = data.local_num_col;
    size_t start_row = divide_row[i];
    size_t end_row = divide_row[i+1];
    vret[i].local_num_row = end_row - start_row;
    size_t start_off = dataoffp[start_row];
    size_t end_off = dataoffp[end_row];
    size_t off_size = end_off - start_off;
    vret[i].val.resize(off_size);
    vret[i].idx.resize(off_size);
    vret[i].off.resize(end_row - start_row + 1); // off[0] == 0 by ctor
    T* valp = &vret[i].val[0];
    I* idxp = &vret[i].idx[0];
    O* offp = &vret[i].off[0];
    for(size_t j = 0; j < off_size; j++) {
      valp[j] = datavalp[j + start_off];
      idxp[j] = dataidxp[j + start_off];
    }
    for(size_t j = 0; j < end_row - start_row; j++) {
      offp[j+1] = offp[j] + (dataoffp[start_row + j + 1] -
                             dataoffp[start_row + j]);
    }
  }
  return vret;
}

#if defined(_SX) || defined(__ve__)
/*
  This version vectorize column dimension of rowmajor matrix
 */
template <class T, class I, class O>
void crs_matrix_spmm_impl2(const crs_matrix_local<T,I,O>& mat,
                           T* retvalp, const T* vvalp, size_t num_col) {
  const T* valp = &mat.val[0];
  const I* idxp = &mat.idx[0];
  const O* offp = &mat.off[0];
  T current_sum[CRS_SPMM_VLEN];
#pragma _NEC vreg(current_sum)
  for(size_t i = 0; i < CRS_SPMM_VLEN; i++) {
    current_sum[i] = 0;
  }
  size_t each = num_col / CRS_SPMM_VLEN;
  size_t rest = num_col % CRS_SPMM_VLEN;
  for(size_t r = 0; r < mat.local_num_row; r++) {
    for(size_t e = 0; e < each; e++) {
      for(O c = offp[r]; c < offp[r+1]; c++) {
        for(size_t mc = 0; mc < CRS_SPMM_VLEN; mc++) {
          current_sum[mc] +=
            valp[c] * vvalp[idxp[c] * num_col + CRS_SPMM_VLEN * e + mc];
        }
      }
      for(size_t mc = 0; mc < CRS_SPMM_VLEN; mc++) {
        retvalp[r * num_col + CRS_SPMM_VLEN * e + mc] += current_sum[mc];
      }
      for(size_t i = 0; i < CRS_SPMM_VLEN; i++) {
        current_sum[i] = 0;
      }
    }
    for(O c = offp[r]; c < offp[r+1]; c++) {
      for(size_t mc = 0; mc < rest; mc++) {
        current_sum[mc] +=
          valp[c] * vvalp[idxp[c] * num_col + CRS_SPMM_VLEN * each + mc];
      }
    }
    for(size_t mc = 0; mc < rest; mc++) {
      retvalp[r * num_col + CRS_SPMM_VLEN * each + mc] += current_sum[mc];
    }
    for(size_t i = 0; i < rest; i++) {
      current_sum[i] = 0;
    }
  }
}

template <class T, class I, class O>
void crs_matrix_spmm_impl(const crs_matrix_local<T,I,O>& mat,
                          T* retvalp, const T* vvalp, size_t num_col) {
  if(num_col < CRS_SPMM_THR) {
    const T* valp = &mat.val[0];
    const I* idxp = &mat.idx[0];
    const O* offp = &mat.off[0];
    for(size_t r = 0; r < mat.local_num_row; r++) {
      size_t mc = 0;
      for(; mc + 15 < num_col; mc += 16) {
        for(O c = offp[r]; c < offp[r+1]; c++) {
          retvalp[r * num_col + mc] += 
            valp[c] * vvalp[idxp[c] * num_col + mc];
          retvalp[r * num_col + mc + 1] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 1];
          retvalp[r * num_col + mc + 2] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 2];
          retvalp[r * num_col + mc + 3] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 3];
          retvalp[r * num_col + mc + 4] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 4];
          retvalp[r * num_col + mc + 5] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 5];
          retvalp[r * num_col + mc + 6] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 6];
          retvalp[r * num_col + mc + 7] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 7];
          retvalp[r * num_col + mc + 8] += 
            valp[c] * vvalp[idxp[c] * num_col + mc + 8];
          retvalp[r * num_col + mc + 9] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 9];
          retvalp[r * num_col + mc + 10] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 10];
          retvalp[r * num_col + mc + 11] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 11];
          retvalp[r * num_col + mc + 12] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 12];
          retvalp[r * num_col + mc + 13] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 13];
          retvalp[r * num_col + mc + 14] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 14];
          retvalp[r * num_col + mc + 15] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 15];
        }
      }
      for(; mc + 7 < num_col; mc += 8) {
        for(O c = offp[r]; c < offp[r+1]; c++) {
          retvalp[r * num_col + mc] += 
            valp[c] * vvalp[idxp[c] * num_col + mc];
          retvalp[r * num_col + mc + 1] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 1];
          retvalp[r * num_col + mc + 2] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 2];
          retvalp[r * num_col + mc + 3] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 3];
          retvalp[r * num_col + mc + 4] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 4];
          retvalp[r * num_col + mc + 5] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 5];
          retvalp[r * num_col + mc + 6] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 6];
          retvalp[r * num_col + mc + 7] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 7];
        }
      }
      for(; mc + 3 < num_col; mc += 4) {
        for(O c = offp[r]; c < offp[r+1]; c++) {
          retvalp[r * num_col + mc] += 
            valp[c] * vvalp[idxp[c] * num_col + mc];
          retvalp[r * num_col + mc + 1] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 1];
          retvalp[r * num_col + mc + 2] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 2];
          retvalp[r * num_col + mc + 3] +=
            valp[c] * vvalp[idxp[c] * num_col + mc + 3];
        }
      }
      for(; mc < num_col; mc++) {
        for(O c = offp[r]; c < offp[r+1]; c++) {
          retvalp[r * num_col + mc] +=
            valp[c] * vvalp[idxp[c] * num_col + mc];
        }
      }
    }
    /*
      for(size_t r = 0; r < mat.local_num_row; r++) {
      O c = offp[r];
      for(; c + CRS_VLEN < offp[r+1]; c += CRS_VLEN) {
      for(size_t mc = 0; mc < num_col; mc++) {
      #pragma cdir on_adb(vvalp)
      for(O i = 0; i < CRS_VLEN; i++) {
      retvalp[r * num_col + mc] +=
      valp[c+i] * vvalp[idxp[c+i] * num_col + mc];
      }
      }
      }
      for(size_t mc = 0; mc < num_col; mc++) {
      #pragma cdir on_adb(vvalp)
      for(O i = 0; c + i < offp[r+1]; i++) {
      retvalp[r * num_col + mc] +=
      valp[c+i] * vvalp[idxp[c+i] * num_col + mc];
      }
      }
      c = offp[r+1];
      }
    */
  } else {
    crs_matrix_spmm_impl2(mat, retvalp, vvalp, num_col);
  }
}
#else
template <class T, class I, class O>
void crs_matrix_spmm_impl(const crs_matrix_local<T,I,O>& mat,
                          T* retvalp, const T* vvalp, size_t num_col) {
  const T* valp = &mat.val[0];
  const I* idxp = &mat.idx[0];
  const O* offp = &mat.off[0];
  for(size_t r = 0; r < mat.local_num_row; r++) {
    for(O c = offp[r]; c < offp[r+1]; c++) {
      for(size_t mc = 0; mc < num_col; mc++) {
        retvalp[r * num_col + mc] +=
          valp[c] * vvalp[idxp[c] * num_col + mc];
      }
    }
  }
}
#endif

template <class T, class I, class O>
rowmajor_matrix_local<T> operator*(const crs_matrix_local<T,I,O>& mat,
                                   const rowmajor_matrix_local<T>& v) {
  rowmajor_matrix_local<T> ret(mat.local_num_row, v.local_num_col);
  T* retvalp = &ret.val[0];
  const T* vvalp = &v.val[0];
  crs_matrix_spmm_impl(mat, retvalp, vvalp, v.local_num_col);
  return ret;
}

template <class T>
template <class I, class O>
crs_matrix_local<T,I,O> rowmajor_matrix_local<T>::to_crs() {
  crs_matrix_local<T,I,O> ret;
  ret.local_num_row = local_num_row;
  ret.local_num_col = local_num_col;
  size_t nnz = 0;
  T* valp = val.data();
  size_t valsize = val.size();
  for(size_t i = 0; i < valsize; i++) {
    if(valp[i] != 0) nnz++;
  }
  ret.val.resize(nnz);
  ret.idx.resize(nnz);
  ret.off.resize(local_num_row + 1);
  size_t current = 0;
  T* retvalp = ret.val.data();
  I* retidxp = ret.idx.data();
  O* retoffp = ret.off.data();
  for(size_t i = 0; i < local_num_row; i++) {
    for(size_t j = 0; j < local_num_col; j++) {
      T v = valp[i * local_num_col + j];
      if(v != 0) {
        retvalp[current] = v;
        retidxp[current] = j;
        current++;
      }
    }
    retoffp[i+1] = current;
  }
  return ret;
}

}
#endif
