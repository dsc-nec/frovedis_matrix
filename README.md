# 1. Introduction

This directory contains matrix module extracted from Frovedis
(http://github.com/frovedis/frovedis). Originally, Frovedis is for MPI
environment, but the extracted module removes MPI related part.

The "matrix" directory contains files for matrix operations; the
"core" directory contains files that are used from matrix files. The
"samples" directory includes samples how to use the module.
The "eval" directory include a simple evaluation program.


# 2. How to build

In the case of x86, please use Makefile.x86.

    $ make -f Makefile.x86

In the case of VE, please use Makefile.ve.

    $ make -f Makefile.ve

Please make sure to call "make -f Makefile.[x86,ve] clean" before
building for different architecture.

The make file also builds samples. They will be explained in the next
section. 


# 3. How to use

## 3.1 binary save and load

Though it is not included Frovedis itself, I created simple binary
save / load routine for convenience. Please look at saveload.cc

    #include <core/utility.hpp>
    ...
    using namespace frovedis;
    ...
    vector<float> a = {0,1,2,3,4,5,6,7,8,9};
    savebinary(a, "./saved_binary");

First you need to include `<core/utility.hpp>` to use the function. 
In addition, you need to link with libfrovedis_core.a. If you use
matrix modules, you also need to link with libfrovedis_matrix.a.

The function savebinary takes `std::vector<T>` as input data and
output filename. It is a template function, so you can use any type of
std::vector of POD (e.g. int, float, double). It just saves the memory
image into the file. All the functions / classes defined in the module
is in the namespace frovedis. 

The function loadbinary<T> loads the data from the file.

    auto r = loadbinary<float>("./saved_binary");

Please note that you need type (<float>) here. The result is
std::vector<T>.

If you run the program, you would see the output like:

    0 1 2 3 4 5 6 7 8 9 


## 3.2 rowmajor matrix

The rowmajor matrix is used as basic dense matrix format. It is also
used as the input of SpMM. Please look at rowmajor.cc.

    #include <matrix/rowmajor_matrix.hpp>
    ...
    auto rm = make_rowmajor_matrix_local_load<float>("rowmajor.txt");

This creates rowmajor format of the matrix by reading from the file.
The type of `rm` is `frovedis::rowmajor_matrix_local<float>`.
It is actually a simple struct, so you can directly access the
internal data structure; it includes `std::vector<T> val` as the data,
`size_t local_num_row` as the number of row, and `size_t
local_num_col` as the number of columns.

The format of input file is like:

    1 2 3
    4 5 6
    7 8 9

Each line represents the each row of the matrix. They are separated by
space. 

    cout << rm << endl;

This produces the same output as the input file.

    rm.debug_print();

The debug_print function shows internal data structure, like:

    local_num_row = 3, local_num_col = 3, val = 1 2 3 4 5 6 7 8 9 

You can save and load the data as the binary file.

    rm.savebinary("./saved_rowmajor");
    auto rm2 = make_rowmajor_matrix_local_loadbinary<float>("./saved_rowmajor");

The format of the saved data is a directory. It contains two files
`nums` and `val`. The file nums contains the shape of the matrix. In
this case, 

    3
    3

This means that the shape of the matrix is 3x3. The file val contains
binary format of the data.

Because your program utilizes OpenMP, you might want to separate the
matrix. 

    auto rms = get_scattered_rowmajor_matrices(rm, 2);

This functions creates vector<rowmajor_matrix_local<T>> by separating
the original rowmajor matrix by row. 

    cout << rms[0] << endl;
    cout << rms[1] << endl;

They print the separated `rowmajor_matrix_local<T>`. In this case,
because the number of row is 3, first one has 2 rows, and the second
one has 1 row.

We also provides column major matrix. You can create it by passing
rowmajor matrix as the argument of the constructor:

  colmajor_matrix_local<float> cm(rm);

The column major matrix does not have interesting methods. It just
contains the column major format of the data. Sometimes it is
convenient to pass the data to the libraries that assume column major
format (e.g. BLAS).

  cm.debug_print();

This shows the internal data of the format.


## 3.3 CRS matrix

CRS (or CSR) format is the most popular sparse matrix format.
Please look at crs.cc

    #include <matrix/crs_matrix.hpp>
    ...
    auto crs = make_crs_matrix_local_load<float,int>("crs.txt");

This creates crs format of sparse matrix by reading from the file.
You need to specify the type of the data as the template argument: in
this case float. The second template argument of the type of the
index. In this example, int is specified. If you omit it, the default
type is `size_t`. There is third template argument (that is omitted
here), which specifies the type of offset (the array that contains the
information of where the next row starts). The default type of offset
is `size_t`. 

The format of input file is like:

    0:1 2:2 5:3
    1:4 4:5
    3:6 5:7
    4:8
    1:9 3:10
    0:11 4:12 5:13

Each line represents the each row of the matrix. Each non-zero item
is separated by space. The non-zero item is represented as "POS:VAL";
POS is 0-based.

    cout << crs << endl;

This produces the same output as the input file.

The `debug_print()` function shows internal data structure, like:

    local_num_row = 6, local_num_col = 6
    val : 1 2 3 4 5 6 7 8 9 10 11 12 13 
    idx : 0 2 5 1 4 3 5 4 1 3 0 4 5 
    off : 0 3 5 7 8 10 13 

The `debug_pretty_print()` function shows the data in dense matrix format:

    1 0 2 0 0 3 
    0 4 0 0 5 0 
    0 0 0 6 0 7 
    0 0 0 0 8 0 
    0 9 0 10 0 0 
    11 0 0 0 12 13 

Like rowmajor matrix, you can save and load the data in binary format:

    crs.savebinary("./saved_crs");
    auto crs2 = make_crs_matrix_local_loadbinary<float,int>("./saved_crs");

Again, you need template argument when loading the file.

The format of the saved data is a directory. It contains two files
`nums` and `val`. The file nums contains the shape of the matrix. In
this case, 

    3
    3

This means that the shape of the matrix is 3x3. The file val contains
binary format of the data.

Because your program utilizes OpenMP, you might want to separate the
matrix. 

    auto rms = get_scattered_rowmajor_matrices(rm, 2);

This functions creates vector<rowmajor_matrix_local<T>> by separating
the original rowmajor matrix by row. 

    cout << rms[0] << endl;
    cout << rms[1] << endl;

They print the separated `rowmajor_matrix_local<T>`. In this case,
because the number of row is 3, first one has 2 rows, and the second
one has 1 row.

We also provides column major matrix. You can create it by passing
rowmajor matrix as the argument of the constructor:

  colmajor_matrix_local<float> cm(rm);

The column major matrix does not have interesting methods. It just
contains the column major format of the data. Sometimes it is
convenient to pass the data to the libraries that assume column major
format (e.g. BLAS).

  cm.debug_print();

This shows the internal data of the format.


3.3 CRS matrix

CRS (or CSR) format is the most popular sparse matrix format.
Please look at crs.cc

    #include <matrix/crs_matrix.hpp>
    ...
    auto crs = make_crs_matrix_local_load<float,int>("crs.txt");

This creates crs format of sparse matrix by reading from the file.
You need to specify the type of the data as the template argument: in
this case float. The second template argument of the type of the
index. In this example, int is specified. If you omit it, the default
type is `size_t`. There is third template argument (that is omitted
here), which specifies the type of offset (the array that contains the
information of where the next row starts). The default type of offset
is `size_t`. 

The format of input file is like:

    0:1 2:2 5:3
    1:4 4:5
    3:6 5:7
    4:8
    1:9 3:10
    0:11 4:12 5:13

Each line represents the each row of the matrix. Each non-zero item
is separated by space. The non-zero item is represented as "POS:VAL";
POS is 0-based.

    cout << crs << endl;

This produces the same output as the input file.

The `debug_print()` function shows internal data structure, like:

    local_num_row = 6, local_num_col = 6
    val : 1 2 3 4 5 6 7 8 9 10 11 12 13 
    idx : 0 2 5 1 4 3 5 4 1 3 0 4 5 
    off : 0 3 5 7 8 10 13 

The `debug_pretty_print()` function shows the data in dense matrix format:

    1 0 2 0 0 3 
    0 4 0 0 5 0 
    0 0 0 6 0 7 
    0 0 0 0 8 0 
    0 9 0 10 0 0 
    11 0 0 0 12 13 

Like rowmajor matrix, you can save and load the data in binary format:

    crs.savebinary("./saved_crs");
    auto crs2 = make_crs_matrix_local_loadbinary<float,int>("./saved_crs");

Again, you need template argument when loading the file.

Like rowmajor matrix, you can
The format of the saved data is a directory. It contains two files
`nums` and `val`. The file nums contains the shape of the matrix. In
this case, 

    3
    3

This means that the shape of the matrix is 3x3. The file val contains
binary format of the data.

Because your program utilizes OpenMP, you might want to separate the
matrix. 

    auto rms = get_scattered_rowmajor_matrices(rm, 2);

This functions creates vector<rowmajor_matrix_local<T>> by separating
the original rowmajor matrix by row. 

    cout << rms[0] << endl;
    cout << rms[1] << endl;

They print the separated `rowmajor_matrix_local<T>`. In this case,
because the number of row is 3, first one has 2 rows, and the second
one has 1 row.

We also provides column major matrix. You can create it by passing
rowmajor matrix as the argument of the constructor:

  colmajor_matrix_local<float> cm(rm);

The column major matrix does not have interesting methods. It just
contains the column major format of the data. Sometimes it is
convenient to pass the data to the libraries that assume column major
format (e.g. BLAS).

  cm.debug_print();

This shows the internal data of the format.


3.3 CRS matrix

CRS (or CSR) format is the most popular sparse matrix format.
Please look at crs.cc

    #include <matrix/crs_matrix.hpp>
    ...
    auto crs = make_crs_matrix_local_load<float,int>("crs.txt");

This creates crs format of sparse matrix by reading from the file.
You need to specify the type of the data as the template argument: in
this case float. The second template argument of the type of the
index. In this example, int is specified. If you omit it, the default
type is `size_t`. There is third template argument (that is omitted
here), which specifies the type of offset (the array that contains the
information of where the next row starts). The default type of offset
is `size_t`. 

The format of input file is like:

    0:1 2:2 5:3
    1:4 4:5
    3:6 5:7
    4:8
    1:9 3:10
    0:11 4:12 5:13

Each line represents the each row of the matrix. Each non-zero item
is separated by space. The non-zero item is represented as "POS:VAL";
POS is 0-based.

    cout << crs << endl;

This produces the same output as the input file.

The `debug_print()` function shows internal data structure, like:

    local_num_row = 6, local_num_col = 6
    val : 1 2 3 4 5 6 7 8 9 10 11 12 13 
    idx : 0 2 5 1 4 3 5 4 1 3 0 4 5 
    off : 0 3 5 7 8 10 13 

The `debug_pretty_print()` function shows the data in dense matrix format:

    1 0 2 0 0 3 
    0 4 0 0 5 0 
    0 0 0 6 0 7 
    0 0 0 0 8 0 
    0 9 0 10 0 0 
    11 0 0 0 12 13 

Like rowmajor matrix, you can save and load the data in binary format:

    crs.savebinary("./saved_crs");
    auto crs2 = make_crs_matrix_local_loadbinary<float,int>("./saved_crs");

Again, you need template argument when loading the file.

The format of the saved data is a directory. It contains two files
`nums` and `val`. The file nums contains the shape of the matrix. In
this case, 

    3
    3

This means that the shape of the matrix is 3x3. The file val contains
binary format of the data.

Because your program utilizes OpenMP, you might want to separate the
matrix. 

    auto rms = get_scattered_rowmajor_matrices(rm, 2);

This functions creates vector<rowmajor_matrix_local<T>> by separating
the original rowmajor matrix by row. 

    cout << rms[0] << endl;
    cout << rms[1] << endl;

They print the separated `rowmajor_matrix_local<T>`. In this case,
because the number of row is 3, first one has 2 rows, and the second
one has 1 row.

We also provides column major matrix. You can create it by passing
rowmajor matrix as the argument of the constructor:

  colmajor_matrix_local<float> cm(rm);

The column major matrix does not have interesting methods. It just
contains the column major format of the data. Sometimes it is
convenient to pass the data to the libraries that assume column major
format (e.g. BLAS).

  cm.debug_print();

This shows the internal data of the format.


3.3 CRS matrix

CRS (or CSR) format is the most popular sparse matrix format.
Please look at crs.cc

    #include <matrix/crs_matrix.hpp>
    ...
    auto crs = make_crs_matrix_local_load<float,int>("crs.txt");

This creates crs format of sparse matrix by reading from the file.
You need to specify the type of the data as the template argument: in
this case float. The second template argument of the type of the
index. In this example, int is specified. If you omit it, the default
type is `size_t`. There is third template argument (that is omitted
here), which specifies the type of offset (the array that contains the
information of where the next row starts). The default type of offset
is `size_t`. 

The format of input file is like:

    0:1 2:2 5:3
    1:4 4:5
    3:6 5:7
    4:8
    1:9 3:10
    0:11 4:12 5:13

Each line represents the each row of the matrix. Each non-zero item
is separated by space. The non-zero item is represented as "POS:VAL";
POS is 0-based column index.

    cout << crs << endl;

This produces the same output as the input file.

The `debug_print()` function shows internal data structure, like:

    local_num_row = 6, local_num_col = 6
    val : 1 2 3 4 5 6 7 8 9 10 11 12 13 
    idx : 0 2 5 1 4 3 5 4 1 3 0 4 5 
    off : 0 3 5 7 8 10 13 

The data val contains the non-zero values. The data idx is the column
index of the non-zero value. The data off contains the information of
where the next row starts. In this case, row 0 starts from index 0 of
val/idx, row 1 starts from index 3 of val/idx, etc (index 3 of val/idx
is 4 and 1). Please note that off always starts from 0 and the size of
off is number of row + 1.

The `debug_pretty_print()` function shows the data in dense matrix format:

    1 0 2 0 0 3 
    0 4 0 0 5 0 
    0 0 0 6 0 7 
    0 0 0 0 8 0 
    0 9 0 10 0 0 
    11 0 0 0 12 13 

Like rowmajor matrix, you can save and load the data in binary format:

    crs.savebinary("./saved_crs");
    auto crs2 = make_crs_matrix_local_loadbinary<float,int>("./saved_crs");

Again, you need template argument when loading the file.

The format of the saved data is a directory. It contains four files
`nums`, `val`, `idx`, and `off`. The file nums contains the shape of
the matrix like rowmajor matrix. The file `val`, `idx`, and `off` is
the data that is included in the data structure.

Like rowmajor matrix, you can separate the matrix:

    auto crss = get_scattered_crs_matrices(crs, 2);

The function tries to make the number of non-zeros of the separated
matrices as same as possible.

You can create crs matrix by loading from COO format of a file.

    auto from_coo = make_crs_matrix_local_loadcoo<float,int>("coo.txt", true);

COO format of the file looks like this:

    0 0 1
    0 2 2
    0 5 3
    1 1 4
    1 4 5
    2 3 6
    2 5 7
    3 4 8
    4 1 9
    4 3 10
    5 0 11
    5 4 12
    5 5 13

Each line represents one nonzero value. First item of the line is row
index, second item is column index, and the last item is the value.
Each item is separated by space or tab. The index can be 0 based or 1
based. If the second argument of the function is true, it is treated
as 0 based; otherwise, treated as 1 base. Please note that you can
omit the second argument and the default value is false (1 based).

You can calculate SpMV using the matrix.

    vector<float> v(num_col);
    for(size_t i = 0; i < num_col; i++) v[i] = 1;
    auto r = crs * v;

The multiplying vector is `std::vector<T>`. As you can see, you can
use `operator*` for SpMV. If you want to pass pointer to the vector
as input and output vector, you can use `crs_matrix_spmv_impl`.
The function is like:

    void crs_matrix_spmv_impl(const crs_matrix_local<T,I,O>& mat, 
                              T* retp, const T* vp);

The second argument is pointer to the return vector, the third
argument is the pointer to the input vector. Please note that memory
of the return vector need to be allocated beforehand. 

You can also calculate SpMM likewise. 

    rowmajor_matrix_local<float> rm(num_col, 2);
    for(size_t i = 0; i < num_col * 2; i++) rm.val[i] = 1;
    auto r2 = crs * rm;

The input matrix is rowmajor matrix. You can specify the shape of the
rowmajor matrix as the input of the constructor. Here, the value of
the matrix is initialized as 1.

Like SpMV, you can use `operator*` here. Again, if you want to pass
pointer to the matrix as input and output matrix, you can use
`crs_matrix_spmm_impl`. The function is like:

    void crs_matrix_spmm_impl(const crs_matrix_local<T,I,O>& mat,
                              T* retvalp, const T* vvalp, size_t num_col);

It is similar to SpMV case, but you need to specify the number of
column of the multiplying rowmajor matrix.


## 3.4 Other sparse matrix format

Frovedis also supports other sparse matrix formats. Please look at
othersparse.cc. 

We support CCS (or CSC) format, JDS (or JAD) format, and the hybrid of
JDS and CRS format. JDS format is known to be good for vector
architecture, because it can make the vector length long in the case
of SpMV. The hybrid format is our original format that works good in
the case of matrix whose distribution of non-zero follows power-law.

You can create these formats of matrix by passing crs matrix as the
argument of the constructor, like:

    ccs_matrix_local<float,int> ccs(crs);
    ...
    jds_matrix_local<float,int> jds(crs);
    ...
    jds_crs_hybrid_local<float,int> hyb(crs,2);

All these formats support SpMV and SpMM.

The constructor of the hybrid format takes the threshold of hybrid
format. This example uses 2 for demonstration, but usually you do not
need to specify it (default value is 256).


# 4. Evaluation

The eval directory contains simple evaluation program. The conv.cc
converts downloaded web-Google data (from
https://snap.stanford.edu/data/web-Google.html) to binary crs file
that can be used as the input of the evaluation program (though this
is a bit too small for VE...).

The evalspmv program evaluates SpMV performance and evalspmm program
evaluates SpMM performance.

In the program `time_spent t` data is created for showing execution
time; `t.show("text ")` shows the spent time from previous `show` or
data creation time.

In both cases, SpMV or SpMM call is parallelized by OpenMP. Please try
using different number of threads.

As for SpMM, CRS format switches the implementation. If the number of
column of multiplying matrix is equal or more than 32, the program
vectorizes the direction of the each row of the multiplying rowmajor
matrix. If the number becomes large (e.g. 256), it becomes quite
fast. Otherwise, using the hybrid format is faster. You can change the
number as the argument of the program.