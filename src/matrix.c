#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails. Remember to set the error messages in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows < 1 || cols < 1) {
        return -1;
    }
    matrix *ptr = malloc(sizeof(matrix));
    if(NULL == ptr) {
        return -2;
    }
    ptr->cols = cols;
    ptr->rows = rows;
    ptr->parent = NULL;
    ptr->ref_cnt = 1;
    ptr->data = calloc(rows * cols, sizeof(double));
    if(NULL == ptr->data) {
        free(ptr);
        return -2;
    }
    *mat = ptr;
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Remember to set the error messages in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    if (rows < 1 || cols < 1) {
        return -1;
    }
    matrix *ptr = malloc(sizeof(matrix));
    if(NULL == ptr) {
        return -2;
    }
    ptr->cols = cols;
    ptr->rows = rows;
    ptr->parent = from;
    ptr->ref_cnt = 1;
    ptr->data = from->data + offset;
    from->ref_cnt++;
    *mat = ptr;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    if(NULL != mat) {
        if( NULL == mat->parent) {
            mat->ref_cnt--;
            if(mat->ref_cnt == 0) {
                free(mat->data);
                free(mat);
            }
        } else {
            mat->data = NULL;
            mat->parent->ref_cnt--;
            if(mat->parent->ref_cnt == 0) {
                free(mat->parent->data);
                free(mat->parent);
            }
            free(mat);
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return *(mat->data + mat->cols * row + col);
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    *(mat->data + mat->cols * row + col) = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {

    __m256d val_256 = _mm256_set1_pd(val);
    unsigned int size = mat->rows * mat->cols;
    #pragma omp parallel for
    for(unsigned int i = 0; i < size / 16 * 16; i += 16) {
        _mm256_storeu_pd(mat->data + i, val_256);
        _mm256_storeu_pd(mat->data + i + 4, val_256);
        _mm256_storeu_pd(mat->data + i + 8, val_256);
        _mm256_storeu_pd(mat->data + i + 12, val_256);
    }
    for(unsigned int i = size - (size % 16); i < size; i++) {
        mat->data[i] = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (mat1->rows != mat2->rows || mat1->rows != result->rows ||
        mat1->cols != mat2->cols || mat1->cols != result->cols) {
        return -1;
    }
    unsigned int size = mat1->rows * mat1->cols;
    __m256d sum = _mm256_setzero_pd();
    #pragma omp parallel for private(sum)
    for(unsigned int i = 0; i < size / 4 * 4; i += 4) {
        sum = _mm256_add_pd(_mm256_loadu_pd(mat1->data + i),
                            _mm256_loadu_pd(mat2->data + i));
        _mm256_storeu_pd(result->data + i, sum);
    }
    for(unsigned int i = size - (size % 4); i < size; i++) {
        result->data[i] = mat1->data[i] + mat2->data[i];
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (mat1->rows != mat2->rows || mat1->rows != result->rows ||
        mat1->cols != mat2->cols || mat1->cols != result->cols) {
        return -1;
    }
    for(int i = 0; i < mat1->rows * mat1->cols; i++) {
        result->data[i] = mat1->data[i] - mat2->data[i];
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if(mat1->cols != mat2->rows || result->rows != mat1->rows || result->cols != mat2->cols) {
        return -1;
    }
    matrix *transp2 = NULL;
    int allocate_fail = allocate_matrix(&transp2, mat2->cols, mat2->rows);
    if(allocate_fail != 0) {
        return allocate_fail;
    }
    #pragma omp parallel for
    for(unsigned int r = 0; r < transp2->rows; r++) {
        for(unsigned int c = 0; c < transp2->cols; c++) {
            *(transp2->data + transp2->cols * r + c) = *(mat2->data + mat2->cols * c + r);
        }
    }

    #pragma omp parallel for
    for(unsigned int r = 0; r < result->rows; r++) {
        for(unsigned int c = 0; c < result->cols; c++) {
            int size = mat1->cols;
            __m256d sum = _mm256_setzero_pd();
            for(unsigned int k = 0; k < size / 16 * 16; k += 16) {
                sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1->data + (mat1->cols * r + k)),
                                      _mm256_loadu_pd(transp2->data + (transp2->cols * c + k)), sum);

                sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1->data + (mat1->cols * r + k + 4)),
                                      _mm256_loadu_pd(transp2->data + (transp2->cols * c + k + 4)), sum);

                sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1->data + (mat1->cols * r + k + 8)),
                                      _mm256_loadu_pd(transp2->data + (transp2->cols * c + k + 8)), sum);

                sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1->data + (mat1->cols * r + k + 12)),
                                      _mm256_loadu_pd(transp2->data + (transp2->cols * c + k + 12)), sum);
            }
            double sum_arr[4];
            _mm256_storeu_pd(sum_arr, sum);
            double temp_sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
            for(unsigned int k = size - (size % 16); k < size; k++) {
                temp_sum += *(mat1->data + mat1->cols * r + k) * *(transp2->data + transp2->cols * c + k);
            }
            *(result->data + result->cols * r + c) = temp_sum;
        }
    }
    deallocate_matrix(transp2);
    return 0;
}


/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * Got help from this link:
 * @source: https://en.wikipedia.org/wiki/Exponentiation_by_squaring
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: check for power of smaller than 2 */
    if (pow > 1) {
        matrix *temp = NULL;
        int allocate_fail = allocate_matrix(&temp, result->rows, result->cols);
        if(allocate_fail) {
            return allocate_fail;
        }
        int mul_fail = mul_matrix(temp, mat, mat);
        if (mul_fail) {
            return mul_fail;
        }
        if(pow % 2 == 0) {
            pow_matrix(result, temp, pow / 2);
        } else {
            pow_matrix(result, temp, (pow - 1) / 2);
            int mul_fail = mul_matrix(temp, result, mat);
            if (mul_fail) {
                return mul_fail;
            }
            memcpy(result->data, temp->data, sizeof(double) * result->rows * result->cols);
        }
        deallocate_matrix(temp);
    } else if (pow == 0){
        fill_matrix(result, 0);
        #pragma omp parallel for
        for(unsigned int i = 0; i < result->cols; i++) {
            result->data[result->cols * i + i] = 1;
        }
    } else if (pow == 1) {
        memcpy(result->data, mat->data, sizeof(double) * result->rows * result->cols);
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    if (mat->rows != result->rows || mat->cols != result->cols) {
        return -1;
    }
    for(int i = 0; i < mat->rows * mat->cols; i++) {
        result->data[i] = -mat->data[i];
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    if (mat->rows != result->rows || mat->cols != result->cols) {
        return -1;
    }
    __m256d all_zeros = _mm256_setzero_pd();
    unsigned int size = mat->rows * mat->cols;
    #pragma omp parallel for
    for(unsigned int i = 0; i < size / 4 * 4; i += 4) {
        __m256d mat_256 = _mm256_loadu_pd(mat->data + i);
        __m256d mat_256_neg = _mm256_sub_pd(all_zeros, mat_256);
        __m256d mask = _mm256_cmp_pd(mat_256, all_zeros, 0x11);
        __m256d pos_and_zeros = _mm256_and_pd(mask, mat_256_neg);
        __m256d all_pos = _mm256_max_pd(pos_and_zeros, mat_256);
        _mm256_storeu_pd(result->data + i, all_pos);
    }
    for(unsigned int i = size - (size % 4); i < size; i++) {
        if (mat->data[i] < 0) {
            result->data[i] = -mat->data[i];
        } else {
            result->data[i] = mat->data[i];
        }
    }
    return 0;
}
