#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

// ==================== 类型定义 ====================
typedef int BLASINT;
typedef double _Complex zdouble;

enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
};
enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};

// ==================== 参赛者需要实现的函数 ====================
void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                 const BLASINT K, const void* alpha, const void* A, const BLASINT lda,
                 const void* B, const BLASINT ldb, const void* beta, void* C, const BLASINT ldc);

// ==================== 辅助函数 ====================

// 生成随机数，线性同余法
double rand_lcg(int* localSeed)
{
    int high = *localSeed / 127773;
    int low = *localSeed % 127773;
    int test = 16807 * low - 2836 * high;
    int value = test + ((test > 0) ? 0 : 2147483647);
    *localSeed = value;
    return (double)value / 2147483647.0;
}

// 生成随机复数矩阵
void random_zmatrix(zdouble* mat, BLASINT rows, BLASINT cols, BLASINT lda, int seed)
{
#pragma omp parallel
    {
        int localSeed = seed + omp_get_thread_num();
#pragma omp for
        for (BLASINT i = 0; i < rows; i++) {
            for (BLASINT j = 0; j < cols; j++) {
                double real_part = rand_lcg(&localSeed) * 10.0 - 5.0;
                double imag_part = rand_lcg(&localSeed) * 10.0 - 5.0;
                mat[i * lda + j] = real_part + imag_part * I;
            }
        }
    }
}

// 辅助函数：获取转置后的维度
static inline BLASINT get_A_rows(enum CBLAS_TRANSPOSE TransA, BLASINT M, BLASINT K)
{
    return (TransA == CblasNoTrans) ? M : K;
}

static inline BLASINT get_A_cols(enum CBLAS_TRANSPOSE TransA, BLASINT M, BLASINT K)
{
    return (TransA == CblasNoTrans) ? K : M;
}

static inline BLASINT get_B_rows(enum CBLAS_TRANSPOSE TransB, BLASINT K, BLASINT N)
{
    return (TransB == CblasNoTrans) ? K : N;
}

static inline BLASINT get_B_cols(enum CBLAS_TRANSPOSE TransB, BLASINT K, BLASINT N)
{
    return (TransB == CblasNoTrans) ? N : K;
}
// 参考实现（使用标准库作为基准）
void reference_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                     const BLASINT K, const zdouble* alpha, const zdouble* A, const BLASINT lda,
                     const zdouble* B, const BLASINT ldb, const zdouble* beta, zdouble* C,
                     const BLASINT ldc)
{
    // 转换为复数指针
    const zdouble* A_ptr = (const zdouble*)A;
    const zdouble* B_ptr = (const zdouble*)B;
    zdouble* C_ptr = (zdouble*)C;
    zdouble alpha_val = *(const zdouble*)alpha;
    zdouble beta_val = *(const zdouble*)beta;

    // 实际维度（考虑转置）
    BLASINT A_rows = get_A_rows(TransA, M, K);
    BLASINT A_cols = get_A_cols(TransA, M, K);
    BLASINT B_rows = get_B_rows(TransB, K, N);
    BLASINT B_cols = get_B_cols(TransB, K, N);

    // 根据存储顺序进行矩阵乘法
    if (Order == CblasRowMajor) {
        // ========== 行主序存储 ==========
        // 首先缩放 C = beta * C
        if (beta_val != 1.0) {
            for (BLASINT i = 0; i < M; i++) {
                for (BLASINT j = 0; j < N; j++) {
                    C_ptr[i * ldc + j] *= beta_val;
                }
            }
        }

// 计算 C = alpha * A * B + C
#pragma omp parallel for
        for (BLASINT i = 0; i < M; i++) {
            for (BLASINT j = 0; j < N; j++) {
                zdouble sum = 0.0;
                for (BLASINT k_idx = 0; k_idx < K; k_idx++) {
                    // 获取 A[i, k_idx]
                    zdouble a_elem;
                    if (TransA == CblasNoTrans) {
                        a_elem = A_ptr[i * lda + k_idx];
                    } else if (TransA == CblasTrans) {
                        a_elem = A_ptr[k_idx * lda + i];
                    } else { // ConjTrans
                        a_elem = conj(A_ptr[k_idx * lda + i]);
                    }

                    // 获取 B[k_idx, j]
                    zdouble b_elem;
                    if (TransB == CblasNoTrans) {
                        b_elem = B_ptr[k_idx * ldb + j];
                    } else if (TransB == CblasTrans) {
                        b_elem = B_ptr[j * ldb + k_idx];
                    } else { // ConjTrans
                        b_elem = conj(B_ptr[j * ldb + k_idx]);
                    }

                    sum += a_elem * b_elem;
                }
                C_ptr[i * ldc + j] = alpha_val * sum + C_ptr[i * ldc + j];
            }
        }
    } else { // CblasColMajor
        // ========== 列主序存储 ==========
        // 首先缩放 C = beta * C
        if (beta_val != 1.0) {
            for (BLASINT i = 0; i < M; i++) {
                for (BLASINT j = 0; j < N; j++) {
                    C_ptr[j * ldc + i] *= beta_val;
                }
            }
        }

// 计算 C = alpha * A * B + C
#pragma omp parallel for
        for (BLASINT i = 0; i < M; i++) {
            for (BLASINT j = 0; j < N; j++) {
                zdouble sum = 0.0;
                for (BLASINT k_idx = 0; k_idx < K; k_idx++) {
                    // 获取 A[i, k_idx] (列主序)
                    zdouble a_elem;
                    if (TransA == CblasNoTrans) {
                        a_elem = A_ptr[k_idx * lda + i];
                    } else if (TransA == CblasTrans) {
                        a_elem = A_ptr[i * lda + k_idx];
                    } else { // ConjTrans
                        a_elem = conj(A_ptr[i * lda + k_idx]);
                    }

                    // 获取 B[k_idx, j] (列主序)
                    zdouble b_elem;
                    if (TransB == CblasNoTrans) {
                        b_elem = B_ptr[j * ldb + k_idx];
                    } else if (TransB == CblasTrans) {
                        b_elem = B_ptr[k_idx * ldb + j];
                    } else { // ConjTrans
                        b_elem = conj(B_ptr[k_idx * ldb + j]);
                    }

                    sum += a_elem * b_elem;
                }
                C_ptr[j * ldc + i] = alpha_val * sum + C_ptr[j * ldc + i];
            }
        }
    }
}

// 结果校验函数
double validate_result(const zdouble* C_result, const zdouble* C_reference, BLASINT M, BLASINT N,
                       BLASINT ldc, enum CBLAS_ORDER Order, double tol, int num_report)
{
    double max_error = 0.0;
    int fail_count = 0;

    for (BLASINT i = 0; i < M; i++) {
        for (BLASINT j = 0; j < N; j++) {
            BLASINT idx = (Order == CblasRowMajor) ? i * ldc + j : j * ldc + i;
            zdouble result_val = C_result[idx];
            zdouble ref_val = C_reference[idx];
            zdouble diff = result_val - ref_val;
            double err = cabs(diff);

            if (err > tol || isnan(err)) {
                if ((++fail_count) <= num_report) {
                    printf("  Error at [%d][%d]: got %.6e%+.6ei, expected %.6e%+.6ei, abs(diff) = "
                           "%.6e\n",
                           i, j, creal(result_val), cimag(result_val), creal(ref_val),
                           cimag(ref_val), err);
                }
            }
            max_error = (max_error < err || isnan(err) ? err : max_error);
        }
    }

    return max_error;
}

// 性能测试函数
void test_performance(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                      const enum CBLAS_TRANSPOSE TransB, BLASINT M, BLASINT N, BLASINT K,
                      int test_runs)
{
    // 分配内存
    zdouble* A = (zdouble*)malloc(M * K * sizeof(zdouble));
    zdouble* B = (zdouble*)malloc(K * N * sizeof(zdouble));
    zdouble* C = (zdouble*)malloc(M * N * sizeof(zdouble));
    zdouble* C_ref = (zdouble*)malloc(M * N * sizeof(zdouble));
    zdouble* C_check = (zdouble*)malloc(M * N * sizeof(zdouble));

    if (!A || !B || !C || C_ref == NULL || C_check == NULL) {
        printf("内存分配失败\n");
        free(A);
        free(B);
        free(C);
        free(C_ref);
        free(C_check);
        return;
    }

    zdouble alpha = 1.0 + 0.5 * I;
    zdouble beta = 0.2 - 0.3 * I;

    // 生成测试数据
    random_zmatrix(A, M, K, K, 12345); // A: M x K
    random_zmatrix(B, K, N, N, 67890); // B: K x N
    random_zmatrix(C, M, N, N, 13579); // C: M x N

    // 复制到参考矩阵
    memcpy(C_ref, C, M * N * sizeof(zdouble));
    memcpy(C_check, C, M * N * sizeof(zdouble));

    // 计算参考结果
    reference_zgemm(Order, TransA, TransB, M, N, K, &alpha, A, K, B, N, &beta, C_ref, N);

    // 运行待测试的实现
    cblas_zgemm(Order, TransA, TransB, M, N, K, &alpha, A, K, B, N, &beta, C_check, N);

    // 验证结果
    int ldc = N; // 行主序
    if (Order == CblasColMajor) {
        ldc = M;
    }
    
    double tol = 1e-10;
    double max_err = validate_result(C_check, C_ref, M, N, ldc, Order, tol, 10);

    if (max_err > tol || isnan(max_err)) {
        printf("%6d x%6d x%6d %12s %12s %12.2e %8s\n", M, N, K, "/", "/", max_err, "FAIL");
        free(A);
        free(B);
        free(C);
        free(C_ref);
        free(C_check);
        return;
    }

    free(C_check);
    free(C_ref);

    // 正式测试
    double total_time = 0.0;
    zdouble* C_copy = (zdouble*)malloc(M * N * sizeof(zdouble));

    for (int run = 0; run < test_runs; run++) {
        memcpy(C_copy, C, M * N * sizeof(zdouble));

        double start_time = omp_get_wtime();
        cblas_zgemm(Order, TransA, TransB, M, N, K, &alpha, A, K, B, N, &beta, C_copy, N);
        double end_time = omp_get_wtime();

        total_time += (end_time - start_time);
    }
    free(C_copy);

    double total_flops = 8.0 * M * N * K;
    double avg_time = total_time / test_runs;
    double gflops = total_flops / avg_time * 1e-9;
    double avg_time_ms = avg_time * 1e3;

    printf("%6d x%6d x%6d %12.2f %12.4f %12.2e %8s\n", M, N, K, avg_time_ms, gflops, max_err,
           "PASS");

    free(A);
    free(B);
    free(C);
}

// ==================== 主测试函数 ====================
int main(int argc, char* argv[])
{
    // 测试参数
    BLASINT M = 512;
    BLASINT N = 512;
    BLASINT K = 512;

    int test_runs = 5;

    // 从命令行读取参数
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5) {
        test_runs = atoi(argv[4]);
    }

    // 测试: 行主序，无转置
    test_performance(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, test_runs);

    return 0;
}
