#if __riscv_vector
#include <riscv_vector.h>
#else
#error("riscv vector not supported")
#endif

#include <assert.h>
#include <stdlib.h>

/* Block sizes */
#define mc 128
#define kc 128
#define DEBUG_PACK_SHAPE
#undef DEBUG_PACK_SHAPE
#define DEBUG_PRINT_DATA
#undef DEBUG_PRINT_DATA

/* Create macros so that the matrices are stored in row-major order */

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define min(i, j) ((i) < (j) ? (i) : (j))

#define GEMM_N (240) // GEMM_R
#define GEMM_M (240) // GEMM_P
#define GEMM_K (240) // GEMM_Q
#define GEMM_UNROLL (4)
#define KERNEL_4x4 kernel_4x4_v2

/* Routine for computing C = A * B + C */
void packB_4(int k, int n, float *from, int ldb, float *to);
void packA_4(int m, int k, float *from, int lda, float *to);
void kernel_4x4_v2(int m, int n, int k, float *sa, float *sb, float *sc,
                   int ldc);

float *fastMalloc(int size) {
  void *ptr = 0;
  int iRet = posix_memalign(&ptr, 64, size * sizeof(float));
  assert(0 == iRet);
  return (float*)ptr;
}

/* Suppose that m%4==0 and n%4==0 and k%4==0, avoiding process boundary !! */
void MY_MMult_4x4_13(int m, int n, int k, float *a, int lda,
              float *b, int ldb, float *c, int ldc) {
#ifdef DEBUG_PRINT_DATA
  printf("\n-------\n");
  print_matrix(m, k, a, lda);
  printf("\n-------\n");
  print_matrix(k, n, b, ldb);
  printf("\n-------\n");
#endif
  float *sa = fastMalloc(m * k);
  float *sb = fastMalloc(k * n);

  int ms, mms, ns, ks;
  int min_m, min_mm, min_n, min_k;
  for (ms = 0; ms < m; ms += GEMM_M) {
    min_m = m - ms;
    if (min_m > GEMM_M) {
      min_m = GEMM_M;
    }

    for (ks = 0; ks < k; ks += min_k) {
      min_k = k - ks;
      if (min_k >= (GEMM_K << 1)) {
        min_k = GEMM_K;
      } else if (min_k > GEMM_K) {
        min_k = (min_k / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
      }

      // first packB
      min_n = n;
      if (n >= GEMM_N * 2) {
        min_n = GEMM_N;
      } else if (n > GEMM_N) {
        min_n = (min_n / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
      }
      packB_4(min_k, min_n, b + ks * ldb, ldb, sb);

      // micro kernel, split A Block to smaller Panel
      for (mms = ms; mms < ms + min_m; mms += min_mm) {
        min_mm = (ms + min_m) - mms;
        if (min_mm >= 3 * GEMM_UNROLL) {
          min_mm = 3 * GEMM_UNROLL;
        } else if (min_mm >= 2 * GEMM_UNROLL) {
          min_mm = 2 * GEMM_UNROLL;
        } else if (min_mm > GEMM_UNROLL) {
          min_mm = GEMM_UNROLL;
        }

        // coninueous packA
        packA_4(min_mm, min_k, a + mms * lda + ks, lda,
                sa + min_k * (mms - ms));

        KERNEL_4x4(min_mm, min_n, min_k, sa + min_k * (mms - ms), sb,
                   c + mms * ldc, ldc);
#ifdef DEBUG_PRINT_DATA
        printf("\n---first kernel----\n");
        print_matrix(m, n, c, ldc);
#endif
      }

      // the first B Block has been packed, proc the others
      for (ns = min_n; ns < n; ns += min_n) {
        min_n = n - ns;
        if (min_n >= GEMM_N * 2) {
          min_n = GEMM_N;
        } else if (min_n > GEMM_N) {
          min_n = (min_n / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
        }

        packB_4(min_k, min_n, b + ns + ldb * ks, ldb, sb);
        KERNEL_4x4(min_m, min_n, min_k, sa, sb, c + ms * ldc + ns, ldc);
#ifdef DEBUG_PRINT_DATA
        printf("\n----second kernel---\n");
        print_matrix(m, n, c, ldc);
#endif
      }
    }
  }

  free(sa);
  free(sb);
}

/**

float* a: A
float* b: (B)T
float* c: C

C = A * (B)T

A1 A2 A3    B1 B4 B7
A4 A5 A6  x B2 B5 B8 => C1 C4 C7 C2 C5 C8 C3 C6 C9 (packed)
A7 A8 A9    B3 B4 B9

Calculation sequence:
1st. calculate C1
2st. calculate C4
3st. calculate C7
...
9st. calculate C9

A1-A9/B1-B9 is packed block, not single number.
C1-C9 is 4x4 block, not single number.

Output
C1 C2 C3
C4 C5 C6
C7 C8 C9

 */
void kernel_4x4_v2(int m, int n, int k, float *sa, float *sb, float *sc,
                   int ldc) {
  assert(m > 0 && n > 0 && k > 0);
  assert(m % 4 == 0 && n % 4 == 0 && k % 4 == 0);

  float *a = sa, *b = sb, *c = sc;
  int i, j, l;
  size_t vlmax = vsetvlmax_e32m1();
  for (i = 0; i < m; i += 4) {
    for (j = 0; j < n; j += 4) {

      vfloat32m1_t v24 = vfmv_v_f_f32m1(0.0f, vlmax);
      vfloat32m1_t v25 = vfmv_v_f_f32m1(0.0f, vlmax);
      vfloat32m1_t v26 = vfmv_v_f_f32m1(0.0f, vlmax);
      vfloat32m1_t v27 = vfmv_v_f_f32m1(0.0f, vlmax);

      for (l = 0; l < k; l += 4) {
        vfloat32m1_t v0 = vle32_v_f32m1(b, vlmax);
        // vfloat32m1_t v16 = vle32_v_f32m1(a, vlmax);

        // v24 = vmlaq_laneq_f32(v24, v0, v16, 0);
        // v25 = vmlaq_laneq_f32(v25, v0, v16, 1);
        // v26 = vmlaq_laneq_f32(v26, v0, v16, 2);
        // v27 = vmlaq_laneq_f32(v27, v0, v16, 3);
        v24 = vfmacc_vf_f32m1(v24, a[0], v0, vlmax);
        v25 = vfmacc_vf_f32m1(v25, a[1], v0, vlmax);
        v26 = vfmacc_vf_f32m1(v26, a[2], v0, vlmax);
        v27 = vfmacc_vf_f32m1(v27, a[3], v0, vlmax);

        vfloat32m1_t v1 = vle32_v_f32m1(b + 4, vlmax);
        vfloat32m1_t v17 = vle32_v_f32m1(a + 4, vlmax);

        v24 = vfmacc_vf_f32m1(v24, a[4], v1, vlmax);
        v25 = vfmacc_vf_f32m1(v25, a[5], v1, vlmax);
        v26 = vfmacc_vf_f32m1(v26, a[6], v1, vlmax);
        v27 = vfmacc_vf_f32m1(v27, a[7], v1, vlmax);

        vfloat32m1_t v2 = vle32_v_f32m1(b + 8, vlmax);
        // vfloat32m1_t v18 = vle32_v_f32m1(a + 8, vlmax);

        v24 = vfmacc_vf_f32m1(v24, a[8], v2, vlmax);
        v25 = vfmacc_vf_f32m1(v25, a[9], v2, vlmax);
        v26 = vfmacc_vf_f32m1(v26, a[10], v2, vlmax);
        v27 = vfmacc_vf_f32m1(v27, a[11], v2, vlmax);

        vfloat32m1_t v3 = vle32_v_f32m1(b + 12, vlmax);
        // vfloat32m1_t v19 = vle32_v_f32m1(a + 12, vlmax);

        v24 = vfmacc_vf_f32m1(v24, a[12], v3, vlmax);
        v25 = vfmacc_vf_f32m1(v25, a[13], v3, vlmax);
        v26 = vfmacc_vf_f32m1(v26, a[14], v3, vlmax);
        v27 = vfmacc_vf_f32m1(v27, a[15], v3, vlmax);

        b += 16;
        a += 16;
      } // endl

      v24 = vfadd_vv_f32m1(vle32_v_f32m1(c, vlmax), v24, vlmax);
      v25 = vfadd_vv_f32m1(vle32_v_f32m1(c + ldc, vlmax), v25, vlmax);
      v26 = vfadd_vv_f32m1(vle32_v_f32m1(c + 2 * ldc, vlmax), v26, vlmax);
      v27 = vfadd_vv_f32m1(vle32_v_f32m1(c + 3 * ldc, vlmax), v27, vlmax);

      vse32_v_f32m1(c, v24, vlmax);
      vse32_v_f32m1(c + ldc, v25, vlmax);
      vse32_v_f32m1(c + 2 * ldc, v26, vlmax);
      vse32_v_f32m1(c + 3 * ldc, v27, vlmax);

      c += 4;
      a -= 4 * k;
    } // endj
    sc += ldc * 4;
    c = sc;
    ;
    a += 4 * k;
    b = sb;
  } // endi
}

/**
pack A means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag

Output:
0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7
8 8 8 8 9 9 9 9 a a a a b b b b
c c c c d d d d e e e e f f f f

Draw it with a line
*/
void packA_4(int m, int k, float *from, int lda, float *to) {
#ifdef DEBUG_PACK_SHAPE
  printf("\n packA_4, m=%d, k=%d", m, k);
#endif
  assert(k != 0 && m != 0 && k % 4 == 0 && m % 4 == 0);
  int i, j;

  float *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
  float *b_offset;
  float ctemp1, ctemp2, ctemp3, ctemp4;
  float ctemp5, ctemp6, ctemp7, ctemp8;
  float ctemp9, ctemp10, ctemp11, ctemp12;
  float ctemp13, ctemp14, ctemp15, ctemp16;

  a_offset = from;
  b_offset = to;

  j = (m >> 2);
  do {
    a_offset1 = a_offset;
    a_offset2 = a_offset1 + lda;
    a_offset3 = a_offset2 + lda;
    a_offset4 = a_offset3 + lda;
    a_offset += 4 * lda;

    i = (k >> 2);
    do {
      ctemp1 = *(a_offset1 + 0);
      ctemp2 = *(a_offset1 + 1);
      ctemp3 = *(a_offset1 + 2);
      ctemp4 = *(a_offset1 + 3);

      ctemp5 = *(a_offset2 + 0);
      ctemp6 = *(a_offset2 + 1);
      ctemp7 = *(a_offset2 + 2);
      ctemp8 = *(a_offset2 + 3);

      ctemp9 = *(a_offset3 + 0);
      ctemp10 = *(a_offset3 + 1);
      ctemp11 = *(a_offset3 + 2);
      ctemp12 = *(a_offset3 + 3);

      ctemp13 = *(a_offset4 + 0);
      ctemp14 = *(a_offset4 + 1);
      ctemp15 = *(a_offset4 + 2);
      ctemp16 = *(a_offset4 + 3);

      *(b_offset + 0) = ctemp1;
      *(b_offset + 1) = ctemp5;
      *(b_offset + 2) = ctemp9;
      *(b_offset + 3) = ctemp13;

      *(b_offset + 4) = ctemp2;
      *(b_offset + 5) = ctemp6;
      *(b_offset + 6) = ctemp10;
      *(b_offset + 7) = ctemp14;

      *(b_offset + 8) = ctemp3;
      *(b_offset + 9) = ctemp7;
      *(b_offset + 10) = ctemp11;
      *(b_offset + 11) = ctemp15;

      *(b_offset + 12) = ctemp4;
      *(b_offset + 13) = ctemp8;
      *(b_offset + 14) = ctemp12;
      *(b_offset + 15) = ctemp16;

      a_offset1 += 4;
      a_offset2 += 4;
      a_offset3 += 4;
      a_offset4 += 4;

      b_offset += 16;
      i--;
    } while (i > 0);
    j--;
  } while (j > 0);
}

/*
suppose that k and n is mutiple of 4
pack B means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag, not like pack A

Output:
0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
8 9 a b 8 9 a b 8 9 a b 8 9 a b
4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7
c d e f c d e f c d e f c d e f
*/
void packB_4(int k, int n, float *from, int ldb, float *to) {
  assert(k != 0 && n != 0 && k % 4 == 0 && n % 4 == 0);
#ifdef DEBUG_PACK_SHAPE
  printf("\n packB_4, k=%d, n=%d", k, n);
#endif

  int i, j;

  float *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
  float *b_offset, *b_offset1;
  float ctemp1, ctemp2, ctemp3, ctemp4;
  float ctemp5, ctemp6, ctemp7, ctemp8;
  float ctemp9, ctemp10, ctemp11, ctemp12;
  float ctemp13, ctemp14, ctemp15, ctemp16;
  a_offset = from;
  b_offset = to;

  j = (k >> 2);
  do {
    a_offset1 = a_offset;
    a_offset2 = a_offset1 + ldb;
    a_offset3 = a_offset2 + ldb;
    a_offset4 = a_offset3 + ldb;
    a_offset += 4 * ldb;

    b_offset1 = b_offset;
    b_offset += 16;

    i = (n >> 2);
    do {
      ctemp1 = *(a_offset1 + 0);
      ctemp2 = *(a_offset1 + 1);
      ctemp3 = *(a_offset1 + 2);
      ctemp4 = *(a_offset1 + 3);

      ctemp5 = *(a_offset2 + 0);
      ctemp6 = *(a_offset2 + 1);
      ctemp7 = *(a_offset2 + 2);
      ctemp8 = *(a_offset2 + 3);

      ctemp9 = *(a_offset3 + 0);
      ctemp10 = *(a_offset3 + 1);
      ctemp11 = *(a_offset3 + 2);
      ctemp12 = *(a_offset3 + 3);

      ctemp13 = *(a_offset4 + 0);
      ctemp14 = *(a_offset4 + 1);
      ctemp15 = *(a_offset4 + 2);
      ctemp16 = *(a_offset4 + 3);

      a_offset1 += 4;
      a_offset2 += 4;
      a_offset3 += 4;
      a_offset4 += 4;

      *(b_offset1 + 0) = ctemp1;
      *(b_offset1 + 1) = ctemp2;
      *(b_offset1 + 2) = ctemp3;
      *(b_offset1 + 3) = ctemp4;

      *(b_offset1 + 4) = ctemp5;
      *(b_offset1 + 5) = ctemp6;
      *(b_offset1 + 6) = ctemp7;
      *(b_offset1 + 7) = ctemp8;

      *(b_offset1 + 8) = ctemp9;
      *(b_offset1 + 9) = ctemp10;
      *(b_offset1 + 10) = ctemp11;
      *(b_offset1 + 11) = ctemp12;

      *(b_offset1 + 12) = ctemp13;
      *(b_offset1 + 13) = ctemp14;
      *(b_offset1 + 14) = ctemp15;
      *(b_offset1 + 15) = ctemp16;

      b_offset1 += k * 4;
      i--;
    } while (i > 0);
    j--;
  } while (j > 0);
}
