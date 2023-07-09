#include <stdio.h>
#include <riscv_vector.h>

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */

void AddDot4x4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     In this version, we use registers for elements in the current row
     of B as well */

  float 
    /* Point to the current elements in the four rows of A */
    *a_0p_pntr, *a_1p_pntr, *a_2p_pntr, *a_3p_pntr;

  a_0p_pntr = &A(0, 0);
  a_1p_pntr = &A(1, 0);
  a_2p_pntr = &A(2, 0);
  a_3p_pntr = &A(3, 0);
  size_t vlmax = vsetvlmax_e32m1();

  vfloat32m1_t c_p0_sum = vfmv_v_f_f32m1(0.0f, vlmax);
  vfloat32m1_t c_p1_sum = vfmv_v_f_f32m1(0.0f, vlmax);
  vfloat32m1_t c_p2_sum = vfmv_v_f_f32m1(0.0f, vlmax);
  vfloat32m1_t c_p3_sum = vfmv_v_f_f32m1(0.0f, vlmax);

  register float
  a_0p_reg,
  a_1p_reg,   
  a_2p_reg,
  a_3p_reg;
  for (int p = 0; p < k; ++p) {
    vfloat32m1_t b_reg = vle32_v_f32m1(&B(p, 0), vlmax);

    a_0p_reg = *a_0p_pntr++;
    a_1p_reg = *a_1p_pntr++;
    a_2p_reg = *a_2p_pntr++;
    a_3p_reg = *a_3p_pntr++;

    c_p0_sum = vfmacc_vf_f32m1(c_p0_sum, a_0p_reg, b_reg, vlmax);
    c_p1_sum = vfmacc_vf_f32m1(c_p1_sum, a_1p_reg, b_reg, vlmax);
    c_p2_sum = vfmacc_vf_f32m1(c_p2_sum, a_2p_reg, b_reg, vlmax);
    c_p3_sum = vfmacc_vf_f32m1(c_p3_sum, a_3p_reg, b_reg, vlmax);
    // 累加
  }

  float *c_pntr = 0;
  c_pntr = &C(0, 0);
  vfloat32m1_t c_reg = vle32_v_f32m1(c_pntr, vlmax);
  c_reg = vfadd_vv_f32m1(c_reg, c_p0_sum, vlmax);
  vse32_v_f32m1(c_pntr, c_reg, vlmax);

  c_pntr = &C(1, 0);
  c_reg = vle32_v_f32m1(c_pntr, vlmax);
  c_reg = vfadd_vv_f32m1(c_reg, c_p1_sum, vlmax);
  vse32_v_f32m1(c_pntr, c_reg, vlmax);

  c_pntr = &C(2, 0);
  c_reg = vle32_v_f32m1(c_pntr, vlmax);
  c_reg = vfadd_vv_f32m1(c_reg, c_p2_sum, vlmax);
  vse32_v_f32m1(c_pntr, c_reg, vlmax);

  c_pntr = &C(3, 0);
  c_reg = vle32_v_f32m1(c_pntr, vlmax);
  c_reg = vfadd_vv_f32m1(c_reg, c_p3_sum, vlmax);
  vse32_v_f32m1(c_pntr, c_reg, vlmax);
}


void MY_MMult_4x4_10( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */

      AddDot4x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}
