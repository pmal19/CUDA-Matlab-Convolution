#include <matrix.h>
#include <mex.h>
#include "gpu/mxGPUArray.h"

/* Definitions to keep compatibility with earlier versions of ML */
#ifndef MWSIZE_MAX
typedef int mwSize;
typedef int mwIndex;
typedef int mwSignedIndex;

#if (defined(_LP64) || defined(_WIN64)) && !defined(MX_COMPAT_32)
/* Currently 2^48 based on hardware limitations */
# define MWSIZE_MAX    281474976710655UL
# define MWINDEX_MAX   281474976710655UL
# define MWSINDEX_MAX  281474976710655L
# define MWSINDEX_MIN -281474976710655L
#else
# define MWSIZE_MAX    2147483647UL
# define MWINDEX_MAX   2147483647UL
# define MWSINDEX_MAX  2147483647L
# define MWSINDEX_MIN -2147483647L
#endif
#define MWSIZE_MIN    0UL
#define MWINDEX_MIN   0UL
#endif


/*
 * Device code
 */
void __global__ TimesTwo(double const * const A,
                         double * const B,
                         int const N)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        B[i] = 2.0 * A[i];
    }
}

void __global__ convolution(double const * const A,double const * const B,double * const C,int const M,int const N,int const K)
{
	int const col = blockDim.x * blockIdx.x + threadIdx.x;
	int const row = blockDim.y * blockIdx.y + threadIdx.y;

	if((row<N)&&(col<M))
	{
		double sum = 0;
		for(int i=0;i<K;i++)
		{
			for(int j=0;j<K;j++)
			{
				int x = row+i+(1-K)/2;
				int y = col+j+(1-K)/2;
				sum = sum + B[j*K+i]*((x<N&&x>=0&&y<M&&y>=0)?A[y*N+x]:0);
			}
		}
		C[col*N+row] = sum;
	}	
}


/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    mxArray *A, *B;
    double *a, *b;
    mxGPUArray const *a_in_mn;
    mxGPUArray const *b_in_k;
    mxGPUArray *c_out_mn;
    double const *d_a;
    double const *d_b;
    double *d_c;
    const mwSize *dims_mn, *dims_k;
    int m, n, k, i, j;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Associate inputs */
    a_in_mn = mxGPUCreateFromMxArray(prhs[0]);
    b_in_k = mxGPUCreateFromMxArray(prhs[1]);

    A = mxDuplicateArray(prhs[0]);
    B = mxDuplicateArray(prhs[1]);

    /* Figure out dimensions */
    dims_mn = mxGPUGetDimensions(a_in_mn);
    n = (int)dims_mn[0]; m = (int)dims_mn[1];

    dims_k = mxGPUGetDimensions(b_in_k);
    k = (int)dims_k[0];
    
    /*
     * Verify that a_in_mn, b_in_k  really is a double array before extracting the pointer.
     */
    if (mxGPUGetClassID(a_in_mn) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    if (mxGPUGetClassID(b_in_k) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_a = (double const *)(mxGPUGetDataReadOnly(a_in_mn));
    d_b = (double const *)(mxGPUGetDataReadOnly(b_in_k));

    a = mxGetPr(A);
    b = mxGetPr(B);
    /*
    mexPrintf("Input Matrix:\n");
    for(i=0;i<n;i++)
    {
        for(j=0;j<m;j++)
        {
            mexPrintf("%f ",a[j*n+i]);
        }
        mexPrintf("\n");
    }
    mexPrintf("Input Kernel:\n");
    for(i=0;i<k;i++)
    {
        for(j=0;j<k;j++)
        {
            mexPrintf("%f ",b[j*k+i]);
        }
        mexPrintf("\n");
    }
    */
    /* Create a GPUArray to hold the result and get its underlying pointer. */
    c_out_mn = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(a_in_mn),
                            	   mxGPUGetDimensions(a_in_mn),
                            	   mxGPUGetClassID(a_in_mn),
                            	   mxGPUGetComplexity(a_in_mn),
                            	   MX_GPU_DO_NOT_INITIALIZE);
    d_c = (double *)(mxGPUGetData(c_out_mn));

    /* Associate outputs */
    plhs[0] = mxGPUCreateMxArrayOnGPU(c_out_mn);


    mexPrintf("Convolving now..\n");
    /* Call kernel here */

    dim3 DimGrid((m-1)/16+1,(n-1)/16+1,1);
    dim3 DimBlock(16,16,1);

    mexPrintf("CUDA kernel launch with %d blocks of %d threads\n", DimGrid.x*DimGrid.y*DimGrid.z, DimBlock.x*DimBlock.y*DimBlock.z);

    convolution<<<DimGrid, DimBlock>>>(d_a, d_b, d_c, m, n, k);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(a_in_mn);
    mxGPUDestroyGPUArray(b_in_k);
    mxGPUDestroyGPUArray(c_out_mn);
}
