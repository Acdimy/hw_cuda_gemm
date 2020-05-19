// studentid: 2016123456
#include "util.h"

template<class T>
double myGEMM(T* A, T* B, T* C, T alpha, T beta)
{
	printf("perform your gemm here on m=%d n=%d k=%d\n", M, N, K);
	bool preprocess = false;
	if(preprocess)
	{
		// your preprocess
		timestamp(t0);
		// your gemm

		checkCudaErrors(cudaDeviceSynchronize());
		timestamp(t1);
		return getDuration(t0, t1);
	}
	else
	{
		// your gemm
		int col = blockIdx.x * blockDim.x + threadIdx.x;
    	int row = blockIdx.y * blockDim.y + threadIdx.y;
		if( (col < N) && (row < M) ) {
        	T tmp = beta * C[row * N + col];
        	for(int i = 0; i < K; ++i) {
            	tmp += alpha * A[row * K + i] * B[col + i * N];
        	}
        	C[row * N + col] = tmp;
    	}
		checkCudaErrors(cudaDeviceSynchronize());
		return 0.f;	
	}
	
}
