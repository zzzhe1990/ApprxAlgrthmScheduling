#include "sameVector.h"

__device__ int gpu_sameVectors(int *vecA, int *vecB, int size)
{
	int same = 1;
	
	for (int i = 0; i < size; i++)
	{		
		if (vecA[i]!= vecB[i])
/*
		asm(
//			".reg .b64 rd<5>;\n\t"
//			".reg .b32 r<3>;\n\t"
//			".reg .pred %p;\n\t"
//			"ld.param.u64 rd1, %1;\n\t"
//			"ld.param.u64 rd2, %2;\n\t"
//			"move.b32 r1, %3;\n\t"
			"cvt.s64.s32	%%rd0, %3;\n\t"
			"shl.b64 	%%rd0, %%rd0, 2;\n\t"
			"add.s64 	%%rd3, %1, %%rd0;\n\t"
			"add.s64 	%%rd4, %2, %%rd0;\n\t"
			"ld.u32 	%%r2, [%%rd3];\n\t"
			"ld.u32 	%%r3, [%%rd4];\n\t"
			"setp.ne.s32 %%p0, %%r2, %%r3;\n\t"
			"@%%p0 mov.s32 %0, 0;\n\t"
			: "=r"(temp) : "l"(vecA), "l"(vecB), "r"(i)
		);
		if (temp)
*/
		{
			same = 0;
			break;
		}
	}
	return same;
}

__device__ int gpu_sameVectors(int *vecA, int choice, int size)
{
	int vecB[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0};

	vecB[15] = choice;

	int same = 1;
	for (int i = 0; i < size; i++)
	{
		if (vecA[i] != vecB[i])
		{
			same = 0;
			break;
		}
	}
	return same;
}


__global__ void gpu_sameVectors(int *A, int *B, const int powK, int *res)
{
	int thread = blockDim.x * blockIdx.x + threadIdx.x;
	int tRes;
//	__shared__ int warpRes[32];

//	if (thread < 32)
//		warpRes[thread] = 0;

	if (thread < powK)
	{
		tRes = __all(A[thread]-B[thread]);
	}

//	if (thread&(32-1) == 0)
//	{
//		warpRes[thread/32] = tRes;
//	}

//	if (thread < 32)
//		tRes = __any( warpRes != 0 );

	if (thread == 0)
		res[0] = tRes;
}
