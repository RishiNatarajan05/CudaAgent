// Minimal CUDA file with intentional anti-patterns for testing.
#include <cuda_runtime.h>

#define BLOCK 256

__device__ float relu(float x) { return x > 0.f ? x : 0.f; }

// Bad: warp divergence + uncoalesced AoS-like strided write + missing __restrict__
__global__ void bad_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x % 2 == 0) {  // warp divergence
        c[i * 32] = a[i] + b[i];  // strided store
    } else {
        c[i * 32] = a[i] - b[i];
    }
}

// Good: coalesced, restrict-qualified
__global__ void good_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = relu(a[i] + b[i]);
}

// Tiled matmul-ish kernel — uses shared memory and syncthreads
__global__ void tiled_matmul(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int N) {
    __shared__ float As[32][32];  // unpadded — bank conflict on column access
    __shared__ float Bs[32][32];
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    float acc = 0.f;
    for (int k = 0; k < 32; k++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + k];
        Bs[threadIdx.y][threadIdx.x] = B[k * N + col];
        __syncthreads();
        for (int kk = 0; kk < 32; kk++) {
            acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// Bad host: kernel launch + memcpy in a loop
void bad_host(float* h, float* d, int N) {
    for (int i = 0; i < N; i++) {
        cudaMemcpy(d, h + i, sizeof(float), cudaMemcpyHostToDevice);
        good_kernel<<<1, BLOCK>>>(d, d, d, 1);
    }
}
