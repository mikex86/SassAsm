__device__ int fibonacci(int n) {
    if (n <= 1)
        return n;

    int prev = 0, curr = 1, next;

    for (int i = 2; i <= n; i++) {
        next = prev + curr;
        prev = curr;
        curr = next;
    }

    return curr;
}

extern "C" __global__ void fibonacciKernel(int* result, int *n) {
    if (threadIdx.x == 0) {
        *result = fibonacci(*n);
    }
}