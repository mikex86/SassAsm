__device__ void get_tid_x() {
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;
    int tidZ = threadIdx.z;
    if ((tidX + tidY + tidZ) == 0) {
        *((int*)0x0) = 0;
    }
}

extern "C" __global__ void get_tid_x_kernel() {
    get_tid_x();
}