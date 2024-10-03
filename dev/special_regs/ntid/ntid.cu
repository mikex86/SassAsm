__device__ void get_ntid() {
    int ntidx = blockDim.x;
    int ntidy = blockDim.y;
    int ntidz = blockDim.z;
    if ((ntidx + ntidy + ntidz) == 0) {
        *((int*)0x0) = 0;
    }
}

extern "C" __global__ void get_ntid_kernel() {
    get_ntid();
}