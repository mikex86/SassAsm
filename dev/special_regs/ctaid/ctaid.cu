__device__ void get_cta_id() {
    int ctax = blockIdx.x;
    int ctay = blockIdx.y;
    int ctaz = blockIdx.z;
    if ((ctax + ctay + ctaz) == 0) {
        *((int*)0x0) = 0;
    }
}

extern "C" __global__ void get_cta_id_kernel() {
    get_cta_id();
}