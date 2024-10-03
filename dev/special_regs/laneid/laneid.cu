__device__ void get_lane_id() {
    int laneId;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneId));  // Directly get the lane ID
    if (laneId == 0) {
        *((int*)0x0) = 0;
    }
}

extern "C" __global__ void get_lane_id_kernel() {
    get_lane_id();
}