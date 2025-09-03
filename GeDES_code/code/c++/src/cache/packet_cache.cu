#include "packet_cache.h"
#include "conf.h"

__global__ void backup_frame_queue_info(VDES::CacheParam *param)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < param->queue_num)
    {
        param->egress_status[tid]->head = param->egress_queues[tid]->head;
        param->egress_status[tid]->size = param->egress_queues[tid]->size;
        /**
         * @TODO: update transmitted_frame_num
         */
        // param->egress_status[tid]->transmitted_frame_num = param->egress_queues[tid]->removed_record;
    }
}

__device__ void update_cache_window(VDES::FrameQueueStatus *status, int expand_size_start, int expand_size_end)
{
    int win_size = status->cache_win_end - status->cache_win_start;

    // update cache window
    int win_size_new = (status->cache_win_end + expand_size_end) - (status->cache_win_start + expand_size_start);
    // update pacekt status in cache window
    // |origin|CPU|GPU|
    int origin_size = max(0, win_size - expand_size_start);
    int cpu_size = max(0, min(status->cache_win_end + expand_size_end, status->last_cache_out_offset) - max(status->cache_win_start + expand_size_start, status->cache_win_end));
    int gpu_size = max(0, min(win_size_new, (status->cache_win_end + expand_size_end) - status->last_cache_out_offset));

    int memset_offset = 0;
    if (origin_size > 0)
    {
        memcpy(status->packet_status_in_cache_win, status->packet_status_in_cache_win + expand_size_start, origin_size * sizeof(uint8_t));
        memset_offset += origin_size;
    }
    if (cpu_size > 0)
    {
        memset(status->packet_status_in_cache_win + memset_offset, 0, cpu_size * sizeof(uint8_t));
        memset_offset += cpu_size;
    }

    if (gpu_size > 0)
    {
        memset(status->packet_status_in_cache_win + memset_offset, 1, gpu_size * sizeof(uint8_t));
    }

    status->cache_win_start += expand_size_start;
    status->cache_win_end += expand_size_end;
}

__global__ void cache_frame_kernel(VDES::CacheParam *param)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < param->queue_num)
    {
        VDES::FrameQueueStatus *status = param->egress_status[tid];
        GPUQueue<VDES::Frame *> *egress_queue = param->egress_queues[tid];
        int *swap_in_num = param->swap_in_frame_num;
        int *swap_out_num = param->swap_out_frame_num;
        VDES::Frame **swap_in_frames = param->swap_in_frame_egress + tid * MAX_TRANSMITTED_PACKET_NUM;
        VDES::Frame **swap_out_frames = param->swap_out_frame_egress + tid * MAX_SWAP_FRAME_NUM;
        VDES::Frame *swap_out_cache = param->swap_out_cache_space + tid * MAX_SWAP_FRAME_NUM;

        // update cache window, remove transmitted frames
        status->cache_win_start -= status->transmitted_frame_num;
        status->cache_win_end -= status->transmitted_frame_num;
        status->last_cache_out_offset -= status->transmitted_frame_num;

        // expand cache window if necessary
        int safe_frame_interval = param->lookahead_timeslot_num * MAX_TRANSMITTED_PACKET_NUM;

        int expand_size_start = 0;
        if (status->cache_win_start < safe_frame_interval + MAX_TRANSMITTED_PACKET_NUM)
        {
            expand_size_start = min(status->size - status->cache_win_start, MAX_TRANSMITTED_PACKET_NUM);
        }
        // exclude gpu frames

        int expand_size_end = 0;
        if (status->cache_win_end < safe_frame_interval * 2 + MAX_TRANSMITTED_PACKET_NUM)
        {
            expand_size_end = min(status->size - status->cache_win_end, MAX_TRANSMITTED_PACKET_NUM);
        }
        int max_swap_out_num = status->size - max(status->last_cache_out_offset, status->cache_win_end + expand_size_end);
        max_swap_out_num = max(max_swap_out_num, 0);
        swap_out_num[tid] = min(max_swap_out_num, MAX_SWAP_FRAME_NUM);

        // swap in frames
        int swap_in_frame_num = 0;
        int win_size = status->cache_win_end - status->cache_win_start;
        for (int i = 0; i < expand_size_start; i++)
        {
            // exchange frame ptr
            int index = status->cache_win_start + i;
            if (i < win_size && status->packet_status_in_cache_win[i] == 0 || index >= status->cache_win_end && index < status->last_cache_out_offset)
            {
                // frame host in cpu memory
                VDES::Frame *temp = egress_queue->get_element(status->head, index);
                egress_queue->set_element(status->head, index, swap_in_frames[swap_in_frame_num]);
                swap_in_frames[swap_in_frame_num] = temp;
                swap_in_frame_num++;
            }
        }
        swap_in_num[tid] = swap_in_frame_num;

        // swap out frames
        int swap_out_offset = max(status->cache_win_end + expand_size_end, status->last_cache_out_offset);
        for (int i = 0; i < swap_out_num[tid]; i++)
        {
            // exchange frame ptr
            int index = swap_out_offset + i;
            VDES::Frame *temp = egress_queue->get_element(status->head, index);
            egress_queue->set_element(status->head, index, swap_out_frames[i]);
            swap_out_frames[i] = temp;

            // copy frame content to cache space
            VDES::Frame *dst_frame = swap_out_cache + i;
            memcpy(dst_frame, temp, sizeof(VDES::Frame));
        }
        // update cache window
        update_cache_window(status, expand_size_start, expand_size_end);

        status->last_cache_out_offset = swap_out_offset + swap_out_num[tid];
    }
}

__global__ void copy_frames_from_cache_space(VDES::CacheParam *param)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < param->queue_num)
    {
        int *swap_in_frame_num = param->swap_in_frame_num;

        int swap_offset = tid * MAX_TRANSMITTED_PACKET_NUM;
        VDES::Frame **swap_in_frame = param->swap_in_frame_egress + swap_offset;
        VDES::Frame *cache_space = param->swap_in_cache_space + swap_offset;

        for (int i = 0; i < swap_in_frame_num[tid]; i++)
        {
            // copy frames from cache space to gpu
            memcpy(swap_in_frame[i], cache_space + i, sizeof(VDES::Frame));
        }
    }
}

namespace VDES
{
    void LaunchCacheFrameKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream)
    {
        cache_frame_kernel<<<grid_dim, block_dim, 0, stream>>>(kernel_param);
    }

    void LaunchCopyFrameCacheSpaceKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream)
    {
        copy_frames_from_cache_space<<<grid_dim, block_dim, 0, stream>>>(kernel_param);
    }

    void LaunchBackupFrameQueueInfoKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream)
    {
        backup_frame_queue_info<<<grid_dim, block_dim, 0, stream>>>(kernel_param);
    }
}

// __global__ void backup_frame_queue_info(VDES::CacheParam *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->queue_num)
//     {
//         param->egress_status[tid]->head = param->egress_queues[tid]->head;
//         param->egress_status[tid]->size = param->egress_queues[tid]->size;
//         /**
//          * @TODO: update transmitted_frame_num
//          */
//         // param->egress_status[tid]->transmitted_frame_num = param->egress_queues[tid]->removed_record;
//     }
// }

// __device__ void update_cache_window(VDES::FrameQueueStatus *status, int expand_size_start, int expand_size_end)
// {
//     int win_size = status->cache_win_end - status->cache_win_start;

//     // update cache window
//     int win_size_new = (status->cache_win_end + expand_size_end) - (status->cache_win_start + expand_size_start);
//     // update pacekt status in cache window
//     // |origin|CPU|GPU|
//     int origin_size = max(0, win_size - expand_size_start);
//     int cpu_size = max(0, min(status->cache_win_end + expand_size_end, status->last_cache_out_offset) - max(status->cache_win_start + expand_size_start, status->cache_win_end));
//     int gpu_size = max(0, min(win_size_new, (status->cache_win_end + expand_size_end) - status->last_cache_out_offset));

//     int memset_offset = 0;
//     if (origin_size > 0)
//     {
//         memcpy(status->packet_status_in_cache_win, status->packet_status_in_cache_win + expand_size_start, origin_size * sizeof(uint8_t));
//         memset_offset += origin_size;
//     }
//     if (cpu_size > 0)
//     {
//         memset(status->packet_status_in_cache_win + memset_offset, 0, cpu_size * sizeof(uint8_t));
//         memset_offset += cpu_size;
//     }

//     if (gpu_size > 0)
//     {
//         memset(status->packet_status_in_cache_win + memset_offset, 1, gpu_size * sizeof(uint8_t));
//     }

//     status->cache_win_start += expand_size_start;
//     status->cache_win_end += expand_size_end;
// }

// __global__ void cache_frame_kernel(VDES::CacheParam *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->queue_num)
//     {
//         VDES::FrameQueueStatus *status = param->egress_status[tid];
//         GPUQueue<VDES::Frame *> *egress_queue = param->egress_queues[tid];
//         int *swap_in_num = param->swap_in_frame_num;
//         int *swap_out_num = param->swap_out_frame_num;
//         VDES::Frame **swap_in_frames = param->swap_in_frame_egress + tid * MAX_TRANSMITTED_PACKET_NUM;
//         VDES::Frame **swap_out_frames = param->swap_out_frame_egress + tid * MAX_SWAP_FRAME_NUM;
//         VDES::Frame *swap_out_cache = param->swap_out_cache_space + tid * MAX_SWAP_FRAME_NUM;

//         // update cache window, remove transmitted frames
//         status->cache_win_start -= status->transmitted_frame_num;
//         status->cache_win_end -= status->transmitted_frame_num;
//         status->last_cache_out_offset -= status->transmitted_frame_num;

//         // expand cache window if necessary
//         int safe_frame_interval = param->lookahead_timeslot_num * MAX_TRANSMITTED_PACKET_NUM;

//         int expand_size_start = 0;
//         if (status->cache_win_start < safe_frame_interval + MAX_TRANSMITTED_PACKET_NUM)
//         {
//             expand_size_start = min(status->size - status->cache_win_start, MAX_TRANSMITTED_PACKET_NUM);
//         }
//         // exclude gpu frames

//         int expand_size_end = 0;
//         if (status->cache_win_end < safe_frame_interval * 2 + MAX_TRANSMITTED_PACKET_NUM)
//         {
//             expand_size_end = min(status->size - status->cache_win_end, MAX_TRANSMITTED_PACKET_NUM);
//         }
//         /**
//          * TODO: Fix the calculation of max_swap_out_num
//          */
//         int max_swap_out_num = status->size - max(status->last_cache_out_offset, status->cache_win_end + expand_size_end);
//         /**
//          * TODO: update max_swap_out_num
//          */
//         max_swap_out_num = max(max_swap_out_num, 0);
//         swap_out_num[tid] = min(max_swap_out_num, MAX_SWAP_FRAME_NUM);

//         // swap in frames
//         int swap_in_frame_num = 0;
//         int win_size = status->cache_win_end - status->cache_win_start;
//         for (int i = 0; i < expand_size_start; i++)
//         {
//             // exchange frame ptr
//             int index = status->cache_win_start + i;
//             if (i < win_size && status->packet_status_in_cache_win[i] == 0 || index >= status->cache_win_end && index < status->last_cache_out_offset)
//             {
//                 // frame host in cpu memory
//                 VDES::Frame *temp = egress_queue->get_element(status->head, index);
//                 egress_queue->set_element(status->head, index, swap_in_frames[swap_in_frame_num]);
//                 swap_in_frames[swap_in_frame_num] = temp;
//                 swap_in_frame_num++;
//             }
//         }
//         swap_in_num[tid] = swap_in_frame_num;

//         // swap out frames
//         int swap_out_offset = max(status->cache_win_end + expand_size_end, status->last_cache_out_offset);
//         for (int i = 0; i < swap_out_num[tid]; i++)
//         {
//             // exchange frame ptr
//             int index = swap_out_offset + i;
//             VDES::Frame *temp = egress_queue->get_element(status->head, index);
//             egress_queue->set_element(status->head, index, swap_out_frames[i]);
//             swap_out_frames[i] = temp;

//             // copy frame content to cache space
//             VDES::Frame *dst_frame = swap_out_cache + i;
//             memcpy(dst_frame, temp, sizeof(VDES::Frame));
//         }
//         // update cache window
//         update_cache_window(status, expand_size_start, expand_size_end);

//         status->last_cache_out_offset = swap_out_offset + swap_out_num[tid];
//     }
// }

// __global__ void copy_frames_from_cache_space(VDES::CacheParam *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->queue_num)
//     {
//         int *swap_in_frame_num = param->swap_in_frame_num;

//         int swap_offset = tid * MAX_TRANSMITTED_PACKET_NUM;
//         VDES::Frame **swap_in_frame = param->swap_in_frame_egress + swap_offset;
//         VDES::Frame *cache_space = param->swap_in_cache_space + swap_offset;

//         for (int i = 0; i < swap_in_frame_num[tid]; i++)
//         {
//             // copy frames from cache space to gpu
//             memcpy(swap_in_frame[i], cache_space + i, sizeof(VDES::Frame));
//         }
//     }
// }

// namespace VDES
// {
//     void LaunchCacheFrameKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream)
//     {
//         cache_frame_kernel<<<grid_dim, block_dim, 0, stream>>>(kernel_param);
//     }

//     void LaunchCopyFrameCacheSpaceKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream)
//     {
//         copy_frames_from_cache_space<<<grid_dim, block_dim, 0, stream>>>(kernel_param);
//     }

//     void LaunchBackupFrameQueueInfoKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream)
//     {
//         backup_frame_queue_info<<<grid_dim, block_dim, 0, stream>>>(kernel_param);
//     }

// }

// #include "packet_cache.h"
// #include "conf.h"

// __global__ void backup_frame_queue_info(VDES::CacheParam *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->queue_num)
//     {
//         param->egress_status[tid]->head = param->egress_queues[tid]->head;
//         param->egress_status[tid]->size = param->egress_queues[tid]->size;
//         param->egress_status[tid]->transmitted_frame_num = param->egress_queues[tid]->removed_record;
//     }
// }

// __device__ void update_cache_window(VDES::FrameQueueStatus *status, int expand_size_start, int expand_size_end)
// {
//     int win_size = status->cache_win_end - status->cache_win_start;

//     // update cache window
//     int win_size_new = (status->cache_win_end + expand_size_end) - (status->cache_win_start + expand_size_start);
//     // update pacekt status in cache window
//     // |origin|CPU|GPU|
//     int origin_size = max(0, win_size - expand_size_start);
//     int cpu_size = max(0, min((status->last_cache_out_offset - status->cache_win_end), (status->cache_win_end + expand_size_end) - status->last_cache_out_offset));
//     int gpu_size = max(0, min(win_size_new, (status->cache_win_end + expand_size_end) - status->last_cache_out_offset));

//     int memset_offset = 0;
//     if (origin_size > 0)
//     {
//         memcpy(status->packet_status_in_cache_win, status->packet_status_in_cache_win + expand_size_start, origin_size * sizeof(uint8_t));
//         memset_offset += origin_size;
//     }
//     if (cpu_size > 0)
//     {
//         memset(status->packet_status_in_cache_win + memset_offset, 1, cpu_size * sizeof(uint8_t));
//         memset_offset += cpu_size;
//     }

//     if (gpu_size > 0)
//     {
//         memset(status->packet_status_in_cache_win + memset_offset, 0, gpu_size * sizeof(uint8_t));
//     }

//     status->cache_win_start += expand_size_start;
//     status->cache_win_end += expand_size_end;
// }

// __global__ void cache_frame_kernel(VDES::CacheParam *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->queue_num)
//     {
//         VDES::FrameQueueStatus *status = param->egress_status[tid];
//         GPUQueue<VDES::Frame *> *egress_queue = param->egress_queues[tid];
//         GPUQueue<int64_t> *egress_ts = param->egress_tss[tid];
//         int *swap_in_num = param->swap_in_frame_num;
//         int *swap_out_num = param->swap_out_frame_num;
//         VDES::Frame **swap_in_frames = param->swap_in_frame_egress;
//         VDES::Frame **swap_out_frames = param->swap_out_frame_egress;
//         VDES::Frame *swap_out_cache = param->swap_out_cache_space;

//         // update cache window, remove transmitted frames
//         status->cache_win_start -= status->transmitted_frame_num;
//         status->cache_win_end -= status->transmitted_frame_num;
//         status->last_cache_out_offset -= status->transmitted_frame_num;

//         // expand cache window if necessary
//         int64_t cache_in_ddl = *(param->timeslot_end) + param->lookahead_timeslot_num * TIMESLOT_LENGTH;
//         int64_t cache_out_threshold = cache_in_ddl + param->lookahead_timeslot_num * TIMESLOT_LENGTH;
//         int safe_frame_interval = param->lookahead_timeslot_num * MAX_TRANSMITTED_PACKET_NUM;

//         int expand_size_start = 0;
//         if (egress_ts->get_element(status->head, status->cache_win_start) < cache_in_ddl && status->cache_win_start < safe_frame_interval + MAX_TRANSMITTED_PACKET_NUM)
//         {
//             expand_size_start = min(status->size - status->cache_win_start, MAX_TRANSMITTED_PACKET_NUM);
//         }
//         // exclude gpu frames
//         // swap_in_num[tid] = expand_size_start;

//         int expand_size_end = 0;
//         if (egress_ts->get_element(status->head, status->cache_win_end) < cache_out_threshold && status->cache_win_end < safe_frame_interval * 2 + MAX_TRANSMITTED_PACKET_NUM)
//         {
//             expand_size_end = min(status->size - status->cache_win_end, MAX_TRANSMITTED_PACKET_NUM);
//         }
//         int max_swap_out_num = status->size - min(status->last_cache_out_offset, status->cache_win_end + expand_size_end);
//         swap_out_num[tid] = min(max_swap_out_num, MAX_SWAP_FRAME_NUM);

//         // swap in frames
//         int swap_in_frame_num = 0;
//         int win_size = status->cache_win_end - status->cache_win_start;
//         for (int i = 0; i < expand_size_start; i++)
//         {
//             // exchange frame ptr
//             int index = status->cache_win_start + i;
//             if (i < win_size && status->packet_status_in_cache_win[i] == 1 || index >= status->cache_win_end && index < status->last_cache_out_offset)
//             {
//                 // frame host in cpu memory
//                 VDES::Frame *temp = egress_queue->get_element(status->head, index);
//                 egress_queue->set_element(status->head, index, swap_in_frames[swap_in_frame_num]);
//                 swap_in_frames[i] = temp;
//                 swap_in_frame_num++;
//             }
//         }
//         swap_in_num[tid] = swap_in_frame_num;

//         // swap out frames
//         int swap_out_offset = max(status->cache_win_end + expand_size_end, status->last_cache_out_offset);
//         for (int i = 0; i < swap_out_num[tid]; i++)
//         {
//             // exchange frame ptr
//             int index = swap_out_offset + i;
//             VDES::Frame *temp = egress_queue->get_element(status->head, index);
//             egress_queue->set_element(status->head, index, swap_out_frames[i]);
//             swap_out_frames[i] = temp;

//             // copy frame content to cache space
//             VDES::Frame *dst_frame = swap_out_cache + i;
//             memcpy(dst_frame, temp, sizeof(VDES::Frame));
//         }
//         // update cache window
//         update_cache_window(status, expand_size_start, expand_size_end);

//         status->last_cache_out_offset = swap_out_offset + swap_out_num[tid];
//     }
// }

// __global__ void copy_frames_from_cache_space(VDES::CacheParam *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->queue_num)
//     {
//         int *swap_in_frame_num = param->swap_in_frame_num;

//         int swap_offset = tid * MAX_TRANSMITTED_PACKET_NUM;
//         VDES::Frame **swap_in_frame = param->swap_in_frame_egress + swap_offset;
//         VDES::Frame *cache_space = param->swap_in_cache_space + swap_offset;

//         for (int i = 0; i < swap_in_frame_num[tid]; i++)
//         {
//             // copy frames from cache space to gpu
//             memcpy(swap_in_frame[i], cache_space + i, sizeof(VDES::Frame));
//         }
//     }
// }

// namespace VDES
// {
//     void LaunchCacheFrameKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream)
//     {
//         cache_frame_kernel<<<grid_dim, block_dim, 0, stream>>>(kernel_param);
//     }

//     void LaunchCopyFrameCacheSpaceKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream)
//     {
//         copy_frames_from_cache_space<<<grid_dim, block_dim, 0, stream>>>(kernel_param);
//     }

//     void LaunchBackupFrameQueueInfoKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream)
//     {
//         backup_frame_queue_info<<<grid_dim, block_dim, 0, stream>>>(kernel_param);
//     }
// }