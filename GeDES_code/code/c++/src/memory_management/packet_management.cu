#include "gpu_packet_manager.h"
#include "conf.h"
#include <cub/cub.cuh>

// // calculate inner offset for each group
// __global__ void calculate_cumsum_divided_rule(int *packet_num, int *packet_offset, int len, int batch_num)
// {
//     __shared__ int offsets[MAX_PACKET_WARP];

//     for (int batch_id = blockIdx.x; batch_id * MAX_PACKET_WARP < len; batch_id += gridDim.x)
//     {
//         int batch_size = min(len - batch_id * MAX_PACKET_WARP, MAX_PACKET_WARP);

//         int *packet_num_block = packet_num + batch_id * MAX_PACKET_WARP;
//         // copy data to shared mem
//         for (int i = threadIdx.x; i < batch_size; i += blockDim.x)
//         {
//             offsets[i] = packet_num_block[i];
//         }

//         __syncthreads();

//         // scan
//         for (int i = 1; i < batch_size; i <<= 1)
//         {
//             for (int j = threadIdx.x; i < batch_size; i += blockDim.x)
//             {
//                 if (j >= i)
//                 {
//                     offsets[j] += offsets[j - i];
//                 }
//             }

//             __syncthreads();
//         }

//         // copy data from shared mem to global mem
//         int *packet_offsets_th = packet_offset + batch_id * MAX_PACKET_WARP;
//         for (int i = threadIdx.x; i < batch_size; i += blockDim.x)
//         {
//             packet_offsets_th[i] = offsets[i];
//         }
//         __syncthreads();
//     }
// }

// // merge divided offsets
// __global__ void merge_offsets(int *packet_offset, int len, int *cumsum)
// {

//     // merge offsets for each batch in a block
//     int tid = threadIdx.y * blockDim.x + threadIdx.x;
//     int batch_num = (len + MAX_PACKET_WARP - 1) / MAX_PACKET_WARP;
//     int thread_num = blockDim.x * blockDim.y;

//     for (int i = 0; tid + i * thread_num < batch_num; i++)
//     {
//         int warp_size = min(MAX_PACKET_WARP, len - i * MAX_PACKET_WARP);
//         cumsum[i] = packet_offset[i * MAX_PACKET_WARP + warp_size - 1];
//     }

//     __syncthreads();

//     for (int i = 1; i < batch_num; i << 2)
//     {
//         int inner_id = tid;
//         for (int j = 0; inner_id < batch_num; j++)
//         {
//             if (inner_id >= i)
//             {
//                 cumsum[inner_id] += cumsum[inner_id - i];
//             }
//             inner_id += thread_num;
//         }
//         __syncthreads();
//     }
// }

__global__ void calculate_offsets(int *packet_num, int *packet_offsets, int len, int *cumsum)
{
    for (int batch_id = 0; batch_id * MAX_PACKET_WARP < len; batch_id += gridDim.x)
    {
        int batch_size = min(MAX_PACKET_WARP, len - batch_id * MAX_PACKET_WARP);

        int offset = 0;
        if (batch_size > 0)
        {
            offset = cumsum[batch_id - 1];
        }

        int *packet_offset_block = packet_offsets + batch_id * MAX_PACKET_WARP;
        int *packet_num_block = packet_num + batch_id * MAX_PACKET_WARP;

        for (int i = threadIdx.x; i < batch_size; i += blockDim.x)
        {
            if (i < batch_size)
            {
                packet_offset_block[i] += (offset - packet_num_block[i]);
            }
        }
    }
}

template <int PACKET_TYPE>
__global__ void recycle_packets(VDES::PacketPoolParams *params)
{
    void **recycle_packets = NULL;
    int *packet_num = NULL;
    GPUQueue<void *> *pool = NULL;
    int *offsets = NULL;
    int interval = 0;

    if constexpr (PACKET_TYPE == VDES::PacketIdentifier::FRAME)
    {
        recycle_packets = params->frame_recycle_queues;
        packet_num = params->frame_recycle_num_per_queue;
        pool = params->frame_pool;
        offsets = params->frame_recycle_offset_per_queue;
        interval = MAX_TRANSMITTED_PACKET_NUM;
    }
    else if constexpr (PACKET_TYPE == VDES::PacketIdentifier::IPV4)
    {
        recycle_packets = params->ipv4_recycle_queues;
        packet_num = params->ipv4_recycle_num_per_queue;
        pool = params->ipv4_pool;
        offsets = params->ipv4_recycle_offset_per_queue;
        interval = MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM;
    }
    else if constexpr (PACKET_TYPE == VDES::PacketIdentifier::TCP)
    {
        recycle_packets = params->tcp_recycle_queues;
        packet_num = params->tcp_recycle_num_per_queue;
        pool = params->tcp_pool;
        offsets = params->tcp_recycle_offset_per_queue;
        interval = MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM;
    }

    int queue_num = params->queue_num;
    int thread_num = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int queue_id = tid; queue_id < queue_num; queue_id += thread_num)
    {
        void **pool_data = pool->queue;
        int offset = (offsets[queue_id] + pool->head + pool->size) % pool->queue_capacity;
        void **packets = recycle_packets + queue_id * interval;

        /**
         * @TODO: min rather than max.
         */
        int copy_size = min(pool->queue_capacity - offset, packet_num[queue_id]);
        if (copy_size > 0)
        {
            memcpy(pool_data + offset, packets, copy_size * 8);
        }

        int remaining_size = packet_num[queue_id] - copy_size;
        if (remaining_size > 0)
        {
            memcpy(pool_data, packets + copy_size, 8 * remaining_size);
        }
    }
}

template <int PACKET_TYPE>
__global__ void update_pool_size_recycle(VDES::PacketPoolParams *params)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0)
    {
        int *offsets;
        GPUQueue<void *> *pool;
        int *packet_num;
        if constexpr (PACKET_TYPE == VDES::PacketIdentifier::FRAME)
        {
            packet_num = params->frame_recycle_num_per_queue;
            pool = params->frame_pool;
            offsets = params->frame_recycle_offset_per_queue;
        }
        else if constexpr (PACKET_TYPE == VDES::PacketIdentifier::IPV4)
        {
            packet_num = params->ipv4_recycle_num_per_queue;
            pool = params->ipv4_pool;
            offsets = params->ipv4_recycle_offset_per_queue;
        }
        else if constexpr (PACKET_TYPE == VDES::PacketIdentifier::TCP)
        {
            packet_num = params->tcp_recycle_num_per_queue;
            pool = params->tcp_pool;
            offsets = params->tcp_recycle_offset_per_queue;
        }

        int queue_num = params->queue_num;
        pool->size += (packet_num[queue_num - 1] + offsets[queue_num - 1]);
        if (pool->size == 0)
        {
            printf("memory exhausted!\n");
        }
    }
}

template <int PACKET_TYPE>
__global__ void allocate_packets(VDES::PacketPoolParams *params)
{
    void **alloc_packets;
    int *packet_num;
    GPUQueue<void *> *pool;
    int *offsets;

    if constexpr (PACKET_TYPE == VDES::PacketIdentifier::FRAME)
    {
        packet_num = params->frame_alloc_num_per_queue;
        pool = params->frame_pool;
        offsets = params->frame_alloc_offset_per_queue;
        alloc_packets = params->frame_alloc_queues;
    }
    else if constexpr (PACKET_TYPE == VDES::PacketIdentifier::IPV4)
    {
        packet_num = params->ipv4_alloc_num_per_queue;
        pool = params->ipv4_pool;
        offsets = params->ipv4_alloc_offset_per_queue;
        alloc_packets = params->ipv4_alloc_queues;
    }
    else if constexpr (PACKET_TYPE == VDES::PacketIdentifier::TCP)
    {
        packet_num = params->tcp_alloc_num_per_queue;
        pool = params->tcp_pool;
        offsets = params->tcp_alloc_offset_per_queue;
        alloc_packets = params->tcp_alloc_queues;
    }

    int queue_num = params->queue_num;
    int thread_num = blockDim.x * gridDim.x;
    if (pool->size < offsets[queue_num - 1] + packet_num[queue_num - 1])
    {
        printf("Packet pool is not enough!\n");
        return;
    }

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int interval = (MAX_GENERATED_PACKET_NUM + MAX_TRANSMITTED_PACKET_NUM);
    for (int queue_id = tid; queue_id < queue_num; queue_id += thread_num)
    {
        int offset = (offsets[queue_id] + pool->head) % (pool->queue_capacity);

        int copy_size = min(pool->queue_capacity - offset, packet_num[queue_id]);
        if (copy_size > 0)
        {
            memcpy(alloc_packets + interval * queue_id, pool->queue + offset, copy_size * 8);
        }

        int remaining_size = packet_num[queue_id] - copy_size;
        if (remaining_size > 0)
        {
            memcpy(alloc_packets + interval * queue_id + copy_size, pool->queue, remaining_size * 8);
        }
    }
}

template <int PACKET_TYPE>
__global__ void update_pool_size_alloc(VDES::PacketPoolParams *params)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        int *packet_num;
        GPUQueue<void *> *pool;
        int *offsets;

        if constexpr (PACKET_TYPE == VDES::PacketIdentifier::FRAME)
        {
            packet_num = params->frame_alloc_num_per_queue;
            pool = params->frame_pool;
            offsets = params->frame_alloc_offset_per_queue;
        }
        else if constexpr (PACKET_TYPE == VDES::PacketIdentifier::IPV4)
        {
            packet_num = params->ipv4_alloc_num_per_queue;
            pool = params->ipv4_pool;
            offsets = params->ipv4_alloc_offset_per_queue;
        }
        else if constexpr (PACKET_TYPE == VDES::PacketIdentifier::TCP)
        {
            packet_num = params->tcp_alloc_num_per_queue;
            pool = params->tcp_pool;
            offsets = params->tcp_alloc_offset_per_queue;
        }

        int queue_num = params->queue_num;
        pool->size -= (packet_num[queue_num - 1] + offsets[queue_num - 1]);
        pool->head += (packet_num[queue_num - 1] + offsets[queue_num - 1]);
        pool->head %= pool->queue_capacity;
    }
}

namespace VDES
{
    void LaunchRecycleFrameKernel(dim3 gird_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream)
    {
        recycle_packets<PacketIdentifier::FRAME><<<gird_dim, block_dim, 0, stream>>>(param);
        update_pool_size_recycle<PacketIdentifier::FRAME><<<1, 1, 0, stream>>>(param);
    }

    void LaunchRecycleIPv4PacketKernel(dim3 gird_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream)
    {
        recycle_packets<PacketIdentifier::IPV4><<<gird_dim, block_dim, 0, stream>>>(param);
        update_pool_size_recycle<PacketIdentifier::IPV4><<<1, 1, 0, stream>>>(param);
    }

    void LaunchRecycleTCPPacketKernel(dim3 gird_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream)
    {
        recycle_packets<PacketIdentifier::TCP><<<gird_dim, block_dim, 0, stream>>>(param);
        update_pool_size_recycle<PacketIdentifier::TCP><<<1, 1, 0, stream>>>(param);
    }

    void LaunchAllocateFrameKernel(dim3 grid_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream)
    {
        allocate_packets<PacketIdentifier::FRAME><<<grid_dim, block_dim, 0, stream>>>(param);
        update_pool_size_alloc<PacketIdentifier::FRAME><<<1, 1, 0, stream>>>(param);
    }

    void LaunchAllocateIPv4PacketKernel(dim3 grid_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream)
    {
        allocate_packets<PacketIdentifier::IPV4><<<grid_dim, block_dim, 0, stream>>>(param);
        update_pool_size_alloc<PacketIdentifier::IPV4><<<1, 1, 0, stream>>>(param);
    }

    void LaunchAllocateTCPPacketKernel(dim3 grid_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream)
    {
        allocate_packets<PacketIdentifier::TCP><<<grid_dim, block_dim, 0, stream>>>(param);
        update_pool_size_alloc<PacketIdentifier::TCP><<<1, 1, 0, stream>>>(param);
    }

    void ExclusivePrefixSum(void *d_temp_storage, size_t &temp_storage_bytes, int *d_in, int *d_out, int size, cudaStream_t stream)
    {
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, size, stream);
    }

} // namespace VDES
