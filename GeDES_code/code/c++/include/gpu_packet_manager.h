#ifndef GPU_PACKET_MANAGER_H
#define GPU_PACKET_MANAGER_H

#include <vector>
#include "packet_definition.h"
#include "gpu_queue.cuh"

#define OFFSET_BLOCK_HEIGHT 8

namespace VDES
{
    typedef struct
    {
        void **frame_alloc_queues;
        void **frame_recycle_queues;
        void **ipv4_alloc_queues;
        void **ipv4_recycle_queues;
        void **tcp_alloc_queues;
        void **tcp_recycle_queues;

        int *frame_alloc_num_per_queue;
        int *frame_recycle_num_per_queue;
        int *ipv4_alloc_num_per_queue;
        int *ipv4_recycle_num_per_queue;
        int *tcp_alloc_num_per_queue;
        int *tcp_recycle_num_per_queue;

        int *frame_alloc_offset_per_queue;
        int *frame_recycle_offset_per_queue;
        int *ipv4_alloc_offset_per_queue;
        int *ipv4_recycle_offset_per_queue;
        int *tcp_alloc_offset_per_queue;
        int *tcp_recycle_offset_per_queue;

        int* frame_alloc_arr_offset;
        int* frame_recycle_arr_offset;
        int* ipv4_alloc_arr_offset;
        int* ipv4_recycle_arr_offset;
        int* tcp_alloc_arr_offset;
        int* tcp_recycle_arr_offset;

        int queue_num;

        GPUQueue<void *> *frame_pool;
        GPUQueue<void *> *ipv4_pool;
        GPUQueue<void *> *tcp_pool;
    } PacketPoolParams;

    typedef enum
    {
        FRAME = 0,
        IPV4,
        TCP
    } PacketIdentifier;

    class GPUPacketManager
    {
    public:
        GPUPacketManager();
        ~GPUPacketManager();

        void InitializeKernelParams();
        void SetFramePacketQueues(void **frame_alloc_queues, int *alloc_frame_num,int* frame_alloc_arr_offset, void **frame_recycle_queues, int *recycle_frame_num, int* frame_recycle_arr_offset);
        void SetIPv4PacketQueues(void **ipv4_alloc_queues, int *alloc_ipv4_num,int* ipv4_alloc_arr_offset, void **ipv4_recycle_queues, int *recycle_ipv4_num, int* ipv4_recycle_arr_offset);
        void SetTCPPacketQueues(void **tcp_alloc_queues, int *alloc_tcp_num, int* tcp_alloc_arr_offset,void **tcp_recycle_queues, int *recycle_tcp_num,int* tcp_recycle_arr_offset);
        void SetFramePool(GPUQueue<void *> *frame_pool);
        void SetIPv4Pool(GPUQueue<void *> *ipv4_pool);
        void SetTCPPool(GPUQueue<void *> *tcp_pool);
        void SetQueueNum(int num);

        void BuildGraphs();
        // two graph for each protocol, recyle and alloc graph
        std::vector<cudaGraph_t> GetGraphs();

        // test interface
        void Run(PacketIdentifier packet_type);

    private:
        std::vector<PacketPoolParams *> m_kernel_params;
        std::vector<cudaStream_t> m_streams;
        std::vector<cudaGraph_t> m_frame_alloc_graph;
        std::vector<cudaGraph_t> m_ipv4_alloc_graph;
        std::vector<cudaGraph_t> m_tcp_alloc_graph;
        std::vector<cudaGraph_t> m_frame_recycle_graph;
        std::vector<cudaGraph_t> m_ipv4_recycle_graph;
        std::vector<cudaGraph_t> m_tcp_recycle_graph;

        std::vector<void **> m_frame_alloc_queues;
        std::vector<void **> m_frame_recycle_queues;
        std::vector<void **> m_ipv4_alloc_queues;
        std::vector<void **> m_ipv4_recycle_queues;
        std::vector<void **> m_tcp_alloc_queues;
        std::vector<void **> m_tcp_recycle_queues;

        std::vector<int *> m_frame_alloc_queues_size;
        std::vector<int *> m_frame_recycle_queues_size;
        std::vector<int *> m_ipv4_alloc_queues_size;
        std::vector<int *> m_ipv4_recycle_queues_size;
        std::vector<int *> m_tcp_alloc_queues_size;
        std::vector<int *> m_tcp_recycle_queues_size;

        std::vector<int *> m_frame_alloc_offset;
        std::vector<int *> m_frame_recycle_offset;
        std::vector<int *> m_ipv4_alloc_offset;
        std::vector<int *> m_ipv4_recycle_offset;
        std::vector<int *> m_tcp_alloc_offset;
        std::vector<int *> m_tcp_recycle_offset;

        std::vector<int*> m_frame_alloc_arr_offset;
        std::vector<int*> m_frame_recycle_arr_offset;
        std::vector<int*> m_ipv4_alloc_arr_offset;
        std::vector<int*> m_ipv4_recycle_arr_offset;
        std::vector<int*> m_tcp_alloc_arr_offset;
        std::vector<int*> m_tcp_recycle_arr_offset;

        GPUQueue<void *> *m_frame_pool;
        GPUQueue<void *> *m_ipv4_pool;
        GPUQueue<void *> *m_tcp_pool;

        int m_queue_num;
    };

    void LaunchRecycleFrameKernel(dim3 gird_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream);
    void LaunchAllocateFrameKernel(dim3 grid_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream);
    void LaunchRecycleIPv4PacketKernel(dim3 gird_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream);
    void LaunchAllocateIPv4PacketKernel(dim3 grid_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream);
    void LaunchRecycleTCPPacketKernel(dim3 gird_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream);
    void LaunchAllocateTCPPacketKernel(dim3 grid_dim, dim3 block_dim, VDES::PacketPoolParams *param, cudaStream_t stream);

    void ExclusivePrefixSum(void *d_temp_storage, size_t &temp_storage_bytes, int *d_in, int *d_out, int size, cudaStream_t stream);

}

#endif