#include "gpu_packet_manager.h"
#include "conf.h"

namespace VDES
{
    GPUPacketManager::GPUPacketManager()
    {
    }

    GPUPacketManager::~GPUPacketManager()
    {
    }

    void GPUPacketManager::InitializeKernelParams()
    {
        PacketPoolParams params_cpu;
        PacketPoolParams *params_gpu;
        cudaMalloc(&params_gpu, sizeof(PacketPoolParams));
        m_kernel_params.push_back(params_gpu);

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        m_streams.push_back(stream);

        cudaMalloc(&params_cpu.frame_alloc_offset_per_queue, sizeof(int) * m_queue_num);
        cudaMalloc(&params_cpu.frame_recycle_offset_per_queue, sizeof(int) * m_queue_num);
        cudaMalloc(&params_cpu.ipv4_alloc_offset_per_queue, sizeof(int) * m_queue_num);
        cudaMalloc(&params_cpu.ipv4_recycle_offset_per_queue, sizeof(int) * m_queue_num);
        cudaMalloc(&params_cpu.tcp_alloc_offset_per_queue, sizeof(int) * m_queue_num);
        cudaMalloc(&params_cpu.tcp_recycle_offset_per_queue, sizeof(int) * m_queue_num);

        /**
         * @TODO: Push the offset into the vectors.
         */
        m_frame_alloc_offset.push_back(params_cpu.frame_alloc_offset_per_queue);
        m_frame_recycle_offset.push_back(params_cpu.frame_recycle_offset_per_queue);
        m_ipv4_alloc_offset.push_back(params_cpu.ipv4_alloc_offset_per_queue);
        m_ipv4_recycle_offset.push_back(params_cpu.ipv4_recycle_offset_per_queue);
        m_tcp_alloc_offset.push_back(params_cpu.tcp_alloc_offset_per_queue);
        m_tcp_recycle_offset.push_back(params_cpu.tcp_recycle_offset_per_queue);

        params_cpu.frame_alloc_queues = m_frame_alloc_queues[0];
        params_cpu.frame_alloc_num_per_queue = m_frame_alloc_queues_size[0];
        params_cpu.frame_recycle_queues = m_frame_recycle_queues[0];
        /**
         * @TODO: frame_recycle_num_per_queue
         */
        params_cpu.frame_recycle_num_per_queue = m_frame_recycle_queues_size[0];
        params_cpu.ipv4_alloc_queues = m_ipv4_alloc_queues[0];
        params_cpu.ipv4_alloc_num_per_queue = m_ipv4_alloc_queues_size[0];
        params_cpu.ipv4_recycle_queues = m_ipv4_recycle_queues[0];
        params_cpu.ipv4_recycle_num_per_queue = m_ipv4_recycle_queues_size[0];
        params_cpu.tcp_alloc_queues = m_tcp_alloc_queues[0];
        params_cpu.tcp_alloc_num_per_queue = m_tcp_alloc_queues_size[0];
        params_cpu.tcp_recycle_queues = m_tcp_recycle_queues[0];
        params_cpu.tcp_recycle_num_per_queue = m_tcp_recycle_queues_size[0];

        params_cpu.queue_num = m_queue_num;
        params_cpu.frame_pool = m_frame_pool;
        params_cpu.ipv4_pool = m_ipv4_pool;
        params_cpu.tcp_pool = m_tcp_pool;

        cudaMemcpy(params_gpu, &params_cpu, sizeof(PacketPoolParams), cudaMemcpyHostToDevice);
    }

    void GPUPacketManager::SetFramePacketQueues(void **frame_alloc_queues, int *alloc_frame_num,int* frame_alloc_arr_offset, void **frame_recycle_queues, int *recycle_frame_num, int* frame_recycle_arr_offset)
    {
        m_frame_alloc_queues.push_back(frame_alloc_queues);
        m_frame_alloc_queues_size.push_back(alloc_frame_num);
        m_frame_alloc_arr_offset.push_back(frame_alloc_arr_offset);
        m_frame_recycle_queues.push_back(frame_recycle_queues);
        m_frame_recycle_queues_size.push_back(recycle_frame_num);
        m_frame_recycle_arr_offset.push_back(frame_recycle_arr_offset);
    }

    void GPUPacketManager::SetIPv4PacketQueues(void **ipv4_alloc_queues, int *alloc_ipv4_num,int* ipv4_alloc_arr_offset, void **ipv4_recycle_queues, int *recycle_ipv4_num, int* ipv4_recycle_arr_offset)
    {
        m_ipv4_alloc_queues.push_back(ipv4_alloc_queues);
        m_ipv4_alloc_queues_size.push_back(alloc_ipv4_num);
        m_ipv4_alloc_arr_offset.push_back(ipv4_alloc_arr_offset);
        m_ipv4_recycle_queues.push_back(ipv4_recycle_queues);
        m_ipv4_recycle_queues_size.push_back(recycle_ipv4_num);
        m_ipv4_recycle_arr_offset.push_back(ipv4_recycle_arr_offset);
    }

    void GPUPacketManager::SetTCPPacketQueues(void **tcp_alloc_queues, int *alloc_tcp_num, int* tcp_alloc_arr_offset,void **tcp_recycle_queues, int *recycle_tcp_num,int* tcp_recycle_arr_offset)
    {
        m_tcp_alloc_queues.push_back(tcp_alloc_queues);
        m_tcp_alloc_queues_size.push_back(alloc_tcp_num);
        m_tcp_alloc_arr_offset.push_back(tcp_alloc_arr_offset);
        m_tcp_recycle_queues.push_back(tcp_recycle_queues);
        m_tcp_recycle_queues_size.push_back(recycle_tcp_num);
        m_tcp_recycle_arr_offset.push_back(tcp_recycle_arr_offset);
    }

    void GPUPacketManager::SetFramePool(GPUQueue<void *> *frame_pool)
    {
        m_frame_pool = frame_pool;
    }

    void GPUPacketManager::SetIPv4Pool(GPUQueue<void *> *ipv4_pool)
    {
        m_ipv4_pool = ipv4_pool;
    }

    void GPUPacketManager::SetTCPPool(GPUQueue<void *> *tcp_pool)
    {
        m_tcp_pool = tcp_pool;
    }

    void GPUPacketManager::SetQueueNum(int num)
    {
        m_queue_num = num;
    }

    void GPUPacketManager::BuildGraphs()
    {
        // build graph for recycling frames
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaGraph_t graph;

        dim3 block_dim(KERNEL_BLOCK_WIDTH);
        dim3 grid_dim((m_queue_num + block_dim.x - 1) / block_dim.x);

        // get the size of temp storage
        void *d_temp_storage = nullptr;
        size_t temp_storage_size = 0;
        ExclusivePrefixSum(d_temp_storage, temp_storage_size, m_frame_alloc_queues_size[0], m_frame_alloc_offset[0], m_queue_num, stream);

        void *frame_recycle_temp_storage = nullptr;
        cudaMalloc(&frame_recycle_temp_storage, temp_storage_size);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        ExclusivePrefixSum(frame_recycle_temp_storage, temp_storage_size, m_frame_recycle_queues_size[0], m_frame_recycle_offset[0], m_queue_num, stream);
        LaunchRecycleFrameKernel(grid_dim, block_dim, m_kernel_params[0], stream);
        cudaStreamEndCapture(stream, &graph);
        m_frame_recycle_graph.push_back(graph);

        void *frame_alloc_temp_storage = nullptr;
        cudaMalloc(&frame_alloc_temp_storage, temp_storage_size);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        ExclusivePrefixSum(frame_alloc_temp_storage, temp_storage_size, m_frame_alloc_queues_size[0], m_frame_alloc_offset[0], m_queue_num, stream);
        LaunchAllocateFrameKernel(grid_dim, block_dim, m_kernel_params[0], stream);
        cudaStreamEndCapture(stream, &graph);
        m_frame_alloc_graph.push_back(graph);

        void *ipv4_recycle_temp_storage = nullptr;
        cudaMalloc(&ipv4_recycle_temp_storage, temp_storage_size);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        ExclusivePrefixSum(ipv4_recycle_temp_storage, temp_storage_size, m_ipv4_recycle_queues_size[0], m_ipv4_recycle_offset[0], m_queue_num, stream);
        LaunchRecycleIPv4PacketKernel(grid_dim, block_dim, m_kernel_params[0], stream);
        cudaStreamEndCapture(stream, &graph);
        m_ipv4_recycle_graph.push_back(graph);

        void *ipv4_alloc_temp_storage = nullptr;
        cudaMalloc(&ipv4_alloc_temp_storage, temp_storage_size);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        ExclusivePrefixSum(ipv4_alloc_temp_storage, temp_storage_size, m_ipv4_alloc_queues_size[0], m_ipv4_alloc_offset[0], m_queue_num, stream);
        LaunchAllocateIPv4PacketKernel(grid_dim, block_dim, m_kernel_params[0], stream);
        cudaStreamEndCapture(stream, &graph);
        m_ipv4_alloc_graph.push_back(graph);

        void *tcp_recycle_temp_storage = nullptr;
        cudaMalloc(&tcp_recycle_temp_storage, temp_storage_size);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        ExclusivePrefixSum(tcp_recycle_temp_storage, temp_storage_size, m_tcp_recycle_queues_size[0], m_tcp_recycle_offset[0], m_queue_num, stream);
        LaunchRecycleTCPPacketKernel(grid_dim, block_dim, m_kernel_params[0], stream);
        cudaStreamEndCapture(stream, &graph);
        m_tcp_recycle_graph.push_back(graph);

        void *tcp_alloc_temp_storage = nullptr;
        cudaMalloc(&tcp_alloc_temp_storage, temp_storage_size);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        ExclusivePrefixSum(tcp_alloc_temp_storage, temp_storage_size, m_tcp_alloc_queues_size[0], m_tcp_alloc_offset[0], m_queue_num, stream);
        LaunchAllocateTCPPacketKernel(grid_dim, block_dim, m_kernel_params[0], stream);
        cudaStreamEndCapture(stream, &graph);
        m_tcp_alloc_graph.push_back(graph);
    }

    std::vector<cudaGraph_t> GPUPacketManager::GetGraphs()
    {
        std::vector<cudaGraph_t> graphs;

        graphs.insert(graphs.end(), m_frame_recycle_graph.begin(), m_frame_recycle_graph.end());
        graphs.insert(graphs.end(), m_frame_alloc_graph.begin(), m_frame_alloc_graph.end());
        graphs.insert(graphs.end(), m_ipv4_recycle_graph.begin(), m_ipv4_recycle_graph.end());
        graphs.insert(graphs.end(), m_ipv4_alloc_graph.begin(), m_ipv4_alloc_graph.end());
        graphs.insert(graphs.end(), m_tcp_recycle_graph.begin(), m_tcp_recycle_graph.end());
        graphs.insert(graphs.end(), m_tcp_alloc_graph.begin(), m_tcp_alloc_graph.end());
        return graphs;
    }

    void GPUPacketManager::Run(PacketIdentifier type)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaGraphExec_t recycle_exec;
        cudaGraphExec_t alloc_exec;

        if (type == PacketIdentifier::FRAME)
        {
            cudaGraphInstantiate(&recycle_exec, m_frame_recycle_graph[0], NULL, NULL, 0);
            cudaGraphLaunch(recycle_exec, stream);

            cudaGraphInstantiate(&alloc_exec, m_frame_alloc_graph[0], NULL, NULL, 0);
            cudaGraphLaunch(alloc_exec, stream);
        }
        else if (type == PacketIdentifier::IPV4)
        {
            cudaGraphInstantiate(&recycle_exec, m_ipv4_recycle_graph[0], NULL, NULL, 0);
            cudaGraphLaunch(recycle_exec, stream);

            cudaGraphInstantiate(&alloc_exec, m_ipv4_alloc_graph[0], NULL, NULL, 0);
            cudaGraphLaunch(alloc_exec, stream);
        }
        else if (type == PacketIdentifier::TCP)
        {
            cudaGraphInstantiate(&recycle_exec, m_tcp_recycle_graph[0], NULL, NULL, 0);
            cudaGraphLaunch(recycle_exec, stream);

            cudaGraphInstantiate(&alloc_exec, m_tcp_alloc_graph[0], NULL, NULL, 0);
            cudaGraphLaunch(alloc_exec, stream);
        }
    }
}