#include "ipv4_controller.h"

namespace VDES
{
    IPv4ProtocolController::IPv4ProtocolController()
    {
    }

    IPv4ProtocolController::~IPv4ProtocolController()
    {
    }

    void IPv4ProtocolController::InitializeKernelParams()
    {
        int batch_num = m_batch_start_index.size();

        for (int i = 0; i < batch_num; i++)
        {
            // initialize kernel configurations
            cudaGraph_t graph;
            cudaGraphCreate(&graph, 0);
            m_graphs.push_back(graph);
            m_graph_execs.emplace_back();

            int node_num = m_batch_end_index[i] - m_batch_start_index[i];
            int egress_num = std::accumulate(m_egress_num_per_node.begin() + m_batch_start_index[i], m_egress_num_per_node.begin() + m_batch_end_index[i], 0);

            // initialize kernel parameters
            IPv4Params cpu_params;
            cudaMallocAsync(&cpu_params.ingresses, sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);
            cudaMallocAsync(&cpu_params.egresses, sizeof(GPUQueue<Ipv4Packet *> *) * egress_num, m_streams[i]);
            cudaMallocAsync(&cpu_params.egress_offset_per_node, sizeof(int) * node_num, m_streams[i]);
            cudaMallocAsync(&cpu_params.egress_num_per_node, sizeof(int) * node_num, m_streams[i]);
            cudaMallocAsync(&cpu_params.local_egresses, sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);
            cudaMallocAsync(&cpu_params.routing_tables, sizeof(GPUQueue<IPv4RoutingRule *> *) * node_num, m_streams[i]);
            cudaMallocAsync(&cpu_params.error_queues, sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);

            int max_swap_out_num = egress_num * MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;

            int egress_offset = std::accumulate(m_egress_num_per_node.begin(), m_egress_num_per_node.begin() + m_batch_start_index[i], 0);
            cudaMemcpyAsync(cpu_params.ingresses, m_ingresses.data() + m_batch_start_index[i], sizeof(GPUQueue<Ipv4Packet *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
            cpu_params.node_num = node_num;
            cudaMemcpyAsync(cpu_params.egresses, m_egresses.data() + egress_offset, sizeof(GPUQueue<Ipv4Packet *> *) * egress_num, cudaMemcpyHostToDevice, m_streams[i]);
            std::vector<int> egress_offset_per_node(node_num);
            std::partial_sum(m_egress_num_per_node.begin() + m_batch_start_index[i], m_egress_num_per_node.begin() + m_batch_end_index[i], egress_offset_per_node.begin());
            std::rotate(egress_offset_per_node.begin(), egress_offset_per_node.end() - 1, egress_offset_per_node.end());

            egress_offset_per_node[0] = 0;
            cudaMemcpyAsync(cpu_params.egress_offset_per_node, egress_offset_per_node.data(), sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_params.egress_num_per_node, m_egress_num_per_node.data() + m_batch_start_index[i], sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

            cudaMemcpyAsync(cpu_params.local_egresses, m_local_egresses.data() + m_batch_start_index[i], sizeof(GPUQueue<Ipv4Packet *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_params.routing_tables, m_routing_tables.data() + m_batch_start_index[i], sizeof(GPUQueue<IPv4RoutingRule *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_params.error_queues, m_error_queues.data() + m_batch_start_index[i], sizeof(GPUQueue<Ipv4Packet *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

            cpu_params.egress_remaining_capacity = m_egress_remaing_capacity_gpu[i];
            IPv4Params *gpu_params;
            cudaMallocAsync(&gpu_params, sizeof(IPv4Params), m_streams[i]);
            cudaMemcpyAsync(gpu_params, &cpu_params, sizeof(IPv4Params), cudaMemcpyHostToDevice, m_streams[i]);
            cudaStreamSynchronize(m_streams[i]);
            m_kernel_params.push_back(gpu_params);
        }
    }

    void IPv4ProtocolController::SetIngressQueues(GPUQueue<Ipv4Packet *> **ingresses, int node_num)
    {
        m_ingresses.insert(m_ingresses.end(), ingresses, ingresses + node_num);
    }

    void IPv4ProtocolController::SetEgressQueues(GPUQueue<Ipv4Packet *> **egresses, int *egress_num_per_node, int node_num)
    {
        int egress_num = std::accumulate(egress_num_per_node, egress_num_per_node + node_num, 0);
        m_egresses.insert(m_egresses.end(), egresses, egresses + egress_num);
        m_egress_num_per_node.insert(m_egress_num_per_node.end(), egress_num_per_node, egress_num_per_node + node_num);
    }

    void IPv4ProtocolController::SetLocalEgressQueues(GPUQueue<Ipv4Packet *> **local_egresses, int node_num)
    {
        m_local_egresses.insert(m_local_egresses.end(), local_egresses, local_egresses + node_num);
    }

    void IPv4ProtocolController::SetRoutingTables(GPUQueue<IPv4RoutingRule *> **routing_table, int node_num)
    {
        m_routing_tables.insert(m_routing_tables.end(), routing_table, routing_table + node_num);
    }

    void IPv4ProtocolController::SetErrorQueues(GPUQueue<Ipv4Packet *> **error_queues, int node_num)
    {
        m_error_queues.insert(m_error_queues.end(), error_queues, error_queues + node_num);
    }

    void IPv4ProtocolController::SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num)
    {
        m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
        m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
    }

    void IPv4ProtocolController::SetStreams(cudaStream_t *streams, int node_num)
    {
        m_streams.insert(m_streams.end(), streams, streams + node_num);
    }

    void IPv4ProtocolController::BuildGraphs(int batch_id)
    {
        int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

        cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
        dim3 block_dim_rt(KERNEL_BLOCK_WIDTH);
        dim3 grid_dim_rt((node_num * m_routing_parallel_degree + block_dim_rt.x - 1) / block_dim_rt.x);
        LaunchRoutingIPv4PacketsKernel(grid_dim_rt, block_dim_rt, m_kernel_params[batch_id], m_streams[batch_id]);
        dim3 block_dim_fw(KERNEL_BLOCK_WIDTH);
        dim3 grid_dim_fw((node_num + block_dim_fw.x - 1) / block_dim_fw.x);
        LaunchForwardIPv4PacketsKernel(grid_dim_fw, block_dim_fw, m_kernel_params[batch_id], m_streams[batch_id]);
        cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
        cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);
    }

    void IPv4ProtocolController::BuildGraphs()
    {
        int batch_num = m_batch_start_index.size();
        for (int i = 0; i < batch_num; i++)
        {
            BuildGraphs(i);
        }
    }

    void IPv4ProtocolController::LaunchInstance(int batch_id)
    {
        cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
    }

    void IPv4ProtocolController::Run(int batch_id)
    {
        cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
        cudaStreamSynchronize(m_streams[batch_id]);
    }

    void IPv4ProtocolController::Run()
    {
    }

    void IPv4ProtocolController::Synchronize(int batch_id)
    {
        cudaStreamSynchronize(m_streams[batch_id]);
    }

    void IPv4ProtocolController::SetEgressRemainingCapacity(int **egress_remaining_capacity, int batch_num)
    {
        m_egress_remaing_capacity_gpu.insert(m_egress_remaing_capacity_gpu.end(), egress_remaining_capacity, egress_remaining_capacity + batch_num);
    }

    cudaGraph_t IPv4ProtocolController::GetGraph(int batch_id)
    {
        return m_graphs[batch_id];
    }

}

// namespace VDES
// {
//     IPv4ProtocolController::IPv4ProtocolController()
//     {
//     }

//     IPv4ProtocolController::~IPv4ProtocolController()
//     {
//     }

//     void IPv4ProtocolController::InitializeKernelParams()
//     {
//         int batch_num = m_batch_start_index.size();

//         for (int i = 0; i < batch_num; i++)
//         {
//             // initialize kernel configurations
//             cudaGraph_t graph;
//             cudaGraphCreate(&graph, 0);
//             m_graphs.push_back(graph);
//             m_graph_execs.emplace_back();

//             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
//             int egress_num = std::accumulate(m_egress_num_per_node.begin() + m_batch_start_index[i], m_egress_num_per_node.begin() + m_batch_end_index[i], 0);

//             // initialize kernel parameters
//             IPv4Params cpu_params;
//             cudaMallocAsync(&cpu_params.ingresses, sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.egresses, sizeof(GPUQueue<Ipv4Packet *> *) * egress_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.egress_offset_per_node, sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.egress_num_per_node, sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.local_egresses, sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.routing_tables, sizeof(GPUQueue<IPv4RoutingRule *> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.error_queues, sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);

//             int max_swap_out_num = egress_num * MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;

//             int egress_offset = std::accumulate(m_egress_num_per_node.begin(), m_egress_num_per_node.begin() + m_batch_start_index[i], 0);
//             cudaMemcpyAsync(cpu_params.ingresses, m_ingresses.data() + m_batch_start_index[i], sizeof(GPUQueue<Ipv4Packet *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cpu_params.node_num = node_num;
//             cudaMemcpyAsync(cpu_params.egresses, m_egresses.data() + egress_offset, sizeof(GPUQueue<Ipv4Packet *> *) * egress_num, cudaMemcpyHostToDevice, m_streams[i]);
//             std::vector<int> egress_offset_per_node(node_num);
//             std::partial_sum(m_egress_num_per_node.begin() + m_batch_start_index[i], m_egress_num_per_node.begin() + m_batch_end_index[i], egress_offset_per_node.begin());
//             std::rotate(egress_offset_per_node.begin(), egress_offset_per_node.end() - 1, egress_offset_per_node.end());
//             /**
//              * TODO: Set the first offset 0.
//              */
//             egress_offset_per_node[0] = 0;
//             cudaMemcpyAsync(cpu_params.egress_offset_per_node, egress_offset_per_node.data(), sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.egress_num_per_node, m_egress_num_per_node.data() + m_batch_start_index[i], sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

//             cudaMemcpyAsync(cpu_params.local_egresses, m_local_egresses.data() + m_batch_start_index[i], sizeof(GPUQueue<Ipv4Packet *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.routing_tables, m_routing_tables.data() + m_batch_start_index[i], sizeof(GPUQueue<IPv4RoutingRule *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.error_queues, m_error_queues.data() + m_batch_start_index[i], sizeof(GPUQueue<Ipv4Packet *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

//             cpu_params.egress_remaining_capacity = m_egress_remaing_capacity_gpu[i];
//             IPv4Params *gpu_params;
//             cudaMallocAsync(&gpu_params, sizeof(IPv4Params), m_streams[i]);
//             cudaMemcpyAsync(gpu_params, &cpu_params, sizeof(IPv4Params), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaStreamSynchronize(m_streams[i]);
//             m_kernel_params.push_back(gpu_params);
//         }
//     }

//     void IPv4ProtocolController::SetIngressQueues(GPUQueue<Ipv4Packet *> **ingresses, int node_num)
//     {
//         m_ingresses.insert(m_ingresses.end(), ingresses, ingresses + node_num);
//     }

//     void IPv4ProtocolController::SetEgressQueues(GPUQueue<Ipv4Packet *> **egresses, int *egress_num_per_node, int node_num)
//     {
//         int egress_num = std::accumulate(egress_num_per_node, egress_num_per_node + node_num, 0);
//         m_egresses.insert(m_egresses.end(), egresses, egresses + egress_num);
//         m_egress_num_per_node.insert(m_egress_num_per_node.end(), egress_num_per_node, egress_num_per_node + node_num);
//     }

//     void IPv4ProtocolController::SetLocalEgressQueues(GPUQueue<Ipv4Packet *> **local_egresses, int node_num)
//     {
//         m_local_egresses.insert(m_local_egresses.end(), local_egresses, local_egresses + node_num);
//     }

//     void IPv4ProtocolController::SetRoutingTables(GPUQueue<IPv4RoutingRule *> **routing_table, int node_num)
//     {
//         m_routing_tables.insert(m_routing_tables.end(), routing_table, routing_table + node_num);
//     }

//     void IPv4ProtocolController::SetErrorQueues(GPUQueue<Ipv4Packet *> **error_queues, int node_num)
//     {
//         m_error_queues.insert(m_error_queues.end(), error_queues, error_queues + node_num);
//     }

//     void IPv4ProtocolController::SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
//     }

//     // void IPv4ProtocolController::CacheOutIpv4Packets(int batch_id)
//     // {
//     //     int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

//     //     /**
//     //      * @TODO: Copy the pointers from the GPU to the CPU.
//     //      */
//     //     cudaMemcpyAsync(m_swap_out_num_cpu[batch_id], m_swap_out_num_gpu[batch_id], sizeof(int) * node_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//     //     cudaMemcpyAsync(m_cache_space_cpu[batch_id], m_cache_space_gpu[batch_id], sizeof(Ipv4Packet) * m_cache_sizes[batch_id], cudaMemcpyDeviceToHost, m_streams[batch_id]);
//     //     // don't need to recycle ipv4 packets
//     //     // cudaMemcpyAsync(m_swap_out_packets_cpu[batch_id], m_swap_out_packets_gpu[batch_id], sizeof(int) * node_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);

//     //     int *swap_out_num = m_swap_out_num_cpu[batch_id];
//     //     cudaStreamSynchronize(m_streams[batch_id]);

//     //     int total_swap_out_num = std::accumulate(swap_out_num, swap_out_num + node_num, 0);
//     //     auto alloc_packets = ipv4_packet_pool_cpu->allocate(total_swap_out_num);
//     //     int packet_num = 0;
//     //     for (int i = 0; i < node_num; i++)
//     //     {
//     //         Ipv4Packet **swap_out_packets = m_swap_out_packets_cpu[batch_id] + m_swap_offset_per_node_cpu[batch_id][i];
//     //         Ipv4Packet *swap_out_cache = m_cache_space_cpu[batch_id] + m_swap_offset_per_node_cpu[batch_id][i];

//     //         for (int j = 0; j < swap_out_num[i]; j++)
//     //         {
//     //             // copy ipv4 packet from cache to discrete packets
//     //             memcpy(swap_out_packets[j], swap_out_cache + j, sizeof(Ipv4Packet));
//     //             swap_out_packets[j] = alloc_packets[packet_num++];
//     //         }
//     //     }

//     //     cudaMemcpyAsync(m_swap_out_packets_gpu[batch_id], m_swap_out_packets_cpu[batch_id], sizeof(Ipv4Packet *) * node_num, cudaMemcpyHostToDevice, m_streams[batch_id]);
//     // }

//     void IPv4ProtocolController::SetStreams(cudaStream_t *streams, int node_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + node_num);
//     }

//     void IPv4ProtocolController::BuildGraphs(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

//         cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
//         dim3 block_dim_rt(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim_rt((node_num * m_routing_parallel_degree + block_dim_rt.x - 1) / block_dim_rt.x);
//         LaunchRoutingIPv4PacketsKernel(grid_dim_rt, block_dim_rt, m_kernel_params[batch_id], m_streams[batch_id]);
//         dim3 block_dim_fw(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim_fw((node_num + block_dim_fw.x - 1) / block_dim_fw.x);
//         LaunchForwardIPv4PacketsKernel(grid_dim_fw, block_dim_fw, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);
//     }

//     void IPv4ProtocolController::BuildGraphs()
//     {
//         int batch_num = m_batch_start_index.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraphs(i);
//         }
//     }

//     void IPv4ProtocolController::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void IPv4ProtocolController::Run(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);
//     }

//     void IPv4ProtocolController::Run()
//     {
//     }

//     void IPv4ProtocolController::Synchronize(int batch_id)
//     {
//         cudaStreamSynchronize(m_streams[batch_id]);
//     }

//     void IPv4ProtocolController::SetEgressRemainingCapacity(int **egress_remaining_capacity, int batch_num)
//     {
//         m_egress_remaing_capacity_gpu.insert(m_egress_remaing_capacity_gpu.end(), egress_remaining_capacity, egress_remaining_capacity + batch_num);
//     }

//     cudaGraph_t IPv4ProtocolController::GetGraph(int batch_id)
//     {
//         return m_graphs[batch_id];
//     }

// }

// #include "ipv4_controller.h"

// namespace VDES
// {
//     IPv4ProtocolController::IPv4ProtocolController()
//     {
//     }

//     IPv4ProtocolController::~IPv4ProtocolController()
//     {
//     }

//     void IPv4ProtocolController::InitializeKernelParams()
//     {
//         int batch_num = m_batch_start_index.size();

//         for (int i = 0; i < batch_num; i++)
//         {
//             // initialize kernel configurations
//             cudaGraph_t graph;
//             cudaGraphCreate(&graph, 0);
//             m_graphs.push_back(graph);
//             m_graph_execs.emplace_back();

//             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
//             int egress_num = std::accumulate(m_egress_num_per_node.begin() + m_batch_start_index[i], m_egress_num_per_node.begin() + m_batch_end_index[i], 0);

//             // initialize kernel parameters
//             IPv4Params cpu_params;
//             cudaMallocAsync(&cpu_params.ingresses, sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.ingress_tss, sizeof(GPUQueue<int64_t> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.egresses, sizeof(GPUQueue<Ipv4Packet *> *) * egress_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.egress_tss, sizeof(GPUQueue<int64_t> *) * egress_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.egress_ip, sizeof(GPUQueue<uint32_t> *) * egress_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.egress_num_per_node, sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.egress_offset_per_node, sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.local_egresses, sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.local_egress_tss, sizeof(GPUQueue<int64_t> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.routing_tables, sizeof(GPUQueue<IPv4RoutingRule *> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.error_queues, sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.error_queue_tss, sizeof(GPUQueue<int64_t> *) * node_num, m_streams[i]);

//             int max_swap_out_num = egress_num * MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_TRANSMITTED_PACKET_NUM;

//             cudaMallocAsync(&cpu_params.swap_out_packets, sizeof(Ipv4Packet *) * max_swap_out_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.swap_out_cache_space, sizeof(Ipv4Packet) * max_swap_out_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.swap_out_num, sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.swap_offset_per_node, sizeof(int) * node_num, m_streams[i]);

//             int egress_offset = std::accumulate(m_egress_num_per_node.begin(), m_egress_num_per_node.begin() + m_batch_start_index[i], 0);
//             cudaMemcpyAsync(cpu_params.ingresses, m_ingresses.data() + m_batch_start_index[i], sizeof(GPUQueue<Ipv4Packet *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.ingress_tss, m_ingress_tss.data() + m_batch_start_index[i], sizeof(GPUQueue<int64_t> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cpu_params.node_num = node_num;
//             cudaMemcpyAsync(cpu_params.egresses, m_egresses.data() + egress_offset, sizeof(GPUQueue<Ipv4Packet *> *) * egress_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.egress_tss, m_egress_tss.data() + egress_offset, sizeof(GPUQueue<int64_t> *) * egress_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.egress_ip, m_egress_ip.data() + egress_offset, sizeof(GPUQueue<uint32_t> *) * egress_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.egress_num_per_node, m_egress_num_per_node.data() + m_batch_start_index[i], sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             // computing offset
//             std::vector<int> egress_offset_per_node(node_num);
//             std::partial_sum(m_egress_num_per_node.begin() + m_batch_start_index[i], m_egress_num_per_node.begin() + m_batch_end_index[i], egress_offset_per_node.begin());
//             std::rotate(egress_offset_per_node.begin(), egress_offset_per_node.end() - 1, egress_offset_per_node.end());
//             cudaMemcpyAsync(cpu_params.egress_offset_per_node, egress_offset_per_node.data(), sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

//             cudaMemcpyAsync(cpu_params.local_egresses, m_local_egresses.data() + m_batch_start_index[i], sizeof(GPUQueue<Ipv4Packet *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.local_egress_tss, m_local_egress_tss.data() + m_batch_start_index[i], sizeof(GPUQueue<int64_t> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.routing_tables, m_routing_tables.data() + m_batch_start_index[i], sizeof(GPUQueue<IPv4RoutingRule *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.error_queues, m_error_queues.data() + m_batch_start_index[i], sizeof(GPUQueue<Ipv4Packet *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.error_queue_tss, m_error_queue_tss.data() + m_batch_start_index[i], sizeof(GPUQueue<int64_t> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

//             // allocate cache space and discrete packets
//             auto alloc_packets = ipv4_packet_pool_cpu->allocate(max_swap_out_num);
//             m_swap_out_packets_cpu.push_back(new Ipv4Packet *[node_num]);
//             memcpy(m_swap_out_packets_cpu[i], alloc_packets.data(), sizeof(Ipv4Packet *) * max_swap_out_num);
//             m_swap_out_packets_gpu.push_back(cpu_params.swap_out_packets);
//             cudaMemcpyAsync(m_swap_out_packets_gpu[i], m_swap_out_packets_cpu[i], sizeof(Ipv4Packet *) * max_swap_out_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemsetAsync(cpu_params.swap_out_cache_space, 0, sizeof(Ipv4Packet) * max_swap_out_num * sizeof(Ipv4Packet), m_streams[i]);
//             m_cache_space_gpu.push_back(cpu_params.swap_out_cache_space);
//             m_cache_space_gpu.push_back(new Ipv4Packet[max_swap_out_num]);
//             int *swap_offset_per_node = new int[node_num];

//             int offset = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 swap_offset_per_node[j] = offset;
//                 int queue_num = m_egress_num_per_node[m_batch_start_index[i] + j];
//                 offset += queue_num * MAX_TRANSMITTED_PACKET_NUM;
//                 offset += MAX_GENERATED_PACKET_NUM;
//             }

//             cudaMemcpyAsync(cpu_params.swap_offset_per_node, swap_offset_per_node, sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             m_swap_offset_per_node_cpu.push_back(swap_offset_per_node);
//             m_swap_offset_per_node_gpu.push_back(cpu_params.swap_offset_per_node);
//             m_cache_sizes.push_back(max_swap_out_num);

//             IPv4Params *gpu_params;
//             cudaMallocAsync(&gpu_params, sizeof(IPv4Params), m_streams[i]);
//             cudaMemcpyAsync(gpu_params, &cpu_params, sizeof(IPv4Params), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaStreamSynchronize(m_streams[i]);
//             m_kernel_params.push_back(gpu_params);
//         }
//     }

//     void IPv4ProtocolController::SetIngressQueues(GPUQueue<Ipv4Packet *> **ingresses, GPUQueue<int64_t> **ingress_tss, int node_num)
//     {
//         m_ingresses.insert(m_ingresses.end(), ingresses, ingresses + node_num);
//         m_ingress_tss.insert(m_ingress_tss.end(), ingress_tss, ingress_tss + node_num);
//     }

//     void IPv4ProtocolController::SetEgressQueues(GPUQueue<Ipv4Packet *> **egresses, GPUQueue<int64_t> **egress_tss, GPUQueue<uint32_t> **egress_ip, int *egress_num_per_node, int node_num)
//     {
//         int egress_num = std::accumulate(egress_num_per_node, egress_num_per_node + node_num, 0);
//         m_egresses.insert(m_egresses.end(), egresses, egresses + egress_num);
//         m_egress_tss.insert(m_egress_tss.end(), egress_tss, egress_tss + egress_num);
//         m_egress_ip.insert(m_egress_ip.end(), egress_ip, egress_ip + egress_num);
//         m_egress_num_per_node.insert(m_egress_num_per_node.end(), egress_num_per_node, egress_num_per_node + node_num);
//     }

//     void IPv4ProtocolController::SetLocalEgressQueues(GPUQueue<Ipv4Packet *> **local_egresses, GPUQueue<int64_t> **local_egress_tss, int node_num)
//     {
//         m_local_egresses.insert(m_local_egresses.end(), local_egresses, local_egresses + node_num);
//         m_local_egress_tss.insert(m_local_egress_tss.end(), local_egress_tss, local_egress_tss + node_num);
//     }

//     void IPv4ProtocolController::SetRoutingTables(GPUQueue<IPv4RoutingRule *> **routing_table, int node_num)
//     {
//         m_routing_tables.insert(m_routing_tables.end(), routing_table, routing_table + node_num);
//     }

//     void IPv4ProtocolController::SetErrorQueues(GPUQueue<Ipv4Packet *> **error_queues, GPUQueue<int64_t> **error_tss, int node_num)
//     {
//         m_error_queues.insert(m_error_queues.end(), error_queues, error_queues + node_num);
//         m_error_queue_tss.insert(m_error_queue_tss.end(), error_tss, error_tss + node_num);
//     }

//     void IPv4ProtocolController::SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
//     }

//     void IPv4ProtocolController::CacheOutIpv4Packets(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

//         cudaMemcpyAsync(m_swap_out_num_cpu[batch_id], m_swap_out_num_cpu[batch_id], sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[batch_id]);
//         cudaMemcpyAsync(m_cache_space_cpu[batch_id], m_cache_space_gpu[batch_id], sizeof(Ipv4Packet) * m_cache_sizes[batch_id], cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         // don't need to recycle ipv4 packets
//         // cudaMemcpyAsync(m_swap_out_packets_cpu[batch_id], m_swap_out_packets_gpu[batch_id], sizeof(int) * node_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);

//         int *swap_out_num = m_swap_out_num_cpu[batch_id];
//         cudaStreamSynchronize(m_streams[batch_id]);

//         int total_swap_out_num = std::accumulate(swap_out_num, swap_out_num + node_num, 0);
//         auto alloc_packets = ipv4_packet_pool_cpu->allocate(total_swap_out_num);
//         int packet_num = 0;
//         for (int i = 0; i < node_num; i++)
//         {
//             Ipv4Packet **swap_out_packets = m_swap_out_packets_cpu[batch_id] + m_swap_offset_per_node_cpu[batch_id][i];
//             Ipv4Packet *swap_out_cache = m_cache_space_cpu[batch_id] + m_swap_offset_per_node_cpu[batch_id][i];

//             for (int j = 0; j < swap_out_num[i]; j++)
//             {
//                 // copy ipv4 packet from cache to discrete packets
//                 memcpy(swap_out_packets[j], swap_out_cache + j, sizeof(Ipv4Packet));
//                 swap_out_packets[j] = alloc_packets[packet_num++];
//             }
//         }

//         cudaMemcpyAsync(m_swap_out_packets_gpu[batch_id], m_swap_out_packets_cpu[batch_id], sizeof(Ipv4Packet *) * node_num, cudaMemcpyHostToDevice, m_streams[batch_id]);
//     }

//     void IPv4ProtocolController::SetStreams(cudaStream_t *streams, int node_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + node_num);
//     }

//     void IPv4ProtocolController::BuildGraphs(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

//         cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
//         dim3 block_dim_rt(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim_rt((node_num * m_routing_parallel_degree + block_dim_rt.x - 1) / block_dim_rt.x);
//         LaunchRoutingIPv4PacketsKernel(grid_dim_rt, block_dim_rt, m_kernel_params[batch_id], m_streams[batch_id]);
//         dim3 block_dim_fw(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim_fw((node_num + block_dim_fw.x - 1) / block_dim_fw.x);
//         LaunchForwardIPv4PacketsKernel(grid_dim_fw, block_dim_fw, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);
//     }

//     void IPv4ProtocolController::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void IPv4ProtocolController::Run(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//         CacheOutIpv4Packets(batch_id);
//         cudaStreamSynchronize(m_streams[batch_id]);
//     }

//     void IPv4ProtocolController::Run()
//     {
//     }

//     void IPv4ProtocolController::Synchronize(int batch_id)
//     {
//         cudaStreamSynchronize(m_streams[batch_id]);
//     }
// }