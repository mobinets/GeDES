#ifndef IPV4_CONTROLLER_H
#define IPV4_CONTROLLER_H

#include <vector>
#include "conf.h"
#include "gpu_queue.cuh"
#include "packet_definition.h"
#include <numeric>
#include <cstring>

namespace VDES
{

    typedef struct
    {
        // packets from nics and local processes, an ingress per node
        GPUQueue<Ipv4Packet *> **ingresses;
        int node_num;

        // normal egress, route packets to nics
        GPUQueue<Ipv4Packet *> **egresses;
        int *egress_offset_per_node;
        int *egress_num_per_node;

        // local delivery, deliver packets to local processes
        GPUQueue<Ipv4Packet *> **local_egresses;

        // routing table, sort rules by priority scores
        GPUQueue<IPv4RoutingRule *> **routing_tables;

        // error packet queue, store packets that cannot be routed
        GPUQueue<Ipv4Packet *> **error_queues;

        // the remaining size of egress queue
        int *egress_remaining_capacity;
    } IPv4Params;

    class IPv4ProtocolController
    {

    private:
        std::vector<IPv4Params *> m_kernel_params;
        std::vector<cudaStream_t> m_streams;
        std::vector<cudaGraph_t> m_graphs;
        std::vector<cudaGraphExec_t> m_graph_execs;

        // ipv4 params
        std::vector<GPUQueue<Ipv4Packet *> *> m_ingresses;

        std::vector<GPUQueue<Ipv4Packet *> *> m_egresses;
        std::vector<int> m_egress_num_per_node;

        // lcoal delivery queue
        std::vector<GPUQueue<Ipv4Packet *> *> m_local_egresses;

        // routing tables
        std::vector<GPUQueue<IPv4RoutingRule *> *> m_routing_tables;

        // error packet queue
        std::vector<GPUQueue<Ipv4Packet *> *> m_error_queues;

        // batch properties
        std::vector<int> m_batch_start_index;
        std::vector<int> m_batch_end_index;

        std::vector<int *> m_egress_remaing_capacity_gpu;

        std::vector<int> m_cache_sizes;

        // parallel degree for routing kernel, default is 1
        int m_routing_parallel_degree{1};

    public:
        IPv4ProtocolController();
        ~IPv4ProtocolController();

        // initialize kernel parameters
        void InitializeKernelParams();
        void SetIngressQueues(GPUQueue<Ipv4Packet *> **ingresses, int node_num);
        void SetEgressQueues(GPUQueue<Ipv4Packet *> **egresses, int *egress_num_per_node, int node_num);
        void SetLocalEgressQueues(GPUQueue<Ipv4Packet *> **local_egresses, int node_num);
        void SetRoutingTables(GPUQueue<IPv4RoutingRule *> **routing_tables, int node_num);
        void SetErrorQueues(GPUQueue<Ipv4Packet *> **error_queues, int node_num);
        void SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num);
        void SetEgressRemainingCapacity(int **egress_remaining_capacity_gpu, int batch_num);

        // set cuda parameters
        void SetStreams(cudaStream_t *streams, int node_num);
        void BuildGraphs(int batch_id);
        void BuildGraphs();
        void LaunchInstance(int batch_id);
        void Synchronize(int batch_id);
        cudaGraph_t GetGraph(int batch_id);

        // for debugging
        void Run(int batch_id);
        void Run();
    };

    void LaunchRoutingIPv4PacketsKernel(dim3 grid_dim, dim3 block_dim, IPv4Params *params, cudaStream_t stream);
    void LaunchForwardIPv4PacketsKernel(dim3 grid_dim, dim3 block_dim, IPv4Params *params, cudaStream_t stream);

} // namespace VDES

#endif

// namespace VDES
// {

//     typedef struct
//     {
//         // packets from nics and local processes, an ingress per node
//         GPUQueue<Ipv4Packet *> **ingresses;
//         int node_num;

//         // normal egress, route packets to nics
//         GPUQueue<Ipv4Packet *> **egresses;
//         /**
//          * @warning: egress_num_per_node Maybe unused in the kernel.
//          */
//         int *egress_offset_per_node;
//         int *egress_num_per_node;

//         // local delivery, deliver packets to local processes
//         GPUQueue<Ipv4Packet *> **local_egresses;

//         // routing table, sort rules by priority scores
//         GPUQueue<IPv4RoutingRule *> **routing_tables;

//         // error packet queue, store packets that cannot be routed
//         GPUQueue<Ipv4Packet *> **error_queues;

//         // cache ipv4 packets, not initialized
//         // Ipv4Packet **swap_out_packets;
//         // Ipv4Packet *swap_out_cache_space;
//         // int *swap_out_num;
//         // int *swap_offset_per_node;

//         // the remaining size of egress queue
//         int *egress_remaining_capacity;

//         // flag identifying that routing upstreaming or downstreaming packets
//         // uint8_t routing_mode; // 0: upstream, 1: downstream

//     } IPv4Params;

//     class IPv4ProtocolController
//     {

//     private:
//         std::vector<IPv4Params *> m_kernel_params;
//         std::vector<cudaStream_t> m_streams;
//         std::vector<cudaGraph_t> m_graphs;
//         std::vector<cudaGraphExec_t> m_graph_execs;

//         // ipv4 params
//         std::vector<GPUQueue<Ipv4Packet *> *> m_ingresses;

//         std::vector<GPUQueue<Ipv4Packet *> *> m_egresses;
//         std::vector<int> m_egress_num_per_node;

//         // lcoal delivery queue
//         std::vector<GPUQueue<Ipv4Packet *> *> m_local_egresses;

//         // routing tables
//         std::vector<GPUQueue<IPv4RoutingRule *> *> m_routing_tables;

//         // error packet queue
//         std::vector<GPUQueue<Ipv4Packet *> *> m_error_queues;

//         // batch properties
//         std::vector<int> m_batch_start_index;
//         std::vector<int> m_batch_end_index;

//         // cache properties
//         // std::vector<Ipv4Packet **> m_swap_out_packets_gpu;
//         // std::vector<Ipv4Packet *> m_cache_space_gpu;
//         // std::vector<int *> m_swap_out_num_gpu;
//         // std::vector<int *> m_swap_offset_per_node_gpu;

//         // std::vector<Ipv4Packet **> m_swap_out_packets_cpu;
//         // std::vector<Ipv4Packet *> m_cache_space_cpu;
//         // std::vector<int *> m_swap_out_num_cpu;
//         // std::vector<int *> m_swap_offset_per_node_cpu;

//         std::vector<int *> m_egress_remaing_capacity_gpu;

//         std::vector<int> m_cache_sizes;

//         // backup original packets, used in cache out, cpu ptr host on cpu
//         // std::vector<Ipv4Packet**> m_swap_out_packets_gpu_backup;

//         // parallel degree for routing kernel, default is 1
//         int m_routing_parallel_degree{1};

//     public:
//         IPv4ProtocolController();
//         ~IPv4ProtocolController();

//         // initialize kernel parameters
//         void InitializeKernelParams();
//         void SetIngressQueues(GPUQueue<Ipv4Packet *> **ingresses, int node_num);
//         void SetEgressQueues(GPUQueue<Ipv4Packet *> **egresses, int *egress_num_per_node, int node_num);
//         void SetLocalEgressQueues(GPUQueue<Ipv4Packet *> **local_egresses, int node_num);
//         void SetRoutingTables(GPUQueue<IPv4RoutingRule *> **routing_tables, int node_num);
//         void SetErrorQueues(GPUQueue<Ipv4Packet *> **error_queues, int node_num);
//         void SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num);
//         void SetEgressRemainingCapacity(int **egress_remaining_capacity_gpu, int batch_num);

//         // // cache packets
//         // void CacheOutIpv4Packets(int batch_id);

//         // set cuda parameters
//         void SetStreams(cudaStream_t *streams, int node_num);
//         void BuildGraphs(int batch_id);
//         void BuildGraphs();
//         void LaunchInstance(int batch_id);
//         void Synchronize(int batch_id);
//         cudaGraph_t GetGraph(int batch_id);

//         // for debugging
//         void Run(int batch_id);
//         void Run();
//     };

//     void LaunchRoutingIPv4PacketsKernel(dim3 grid_dim, dim3 block_dim, IPv4Params *params, cudaStream_t stream);
//     void LaunchForwardIPv4PacketsKernel(dim3 grid_dim, dim3 block_dim, IPv4Params *params, cudaStream_t stream);

// } // namespace VDES

// #endif