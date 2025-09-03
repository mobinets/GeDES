#ifndef IPV4_DECAPSULATION_H
#define IPV4_DECAPSULATION_H

#include "gpu_queue.cuh"
#include "packet_definition.h"
#include <vector>
#include "conf.h"
#include "protocol_type.h"

namespace VDES
{
        typedef struct
        {
                // ipv4 packets
                GPUQueue<Ipv4Packet *> **ipv4_queues;

                // l4 packets, node_num * l4_protocol_num, |TCP|UDP|
                GPUQueue<uint8_t *> **l4_queues;

                // recycle packets
                Ipv4Packet **recycle_ipv4_packets;
                int *recycle_offset_per_node;
                int *recycle_ipv4_packets_num;

                // timestamp offset, the offset of tiemstamp filed in l4 packets
                int *l4_timestamp_offset;
                int *l4_src_ip_offset;
                int *l4_dst_ip_offset;

                // cache in packets,|TCP|UDP|
                uint8_t **l4_swap_in_packets;
                int *l4_swap_in_offset_per_node;
                // node*l4_protocol_num, |TCP|UDP|
                int *l4_swap_in_packets_num;
                // the max number of packets for a single protocol
                int cache_packet_num;

                int node_num;

        } IPv4DecapsulationParam;

        class IPv4DecapsulationController
        {

        private:
                std::vector<IPv4DecapsulationParam *> m_kernel_params;
                std::vector<cudaStream_t> m_streams;
                std::vector<cudaGraph_t> m_graphs;
                std::vector<cudaGraphExec_t> m_graph_execs;

                // decapsulation params
                std::vector<GPUQueue<Ipv4Packet *> *> m_ipv4_queues;
                std::vector<GPUQueue<uint8_t *> *> m_l4_queues;

                // node info
                std::vector<int *> m_packet_offset_per_node_cpu;
                std::vector<int> m_nic_num_per_node;

                // recycle ipv4 packets
                std::vector<Ipv4Packet **> m_recycle_ipv4_packets_gpu;
                std::vector<int *> m_recycle_ipv4_packets_num_gpu;
                std::vector<Ipv4Packet **> m_recycle_ipv4_packets_cpu;
                std::vector<int *> m_recycle_ipv4_packets_num_cpu;

                // swap in cache
                std::vector<uint8_t *> m_cache_space_gpu;
                std::vector<uint8_t *> m_cache_space_cpu;
                std::vector<uint8_t **> m_l4_swap_in_packets_gpu;
                std::vector<uint8_t **> m_l4_swap_in_packets_cpu;
                // gpu ptr host in cpu
                std::vector<uint8_t **> m_l4_swap_in_packets_gpu_backup;
                // cpu ptr host in gpu
                std::vector<uint8_t **> m_l4_swap_in_packets_cpu_backup;
                std::vector<int> m_cache_space_sizes;

                std::vector<int *> m_l4_swap_in_packet_num_gpu;
                std::vector<int *> m_l4_swap_in_packet_num_cpu;

                // batch properties
                std::vector<int> m_batch_start_index;
                std::vector<int> m_batch_end_index;

                std::vector<int> m_max_packet_num_per_batch;

                // the native size of each packet
                std::vector<int> m_native_packet_size;
                std::vector<int> m_l4_packet_size;

                std::vector<Ipv4Packet **> m_recycle_packets_tmp;

#if ENABLE_HUGE_GRAPH

                std::vector<cudaMemcpy3DParms> m_memcpy_param;
                std::vector<cudaHostNodeParams> m_host_param;

#endif

        public:
                IPv4DecapsulationController();
                ~IPv4DecapsulationController();

                void InitalizeKernelParams();

                void SetStreams(cudaStream_t *streams, int num);
                void SetIPv4Queues(GPUQueue<Ipv4Packet *> **ipv4_queues, int node_num);
                void SetL4Queues(GPUQueue<uint8_t *> **l4_queues, int node_num);
                void SetNICNum(int *nic_num_per_node, int node_num);
                void SetBatchProperties(int *batch_start_index, int *batch_end_index, int node_num);

                // cache in packets
                void CacheInL4Packets(int btach_id);
                void RecycleIPv4Packets(int batch_id);

                // build graph
                void BuildGraph(int batch_id);
                void BuildGraph();
                void LaunchInstance(int batch_id);

                cudaGraph_t GetGraph(int batch_id);

                void Run(int batch_id);
                void Run();

#if ENABLE_HUGE_GRAPH
                std::vector<cudaMemcpy3DParms> &GetMemcpyParams();
                std::vector<cudaHostNodeParams> &GetHostParams();
#endif

                std::vector<void*> GetRecycleInfo();
        };

        void LaunchIPv4DecapsulationKernel(dim3 grid, dim3 block, IPv4DecapsulationParam *param, cudaStream_t stream);

} // namespace VDES

#endif