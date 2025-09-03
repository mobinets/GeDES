#ifndef FRAME_ENCAPSULATION_H
#define FRAME_ENCAPSULATION_H

#include <vector>
#include "conf.h"
#include "gpu_queue.cuh"
#include "packet_definition.h"
#include "protocol_type.h"
#include <algorithm>
#include <numeric>
#include <cstring>

namespace VDES
{
        typedef struct
        {
                // packets from up protocols, protocol_num*frame_queue,structured by |ipv4|ipv6|...
                GPUQueue<uint8_t *> **packets_egresses;

                // frame queue info
                GPUQueue<Frame *> **frame_queues;
                uint8_t **frame_queue_macs;
                int *node_id_per_frame_queue;
                int *frame_queue_offset_per_node;

                int *l3_packet_len_offset;
                int *l3_packet_timestamp_offset;
                int *l3_dst_ip_offset;
                int *l3_packet_size;

                // null mac address
                uint8_t null_mac[6];

                // arp table, a table for a frame queue
                GPUQueue<ARPRule *> **arp_tables;

                // fattree arp
                uint16_t k;
                uint32_t base_ip;
                uint32_t ip_group_size;
                uint8_t **mac_addr_ft;

                // frame packet,
                Frame **alloc_frames;
                int *alloc_num_per_frame_queue;
                int *alloc_offset_per_node;

#if ENABLE_CACHE
                // swap out l4 packets
                uint8_t **swap_out_l3_packets;
                int *swap_out_l3_packets_num;
                uint8_t **l3_cache_ptr;
#endif

                int queue_num;
                int node_num;
                int max_packet_num;

        } FrameEncapsulationParams;

        class FrameEncapsulationController
        {
        private:
                std::vector<FrameEncapsulationParams *> m_kernel_params;
                std::vector<cudaStream_t> m_streams;
                std::vector<cudaGraph_t> m_graphs;
                std::vector<cudaGraphExec_t> m_graph_execs;
                std::vector<dim3> m_grid_dim;
                std::vector<dim3> m_block_dim;

                // packet queues, node_num
                std::vector<GPUQueue<void *> *> m_packets_egresses;

                // frame queues
                std::vector<GPUQueue<Frame *> *> m_frame_queues;
                // cpu ptr host on cpu
                std::vector<uint8_t *> m_frame_queue_macs_cpu;
                // gpu ptr host on cpu
                std::vector<uint8_t *> m_frame_queue_macs_gpu;
                std::vector<int *> m_node_id_per_frame_queue;
                std::vector<uint8_t *> m_first_frame_id_per_queue;
                std::vector<int> m_frame_num_per_node;

                // arp tables
                std::vector<GPUQueue<ARPRule *> *> m_arp_tables;

                // fattree arp
                uint16_t m_ft_k;
                uint32_t m_ft_base_ip;
                uint32_t m_ft_ip_group_size;

                // batch properties
                std::vector<int> m_batch_start_index;
                std::vector<int> m_batch_end_index;
                std::vector<int> m_frame_num_per_batch;

                // alloc frames
                std::vector<Frame **> m_alloc_frames_cpu;
                std::vector<Frame **> m_alloc_frames_gpu;
                std::vector<int *> m_alloc_num_per_frame_queue_gpu;
                std::vector<int *> m_alloc_num_per_frame_queue_cpu;
                std::vector<int *> m_alloc_offset_per_node_cpu;
                std::vector<int *> m_alloc_offset_per_node_gpu;
                std::vector<int> m_total_alloc_frame_num;

#if ENABLE_HUGE_GRAPH

                std::vector<cudaMemcpy3DParms> m_memcpy_param;
                std::vector<cudaHostNodeParams> m_host_param;

#endif

#if ENABLE_CACHE
                // cache
                std::vector<uint8_t *> m_l3_cache_gpu;
                std::vector<uint8_t *> m_l3_cache_cpu;
                // gpu ptr host on gpu
                std::vector<uint8_t **> m_l3_cache_ptr_gpu;
                std::vector<uint8_t **> m_l3_cache_ptr_cpu;
                std::vector<uint8_t **> m_l3_swap_out_packet_gpu;
                std::vector<uint8_t **> m_l3_swap_out_packet_cpu;
                std::vector<uint8_t **> m_l3_swap_out_packet_cpu_backup;
                std::vector<int *> m_l3_swap_out_packet_num_gpu;
                std::vector<int *> m_l3_swap_out_packet_num_cpu;
#endif

                std::vector<int> m_packet_sizes;
                std::vector<int> m_cache_sizes;
                std::vector<int> m_max_packet_num;

        public:
                FrameEncapsulationController();
                ~FrameEncapsulationController();

                // init
                void InitializeKernelParams();
                void SetStreams(cudaStream_t *streams, int num);

                // set packet propeties
                void SetPacketProperties(GPUQueue<void *> **packet_queues, int node_num);
                void SetFrameProperties(GPUQueue<Frame *> **frame_queues, uint8_t **frame_queue_macs, int *frame_num_per_node, int node_num);
                void SetArpProperties(GPUQueue<ARPRule *> **arp_tables, int node_num);
                void SetFatTreeArpProperties(uint16_t k, uint32_t base_ip, uint32_t ip_group_size);

                void SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num);

                void CacheOutL3Packets(int batch_id);
                void BuildGraph(int batch_id);
                void BuildGraph();
                cudaGraph_t GetGraph(int batch_id);

                void LaunchInstance(int batch_id);
                void Run(int batch_id);
                void Run();

                // complement alloted frames
                void UpdateComsumedFrames(int batch_id);

#if ENABLE_HUGE_GRAPH
                std::vector<cudaMemcpy3DParms> &GetMemcpyParams();
                std::vector<cudaHostNodeParams> &GetHostParams();
#endif
                std::vector<void*> GetAllocateInfo();
        };

        void LaunchEncapsulateFrameKernel(dim3 grid_dim, dim3 block_dim, VDES::FrameEncapsulationParams *params, cudaStream_t stream);

} // namespace VDES

#endif
