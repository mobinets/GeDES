#ifndef FRAME_DECAPSULATION_H
#define FRAME_DECAPSULATION_H

#include "gpu_queue.cuh"
#include "packet_definition.h"
#include <vector>
#include <cuda_runtime.h>
#include "conf.h"
#include "protocol_type.h"
#include <numeric>
#include <cstring>
#include "thread_pool.h"

namespace VDES
{
        typedef struct
        {
                // frame info
                GPUQueue<Frame *> **frame_ingresses;
                int queue_num;

                // node info
                int *frame_queue_num_per_node;
                int *frame_queue_offset_per_node;
                int node_num;

                // protocol ingress queues, structure: |ipv4 queues|ipv6 queues|arp queues|rarp queues|
                GPUQueue<uint8_t *> **packet_ingresses;
                int *l3_timestamp_offset;

                // swap-in packet buffer, protocol_type *frame_queue_num * max_frame_size
                uint8_t **swap_in_packet_buffer;
                int *swap_packet_num;

                // recycle frame buffer, frame_queue_num * max_frame_size
                Frame **recycle_frames;
                int *recycle_num_per_frame_queue;

        } FrameDecapsulationParams;

        class FrameDecapsulationConatroller
        {

        private:
                // kernel params
                std::vector<FrameDecapsulationParams *> m_kernel_params;
                std::vector<cudaStream_t> m_streams;
                std::vector<cudaStream_t> m_cache_streams;
                std::vector<cudaEvent_t> m_events;
                std::vector<cudaGraph_t> m_graphs;
                std::vector<cudaGraphExec_t> m_graph_execs;

                // frame ingress properties
                std::vector<GPUQueue<Frame *> *> m_frame_ingresses;
                std::vector<int> m_frame_ingress_num;

                // node properties
                std::vector<int> m_frame_queue_num_per_node;

                // batch properties, divided in node-level
                std::vector<int> m_batch_start;
                std::vector<int> m_batch_end;

                // packet ingress properties
                std::vector<GPUQueue<uint8_t *> *> m_packet_ingresses;

                // swap-in packet buffer properties
                std::vector<uint8_t **> m_swap_in_packet_buffer_gpu;
                std::vector<uint8_t **> m_swap_in_packet_buffer_cpu;

                // cpu ptrs
                std::vector<uint8_t **> m_swap_in_packet_buffer_cpu_backup;
                // gpu ptrs
                std::vector<uint8_t **> m_swap_in_packet_buffer_gpu_backup;
                std::vector<int *> m_swap_packet_num_gpu;
                std::vector<int *> m_swap_packet_num_cpu;

                // cache space for packets
                std::vector<uint8_t *> m_cache_space_gpu;
                std::vector<uint8_t *> m_cache_space_cpu;
                std::vector<int> m_cache_space_size;

                // packet size
                std::vector<int> m_packet_size;
                std::vector<int> m_native_packet_size;

                // recycle frame buffer properties
                std::vector<Frame **> m_recycle_frames_gpu;
                std::vector<int *> m_recycle_num_per_frame_queue_gpu;
                std::vector<Frame **> m_recycle_frames_cpu;
                std::vector<int *> m_recycle_num_per_frame_queue_cpu;

                /**
                 * @TODO: ADD TMP RECYCLE PACKETS.
                 */
                std::vector<Frame **> m_recycle_frame_tmp;

                // recycle packets
                std::vector<std::vector<void *>> m_recycle_packets;

#if ENABLE_HUGE_GRAPH

                std::vector<cudaMemcpy3DParms> m_memcpy_param;
                std::vector<cudaHostNodeParams> m_host_param;

#endif

        public:
                FrameDecapsulationConatroller();
                ~FrameDecapsulationConatroller();

                // initialize the controller
                void InitializeKernelParams();
                void SetFrameIngress(GPUQueue<Frame *> **frame_ingresses, int frame_queue_num);
                void SetPacketIngress(GPUQueue<void *> **packet_ingresses, int node_num);
                void SetNodeProperties(int *frame_queue_num_per_node, int node_num);
                void SetBatchProperties(int *batch_start, int *batch_end, int batch_num);
                void SetStreams(cudaStream_t *streams, int stream_num);
                void BuildGraph(int batch_id);
                void BuildGraphs();

                void RecycleFrames(int batch_id);
                void LaunchInstance(int batch_id);
                void SynchronizeCache(int batch_id);
                // void HandleCacheAndRecycle(int batch_id);

                cudaGraph_t GetGraph(int batch_id);

#if ENBALE_CACHE
                void CacheInPackets(int batch_id);
                void RecyclePackets(int batch_id);
#endif

                // test interfaces
                void Run(int batch_id);
#if ENABLE_HUGE_GRAPH
                std::vector<cudaMemcpy3DParms> &GetMemcpyParams();
                std::vector<cudaHostNodeParams> &GetHostParams();
#endif
                // void Run();
                std::vector<void*> GetRecycleInfo();
        };

        void LaunchFrameDecapsulationKernel(dim3 grid_dim, dim3 block_dim, FrameDecapsulationParams *kernel_params, cudaStream_t stream);
        void LaunchSortPacketKernel(dim3 grid_dim, dim3 block_dim, FrameDecapsulationParams *kernel_params, cudaStream_t stream);

}

#endif