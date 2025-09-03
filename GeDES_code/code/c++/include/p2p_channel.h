#ifndef P2P_CHANNEL_H
#define P2P_CHANNEL_H

#include "gpu_queue.cuh"
#include "packet_definition.h"
#include "conf.h"
#include <vector>
#include <cstring>
#include <numeric>

namespace VDES
{
    typedef struct
    {
        GPUQueue<Frame *> **ingresses;
        GPUQueue<Frame *> **egresses;
        int queue_num;
        /** To examinate the switches. */
        int ft_k;
        /**Each block chekc once. */
        uint8_t *is_completed;

        int *propogation_delay_per_ch;
        int *link_transmission_rate;

        // the timestamp of the last transmitted packet
        int64_t *last_tx_end_time;
        uint16_t *remainder_of_packet_len;

        int64_t *timeslot_start_time;
        int64_t *timeslot_end_time;

    } P2PParams;

    class P2PChannelController
    {

    private:
        std::vector<P2PParams *> m_kernel_params;
        std::vector<cudaStream_t> m_streams;
        std::vector<cudaGraph_t> m_graphs;
        std::vector<cudaGraphExec_t> m_graph_execs;

        std::vector<GPUQueue<Frame *> *> m_ingress_queues;
        std::vector<GPUQueue<Frame *> *> m_egress_queues;
        // std::vector<int> m_nic_num_per_node;

        std::vector<int> m_propogation_delay_per_ch;
        std::vector<int> m_link_transmission_rate;

        std::vector<int> m_batch_start_index;
        std::vector<int> m_batch_end_index;

        int64_t *m_timeslot_start_time;
        int64_t *m_timeslot_end_time;

        uint8_t *m_transmission_completed;
        uint32_t m_total_block_num;

    public:
        P2PChannelController();
        ~P2PChannelController();

        void SetIngressAndEgress(GPUQueue<Frame *> **ingresses, GPUQueue<Frame *> **egresses, int *propogation_delay_per_ch, int *link_transmission_rate, int ch_num);
        void SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num);
        void SetTimeslot(int64_t *timeslot_start_time, int64_t *timeslot_end_time);
        void SetStreams(cudaStream_t *streams, int stream_num);

        uint8_t *GetTransmissionCompletedAddr();
        uint32_t GetTotalBlockNum();

        void InitializeKernelParams();
        void BuildGraph(int batch_id);
        void BuildGraph();
        void LaunchInstance(int batch_id);
        void Run(int batch_id);
        void Run();

        cudaGraph_t GetGraph(int batch_id);
    };

    void LuanchP2PTransmiFramesKernel(dim3 grid, dim3 block, VDES::P2PParams *p2p_params, cudaStream_t stream);

} // namespace VDES

#endif