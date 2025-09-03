#ifndef VDES_TIMER_H
#define VDES_TIMER_H

#include <cuda_runtime.h>
#include <vector>
#include "gpu_queue.cuh"
#include "packet_definition.h"

namespace VDES
{
    typedef struct
    {
        int64_t *time_start;
        int64_t *time_end;

        // identify whether the nodes completed
        // temporarily completed
        uint8_t *is_node_temporary_completed;
        // forever finished
        uint8_t *is_node_payload_completed;
        // identify whether switches completed
        uint8_t *is_transmission_completed;

        // identify whether channels completed
        uint8_t *is_ch_completed;

        // flow start instants
        GPUQueue<int64_t> *flow_start_instants;

        int64_t *temporary_completed_node_num;
        int64_t *completed_node_num;
        int64_t *transmission_completed_sw_num;

        int64_t *ch_completed_ch_num;

        bool *is_finished;

        int node_num;
        int sw_node;

        int ch_block_num;

    } TimerParam;

    class VDESTimer
    {

    public:
        /* data */
        VDESTimer();
        ~VDESTimer();

        void InitKernelParams();
        void SetTemporaryCompletedNodeNum(uint8_t *temporary_state);
        void SetCompletedNodeNum(uint8_t *completed_state);
        void SetTransmissionCompleted(uint8_t *transmission_completed);
        void SetNodeNum(int node_num);
        void SetSwitchNode(int sw_node);
        void SetTimestamp(int64_t *time_start, int64_t *time_end);
        void SetFlowStartInstants(GPUQueue<int64_t> *flow_start_instants);
        void SetIsFinished(bool *is_finished);
        bool IsFinished();

        void BuildGraphs();
        cudaGraph_t GetGraph();

        void SetChannelBlockNum(int ch_block_num);

    private:
        TimerParam *m_kernel_params;
        std::vector<cudaGraph_t> m_graphs;
        std::vector<cudaGraphExec_t> m_graph_execs;

        int64_t *m_temporary_completed_node_num_gpu;
        int64_t *m_completed_node_num_gpu;
        int64_t *m_transmission_completed_sw_num_gpu;

        int64_t *m_ch_completed_ch_num_gpu;

        // identify whether the nodes completed
        // temporarily completed
        uint8_t *m_is_node_temporary_completed_gpu;
        // forever finished
        uint8_t *m_is_node_payload_completed_gpu;
        // identify whether switches completed
        uint8_t *m_is_transmission_completed_gpu;

        // identify whether channels completed
        uint8_t *m_is_ch_completed_gpu;

        int64_t m_sw_num_gpu;
        int64_t m_node_num_gpu;

        int64_t m_ch_block_num_gpu;

        int64_t *m_timestamp_start_gpu;
        int64_t *m_timestamp_end_gpu;

        bool *m_is_finished_gpu;
        bool m_is_finished;

        GPUQueue<int64_t> *m_flow_start_instants_gpu;
    };

} // namespace VDES

cudaGraph_t create_timer_graph(int64_t *time_start, int64_t *time_end);

void LaunchTimer(int64_t *time_start, int64_t *time_end, cudaStream_t stream);

void UpdateTimer(VDES::TimerParam *param, cudaStream_t stream);

void Sum(void *d_temp_storage, size_t &temp_storage_bytes, uint8_t *d_in, int64_t *d_out, int64_t num, cudaStream_t stream);

#endif