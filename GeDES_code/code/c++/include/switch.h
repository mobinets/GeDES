#ifndef SWITCH_H
#define SWITCH_H

#include "gpu_queue.cuh"
#include "packet_definition.h"
#include "conf.h"
#include <vector>
#include <numeric>
#include <cstring>

namespace VDES
{

    typedef struct
    {
        uint8_t protocol;
        uint8_t *payload;
    } RecycleFramePayload;

    typedef struct
    {
        // forward packets from ingress to egress
        GPUQueue<Frame *> **ingresses;
        GPUQueue<Frame *> **egresses;

        int *queue_num_per_node;
        int *queue_offset_per_node;
        int *node_offset_per_queue;
        int *sw_id_per_node;

        // received packets per queue, node_num * queue_num * (queue_num+1)
        int **received_packets_per_queue;

        // mac forwarding table
        GPUQueue<MacForwardingRule *> **mac_forwarding_table;

        // fattree mode
        int ft_k;
        // k^2/4
        int ft_k_sq_quarter;
        uint8_t *ft_current_port_up_forward;

        // error frames, frame_queue * MAX_Transmission_num
        Frame **drop_frames;
        int *drop_frame_num;
        RecycleFramePayload *drop_cache;

        int node_num;
        int queue_num;

        uint8_t *is_completed;

    } SwitchParams;

    class SwitchController
    {

    private:
        std::vector<SwitchParams *> m_kernel_params;
        std::vector<cudaStream_t> m_streams;
        std::vector<cudaGraph_t> m_graphs;
        std::vector<cudaGraphExec_t> m_graph_execs;

        // kernel params
        std::vector<GPUQueue<Frame *> *> m_ingresses;
        std::vector<GPUQueue<Frame *> *> m_egresses;
        std::vector<int> m_queue_num_per_node;
        std::vector<int> m_sw_id_per_node;

        // ft properties
        int m_ft_k;
        int m_ft_k_sq_quarter;

        // drop frames
        std::vector<Frame **> m_drop_frames_gpu;
        std::vector<Frame **> m_drop_frames_cpu;
        std::vector<int *> m_drop_frame_num_gpu;
        std::vector<int *> m_drop_frame_num_cpu;
        std::vector<RecycleFramePayload *> m_drop_cache_gpu;
        std::vector<RecycleFramePayload *> m_drop_cache_cpu;

        // mac forwarding table
        std::vector<GPUQueue<MacForwardingRule *> *> m_mac_forwarding_table;

        // batch properties
        std::vector<int> m_batch_start_index;
        std::vector<int> m_batch_end_index;
        std::vector<int> m_queue_num_per_batch;

        uint8_t *m_transmission_completed;

    public:
        SwitchController();
        ~SwitchController();

        void SetIngresAndEgress(GPUQueue<Frame *> **ingresses, GPUQueue<Frame *> **egresses, int *queue_num_per_node, int *sw_id_per_node, int node_num);
        void SetMacForwardingTable(GPUQueue<MacForwardingRule *> **mac_forwarding_table, int node_num);
        void SetBatchproperties(int *batch_start_index, int *batch_end_index, int batch_num);
        void SetFtProperties(int ft_k);
        void SetStreams(cudaStream_t *streams, int num);

        void InitalizeKernelParams();
        void RecycleDropFrames(int batch_id);
        void BuildGraph(int batch_id);
        void BuildGraph();
        void LaunchInstance(int batch_id);
        void Run(int batch_id);
        void Run();

        cudaGraph_t GetGraph(int batch_id);

        uint8_t* GetTransmissionCompletedArr();
    };

    void LaunchForwardFramesKernel(dim3 grid, dim3 block, SwitchParams *params, cudaStream_t stream);

}

#endif

// namespace VDES
// {

//     typedef struct
//     {
//         uint8_t protocol;
//         uint8_t *payload;
//     } RecycleFramePayload;

//     typedef struct
//     {
//         // forward packets from ingress to egress
//         GPUQueue<Frame *> **ingresses;
//         GPUQueue<Frame *> **egresses;

//         int *queue_num_per_node;
//         int *queue_offset_per_node;
//         int *node_offset_per_queue;
//         int *sw_id_per_node;

//         // received packets per queue, node_num * queue_num * (queue_num+1)
//         int **received_packets_per_queue;

//         // mac forwarding table
//         GPUQueue<MacForwardingRule *> **mac_forwarding_table;

//         // fattree mode
//         int ft_k;
//         // k^2/4
//         int ft_k_sq_quarter;
//         uint8_t *ft_current_port_up_forward;

//         // error frames, frame_queue * MAX_Transmission_num
//         Frame **drop_frames;
//         int *drop_frame_num;
//         RecycleFramePayload *drop_cache;

//         int node_num;
//         int queue_num;

//     } SwitchParams;

//     class SwitchController
//     {

//     private:
//         std::vector<SwitchParams *> m_kernel_params;
//         std::vector<cudaStream_t> m_streams;
//         std::vector<cudaGraph_t> m_graphs;
//         std::vector<cudaGraphExec_t> m_graph_execs;

//         // kernel params
//         std::vector<GPUQueue<Frame *> *> m_ingresses;
//         std::vector<GPUQueue<Frame *> *> m_egresses;
//         std::vector<int> m_queue_num_per_node;
//         std::vector<int> m_sw_id_per_node;

//         // ft properties
//         int m_ft_k;
//         int m_ft_k_sq_quarter;

//         // drop frames
//         std::vector<Frame **> m_drop_frames_gpu;
//         std::vector<Frame **> m_drop_frames_cpu;
//         std::vector<int *> m_drop_frame_num_gpu;
//         std::vector<int *> m_drop_frame_num_cpu;
//         std::vector<RecycleFramePayload *> m_drop_cache_gpu;
//         std::vector<RecycleFramePayload *> m_drop_cache_cpu;

//         // mac forwarding table
//         std::vector<GPUQueue<MacForwardingRule *> *> m_mac_forwarding_table;

//         // batch properties
//         std::vector<int> m_batch_start_index;
//         std::vector<int> m_batch_end_index;
//         std::vector<int> m_queue_num_per_batch;

//     public:
//         SwitchController();
//         ~SwitchController();

//         void SetIngresAndEgress(GPUQueue<Frame *> **ingresses, GPUQueue<Frame *> **egresses, int *queue_num_per_node, int *sw_id_per_node, int node_num);
//         void SetMacForwardingTable(GPUQueue<MacForwardingRule *> **mac_forwarding_table, int node_num);
//         void SetBatchproperties(int *batch_start_index, int *batch_end_index, int batch_num);
//         void SetFtProperties(int ft_k);
//         void SetStreams(cudaStream_t *streams, int num);

//         void InitalizeKernelParams();
//         void RecycleDropFrames(int batch_id);
//         void BuildGraph(int batch_id);
//         void BuildGraph();
//         void LaunchInstance(int batch_id);
//         void Run(int batch_id);
//         void Run();

//         cudaGraph_t GetGraph(int batch_id);
//     };

//     void LaunchForwardFramesKernel(dim3 grid, dim3 block, SwitchParams *params, cudaStream_t stream);

// }

// #endif

// #ifndef SWITCH_H
// #define SWITCH_H

// #include "gpu_queue.cuh"
// #include "packet_definition.h"
// #include "conf.h"
// #include <vector>
// #include <numeric>

// namespace VDES
// {

//     typedef struct
//     {
//         // forward packets from ingress to egress
//         GPUQueue<Frame *> **ingresses;
//         GPUQueue<Frame *> **egresses;

//         int *queue_num_per_node;
//         int *queue_offset_per_node;
//         int *node_offset_per_queue;
//         int *sw_id_per_node;

//         // received packets per queue, node_num * queue_num * (queue_num+1)
//         int **received_packets_per_queue;

//         // mac forwarding table
//         GPUQueue<MacForwardingRule *> **mac_forwarding_table;

//         // fattree mode
//         int ft_k;
//         // k^2/4
//         int ft_k_sq_quarter;
//         uint8_t *ft_current_port_up_forward;
//         uint8_t *ft_current_port_down_forward;
//         uint8_t *ft_port_num_per_direction;

//         // error frames, frame_queue * MAX_Transmission_num
//         Frame **drop_frames;
//         int *drop_frame_num;

//         int node_num;
//         int queue_num;

//     } SwitchParams;

//     class SwitchController
//     {

//     private:
//         std::vector<SwitchParams *> m_kernel_params;
//         std::vector<cudaStream_t> m_streams;
//         std::vector<cudaGraph_t> m_graphs;
//         std::vector<cudaGraphExec_t> m_graph_execs;

//         // kernel params
//         std::vector<GPUQueue<Frame *> *> m_ingresses;
//         std::vector<GPUQueue<Frame *> *> m_egresses;
//         std::vector<int> m_queue_num_per_node;
//         std::vector<int> m_sw_id_per_node;

//         // ft properties
//         int m_ft_k;
//         int m_ft_k_sq_quarter;

//         // drop frames
//         std::vector<Frame **> m_drop_frames_gpu;
//         std::vector<Frame **> m_drop_frames_cpu;
//         std::vector<int *> m_drop_frame_num_gpu;
//         std::vector<int *> m_drop_frame_num_cpu;

//         // mac forwarding table
//         std::vector<GPUQueue<MacForwardingRule *> *> m_mac_forwarding_table;

//         // batch properties
//         std::vector<int> m_batch_start_index;
//         std::vector<int> m_batch_end_index;
//         std::vector<int> m_queue_num_per_batch;

//     public:
//         SwitchController();
//         ~SwitchController();

//         void SetIngresAndEgress(GPUQueue<Frame *> **ingresses, GPUQueue<Frame *> **egresses, int *queue_num_per_node, int *sw_id_per_node, int node_num);
//         void SetMacForwardingTable(GPUQueue<MacForwardingRule *> **mac_forwarding_table, int node_num);
//         void SetBatchproperties(int *batch_start_index, int *batch_end_index, int batch_num);
//         void SetFtProperties(int ft_k);
//         void SetStreams(cudaStream_t* streams, int num);

//         void InitalizeKernelParams();
//         void RecycleDropFrames(int batch_id);
//         void BuildGraph(int batch_id);
//         void LaunchInstance(int batch_id);
//         void Run(int batch_id);
//         void Run();

//     };

//     void LaunchForwardFramesKernel(dim3 grid, dim3 block, SwitchParams *params, cudaStream_t stream);

// }

// #endif