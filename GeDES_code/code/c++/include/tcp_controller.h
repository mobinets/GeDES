#ifndef TCP_CONTROLLER_H
#define TCP_CONTROLLER_H

#include "gpu_queue.cuh"
#include <vector>
#include <memory>
#include "packet_definition.h"
#include <cuda_runtime.h>
#include "conf.h"

namespace VDES
{

    typedef struct RecvPacketRecord
    {
        int64_t start_id;
        int64_t end_id;
        RecvPacketRecord *next;
    } RecvPacketRecord;

    // the definition of TCPConnection
    typedef struct
    {
        uint32_t src_ip;
        uint32_t dst_ip;
        uint16_t src_port;
        uint16_t dst_port;

        // congestion window
        uint32_t cwnd;

        // slow start threshold
        uint32_t ssthresh;

        // send and receive window size
        uint32_t rwnd;
        uint32_t swnd;

        // RTO, the retransmission timeout, default is 100 * 1000 ns
        uint32_t rto;
        uint32_t rtt;
        // the timestamp of the last packet sent
        int64_t retrans_timer;

        // maximal segment size, default is 1460 bytes
        uint32_t mss;

        // send bytes
        uint32_t snd;

        // sent but unacked bytes
        uint32_t una;

        // delayed ack packets, adopt delayed ack
        uint32_t unacked_packets;
        // delayed ack, ack per 5 packets by default
        uint32_t packets_num_per_ack;
        // the number of non-transmitted repeat ack
        uint32_t repeat_non_ack_num;

        // the number of received ack packets, used to update cwnd
        uint32_t ack_count;

        // expected sequence number
        uint32_t expected_seq;
        uint32_t former_repeat_ack_count;
        uint32_t repeat_ack_count;

        // send and recv cache
        uint64_t tx_num;
        uint64_t rx_num;

        //
        uint64_t planned_bytes;
        uint64_t acked_bytes;

        GPUQueue<Flow> *flows;
        GPUQueue<RecvPacketRecord *> *records_pool;
        RecvPacketRecord *records;
        int record_num;

        // VDES::Payload* for non-null mode, int64_t for null mode (timestamp),
        GPUQueue<void *> *recv_cache;
        GPUQueue<VDES::TCPPacket *> *send_cache;

#if ENABLE_DCTCP_PACKET
        // used for dctcp
        uint32_t dctcp_ecn_masked_bytes;
        uint32_t dctcp_acked_bytes;
        uint32_t dctcp_window_end;
        float dctcp_alpha;

        uint8_t ece;
#endif
        uint32_t acked_flows;
    } TCPConnection;

    typedef struct
    {
        // tcp connections, node_num*max_tcp_num
        TCPConnection **tcp_cons;
        int *tcp_cons_num_per_node;
        int node_num;

        // the maxinum number of tcp connections of a node
        int max_tcp_num;

        // for receivers, a queue per node
        GPUQueue<VDES::TCPPacket *> **recv_queues;

        // for senders
        GPUQueue<VDES::TCPPacket *> **send_queues;

        // MAX_GENERATED_PACKET_NUM
        TCPPacket **alloc_packets;
        int *packet_offset_per_node;
        int *used_packet_num_per_node;
        int *remaining_nic_cache_space_per_node;

        int *recycle_tcp_packet_num;
        TCPPacket **recycle_packets;

        int64_t *timeslot_start_time;
        int64_t *timeslot_end_time;

        uint8_t *is_completed_traffic_plan;
        uint8_t *is_completed_temporary_traffic_plan;

    } TCPParams;

    class TCPController
    {
    private:
        std::vector<TCPParams *> m_kernel_params;
        std::vector<cudaStream_t> m_streams;

        // cuda graph
        std::vector<cudaGraph_t> m_receiver_graphs;
        std::vector<cudaGraph_t> m_sender_graphs;

        // instance
        std::vector<cudaGraphExec_t> m_receiver_instances;
        std::vector<cudaGraphExec_t> m_sender_instances;

        // tcp params
        std::vector<TCPConnection *> m_tcp_cons;
        std::vector<int> m_tcp_num_per_node;
        std::vector<GPUQueue<VDES::TCPPacket *> *> m_recv_queues;
        std::vector<GPUQueue<VDES::TCPPacket *> *> m_send_queues;
        std::vector<int> m_nic_num_per_node;

        // gpu memory
        std::vector<TCPPacket *> m_packet_cache_space;
        // gpu ptr hosted in GPU
        std::vector<TCPPacket **> m_alloc_packets_gpu;
        // gpu ptr hosted in CPU
        std::vector<TCPPacket **> m_alloc_packets_cpu;
        // sharing with frame decapsulator
        std::vector<int *> m_remainming_nic_cache_space_per_node;

        std::vector<int *> m_used_packet_num_per_node_gpu;
        std::vector<int *> m_used_packet_num_per_node_cpu;

        // recycle packets
        std::vector<int *> m_recycle_tcp_packet_num_gpu;
        std::vector<int *> m_recycle_tcp_packet_num_cpu;
        std::vector<TCPPacket **> m_recycle_packets_gpu;
        std::vector<TCPPacket **> m_recycle_packets_cpu;
        std::vector<int *> m_packet_offsets;

        std::vector<int> m_max_packet_num;

        // batch properties
        std::vector<int> m_batch_start_index;
        std::vector<int> m_batch_end_index;

        // timeslot information
        int64_t *m_timeslot_start_time;
        int64_t *m_timeslot_end_time;

        // examine whether the packets are completed
        uint8_t *m_packets_completed_gpu;
        uint8_t *m_packets_completed_temporary_gpu;

        std::vector<TCPPacket **> m_recycle_tcp_tmp;

#if ENABLE_HUGE_GRAPH

        std::vector<cudaMemcpy3DParms> m_receive_memcpy_param;
        std::vector<cudaHostNodeParams> m_receive_host_param;
        std::vector<cudaMemcpy3DParms> m_send_memcpy_param;
        std::vector<cudaHostNodeParams> m_send_host_param;

#endif

    public:
        TCPController();
        ~TCPController();

        // initialize kernel params
        void InitKernelParams();

        void SetTCPConnections(TCPConnection **tcp_cons, int *tcp_cons_num_per_node, int node_num);
        void SetRecvQueues(GPUQueue<VDES::TCPPacket *> **recv_queues, int node_num);
        void SetSendQueues(GPUQueue<VDES::TCPPacket *> **send_queues, int node_num);
        void SetRemainingCacheSizeArray(int **remaining_nic_cache_space_per_node, int node_num);
        void SetTimeslotInfo(int64_t *timeslot_start_time, int64_t *timeslot_end_time);
        void SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num);
        void SetStreams(cudaStream_t *streams, int num);
        void SetNicNumPerNode(int *nic_num_per_node, int node_num);
        uint8_t *GetCompletedArr();
        uint8_t *GetTemporaryCompletedArr();

        void RecycleTCPPackets(int batch_id);
        void UpdateUsedTCPPackets(int batch_id);

        std::vector<void *> GetAllocInfo();
        std::vector<void *> GetRecycleInfo();

        // build graph
        void BuildGraph(int batch_id);
        void BuildGraph();

        cudaGraph_t GetReceiveGraph(int batch_id);
        cudaGraph_t GetSendGraph(int batch_id);

        // launch kernels
        void LaunchReceiveInstance(int batch_id);
        void LaunchSendInstance(int batch_id);
#if ENABLE_HUGE_GRAPH

        std::vector<cudaMemcpy3DParms> &GetReceiveMemcpyParam();
        std::vector<cudaHostNodeParams> &GetReceiveHostParam();
        std::vector<cudaMemcpy3DParms> &GetSendMemcpyParam();
        std::vector<cudaHostNodeParams> &GetSendHostParam();

#endif

        void RecordFlowResults(std::string file_name);
    };

    void LaunchReceiveTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream);
    void LaunchSendTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream);

}

#endif

// namespace VDES
// {
//     // the definition of TCPConnection
//     typedef struct
//     {
//         uint32_t src_ip;
//         uint32_t dst_ip;
//         uint16_t src_port;
//         uint16_t dst_port;

//         // congestion window
//         uint32_t cwnd;

//         // slow start threshold
//         uint32_t ssthresh;

//         // send and receive window size
//         // uint32_t rwnd;
//         uint32_t swnd;

//         // RTO, the retransmission timeout, default is 100 * 1000 ns
//         uint32_t rto;
//         // uint32_t rtt;
//         // the timestamp of the last packet sent
//         int64_t retrans_timer;

//         // maximal segment size, default is 1460 bytes
//         uint32_t mss;

//         // send bytes
//         uint32_t snd;

//         // sent but unacked bytes
//         uint32_t una;

//         // delayed ack packets, adopt delayed ack
//         uint32_t unacked_packets;
//         // delayed ack, ack per 5 packets by default
//         uint32_t packets_num_per_ack;

//         // the number of received ack packets, used to update cwnd
//         uint32_t ack_count;

//         // expected sequence number
//         uint32_t expected_seq;
//         uint32_t former_repeat_ack_count;
//         uint32_t repeat_ack_count;

//         // send and recv cache
//         uint64_t tx_num;
//         uint64_t rx_num;

//         //
//         uint64_t planned_bytes;
//         uint64_t acked_bytes;

//         // VDES::Payload* for non-null mode, int64_t for null mode (timestamp),
//         // GPUQueue<void *> *recv_cache;
//         // GPUQueue<VDES::TCPPacket *> *send_cache;
//     } TCPConnection;

//     typedef struct
//     {
//         // tcp connections, node_num*max_tcp_num
//         TCPConnection **tcp_cons;
//         int *tcp_cons_num_per_node;
//         int node_num;

//         // the maxinum number of tcp connections of a node
//         int max_tcp_num;

//         // for receivers, a queue per node
//         GPUQueue<VDES::TCPPacket *> **recv_queues;

//         // for senders
//         GPUQueue<VDES::TCPPacket *> **send_queues;

//         // MAX_GENERATED_PACKET_NUM
//         // TCPPacket *packet_cache_space;
//         TCPPacket **alloc_packets;
//         int *packet_offset_per_node;
//         int *used_packet_num_per_node;
//         int *remaining_nic_cache_space_per_node;

//         int *recycle_tcp_packet_num;
//         TCPPacket **recycle_packets;

//         int64_t *timeslot_start_time;
//         int64_t *timeslot_end_time;

//     } TCPParams;

//     class TCPController
//     {
//     private:
//         std::vector<TCPParams *> m_kernel_params;
//         std::vector<cudaStream_t> m_streams;

//         // cuda graph
//         std::vector<cudaGraph_t> m_receiver_graphs;
//         std::vector<cudaGraph_t> m_sender_graphs;

//         // instance
//         std::vector<cudaGraphExec_t> m_receiver_instances;
//         std::vector<cudaGraphExec_t> m_sender_instances;

//         // tcp params
//         std::vector<TCPConnection *> m_tcp_cons;
//         std::vector<int> m_tcp_num_per_node;
//         std::vector<GPUQueue<VDES::TCPPacket *> *> m_recv_queues;
//         std::vector<GPUQueue<VDES::TCPPacket *> *> m_send_queues;
//         std::vector<int> m_nic_num_per_node;

//         // gpu memory
//         std::vector<TCPPacket *> m_packet_cache_space;
//         // gpu ptr hosted in GPU
//         std::vector<TCPPacket **> m_alloc_packets_gpu;
//         // gpu ptr hosted in CPU
//         std::vector<TCPPacket **> m_alloc_packets_cpu;
//         // sharing with frame decapsulator
//         std::vector<int *> m_remainming_nic_cache_space_per_node;

//         std::vector<int *> m_used_packet_num_per_node_gpu;
//         std::vector<int *> m_used_packet_num_per_node_cpu;

//         // recycle packets
//         std::vector<int *> m_recycle_tcp_packet_num_gpu;
//         std::vector<int *> m_recycle_tcp_packet_num_cpu;
//         std::vector<TCPPacket **> m_recycle_packets_gpu;
//         std::vector<TCPPacket **> m_recycle_packets_cpu;
//         std::vector<int *> m_packet_offsets;

//         std::vector<int> m_max_packet_num;

//         // batch properties
//         std::vector<int> m_batch_start_index;
//         std::vector<int> m_batch_end_index;

//         // timeslot information
//         int64_t *m_timeslot_start_time;
//         int64_t *m_timeslot_end_time;

//         std::vector<TCPPacket **> m_recycle_tcp_tmp;

// #if ENABLE_HUGE_GRAPH

//         std::vector<cudaMemcpy3DParms> m_receive_memcpy_param;
//         std::vector<cudaHostNodeParams> m_receive_host_param;
//         std::vector<cudaMemcpy3DParms> m_send_memcpy_param;
//         std::vector<cudaHostNodeParams> m_send_host_param;

// #endif

//     public:
//         TCPController();
//         ~TCPController();

//         // initialize kernel params
//         void InitKernelParams();

//         void SetTCPConnections(TCPConnection **tcp_cons, int *tcp_cons_num_per_node, int node_num);
//         void SetRecvQueues(GPUQueue<VDES::TCPPacket *> **recv_queues, int node_num);
//         void SetSendQueues(GPUQueue<VDES::TCPPacket *> **send_queues, int node_num);
//         void SetRemainingCacheSizeArray(int **remaining_nic_cache_space_per_node, int node_num);
//         void SetTimeslotInfo(int64_t *timeslot_start_time, int64_t *timeslot_end_time);
//         void SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num);
//         void SetStreams(cudaStream_t *streams, int num);
//         void SetNicNumPerNode(int *nic_num_per_node, int node_num);

//         void RecycleTCPPackets(int batch_id);
//         void UpdateUsedTCPPackets(int batch_id);

//         // build graph
//         void BuildGraph(int batch_id);
//         void BuildGraph();

//         cudaGraph_t GetReceiveGraph(int batch_id);
//         cudaGraph_t GetSendGraph(int batch_id);

//         // void Run(int batch_id);

//         // launch kernels
//         void LaunchReceiveInstance(int batch_id);
//         void LaunchSendInstance(int batch_id);

// #if ENABLE_HUGE_GRAPH

//         std::vector<cudaMemcpy3DParms> &GetReceiveMemcpyParam();
//         std::vector<cudaHostNodeParams> &GetReceiveHostParam();
//         std::vector<cudaMemcpy3DParms> &GetSendMemcpyParam();
//         std::vector<cudaHostNodeParams> &GetSendHostParam();

// #endif
//     };

//     void LaunchReceiveTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream);
//     void LaunchSendTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream);
// }

// #endif
// #ifndef TCP_CONTROLLER_H
// #define TCP_CONTROLLER_H

// #include "gpu_queue.cuh"
// #include <vector>
// #include <memory>
// #include "packet_definition.h"
// #include <cuda_runtime.h>
// #include "conf.h"

// namespace VDES
// {
//     // the definition of TCPConnection
//     typedef struct
//     {
//         uint32_t src_ip;
//         uint32_t dst_ip;
//         uint16_t src_port;
//         uint16_t dst_port;

//         // congestion window
//         uint32_t cwnd;

//         // slow start threshold
//         uint32_t ssthresh;

//         // send and receive window size
//         uint32_t rwnd;
//         uint32_t cwnd;

//         // RTO, the retransmission timeout, default is 100 * 1000 ns
//         uint32_t rto;
//         uint32_t rtt;
//         // the timestamp of the last packet sent
//         int64_t retrans_timer;

//         // maximal segment size, default is 1460 bytes
//         uint32_t mss;

//         // send bytes
//         uint32_t snd;

//         // sent but unacked bytes
//         uint32_t una;

//         // delayed ack packets, adopt delayed ack
//         uint32_t unacked_packets;
//         // delayed ack, ack per 5 packets by default
//         uint32_t packets_num_per_ack;

//         // the number of received ack packets, used to update cwnd
//         uint32_t ack_count;

//         // expected sequence number
//         uint32_t expected_seq;
//         uint32_t former_repeat_ack_count;
//         uint32_t repeat_ack_count;

//         // send and recv cache
//         uint64_t tx_num;
//         // not used now
//         uint64_t rx_num;

//         //
//         uint64_t planned_bytes;
//         uint64_t acked_bytes;

//         // VDES::Payload* for non-null mode, int64_t for null mode (timestamp),
//         GPUQueue<void *> *recv_cache;
//         GPUQueue<VDES::TCPPacket *> *send_cache;
//     } TCPConnection;

//     typedef struct
//     {
//         // tcp connections, node_num*max_tcp_num
//         TCPConnection **tcp_cons;
//         int *tcp_cons_num_per_node;
//         int node_num;

//         // the maxinum number of tcp connections of a node
//         int max_tcp_num;

//         // for receivers, a queue per node
//         GPUQueue<VDES::TCPPacket *> **recv_queues;

//         // for senders
//         GPUQueue<VDES::TCPPacket *> **send_queues;

//         // MAX_GENERATED_PACKET_NUM
//         // TCPPacket *packet_cache_space;
//         TCPPacket **alloc_packets;
//         int *packet_offset_per_node;
//         int *used_packet_num_per_node;
//         int *remaining_nic_cache_space_per_node;

//         int64_t *timeslot_start_time;
//         int64_t *timeslot_end_time;

//     } TCPParams;

//     class TCPController
//     {
//     private:
//         std::vector<TCPParams *> m_kernel_params;
//         std::vector<cudaStream_t> m_streams;

//         // cuda graph
//         std::vector<cudaGraph_t> m_receiver_graphs;
//         std::vector<cudaGraph_t> m_sender_graphs;

//         // instance
//         std::vector<cudaGraphExec_t> m_receiver_instances;
//         std::vector<cudaGraphExec_t> m_sender_instances;

//         // tcp params
//         std::vector<TCPConnection *> m_tcp_cons;
//         std::vector<int> m_tcp_num_per_node;
//         std::vector<GPUQueue<VDES::TCPPacket *> *> m_recv_queues;
//         std::vector<GPUQueue<VDES::TCPPacket *> *> m_send_queues;
//         std::vector<int> m_nic_num_per_node;

//         // gpu memory
//         std::vector<TCPPacket *> m_packet_cache_space;
//         // gpu ptr hosted in GPU
//         std::vector<TCPPacket **> m_alloc_packets_gpu;
//         // gpu ptr hosted in CPU
//         std::vector<TCPPacket **> m_alloc_packets_cpu;
//         // sharing with frame decapsulator
//         std::vector<int *> m_remainming_nic_cache_space_per_node;

//         // batch properties
//         std::vector<int> m_batch_start_index;
//         std::vector<int> m_batch_end_index;

//         // timeslot information
//         int64_t *m_timeslot_start_time;
//         int64_t *m_timeslot_end_time;

//     public:
//         TCPController();
//         ~TCPController();

//         // initialize kernel params
//         void InitKernelParams();

//         void SetTCPConnections(TCPConnection **tcp_cons, int *tcp_cons_num_per_node, int node_num);
//         void SetRecvQueues(GPUQueue<VDES::TCPPacket *> **recv_queues, int node_num);
//         void SetSendQueues(GPUQueue<VDES::TCPPacket *> **send_queues, int node_num);
//         void SetRemainingCacheSizeArray(int **remaining_nic_cache_space_per_node, int node_num);
//         void SetTimeslotInfo(int64_t *timeslot_start_time, int64_t *timeslot_end_time);
//         void SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num);
//         void SetStreams(cudaStream_t *streams, int num);
//         void SetNicNumPerNode(int *nic_num_per_node, int node_num);

//         // build graph
//         void BuildGraph(int batch_id);

//         // launch kernels
//         void LaunchReceiveInstance(int batch_id);
//         void LaunchSendInstance(int batch_id);
//     };

//     void LaunchReceiveTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream);
//     void LaunchSendTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream);
// }

// #endif