#ifndef TOPOLOGY_VDES_H
#define TOPOLOGY_VDES_H

#include <random>
#include <numeric>
#include <cmath>
#include "component.h"
#include "frame_decapsulation.h"
#include "frame_encapsulation.h"
#include "gpu_packet_manager.h"
#include "ipv4_controller.h"
#include "ipv4_decapsulation.h"
#include "ipv4_encapsulation.h"
#include "p2p_channel.h"
#include "packet_cache.h"
#include "switch.h"
#include "tcp_controller.h"
#include <vector>
#include "gpu_packet_manager.h"
#include "vdes_timer.h"

namespace VDES
{

  class Topology
  {
  private:
    // network devices
    std::vector<VDES::Node *> m_nodes;
    std::vector<VDES::Switch *> m_switches;

    // channels
    std::vector<VDES::P2PChanenl *> m_channels;
    int m_channel_tx_rate;
    int m_channel_popogation_delay;

    // topology
    std::vector<std::vector<int>> m_topology;

    // fattree topology
    int m_ft_k;

    uint32_t m_ip_group_size;
    uint32_t m_ip_base_addr;
    uint32_t m_ip_mask;

    std::map<int64_t, std::pair<int64_t, int64_t>> m_transmission_plan;

    // controller params
    std::vector<GPUQueue<Frame *> *> m_node_frame_ingress_queues;
    std::vector<GPUQueue<Frame *> *> m_node_frame_egress_queues;
    std::vector<uint8_t *> m_node_nic_mac;
    std::vector<GPUQueue<Frame *> *> m_sw_frame_ingress_queues;
    std::vector<GPUQueue<Frame *> *> m_sw_frame_egress_queues;
    std::vector<GPUQueue<Frame *> *> m_ch_frame_ingress_queues;
    std::vector<GPUQueue<Frame *> *> m_ch_frame_egress_queues;
    std::vector<int> m_ch_tx_rate;
    std::vector<int> m_ch_popogation_delay;
    std::vector<GPUQueue<Ipv4Packet *> *> m_node_ipv4_ingress_queues;
    std::vector<GPUQueue<Ipv4Packet *> *> m_node_ipv4_egress_queues;
    std::vector<GPUQueue<Ipv4Packet *> *> m_node_ipv4_error_queues;
    std::vector<GPUQueue<Ipv4Packet *> *> m_node_ipv4_local_delivery_queues;
    std::vector<GPUQueue<IPv4RoutingRule *> *> m_node_ipv4_routing_table_queues;
    std::vector<int *> m_node_ipv4_egress_remaing_capacity;

    std::vector<GPUQueue<TCPPacket *> *> m_node_tcp_ingress_queues;
    std::vector<GPUQueue<TCPPacket *> *> m_node_tcp_egress_queues;
    std::vector<TCPConnection *> m_tcp_connections;
    std::vector<int> m_tcp_num_per_node;

    std::vector<int> m_nic_num_per_node;
    std::vector<int> m_nic_num_per_sw;
    std::vector<int> m_sw_id_per_node;

    std::vector<GPUQueue<Frame *> *> m_cache_frame_queues;
    std::vector<int> m_cache_frame_num_per_node;
    std::vector<FrameQueueStatus *> m_cache_frame_status;

    std::vector<int> m_node_batch_start_index;
    std::vector<int> m_node_batch_end_index;
    std::vector<int> m_sw_batch_start_index;
    std::vector<int> m_sw_batch_end_index;
    std::vector<int> m_ch_batch_start_index;
    std::vector<int> m_ch_batch_end_index;
    std::vector<int> m_cache_batch_start_index;
    std::vector<int> m_cache_batch_end_index;

    std::vector<cudaStream_t> m_node_streams;
    std::vector<cudaStream_t> m_sw_streams;
    std::vector<cudaStream_t> m_ch_streams;
    std::vector<cudaStream_t> m_cache_streams;

    // controllers
    P2PChannelController m_p2p_controller;
    SwitchController m_sw_controller;
    FrameDecapsulationConatroller m_frame_decapsulation_controller;
    FrameEncapsulationController m_frame_encapsulation_controller;
    IPv4ProtocolController m_ipv4_controller;
    IPv4DecapsulationController m_ipv4_decapsulation_controller;
    IPv4EncapsulationController m_ipv4_encapsulation_controller;
    TCPController m_tcp_controller;
    CacheController m_cache_controller;
    GPUPacketManager m_pool_controller;
    VDESTimer m_timer_comtroller;

    // timeslot info
    int64_t *m_timeslot_start_gpu;
    int64_t *m_timeslot_end_gpu;
    int64_t m_timeslot_start_cpu;
    int64_t m_timeslot_end_cpu;

    // look ahead timeslot
    int m_look_ahead_timeslot;

    // transmission plan params
    double m_expected_packets_per_flow;
    double m_sigma;
    uint32_t m_distribution_seed;

    cudaGraph_t m_graph;
    cudaGraphExec_t m_graph_exec;
    cudaStream_t m_graph_stream;

    // traffic flows
    std::vector<std::vector<std::vector<Flow>>> m_flows;
    std::vector<int64_t> m_flow_start_instants;

    // build ft topology
    void InitializeFtNodeAndSwitch();
    void InstallIPv4ProtocolForAllNodes();
    void InstallTCPProtocolForAllNodes();
    void AllocateIPAddrForAllNodes();
    void BuildTCPConnections();
    void BuildIPv4RoutingTable();
    void ExractParamsForController();
    void InitializeControllers();
    void GenerateBatches();
    void CreateFlows();

  public:
    Topology();
    ~Topology();

    void SetNodes(Node **nodes, int node_num);
    void SetSwitches(Switch **switches, int sw_num);
    void SetFattreeProperties(int k, uint32_t ip_group_size,
                              uint32_t ip_base_addr, uint32_t ip_mask);
    void SetTimeslotInfo(int64_t timeslot_start, int64_t timeslot_end);
    void SetLookAheadTimeslot(int look_ahead_timeslot);

    void SetTransmissionPlanParams(int expected_packets_per_flow, double deviate, uint32_t seed = 42);

    void GenerateTransmissionPlan();
    void SetChannelParams(int tx_rate, int popogation_delay);

    void RecordFlowResults(std::string file_name);

    void BuildTopology();
#if ENABLE_HUGE_GRAPH
    void BuildHugeGraph();
#endif

    void BuildGraphOnlyGPU();

    // test interface
    void Run(int time);
    void RunGraph();
    /** @TODO: Check if all packets are received. */
    void CheckReceivedPackets();

    bool IsFinished();

    /** @brief: Create flows which timestamp increased with the system time. This is only used for validate the fidelity of the simulation. */
    void CreateFlows(int64_t start_timestamp);
    /**
     * @brief: Store the flows.
     * key: src node, val1: dst node, val2: flow info.
     */
    std::map<int64_t, std::vector<int64_t>> connectionss_map_;
    std::map<int64_t, std::vector<Flow>> flows_map_;
  };
  // Function used to generate traffic.
  extern std::vector<int64_t> generateTraffic(double mu, double sigma, int num_samples, double total_sum, unsigned int seed);

  extern std::vector<int64_t> generatePoissonTraffic(double flow_lambda, int num_samples, int64_t total_sum, unsigned int seed);

  extern std::vector<int64_t> generateAvgTraffic(double mu, double sigma, int num_samples, double total_sum, unsigned int seed);

  void SetFlowTimeRange(int range);

} // namespace VDES

// namespace VDES
// {

//     class Topology
//     {
//     private:
//         // network devices
//         std::vector<VDES::Node *> m_nodes;
//         std::vector<VDES::Switch *> m_switches;

//         // channels
//         std::vector<VDES::P2PChanenl *> m_channels;
//         int m_channel_tx_rate;
//         int m_channel_popogation_delay;

//         // topology
//         std::vector<std::vector<int>> m_topology;

//         // fattree topology
//         int m_ft_k;

//         uint32_t m_ip_group_size;
//         uint32_t m_ip_base_addr;
//         uint32_t m_ip_mask;

//         std::vector<int64_t> m_transmission_plan;

//         // controller params
//         std::vector<GPUQueue<Frame *> *> m_node_frame_ingress_queues;
//         std::vector<GPUQueue<Frame *> *> m_node_frame_egress_queues;
//         std::vector<uint8_t *> m_node_nic_mac;
//         std::vector<GPUQueue<Frame *> *> m_sw_frame_ingress_queues;
//         std::vector<GPUQueue<Frame *> *> m_sw_frame_egress_queues;
//         std::vector<GPUQueue<Frame *> *> m_ch_frame_ingress_queues;
//         std::vector<GPUQueue<Frame *> *> m_ch_frame_egress_queues;
//         std::vector<int> m_ch_tx_rate;
//         std::vector<int> m_ch_popogation_delay;
//         std::vector<GPUQueue<Ipv4Packet *> *> m_node_ipv4_ingress_queues;
//         std::vector<GPUQueue<Ipv4Packet *> *> m_node_ipv4_egress_queues;
//         std::vector<GPUQueue<Ipv4Packet *> *> m_node_ipv4_error_queues;
//         std::vector<GPUQueue<Ipv4Packet *> *>
//         m_node_ipv4_local_delivery_queues;
//         std::vector<GPUQueue<IPv4RoutingRule *> *>
//         m_node_ipv4_routing_table_queues; std::vector<int *>
//         m_node_ipv4_egress_remaing_capacity;

//         std::vector<GPUQueue<TCPPacket *> *> m_node_tcp_ingress_queues;
//         std::vector<GPUQueue<TCPPacket *> *> m_node_tcp_egress_queues;
//         std::vector<TCPConnection *> m_tcp_connections;
//         std::vector<int> m_tcp_num_per_node;

//         std::vector<int> m_nic_num_per_node;
//         std::vector<int> m_nic_num_per_sw;
//         std::vector<int> m_sw_id_per_node;

//         std::vector<GPUQueue<Frame *> *> m_cache_frame_queues;
//         std::vector<int> m_cache_frame_num_per_node;
//         std::vector<FrameQueueStatus *> m_cache_frame_status;

//         std::vector<int> m_node_batch_start_index;
//         std::vector<int> m_node_batch_end_index;
//         std::vector<int> m_sw_batch_start_index;
//         std::vector<int> m_sw_batch_end_index;
//         std::vector<int> m_ch_batch_start_index;
//         std::vector<int> m_ch_batch_end_index;
//         std::vector<int> m_cache_batch_start_index;
//         std::vector<int> m_cache_batch_end_index;

//         std::vector<cudaStream_t> m_node_streams;
//         std::vector<cudaStream_t> m_sw_streams;
//         std::vector<cudaStream_t> m_ch_streams;
//         std::vector<cudaStream_t> m_cache_streams;

//         // controllers
//         P2PChannelController m_p2p_controller;
//         SwitchController m_sw_controller;
//         FrameDecapsulationConatroller m_frame_decapsulation_controller;
//         FrameEncapsulationController m_frame_encapsulation_controller;
//         IPv4ProtocolController m_ipv4_controller;
//         IPv4DecapsulationController m_ipv4_decapsulation_controller;
//         IPv4EncapsulationController m_ipv4_encapsulation_controller;
//         TCPController m_tcp_controller;
//         CacheController m_cache_controller;

//         // timeslot info
//         int64_t *m_timeslot_start_gpu;
//         int64_t *m_timeslot_end_gpu;
//         int64_t m_timeslot_start_cpu;
//         int64_t m_timeslot_end_cpu;

//         // look ahead timeslot
//         int m_look_ahead_timeslot;

//         cudaGraph_t m_graph;
//         cudaGraphExec_t m_graph_exec;
//         cudaStream_t m_graph_stream;

//         // build ft topology
//         void InitializeFtNodeAndSwitch();
//         void InstallIPv4ProtocolForAllNodes();
//         void InstallTCPProtocolForAllNodes();
//         void AllocateIPAddrForAllNodes();
//         void BuildTCPConnections();
//         void BuildIPv4RoutingTable();
//         void ExractParamsForController();
//         void IntiializeControllers();
//         void GenerateBatches();

//     public:
//         Topology();
//         ~Topology();

//         void SetNodes(Node **nodes, int node_num);
//         void SetSwitches(Switch **switches, int sw_num);
//         void SetFattreeProperties(int k, uint32_t ip_group_size, uint32_t
//         ip_base_addr, uint32_t ip_mask);
//         /**
//          * TODO: Set and Initialize timeslot info.
//          */
//         void SetTimeslotInfo(int64_t timeslot_start, int64_t timeslot_end);
//         /**
//          * TODO: Set the look timeslot.
//          */
//         void SetLookAheadTimeslot(int look_ahead_timeslot);
//         void GenerateTransmissionPlan();
//         void SetChannelParams(int tx_rate, int popogation_delay);

//         void BuildTopology();
// #if ENABLE_HUGE_GRAPH
//         void BuildHugeGraph();
// #endif
//         // test interface
//         void Run();
//         void RunGraph();
//     };

// } // namespace VDES

#endif