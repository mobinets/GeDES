#include "topology.h"
#include "vdes_timer.h"
#include <cuda_runtime.h>
#include <iostream>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <getopt.h>
#include <cstdlib>
#include <cmath>
#include <random>
#include <ctime>

namespace VDES
{
  int flow_time_range = 10000000;

  void SetFlowTimeRange(int range)
  {
    flow_time_range = range;
  }

  void HostNodeCallback(void *user_data)
  {
    auto func = static_cast<std::function<void()> *>(user_data);
    (*func)();
  }

  template <typename T>
  GPUQueue<T *> *CreateGPUMemPool(int size)
  {
    GPUQueue<T *> cpu_queue;
    cudaMalloc(&cpu_queue.queue, size * sizeof(T *));
    std::vector<T *> alloc_packets;
    T *packets;
    cudaMalloc(&packets, size * sizeof(T));
    for (int i = 0; i < size; i++)
    {
      alloc_packets.push_back(packets + i);
    }
    cudaMemcpy(cpu_queue.queue, alloc_packets.data(), size * sizeof(T *), cudaMemcpyHostToDevice);
    cpu_queue.queue_capacity = size;
    cpu_queue.size = size;
    cpu_queue.head = 0;
    cpu_queue.removed_record = 0;

    GPUQueue<T *> *gpu_queue;
    cudaMalloc(&gpu_queue, sizeof(GPUQueue<T *>));
    cudaMemcpy(gpu_queue, &cpu_queue, sizeof(GPUQueue<T *>), cudaMemcpyHostToDevice);
    return gpu_queue;
  }

  template <typename T>
  void CopyDataToGPUQueue(GPUQueue<T> *gpu_queue, T *data, int num)
  {
    if (num == 0)
    {
      return;
    }

    GPUQueue<T> cpu_queue;
    cudaMemcpy(&cpu_queue, gpu_queue, sizeof(GPUQueue<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_queue.queue, data, num * sizeof(T), cudaMemcpyHostToDevice);
    cpu_queue.size = num;
    cudaMemcpy(gpu_queue, &cpu_queue, sizeof(GPUQueue<T>), cudaMemcpyHostToDevice);
  }

  Topology::Topology()
  {
    cudaStreamCreate(&m_graph_stream);
  }

  Topology::~Topology() {}

  void Topology::SetNodes(Node **nodes, int node_num)
  {
    m_nodes.insert(m_nodes.end(), nodes, nodes + node_num);
  }

  void Topology::SetSwitches(Switch **switches, int sw_num)
  {
    m_switches.insert(m_switches.end(), switches, switches + sw_num);
  }

  void Topology::SetFattreeProperties(int k, uint32_t ip_group_size,
                                      uint32_t ip_base_addr, uint32_t ip_mask)
  {
    m_ft_k = k;
    m_ip_group_size = ip_group_size;
    m_ip_base_addr = ip_base_addr;
    m_ip_mask = ip_mask;
  }

  void Topology::InitializeFtNodeAndSwitch()
  {
    int node_num = m_ft_k * m_ft_k * m_ft_k / 4;
    int sw_num = m_ft_k * m_ft_k + m_ft_k * m_ft_k / 4;
    int half_of_ft_k_sq = m_ft_k * m_ft_k / 2;

    for (uint32_t i = 0; i < node_num; i++)
    {
      Node *node = CreateNode(1, NODE_DEFAULT_INGRESS_QUEUE_SIZE,
                              NODE_DEFAULT_EGRESS_QUEUE_SIZE);
      node->node_id = i;
      memcpy(node->nics[0]->mac_addr, &i, 4);
      m_nodes.push_back(node);
    }

    for (int i = 0; i < sw_num; i++)
    {
      Switch *sw = CreateSwitch(m_ft_k, Switch_DEFAULT_INGRESS_QUEUE_SIZE,
                                Switch_DEFAULT_EGRESS_QUEUE_SIZE);
      m_switches.push_back(sw);
    }

    int sw_id = 0;
    for (int i = 0; i < m_ft_k; i++)
    {
      for (int j = 0; j < m_ft_k / 2; j++)
      {
        m_switches[sw_id]->sw_id = i * m_ft_k + j;
        m_switches[sw_id + half_of_ft_k_sq]->sw_id = i * m_ft_k + j + m_ft_k / 2;
        sw_id++;
      }
    }

    for (int i = m_ft_k * m_ft_k; i < sw_num; i++)
    {
      m_switches[i]->sw_id = i;
    }
  }

  void Topology::InstallIPv4ProtocolForAllNodes()
  {
    for (int i = 0; i < m_nodes.size(); i++)
    {
      InstallIPv4Protocol(m_nodes[i]);
    }
  }

  void Topology::InstallTCPProtocolForAllNodes()
  {
    for (int i = 0; i < m_nodes.size(); i++)
    {
      InstallTCPProtocol(m_nodes[i]);
    }
  }

  void Topology::AllocateIPAddrForAllNodes()
  {
    int group_num = m_ft_k * m_ft_k / 2;
    for (int i = 0; i < group_num; i++)
    {
      AllocateIPAddr(m_nodes.data() + i * m_ft_k / 2, m_ft_k / 2,
                     m_ip_base_addr + i * m_ip_group_size, m_ip_mask);
    }
  }

  void Topology::BuildTCPConnections()
  {
    size_t node_num = m_nodes.size();
    // std::vector<bool> is_visited(node_num, false);
    // for (size_t i = 0; i < node_num; i++)
    // {
    //   if (!is_visited.at(i))
    //   {
    //     auto [dst_node, plan_bytes] = m_transmission_plan.at(i);
    //     ConnectTCPConnection(m_nodes[i], plan_bytes, 0, m_nodes[dst_node], m_transmission_plan[dst_node].second, 0);
    //     is_visited[i] = true;
    //     is_visited[dst_node] = true;
    //   }
    // }
    for (int i = 0; i < node_num / 2; i++)
    {
      ConnectTCPConnection(m_nodes[i], 0, 0, m_nodes[i + node_num / 2], 0, 0);
    }
  }

  void Topology::BuildIPv4RoutingTable()
  {

    for (int i = 0; i < m_nodes.size(); i++)
    {
      auto ipv4 = GetIPv4Protocol(m_nodes[i]);
      GPUQueue<IPv4RoutingRule *> routing_table;
      cudaMemcpy(&routing_table, ipv4->routing_table,
                 sizeof(GPUQueue<IPv4RoutingRule *>), cudaMemcpyDeviceToHost);

      int nic_num = ipv4->nic_num;
      for (int j = 0; j < nic_num; j++)
      {
        IPv4RoutingRule rule_cpu;
        IPv4RoutingRule *rule_gpu;
        cudaMalloc(&rule_gpu, sizeof(IPv4RoutingRule));
        rule_cpu.dst = ipv4->ipv4_interfaces[j].ip;
        rule_cpu.mask = -1;
        rule_cpu.gw = 0;
        cudaMemcpy(rule_gpu, &rule_cpu, sizeof(IPv4RoutingRule),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(routing_table.queue + j, &rule_gpu, sizeof(IPv4RoutingRule *),
                   cudaMemcpyHostToDevice);
      }

      // add default gateway
      IPv4RoutingRule rule_cpu;
      IPv4RoutingRule *rule_gpu;
      cudaMalloc(&rule_gpu, sizeof(IPv4RoutingRule));
      rule_cpu.dst = 0;
      rule_cpu.mask = 0;
      // next hop is the dst_ip
      rule_cpu.gw = 1;
      rule_cpu.if_id = 0;
      cudaMemcpy(rule_gpu, &rule_cpu, sizeof(IPv4RoutingRule),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(routing_table.queue + nic_num, &rule_gpu,
                 sizeof(IPv4RoutingRule *), cudaMemcpyHostToDevice);
      routing_table.size = nic_num + 1;
      cudaMemcpy(ipv4->routing_table, &routing_table,
                 sizeof(GPUQueue<IPv4RoutingRule>), cudaMemcpyHostToDevice);
    }
  }

  void Topology::InitializeControllers()
  {
    m_p2p_controller.SetIngressAndEgress(
        m_ch_frame_ingress_queues.data(), m_ch_frame_egress_queues.data(),
        m_ch_popogation_delay.data(), m_ch_tx_rate.data(), m_channels.size() * 2);
    m_p2p_controller.SetStreams(m_ch_streams.data(), m_ch_streams.size());
    m_p2p_controller.SetBatchProperties(m_ch_batch_start_index.data(),
                                        m_ch_batch_end_index.data(),
                                        m_ch_batch_start_index.size());
    m_p2p_controller.SetTimeslot(m_timeslot_start_gpu, m_timeslot_end_gpu);
    m_p2p_controller.InitializeKernelParams();
    m_p2p_controller.BuildGraph();

    m_frame_encapsulation_controller.SetStreams(m_node_streams.data(),
                                                m_node_streams.size());
    m_frame_encapsulation_controller.SetFrameProperties(
        m_node_frame_egress_queues.data(), m_node_nic_mac.data(),
        m_nic_num_per_node.data(), m_nodes.size());
    m_frame_encapsulation_controller.SetFatTreeArpProperties(
        m_ft_k, m_ip_base_addr, m_ip_group_size);
    m_frame_encapsulation_controller.SetPacketProperties(
        (GPUQueue<void *> **)m_node_ipv4_egress_queues.data(), m_nodes.size());
    m_frame_encapsulation_controller.SetBatchProperties(
        m_node_batch_start_index.data(), m_node_batch_end_index.data(),
        m_node_batch_start_index.size());
    m_frame_encapsulation_controller.InitializeKernelParams();
    m_frame_encapsulation_controller.BuildGraph();

    m_frame_decapsulation_controller.SetStreams(m_node_streams.data(),
                                                m_node_streams.size());
    m_frame_decapsulation_controller.SetFrameIngress(
        m_node_frame_ingress_queues.data(), m_node_frame_ingress_queues.size());
    m_frame_decapsulation_controller.SetPacketIngress(
        (GPUQueue<void *> **)m_node_ipv4_ingress_queues.data(),
        m_node_ipv4_ingress_queues.size());
    m_frame_decapsulation_controller.SetNodeProperties(m_nic_num_per_node.data(),
                                                       m_nodes.size());
    m_frame_decapsulation_controller.SetBatchProperties(
        m_node_batch_start_index.data(), m_node_batch_end_index.data(),
        m_node_batch_start_index.size());
    m_frame_decapsulation_controller.InitializeKernelParams();
    m_frame_decapsulation_controller.BuildGraphs();

    m_ipv4_controller.SetEgressQueues(m_node_ipv4_egress_queues.data(),
                                      m_nic_num_per_node.data(), m_nodes.size());
    m_ipv4_controller.SetIngressQueues(m_node_ipv4_ingress_queues.data(),
                                       m_node_ipv4_ingress_queues.size());
    m_ipv4_controller.SetErrorQueues(m_node_ipv4_error_queues.data(),
                                     m_node_ipv4_error_queues.size());
    m_ipv4_controller.SetLocalEgressQueues(
        m_node_ipv4_local_delivery_queues.data(),
        m_node_ipv4_local_delivery_queues.size());
    m_ipv4_controller.SetRoutingTables(
        (GPUQueue<IPv4RoutingRule *> **)m_node_ipv4_routing_table_queues.data(),
        m_node_ipv4_routing_table_queues.size());
    m_ipv4_controller.SetStreams(m_node_streams.data(), m_node_streams.size());
    m_ipv4_controller.SetBatchProperties(m_node_batch_start_index.data(),
                                         m_node_batch_end_index.data(),
                                         m_node_batch_start_index.size());
    m_ipv4_controller.SetEgressRemainingCapacity(
        m_node_ipv4_egress_remaing_capacity.data(),
        m_node_ipv4_egress_remaing_capacity.size());
    m_ipv4_controller.InitializeKernelParams();
    m_ipv4_controller.BuildGraphs();

    m_ipv4_decapsulation_controller.SetStreams(m_node_streams.data(),
                                               m_node_streams.size());
    m_ipv4_decapsulation_controller.SetIPv4Queues(
        m_node_ipv4_local_delivery_queues.data(), m_nodes.size());
    m_ipv4_decapsulation_controller.SetL4Queues(
        (GPUQueue<uint8_t *> **)m_node_tcp_ingress_queues.data(), m_nodes.size());
    m_ipv4_decapsulation_controller.SetNICNum(m_nic_num_per_node.data(),
                                              m_nodes.size());
    m_ipv4_decapsulation_controller.SetBatchProperties(
        m_node_batch_start_index.data(), m_node_batch_end_index.data(),
        m_node_batch_start_index.size());
    m_ipv4_decapsulation_controller.InitalizeKernelParams();
    m_ipv4_decapsulation_controller.BuildGraph();

    m_ipv4_encapsulation_controller.SetStreams(m_node_streams.data(),
                                               m_node_streams.size());
    m_ipv4_encapsulation_controller.SetIPv4PacketQueue(
        m_node_ipv4_ingress_queues.data(), m_nodes.size());
    m_ipv4_encapsulation_controller.SetL4PacketQueue(
        (GPUQueue<uint8_t *> **)m_node_tcp_egress_queues.data(), m_nodes.size());
    m_ipv4_encapsulation_controller.SetNICNumPerNode(m_nic_num_per_node.data(),
                                                     m_nodes.size());
    m_ipv4_encapsulation_controller.SetBatchProperties(
        m_node_batch_start_index.data(), m_node_batch_end_index.data(),
        m_node_batch_start_index.size());
    m_ipv4_encapsulation_controller.InitalizeKernelParams();
    m_ipv4_encapsulation_controller.BuildGraph();

    m_tcp_controller.SetStreams(m_node_streams.data(), m_node_streams.size());
    m_tcp_controller.SetNicNumPerNode(m_nic_num_per_node.data(), m_nodes.size());
    m_tcp_controller.SetBatchProperties(m_node_batch_start_index.data(),
                                        m_node_batch_end_index.data(),
                                        m_node_batch_start_index.size());
    m_tcp_controller.SetRecvQueues(m_node_tcp_ingress_queues.data(),
                                   m_nodes.size());
    m_tcp_controller.SetSendQueues(m_node_tcp_egress_queues.data(),
                                   m_node_tcp_egress_queues.size());
    m_tcp_controller.SetTCPConnections(m_tcp_connections.data(),
                                       m_tcp_num_per_node.data(), m_nodes.size());
    m_tcp_controller.SetRemainingCacheSizeArray(
        m_node_ipv4_egress_remaing_capacity.data(),
        m_node_ipv4_egress_remaing_capacity.size());
    m_tcp_controller.SetTimeslotInfo(m_timeslot_start_gpu, m_timeslot_end_gpu);
    m_tcp_controller.InitKernelParams();
    m_tcp_controller.BuildGraph();

    m_sw_controller.SetStreams(m_sw_streams.data(), m_sw_streams.size());
    m_sw_controller.SetFtProperties(m_ft_k);
    m_sw_controller.SetIngresAndEgress(
        m_sw_frame_ingress_queues.data(), m_sw_frame_egress_queues.data(),
        m_nic_num_per_sw.data(), m_sw_id_per_node.data(), m_switches.size());
    m_sw_controller.SetBatchproperties(m_sw_batch_start_index.data(),
                                       m_sw_batch_end_index.data(),
                                       m_sw_batch_start_index.size());
    m_sw_controller.InitalizeKernelParams();
    m_sw_controller.BuildGraph();

#if ENABLE_CACHE
    m_cache_controller.SetStreams(m_cache_streams.data(), m_cache_streams.size());
    m_cache_controller.SetEgressProperties(
        m_cache_frame_queues.data(), m_cache_frame_status.data(),
        m_cache_frame_num_per_node.data(), m_nodes.size() + m_switches.size());
    m_cache_controller.SetLookaheadTimeslotNum(m_look_ahead_timeslot);
    m_cache_controller.SetBatches(m_cache_batch_start_index.data(),
                                  m_cache_batch_end_index.data(),
                                  m_cache_batch_start_index.size());
    m_cache_controller.SetTimeSlot(m_timeslot_start_gpu, m_timeslot_end_gpu);
    m_cache_controller.InitializeKernelParams();
    m_cache_controller.BuildGraph();
#endif

    auto frame_alloc_info = m_frame_encapsulation_controller.GetAllocateInfo();
    auto frame_recycle_info = m_frame_decapsulation_controller.GetRecycleInfo();

    m_pool_controller.SetFramePacketQueues((void **)frame_alloc_info[0], (int *)frame_alloc_info[1], NULL, (void **)frame_recycle_info[0], (int *)frame_recycle_info[1], NULL);

    auto ipv4_alloc_info = m_ipv4_encapsulation_controller.GetAllocateInfo();
    auto ipv4_recycle_info = m_ipv4_decapsulation_controller.GetRecycleInfo();
    m_pool_controller.SetIPv4PacketQueues((void **)ipv4_alloc_info[0], (int *)ipv4_alloc_info[1], NULL, (void **)ipv4_recycle_info[0], (int *)ipv4_recycle_info[1], NULL);

    auto tcp_alloc_info = m_tcp_controller.GetAllocInfo();
    auto tcp_recycle_info = m_tcp_controller.GetRecycleInfo();
    m_pool_controller.SetTCPPacketQueues((void **)tcp_alloc_info[0], (int *)tcp_alloc_info[1], NULL, (void **)tcp_recycle_info[0], (int *)tcp_recycle_info[1], NULL);
    m_pool_controller.SetQueueNum(m_nodes.size());

#if ENABLE_GPU_MEM_POOL
    GPUQueue<TCPPacket *> *tcp_pool = CreateGPUMemPool<TCPPacket>(INITIALIZED_MEMORY_POOL_SIZE);
    GPUQueue<Ipv4Packet *> *ipv4_pool = CreateGPUMemPool<Ipv4Packet>(INITIALIZED_MEMORY_POOL_SIZE);
    GPUQueue<Frame *> *frame_pool = CreateGPUMemPool<Frame>(INITIALIZED_MEMORY_POOL_SIZE);
    m_pool_controller.SetTCPPool((GPUQueue<void *> *)tcp_pool);
    m_pool_controller.SetIPv4Pool((GPUQueue<void *> *)ipv4_pool);
    m_pool_controller.SetFramePool((GPUQueue<void *> *)frame_pool);

#endif
    m_pool_controller.InitializeKernelParams();
    m_pool_controller.BuildGraphs();

    GPUQueue<int64_t> *flow_start_instants = create_gpu_queue<int64_t>(100);
    CopyDataToGPUQueue(flow_start_instants, m_flow_start_instants.data(), m_flow_start_instants.size());
    m_timer_comtroller.SetFlowStartInstants(flow_start_instants);
    m_timer_comtroller.SetCompletedNodeNum(m_tcp_controller.GetCompletedArr());
    m_timer_comtroller.SetTemporaryCompletedNodeNum(m_tcp_controller.GetTemporaryCompletedArr());
    // m_timer_comtroller.SetTransmissionCompleted(m_sw_controller.GetTransmissionCompletedArr());
    m_timer_comtroller.SetTransmissionCompleted(m_p2p_controller.GetTransmissionCompletedAddr());
    /**
     * @TODO: Set the node and switch size, and init the kernel params of timer.
     */
    m_timer_comtroller.SetNodeNum(m_nodes.size());
    m_timer_comtroller.SetSwitchNode(m_switches.size());
    m_timer_comtroller.SetChannelBlockNum(m_p2p_controller.GetTotalBlockNum());
    m_timer_comtroller.SetTimestamp(m_timeslot_start_gpu, m_timeslot_end_gpu);
    m_timer_comtroller.InitKernelParams();
    m_timer_comtroller.BuildGraphs();
  }

  void Topology::ExractParamsForController()
  {
    for (int i = 0; i < m_nodes.size(); i++)
    {
      int nic_num = m_nodes[i]->nics.size();
      auto nics = m_nodes[i]->nics;
      auto ipv4 = GetIPv4Protocol(m_nodes[i]);
      auto tcp = GetTCPProtocol(m_nodes[i]);
      for (int j = 0; j < nic_num; j++)
      {
        m_node_frame_ingress_queues.push_back(nics[j]->ingress);
        m_node_frame_egress_queues.push_back(nics[j]->egress);
        m_node_nic_mac.push_back(nics[j]->mac_addr);
        m_cache_frame_queues.push_back(nics[j]->egress);
      }
      m_nic_num_per_node.push_back(nic_num);
      m_node_ipv4_ingress_queues.push_back(ipv4->ingress);
      m_node_ipv4_egress_queues.insert(m_node_ipv4_egress_queues.end(),
                                       ipv4->egresses, ipv4->egresses + nic_num);
      m_node_ipv4_local_delivery_queues.push_back(ipv4->local_delivery);
      m_node_ipv4_error_queues.push_back(ipv4->error_queue);
      m_node_ipv4_routing_table_queues.push_back(ipv4->routing_table);
      m_node_tcp_egress_queues.push_back(tcp->egress);
      m_node_tcp_ingress_queues.push_back(tcp->ingress);
      m_cache_frame_num_per_node.push_back(nic_num);

      TCPConnection **tcp_cons;
      for (int j = 0; j < MAX_TCP_CONNECTION_NUM; j++)
      {
        if (j < tcp->tcp_cons_num)
        {
          TCPConnection *tcp_con;
          cudaMalloc(&tcp_con, sizeof(TCPConnection));
          cudaMemcpy(tcp_con, tcp->tcp_cons[j], sizeof(TCPConnection),
                     cudaMemcpyHostToDevice);
          m_tcp_connections.push_back(tcp_con);
        }
        else
        {
          m_tcp_connections.push_back(NULL);
        }
      }

      m_tcp_num_per_node.push_back(tcp->tcp_cons_num);
    }

    for (int i = 0; i < m_switches.size(); i++)
    {
      int port_num = m_switches[i]->port_num;
      auto nics = m_switches[i]->nics;
      for (int j = 0; j < port_num; j++)
      {
        m_sw_frame_egress_queues.push_back(nics[j]->egress);
        m_sw_frame_ingress_queues.push_back(nics[j]->ingress);
        m_cache_frame_queues.push_back(nics[j]->egress);
      }
      m_nic_num_per_sw.push_back(port_num);
      m_sw_id_per_node.push_back(m_switches[i]->sw_id);
      m_cache_frame_num_per_node.push_back(port_num);
    }

    for (int i = 0; i < m_cache_frame_queues.size(); i++)
    {
      FrameQueueStatus *status_gpu;
      cudaMalloc(&status_gpu, sizeof(FrameQueueStatus));
      FrameQueueStatus status_cpu;
      memset(&status_cpu, 0, sizeof(FrameQueueStatus));
      cudaMalloc(&status_cpu.packet_status_in_cache_win,
                 sizeof(uint8_t) * MAX_TRANSMITTED_PACKET_NUM *
                     (m_look_ahead_timeslot * 2 + 2));
      cudaMemcpy(status_gpu, &status_cpu, sizeof(FrameQueueStatus),
                 cudaMemcpyHostToDevice);
      m_cache_frame_status.push_back(status_gpu);
    }

    for (int i = 0; i < m_nodes.size(); i++)
    {
      m_ch_frame_ingress_queues.push_back(m_channels[i]->nic2->egress);
      m_ch_frame_egress_queues.push_back(m_channels[i]->nic1->ingress);
      m_ch_tx_rate.push_back(m_channels[i]->tx_rate);
      m_ch_popogation_delay.push_back(m_channels[i]->popogation_delay);
    }

    for (int i = 0; i < m_nodes.size(); i++)
    {
      m_ch_frame_ingress_queues.push_back(m_channels[i]->nic1->egress);
      m_ch_frame_egress_queues.push_back(m_channels[i]->nic2->ingress);
      m_ch_tx_rate.push_back(m_channels[i]->tx_rate);
      m_ch_popogation_delay.push_back(m_channels[i]->popogation_delay);
    }

    int node_ch_num = m_ft_k * m_ft_k * m_ft_k / 4;
    for (int i = 0; i < m_ft_k * m_ft_k * m_ft_k / 4; i++)
    {
      m_ch_frame_ingress_queues.push_back(
          m_channels[node_ch_num + i]->nic1->egress);
      m_ch_frame_egress_queues.push_back(
          m_channels[node_ch_num + i]->nic2->ingress);
      m_ch_tx_rate.push_back(m_channels[node_ch_num + i]->tx_rate);
      m_ch_popogation_delay.push_back(
          m_channels[node_ch_num + i]->popogation_delay);
    }

    for (int i = 0; i < m_ft_k * m_ft_k * m_ft_k / 4; i++)
    {
      m_ch_frame_ingress_queues.push_back(
          m_channels[node_ch_num + i]->nic2->egress);
      m_ch_frame_egress_queues.push_back(
          m_channels[node_ch_num + i]->nic1->ingress);
      m_ch_tx_rate.push_back(m_channels[node_ch_num + i]->tx_rate);
      m_ch_popogation_delay.push_back(
          m_channels[node_ch_num + i]->popogation_delay);
    }

    int core_sw_ch_num = m_ft_k * m_ft_k * m_ft_k / 2;
    for (int i = 0; i < m_ft_k * m_ft_k * m_ft_k / 4; i++)
    {
      m_ch_frame_ingress_queues.push_back(
          m_channels[core_sw_ch_num + i]->nic1->egress);
      m_ch_frame_egress_queues.push_back(
          m_channels[core_sw_ch_num + i]->nic2->ingress);
      m_ch_tx_rate.push_back(m_channels[core_sw_ch_num + i]->tx_rate);
      m_ch_popogation_delay.push_back(
          m_channels[core_sw_ch_num + i]->popogation_delay);
    }

    for (int i = 0; i < m_ft_k * m_ft_k * m_ft_k / 4; i++)
    {
      m_ch_frame_ingress_queues.push_back(
          m_channels[core_sw_ch_num + i]->nic2->egress);
      m_ch_frame_egress_queues.push_back(
          m_channels[core_sw_ch_num + i]->nic1->ingress);
      m_ch_tx_rate.push_back(m_channels[core_sw_ch_num + i]->tx_rate);
      m_ch_popogation_delay.push_back(
          m_channels[core_sw_ch_num + i]->popogation_delay);
    }
  }

  void Topology::GenerateBatches()
  {
    m_node_batch_start_index.push_back(0);
    m_node_batch_end_index.push_back(m_nodes.size());

    int node_batch_num = m_node_batch_start_index.size();
    for (int i = 0; i < node_batch_num; i++)
    {
      int *remaining_capacity;
      cudaMalloc(&remaining_capacity,
                 sizeof(int) *
                     (m_node_batch_end_index[i] - m_node_batch_start_index[i]));
      m_node_ipv4_egress_remaing_capacity.push_back(remaining_capacity);
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      m_node_streams.push_back(stream);
    }

    // access layer
    m_sw_batch_start_index.push_back(0);
    m_sw_batch_end_index.push_back(m_ft_k * m_ft_k / 2);

    // aggregation layer
    m_sw_batch_start_index.push_back(m_ft_k * m_ft_k / 2);
    m_sw_batch_end_index.push_back(m_ft_k * m_ft_k);

    // core layer
    m_sw_batch_start_index.push_back(m_ft_k * m_ft_k);
    m_sw_batch_end_index.push_back(m_ft_k * m_ft_k + m_ft_k * m_ft_k / 4);

    for (int i = 0; i < m_sw_batch_start_index.size(); i++)
    {
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      m_sw_streams.push_back(stream);
    }

    // channel
    m_ch_batch_start_index.push_back(0);
    m_ch_batch_end_index.push_back(m_ch_frame_ingress_queues.size());

    for (int i = 0; i < m_ch_batch_start_index.size(); i++)
    {
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      m_ch_streams.push_back(stream);
    }

    // cache
    m_cache_batch_start_index.push_back(0);
    m_cache_batch_end_index.push_back(m_cache_frame_num_per_node.size());
    for (int i = 0; i < m_cache_batch_start_index.size(); i++)
    {
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      m_cache_streams.push_back(stream);
    }
  }

  void Topology::BuildTopology()
  {
#if ENABLE_FATTREE_MODE

    // build fat-tree topology
    InitializeFtNodeAndSwitch();

    // networking between nodes and switches
    int access_sw_num = m_ft_k * m_ft_k / 2;
    for (int i = 0; i < access_sw_num; i++)
    {
      for (int j = 0; j < m_ft_k / 2; j++)
      {
        P2PChanenl *ch =
            CreateP2PChannel(m_channel_tx_rate, m_channel_popogation_delay);
        bool is_success =
            ConnectDevices(m_nodes[i * m_ft_k / 2 + j], 0, m_switches[i], j, ch);
        if (!is_success)
        {
          std::cout << "ConnectDevices with nodes failed!" << i << j << std::endl;
          exit(1);
        }
        m_channels.push_back(ch);
      }
    }

    for (int i = 0; i < m_ft_k; i++)
    {
      for (int j = 0; j < m_ft_k / 2; j++)
      {
        for (int k = 0; k < m_ft_k / 2; k++)
        {
          P2PChanenl *ch =
              CreateP2PChannel(m_channel_tx_rate, m_channel_popogation_delay);
          /**
           * @TODO: Port id + m_ft_k / 2.
           */
          bool is_success =
              ConnectDevices(m_switches[access_sw_num + i * m_ft_k / 2 + j], k,
                             m_switches[i * m_ft_k / 2 + k], j + m_ft_k / 2, ch);
          if (!is_success)
          {
            std::cout << "ConnectDevices failed!" << i << j << k << std::endl;
            exit(1);
          }
          m_channels.push_back(ch);
        }
      }
    }

    int core_sw_offset = m_ft_k * m_ft_k;
    int core_sw_num = m_ft_k * m_ft_k / 4;
    for (int i = 0; i < m_ft_k / 2; i++)
    {
      for (int j = 0; j < m_ft_k / 2; j++)
      {
        for (int k = 0; k < m_ft_k; k++)
        {
          P2PChanenl *ch =
              CreateP2PChannel(m_channel_tx_rate, m_channel_popogation_delay);
          bool is_success = ConnectDevices(
              m_switches[core_sw_offset + i * m_ft_k / 2 + j], k,
              m_switches[access_sw_num + k * m_ft_k / 2 + i], j + m_ft_k / 2, ch);
          if (!is_success)
          {
            std::cout << "ConnectDevices failed!" << i << j << k << std::endl;
            exit(1);
          }
          m_channels.push_back(ch);
        }
      }
    }

    // intall protocols
    // GenerateTransmissionPlan();
    InstallIPv4ProtocolForAllNodes();
    InstallTCPProtocolForAllNodes();
    AllocateIPAddrForAllNodes();
    BuildTCPConnections();
    CreateFlows(0);
    BuildIPv4RoutingTable();
    ExractParamsForController();
    GenerateBatches();
    InitializeControllers();
#else
// build network according to m_topolgoy
#endif
  }

  void Topology::Run(int time)
  {
    int cache_batch_num = m_cache_batch_start_index.size();
#if ENABLE_CACHE
    for (int i = 0; i < cache_batch_num; i++)
    {
      m_cache_controller.Run(i);
    }
#endif

    int node_batch_num = m_node_batch_start_index.size();
    for (int i = 0; i < node_batch_num; i++)
    {
      m_frame_decapsulation_controller.Run(i);
      m_ipv4_controller.Run(i);
      m_ipv4_decapsulation_controller.Run(i);
      m_tcp_controller.LaunchReceiveInstance(i);
      m_tcp_controller.LaunchSendInstance(i);
      m_ipv4_encapsulation_controller.Run(i);
      m_ipv4_controller.Run(i);
      m_frame_encapsulation_controller.Run(i);
    }

    int sw_batch_num = m_sw_batch_start_index.size();
    for (int i = 0; i < sw_batch_num; i++)
    {
      m_sw_controller.Run(i);
    }

    int ch_batch_num = m_ch_batch_start_index.size();
    for (int i = 0; i < ch_batch_num; i++)
    {
      m_p2p_controller.Run(i);
    }
    cudaStreamSynchronize(m_graph_stream);
    LaunchTimer(m_timeslot_start_gpu, m_timeslot_end_gpu, m_graph_stream);
    cudaStreamSynchronize(m_graph_stream);
  }

  void Topology::SetTransmissionPlanParams(int expected_packets_per_flow, double deviate, uint32_t seed)
  {
    m_expected_packets_per_flow = expected_packets_per_flow;
    // m_sigma = expected_packets_per_flow * deviate / 3;
    m_sigma = deviate;
    m_distribution_seed = seed;
  }

  void Topology::GenerateTransmissionPlan()
  {
    size_t node_num = m_nodes.size();
    // m_transmission_plan.resize(node_num * node_num);
    std::vector<int> tcp_con_num_per_node(node_num, 0);
    int planned_bytes = 1460 * 20000;
    // int planned_bytes[8] = {0, 1460 * 1000, 1460 * 15000, 1460 * 3000, 1460 * 0, 1460 * 0, 1460 * 0, 1460 * 10000};
    // int plan_len = 8;
    // std::vector<int> global_plan_bytes(node_num, 0);
    // for (size_t i = 0; i < node_num; i++)
    // {
    //   global_plan_bytes[i] = planned_bytes[i % plan_len];
    // }
    // int plan_index = 0;
    // std::sort(global_plan_bytes.begin(), global_plan_bytes.end(), [](const int a, const int b)
    //           { return a > b; });

    // for (size_t i = 0; i < node_num / 2; i++)
    // {
    //   m_transmission_plan[i] = std::make_pair(i + node_num / 2, global_plan_bytes[plan_index++]);
    //   // tcp_con_num_per_node[i]++;
    // }

    // for (size_t i = node_num / 2; i < node_num; i++)
    // {
    //   m_transmission_plan[i] = std::make_pair(i - node_num / 2, global_plan_bytes[plan_index++]);
    //   // tcp_con_num_per_node[i]++;
    // }

    // for (size_t i = 0; i < node_num / 2; i++)
    // {
    //   m_transmission_plan[i] = std::make_pair(i + node_num / 2, planned_bytes);
    //   tcp_con_num_per_node[i]++;
    // }

    // for (size_t i = node_num / 2; i < node_num; i++)
    // {
    //   m_transmission_plan[i] = std::make_pair(i - node_num / 2, planned_bytes);
    //   tcp_con_num_per_node[i]++;
    // }

    // for (size_t i = 0; i < node_num; i++)
    // {
    //   std::cout << "Node " << i << " flows: " << std::endl;
    //   std::cout << "start at: " << 0 << " packets: " << 20000 << std::endl;
    //   std::cout << std::endl;
    // }

    // for (auto &[src, plan] : m_transmission_plan)
    // {
    //   auto &[dst, bytes] = plan;
    //   std::cout << "src: " << src << " plan: " << bytes << std::endl;
    // }

    int planned_flow_sum = node_num * m_expected_packets_per_flow;
    std::vector<int64_t> traffic = generateTraffic(m_expected_packets_per_flow,
                                                   m_sigma,
                                                   node_num,
                                                   planned_flow_sum,
                                                   m_distribution_seed);
    int traffic_index = 0;

    // maintain available nodes.
    std::vector<int> available_nodes(node_num);
    for (int i = 0; i < node_num; ++i)
    {
      available_nodes[i] = i;
    }

    // random generator.
    std::mt19937 rng(m_distribution_seed);

    for (int i = 0; i < node_num; ++i)
    {
      if (tcp_con_num_per_node[i] == 0)
      {
        // randomly select a node from available nodes.
        std::uniform_int_distribution<int> dist(0, available_nodes.size() - 1);
        int random_index = dist(rng);
        int j = available_nodes[random_index];

        // make sure that j is not the same as i.
        while (j == i || tcp_con_num_per_node[j] != 0)
        {
          random_index = dist(rng);
          j = available_nodes[random_index];
        }

        // assign the traffic to the two nodes.
        m_transmission_plan.insert(std::make_pair(i, std::make_pair(j, traffic[traffic_index++] * 1460)));
        m_transmission_plan.insert(std::make_pair(j, std::make_pair(i, traffic[traffic_index++] * 1460)));
        tcp_con_num_per_node[i]++;
        tcp_con_num_per_node[j]++;

        // remove the selected nodes from available nodes.
        std::swap(available_nodes[random_index], available_nodes.back());
        available_nodes.pop_back();
      }
    }
  }

  void Topology::SetChannelParams(int tx_rate, int popogation_delay)
  {
    m_channel_tx_rate = tx_rate;
    m_channel_popogation_delay = popogation_delay;
  }

  void Topology::SetTimeslotInfo(int64_t timeslot_start, int64_t timeslot_end)
  {
    m_timeslot_start_cpu = timeslot_start;
    m_timeslot_end_cpu = timeslot_end;
    cudaMalloc(&m_timeslot_start_gpu, sizeof(int64_t));
    cudaMalloc(&m_timeslot_end_gpu, sizeof(int64_t));
    cudaMemcpy(m_timeslot_start_gpu, &m_timeslot_start_cpu, sizeof(int64_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(m_timeslot_end_gpu, &m_timeslot_end_cpu, sizeof(int64_t),
               cudaMemcpyHostToDevice);
  }

  void Topology::SetLookAheadTimeslot(int look_ahead_timeslot)
  {
    m_look_ahead_timeslot = look_ahead_timeslot;
  }
#if ENABLE_HUGE_GRAPH
  void Topology::BuildHugeGraph()
  {
    cudaGraph_t main_graph;
    cudaGraphCreate(&main_graph, 0);

    // frame decapsulation nodes
    int node_batch_num = m_node_batch_start_index.size();
    std::vector<cudaGraphNode_t> frame_decapsulation_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      cudaGraph_t frame_decapsulation_graph =
          m_frame_decapsulation_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph, NULL, 0,
                                 frame_decapsulation_graph);

      auto &memcpy_param = m_frame_decapsulation_controller.GetMemcpyParams();
      auto &host_param = m_frame_decapsulation_controller.GetHostParams();

      std::vector<cudaGraphNode_t> mempcy_nodes(2);
      cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph, &child_graph_node, 1,
                             &memcpy_param[0 + i * 2]);
      cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph, &child_graph_node, 1,
                             &memcpy_param[1 + i * 2]);

      cudaGraphNode_t host_node;
      cudaGraphAddHostNode(&host_node, main_graph, mempcy_nodes.data(),
                           mempcy_nodes.size(), &host_param[i]);

      frame_decapsulation_nodes.push_back(child_graph_node);
    }

    // ipv4 nodes
    std::vector<cudaGraphNode_t> ipv4_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      cudaGraph_t ipv4_graph = m_ipv4_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
                                 &frame_decapsulation_nodes[i], 1, ipv4_graph);
      ipv4_nodes.push_back(child_graph_node);
    }

    // ipv4 decapsulation nodes
    std::vector<cudaGraphNode_t> ipv4_decapsulation_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      cudaGraph_t ipv4_decapsulation_graph =
          m_ipv4_decapsulation_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph, &ipv4_nodes[i], 1,
                                 ipv4_decapsulation_graph);

      auto memcpy_param = m_ipv4_decapsulation_controller.GetMemcpyParams();
      auto host_param = m_ipv4_decapsulation_controller.GetHostParams();

      std::vector<cudaGraphNode_t> mempcy_nodes(2);
      cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph, &child_graph_node, 1,
                             &memcpy_param[0 + i * 2]);
      cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph, &child_graph_node, 1,
                             &memcpy_param[1 + i * 2]);

      cudaGraphNode_t host_node;
      cudaGraphAddHostNode(&host_node, main_graph, mempcy_nodes.data(),
                           mempcy_nodes.size(), &host_param[i]);

      ipv4_decapsulation_nodes.push_back(child_graph_node);
    }

    // tcp nodes
    std::vector<cudaGraphNode_t> tcp_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {

      std::vector<cudaGraphNode_t> send_mempcy_nodes(2);
      auto send_memcpy_param = m_tcp_controller.GetSendMemcpyParam();
      auto send_host_param = m_tcp_controller.GetSendHostParam();
      cudaGraphAddMemcpyNode(&send_mempcy_nodes[1], main_graph, NULL, 0, &send_memcpy_param[1 + i * 2]);

      cudaGraph_t receive_graph = m_tcp_controller.GetReceiveGraph(i);
      cudaGraphNode_t receive_node;
      std::vector<cudaGraphNode_t> receive_dependencies;
      // fixed bug here
      receive_dependencies.push_back(ipv4_decapsulation_nodes[i]);
      receive_dependencies.push_back(send_mempcy_nodes[1]);
      cudaGraphAddChildGraphNode(&receive_node, main_graph, receive_dependencies.data(), receive_dependencies.size(), receive_graph);

      auto memcpy_param = m_tcp_controller.GetReceiveMemcpyParam();
      auto host_param = m_tcp_controller.GetReceiveHostParam();

      std::vector<cudaGraphNode_t> receive_mempcy_nodes(2);
      cudaGraphAddMemcpyNode(&receive_mempcy_nodes[0], main_graph, &receive_node,
                             1, &memcpy_param[0 + i * 2]);
      cudaGraphAddMemcpyNode(&receive_mempcy_nodes[1], main_graph, &receive_node,
                             1, &memcpy_param[1 + i * 2]);
      cudaGraphNode_t receive_host_node;
      cudaGraphAddHostNode(&receive_host_node, main_graph,
                           receive_mempcy_nodes.data(),
                           receive_mempcy_nodes.size(), &host_param[i]);

      cudaGraph_t send_graph = m_tcp_controller.GetSendGraph(i);

      cudaGraphNode_t send_node;
      std::vector<cudaGraphNode_t> send_dependencies;
      send_dependencies.push_back(receive_node);
      send_dependencies.push_back(send_mempcy_nodes[1]);

      cudaGraphAddChildGraphNode(&send_node, main_graph, send_dependencies.data(),
                                 send_dependencies.size(), send_graph);

      cudaGraphAddMemcpyNode(&send_mempcy_nodes[0], main_graph, &send_node, 1,
                             &send_memcpy_param[0 + i * 2]);
      cudaGraphNode_t send_host_node;
      cudaGraphAddHostNode(&send_host_node, main_graph, &send_mempcy_nodes[0], 1,
                           &send_host_param[i]);

      tcp_nodes.push_back(send_node);
    }

    // ipv4 encapsulation nodes
    std::vector<cudaGraphNode_t> ipv4_encapsulation_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      auto memcpy_param = m_ipv4_encapsulation_controller.GetMemcpyParams();
      auto host_param = m_ipv4_encapsulation_controller.GetHostParams();
      std::vector<cudaGraphNode_t> mempcy_nodes(2);
      cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph, NULL, 0,
                             &memcpy_param[1 + i * 2]);

      cudaGraph_t ipv4_encapsulation_graph =
          m_ipv4_encapsulation_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      std::vector<cudaGraphNode_t> dependencies;
      dependencies.push_back(tcp_nodes[i]);
      dependencies.push_back(mempcy_nodes[1]);
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
                                 dependencies.data(), dependencies.size(),
                                 ipv4_encapsulation_graph);

      cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph, &child_graph_node, 1,
                             &memcpy_param[0 + i * 2]);
      cudaGraphNode_t host_node;
      cudaGraphAddHostNode(&host_node, main_graph, &mempcy_nodes[0], 1,
                           &host_param[i]);

      ipv4_encapsulation_nodes.push_back(child_graph_node);
    }

    // relaunch ipv4
    std::vector<cudaGraphNode_t> relaunch_ipv4_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      cudaGraph_t relaunch_ipv4_graph = m_ipv4_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
                                 &ipv4_encapsulation_nodes[i], 1,
                                 relaunch_ipv4_graph);
      relaunch_ipv4_nodes.push_back(child_graph_node);
    }

    // frame encapsulation nodes
    std::vector<cudaGraphNode_t> frame_encapsulation_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      std::vector<cudaGraphNode_t> mempcy_nodes(2);
      auto memcpy_param = m_frame_encapsulation_controller.GetMemcpyParams();
      auto host_param = m_frame_encapsulation_controller.GetHostParams();
      cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph, NULL, 0,
                             &memcpy_param[1 + i * 2]);

      cudaGraph_t frame_encapsulation_graph =
          m_frame_encapsulation_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      std::vector<cudaGraphNode_t> dependencies;
      dependencies.push_back(relaunch_ipv4_nodes[i]);
      dependencies.push_back(mempcy_nodes[1]);
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
                                 dependencies.data(), dependencies.size(),
                                 frame_encapsulation_graph);

      cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph, &child_graph_node, 1,
                             &memcpy_param[0 + i * 2]);
      cudaGraphNode_t host_node;
      cudaGraphAddHostNode(&host_node, main_graph, &mempcy_nodes[0], 1,
                           &host_param[i]);

      frame_encapsulation_nodes.push_back(child_graph_node);
    }

    // sw nodes
    int sw_batch_num = m_sw_batch_start_index.size();
    std::vector<cudaGraphNode_t> sw_nodes;
    for (int i = 0; i < sw_batch_num; i++)
    {
      cudaGraph_t sw_graph = m_sw_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph, NULL, 0,
                                 sw_graph);
      sw_nodes.push_back(child_graph_node);
    }

    // add p2p cuda graph nodes
    int p2p_batch_num = m_ch_batch_start_index.size();
    std::vector<cudaGraphNode_t> p2p_predecessor_nodes;
    p2p_predecessor_nodes.insert(p2p_predecessor_nodes.end(),
                                 frame_encapsulation_nodes.begin(),
                                 frame_encapsulation_nodes.end());
    p2p_predecessor_nodes.insert(p2p_predecessor_nodes.end(), sw_nodes.begin(),
                                 sw_nodes.end());

    std::vector<cudaGraphNode_t> p2p_nodes;
    for (int i = 0; i < p2p_batch_num; i++)
    {
      cudaGraph_t p2p_graph = m_p2p_controller.GetGraph(i);
      cudaGraphNode_t p2p_child_node;

      cudaGraphAddChildGraphNode(&p2p_child_node, main_graph,
                                 p2p_predecessor_nodes.data(),
                                 p2p_predecessor_nodes.size(), p2p_graph);
      p2p_nodes.push_back(p2p_child_node);
    }

    // add timer
    cudaGraph_t timer_graph =
        create_timer_graph(m_timeslot_start_gpu, m_timeslot_end_gpu);
    cudaGraphNode_t timer_node;
    cudaGraphAddChildGraphNode(&timer_node, main_graph, p2p_nodes.data(),
                               p2p_nodes.size(), timer_graph);

    m_graph = main_graph;
    cudaGraphInstantiate(&m_graph_exec, m_graph, 0);

    // exportGraph
  }
#endif

  void Topology::CreateFlows()
  {
    int node_num = m_nodes.size();

    for (int i = 0; i < node_num / 2; i++)
    {
      m_flows.emplace_back();
      for (int j = 0; j < node_num; j++)
      {
        m_flows[i].emplace_back();

        if (j >= node_num / 2)
        {
          for (int k = 0; k < 2; k++)
          {
            Flow flow;
            flow.timestamp = k * 10000000;
            flow.flow_size = 10000 * 1460;
          }
        }
      }
    }
  }

  void Topology::RunGraph()
  {
    cudaGraphLaunch(m_graph_exec, m_graph_stream);
    cudaStreamSynchronize(m_graph_stream);
  }

  void Topology::BuildGraphOnlyGPU()
  {
    cudaGraph_t main_graph;
    cudaGraphCreate(&main_graph, 0);
    auto recyle_and_alloc_graph = m_pool_controller.GetGraphs();

    // frame decapsulation nodes
    int node_batch_num = m_node_batch_start_index.size();
    std::vector<cudaGraphNode_t> frame_decapsulation_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      cudaGraph_t frame_decapsulation_graph = m_frame_decapsulation_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph, NULL, 0, frame_decapsulation_graph);
      frame_decapsulation_nodes.push_back(child_graph_node);
    }

    cudaGraphNode_t frame_recycle_node;
    cudaGraphAddChildGraphNode(&frame_recycle_node, main_graph, frame_decapsulation_nodes.data(), node_batch_num, recyle_and_alloc_graph[0]);

    // ipv4 nodes
    std::vector<cudaGraphNode_t> ipv4_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      cudaGraph_t ipv4_graph = m_ipv4_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
                                 &frame_decapsulation_nodes[i], 1, ipv4_graph);
      ipv4_nodes.push_back(child_graph_node);
    }

    // ipv4 decapsulation nodes
    std::vector<cudaGraphNode_t> ipv4_decapsulation_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      cudaGraph_t ipv4_decapsulation_graph =
          m_ipv4_decapsulation_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph, &ipv4_nodes[i], 1,
                                 ipv4_decapsulation_graph);
      ipv4_decapsulation_nodes.push_back(child_graph_node);
    }

    cudaGraphNode_t ipv4_recycle_node;
    cudaGraphAddChildGraphNode(&ipv4_recycle_node, main_graph, ipv4_decapsulation_nodes.data(), node_batch_num, recyle_and_alloc_graph[2]);

    // tcp nodes
    std::vector<cudaGraphNode_t> tcp_receive_nodes;
    std::vector<cudaGraphNode_t> tcp_send_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {

      cudaGraph_t receive_graph = m_tcp_controller.GetReceiveGraph(i);
      cudaGraphNode_t receive_node;

      cudaGraphAddChildGraphNode(&receive_node, main_graph, &ipv4_decapsulation_nodes[i], 1, receive_graph);
      tcp_receive_nodes.push_back(receive_node);

      cudaGraph_t send_graph = m_tcp_controller.GetSendGraph(i);

      cudaGraphNode_t send_node;
      cudaGraphAddChildGraphNode(&send_node, main_graph, &receive_node, 1, send_graph);
      tcp_send_nodes.push_back(send_node);
    }
    cudaGraphNode_t tcp_recycle_node;
    cudaGraphNode_t tcp_alloc_node;
    cudaGraphAddChildGraphNode(&tcp_recycle_node, main_graph, tcp_receive_nodes.data(), tcp_receive_nodes.size(), recyle_and_alloc_graph[4]);
    std::vector<cudaGraphNode_t> tcp_alloc_dependencies;
    tcp_alloc_dependencies.push_back(tcp_recycle_node);
    tcp_alloc_dependencies.insert(tcp_alloc_dependencies.end(), tcp_send_nodes.begin(), tcp_send_nodes.end());
    cudaGraphAddChildGraphNode(&tcp_alloc_node, main_graph, tcp_alloc_dependencies.data(), tcp_alloc_dependencies.size(), recyle_and_alloc_graph[5]);

    // ipv4 encapsulation nodes
    std::vector<cudaGraphNode_t> ipv4_encapsulation_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      cudaGraph_t ipv4_encapsulation_graph = m_ipv4_encapsulation_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph, &tcp_send_nodes[i], 1, ipv4_encapsulation_graph);
      ipv4_encapsulation_nodes.push_back(child_graph_node);
    }

    cudaGraphNode_t ipv4_alloc_node;
    std::vector<cudaGraphNode_t> ipv4_alloc_dependencies;
    ipv4_alloc_dependencies.push_back(ipv4_recycle_node);
    ipv4_alloc_dependencies.insert(ipv4_alloc_dependencies.end(), ipv4_encapsulation_nodes.begin(), ipv4_encapsulation_nodes.end());
    cudaGraphAddChildGraphNode(&ipv4_alloc_node, main_graph, ipv4_alloc_dependencies.data(), ipv4_alloc_dependencies.size(), recyle_and_alloc_graph[3]);

    // relaunch ipv4
    std::vector<cudaGraphNode_t> relaunch_ipv4_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      cudaGraph_t relaunch_ipv4_graph = m_ipv4_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph, &ipv4_encapsulation_nodes[i], 1, relaunch_ipv4_graph);
      relaunch_ipv4_nodes.push_back(child_graph_node);
    }

    // frame encapsulation nodes
    std::vector<cudaGraphNode_t> frame_encapsulation_nodes;
    for (int i = 0; i < node_batch_num; i++)
    {
      cudaGraph_t frame_encapsulation_graph = m_frame_encapsulation_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph, &relaunch_ipv4_nodes[i], 1, frame_encapsulation_graph);
      frame_encapsulation_nodes.push_back(child_graph_node);
    }
    cudaGraphNode_t frame_alloc_node;
    std::vector<cudaGraphNode_t> frame_alloc_dependencies;
    frame_alloc_dependencies.push_back(frame_recycle_node);
    frame_alloc_dependencies.insert(frame_alloc_dependencies.end(), frame_encapsulation_nodes.begin(), frame_encapsulation_nodes.end());
    cudaGraphAddChildGraphNode(&frame_alloc_node, main_graph, frame_alloc_dependencies.data(), frame_alloc_dependencies.size(), recyle_and_alloc_graph[1]);

    // sw nodes
    int sw_batch_num = m_sw_batch_start_index.size();
    std::vector<cudaGraphNode_t> sw_nodes;
    for (int i = 0; i < sw_batch_num; i++)
    {
      cudaGraph_t sw_graph = m_sw_controller.GetGraph(i);
      cudaGraphNode_t child_graph_node;
      cudaGraphAddChildGraphNode(&child_graph_node, main_graph, NULL, 0,
                                 sw_graph);
      sw_nodes.push_back(child_graph_node);
    }

    // add p2p cuda graph nodes
    int p2p_batch_num = m_ch_batch_start_index.size();
    std::vector<cudaGraphNode_t> p2p_predecessor_nodes;
    p2p_predecessor_nodes.insert(p2p_predecessor_nodes.end(),
                                 frame_encapsulation_nodes.begin(),
                                 frame_encapsulation_nodes.end());
    p2p_predecessor_nodes.insert(p2p_predecessor_nodes.end(), sw_nodes.begin(),
                                 sw_nodes.end());

    std::vector<cudaGraphNode_t> p2p_nodes;
    for (int i = 0; i < p2p_batch_num; i++)
    {
      cudaGraph_t p2p_graph = m_p2p_controller.GetGraph(i);
      cudaGraphNode_t p2p_child_node;

      cudaGraphAddChildGraphNode(&p2p_child_node, main_graph,
                                 p2p_predecessor_nodes.data(),
                                 p2p_predecessor_nodes.size(), p2p_graph);
      p2p_nodes.push_back(p2p_child_node);
    }

    // add timer
    // cudaGraph_t timer_graph =
    // create_timer_graph(m_timeslot_start_gpu, m_timeslot_end_gpu);
    cudaGraph_t timer_graph = m_timer_comtroller.GetGraph();
    cudaGraphNode_t timer_node;
    cudaGraphAddChildGraphNode(&timer_node, main_graph, p2p_nodes.data(),
                               p2p_nodes.size(), timer_graph);

    m_graph = main_graph;
    cudaGraphInstantiate(&m_graph_exec, m_graph, 0);
  }

  void Topology::CheckReceivedPackets()
  {
    TCPConnection *tcp_con = new TCPConnection;
    for (int i = 0; i < m_nodes.size(); i++)
    {
      for (int j = 0; j < MAX_TCP_CONNECTION_NUM; j++)
      {
        if (m_tcp_connections[i * MAX_TCP_CONNECTION_NUM + j] == nullptr)
        {
          continue;
        }
        cudaMemcpy(tcp_con, m_tcp_connections[i * MAX_TCP_CONNECTION_NUM + j], sizeof(TCPConnection), cudaMemcpyDeviceToHost);
        if (tcp_con != nullptr && tcp_con->planned_bytes != tcp_con->acked_bytes && tcp_con->planned_bytes / 1460 - tcp_con->acked_bytes / 1460 >= 5)
        {
          std::cout << "Node " << i << " TCP connection " << j << " received " << tcp_con->acked_bytes << " bytes," << tcp_con->acked_bytes / 1460 << " packets"
                    << ", but expected " << tcp_con->planned_bytes << " bytes. " << tcp_con->planned_bytes / 1460 << " packets." << std::endl;
        }
      }
    }
  }

  bool Topology::IsFinished()
  {
    return m_timer_comtroller.IsFinished();
  }

  int biased_random_choice(const int64_t total_traffic[], int size, double lambda = 0.8)
  {
    // Step 1: 计算指数衰减权重
    std::vector<double> weights(size);
    double total_weight = 0.0;
    for (int i = 0; i < size; ++i)
    {
      weights[i] = std::exp(-lambda * i); // 越后面越小
      total_weight += weights[i];
    }

    // Step 2: 归一化为概率分布
    for (int i = 0; i < size; ++i)
    {
      weights[i] /= total_weight;
    }

    // Step 3: 随机采样
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    double r = dist(gen);

    double cumulative = 0.0;
    for (int i = 0; i < size; ++i)
    {
      cumulative += weights[i];
      if (r < cumulative)
      {
        return i; // 选中索引 i
      }
    }
    return size - 1; // 防止精度误差
  }

  // void Topology::CreateFlows(int64_t start_timestamp)
  // {
  //   auto poison_inter = [](int64_t avg) -> int64_t
  //   {
  //     static std::random_device rd;
  //     static std::mt19937 gen(rd());
  //     static std::uniform_real_distribution<> dis(0.0, 1.0);
  //     return -std::log(1.0 - dis(gen)) * avg;
  //   };
  //   size_t node_num = m_nodes.size();
  //   int64_t gaps[3] = {5000, 10000, 50000};
  //   size_t traffic_pattern = 3;
  //   int64_t total_traffic[6] = {100, 500, 800, 1500, 2000, 6000};
  //   int parition[6] = {5, 2, 4, 3, 5, 3};
  //   size_t flow_index = 0;
  //   for (size_t i = 0; i < node_num / 2; i++)
  //   {
  //     int64_t cur_time = start_timestamp + i * 5000;
  //     auto *node = m_nodes[i];
  //     auto *node_reverse = m_nodes[i + node_num / 2];
  //     flows_map_.insert(std::make_pair(i, std::vector<Flow>()));
  //     flows_map_.insert(std::make_pair(i + node_num / 2, std::vector<Flow>()));
  //     auto &flow_vec = flows_map_[i];
  //     auto &flow_vec_reverse = flows_map_[i + node_num / 2];
  //     flow_index = biased_random_choice(total_traffic, 6);
  //     for (size_t traffic = 0; traffic < parition[flow_index]; traffic++)
  //     {
  //       Flow flow;
  //       flow.flow_size = 1460 * (total_traffic[flow_index] / parition[flow_index]);
  //       flow.timestamp = cur_time;
  //       flow_vec.emplace_back(flow);
  //       flow_vec_reverse.emplace_back(flow);
  //       /**
  //        * @brief: 80 was given by 1 / (bandwidth / 8.0 / 1000) * 1000000000
  //        */
  //       // cur_time += poison_inter(500);
  //       cur_time += gaps[traffic % traffic_pattern];
  //       m_flow_start_instants.emplace_back(flow.timestamp);
  //       m_flow_start_instants.emplace_back(flow.timestamp);
  //     }
  //     auto *tcp_protocol = GetTCPProtocol(node);
  //     // We assume that the TCP connection is already established and there is only 1 connection per node.
  //     GPUQueue<Flow> flow_queue;
  //     flow_queue.size = flow_vec.size();
  //     flow_queue.queue_capacity = 100;
  //     cudaMalloc(&flow_queue.queue, sizeof(Flow) * 100);
  //     cudaMemcpy(flow_queue.queue, flow_vec.data(), sizeof(Flow) * flow_vec.size(), cudaMemcpyHostToDevice);

  //     // We assume only send to 1 node.
  //     cudaMemcpy(tcp_protocol->tcp_cons[0]->flows, &flow_queue, sizeof(GPUQueue<Flow>), cudaMemcpyHostToDevice);

  //     auto *tcp_protocol_reverse = GetTCPProtocol(node_reverse);
  //     GPUQueue<Flow> flow_queue_reverse;
  //     flow_queue_reverse.size = flow_vec_reverse.size();
  //     flow_queue_reverse.queue_capacity = 100;
  //     cudaMalloc(&flow_queue_reverse.queue, sizeof(Flow) * 100);
  //     cudaMemcpy(flow_queue_reverse.queue, flow_vec_reverse.data(), sizeof(Flow) * flow_vec_reverse.size(), cudaMemcpyHostToDevice);
  //     cudaMemcpy(tcp_protocol_reverse->tcp_cons[0]->flows, &flow_queue_reverse, sizeof(GPUQueue<Flow>), cudaMemcpyHostToDevice);

  //     // flow_index = (flow_index + 1) % 6;
  //   }

  class CustomRand
  {
  public:
    bool setCdf(const std::vector<std::pair<double, double>> &input_cdf)
    {
      cdf = input_cdf;
      return !cdf.empty();
    }

    double rand()
    {
      double r = uniform_dist(rng);
      for (const auto &p : cdf)
      {
        if (r <= p.second)
          return p.first;
      }
      return cdf.back().first;
    }

    double getAvg()
    {
      double sum = 0.0;
      for (size_t i = 1; i < cdf.size(); ++i)
      {
        sum += (cdf[i].first + cdf[i - 1].first) / 2 * (cdf[i].second - cdf[i - 1].second);
      }
      return sum;
    }

  private:
    std::vector<std::pair<double, double>> cdf;
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> uniform_dist{0.0, 1.0};
  };

  class ParetoRand
  {
  public:
    bool setParameters(double alpha, int xm)
    {
      if (alpha <= 1 || xm <= 0)
        return false;
      this->alpha = alpha;
      this->xm = xm;
      return true;
    }

    double rand()
    {
      double u = uniform_dist(rng);
      return xm / pow(u, 1.0 / alpha);
    }

    double getAvg()
    {
      return (alpha * xm) / (alpha - 1);
    }

  private:
    double alpha;
    int xm;
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> uniform_dist{0.0, 1.0};
  };

  int poisson(double mean)
  {
    static std::default_random_engine gen(std::random_device{}());
    std::poisson_distribution<int> dist(mean);
    return dist(gen);
  }

  void Topology::CreateFlows(int64_t start_timestamp)
  {
    auto poison_inter = [](int64_t avg) -> int64_t
    {
      static std::random_device rd;
      static std::mt19937 gen(rd());
      static std::uniform_real_distribution<> dis(0.0, 1.0);
      return -std::log(1.0 - dis(gen)) * avg;
    };
    size_t node_num = m_nodes.size();
    // // int64_t gaps[3] = {5000, 10000, 50000};
    // size_t traffic_pattern = 3;
    // // int64_t total_traffic[6] = {100, 500, 800, 1500, 2000, 6000};
    // int64_t total_traffic[6] = {1000, 2000, 5000, 7000, 10000, 20000};
    // // int parition[6] = {5, 2, 4, 3, 5, 2};
    // int samples[6] = {2, 1, 2, 1, 2, 1};
    // size_t flow_index = 0;
    // for (size_t i = 0; i < node_num / 2; i++)
    // {
    //   int64_t cur_time = start_timestamp + i * 5000;
    //   auto *node = m_nodes[i];
    //   auto *node_reverse = m_nodes[i + node_num / 2];
    //   flows_map_.insert(std::make_pair(i, std::vector<Flow>()));
    //   flows_map_.insert(std::make_pair(i + node_num / 2, std::vector<Flow>()));
    //   auto &flow_vec = flows_map_[i];
    //   auto &flow_vec_reverse = flows_map_[i + node_num / 2];
    //   flow_index = biased_random_choice(total_traffic, 6);
    //   for (size_t traffic = 0; traffic < samples[flow_index]; traffic++)
    //   {
    //     Flow flow;
    //     flow.flow_size = 1460 * (total_traffic[flow_index]);
    //     flow.timestamp = cur_time;
    //     flow_vec.emplace_back(flow);
    //     flow_vec_reverse.emplace_back(flow);
    //     /**
    //      * @brief: 80 was given by 1 / (bandwidth / 8.0 / 1000) * 1000000000
    //      */
    //     cur_time += poison_inter(500);
    //     // cur_time += gaps[traffic % traffic_pattern];
    //     m_flow_start_instants.emplace_back(flow.timestamp);
    //     m_flow_start_instants.emplace_back(flow.timestamp);
    //   }
    //   auto *tcp_protocol = GetTCPProtocol(node);
    //   // We assume that the TCP connection is already established and there is only 1 connection per node.
    //   GPUQueue<Flow> flow_queue;
    //   flow_queue.size = flow_vec.size();
    //   flow_queue.queue_capacity = 100;
    //   cudaMalloc(&flow_queue.queue, sizeof(Flow) * 100);
    //   cudaMemcpy(flow_queue.queue, flow_vec.data(), sizeof(Flow) * flow_vec.size(), cudaMemcpyHostToDevice);

    //   // We assume only send to 1 node.
    //   cudaMemcpy(tcp_protocol->tcp_cons[0]->flows, &flow_queue, sizeof(GPUQueue<Flow>), cudaMemcpyHostToDevice);

    //   auto *tcp_protocol_reverse = GetTCPProtocol(node_reverse);
    //   GPUQueue<Flow> flow_queue_reverse;
    //   flow_queue_reverse.size = flow_vec_reverse.size();
    //   flow_queue_reverse.queue_capacity = 100;
    //   cudaMalloc(&flow_queue_reverse.queue, sizeof(Flow) * 100);
    //   cudaMemcpy(flow_queue_reverse.queue, flow_vec_reverse.data(), sizeof(Flow) * flow_vec_reverse.size(), cudaMemcpyHostToDevice);
    //   cudaMemcpy(tcp_protocol_reverse->tcp_cons[0]->flows, &flow_queue_reverse, sizeof(GPUQueue<Flow>), cudaMemcpyHostToDevice);

    //   // flow_index = (flow_index + 1) % 6;
    // }

    int planned_flow_sum = node_num * m_expected_packets_per_flow;
    // std::vector<int64_t> traffic = generateTraffic(m_expected_packets_per_flow,
    //                                                m_sigma,
    //                                                node_num,
    //                                                planned_flow_sum,
    //                                                m_distribution_seed);

    // std::vector<int64_t> traffic = generateAvgTraffic(m_expected_packets_per_flow,
    //                                                   m_sigma,
    //                                                   node_num,
    //                                                   planned_flow_sum,
    //                                                   m_distribution_seed);

    // std::vector<int64_t> traffic(node_num, m_expected_packets_per_flow);

    // std::vector<int64_t> traffic = generatePoissonTraffic(
    //     m_sigma, node_num, planned_flow_sum, m_distribution_seed);

    std::vector<int64_t> traffic(node_num, 0);
    for (int i = 0; i < node_num; i++)
    {
      traffic[i] = std::min(poison_inter(m_expected_packets_per_flow) + 1, (int64_t)m_expected_packets_per_flow * 2);
      // traffic[i] = m_expected_packets_per_flow;
    }

    // maintain available nodes.
    std::vector<int> available_nodes(node_num);
    for (int i = 0; i < node_num; ++i)
    {
      available_nodes[i] = i;
    }

    // random generator.
    std::mt19937 rng(m_distribution_seed);

    int64_t cur_time = 0;
    std::vector<int> tcp_con_num_per_node(node_num, 0);
    int flow_index = 0;
    for (int i = 0; i < node_num; ++i)
    {
      if (tcp_con_num_per_node[i] == 0)
      {
        // randomly select a node from available nodes.
        std::uniform_int_distribution<int> dist(0, available_nodes.size() - 1);
        int random_index = dist(rng);
        // int j = available_nodes[random_index];
        int j = i + node_num / 2;

        // make sure that j is not the same as i.
        while (j == i || tcp_con_num_per_node[j] != 0)
        {
          random_index = dist(rng);
          j = available_nodes[random_index];
        }

        auto *node1 = m_nodes[i];
        auto *node2 = m_nodes[j];
        flows_map_.insert(std::make_pair(i, std::vector<Flow>()));
        flows_map_.insert(std::make_pair(j, std::vector<Flow>()));
        auto &flow_vec = flows_map_[i];
        auto &flow_vec_reverse = flows_map_[j];

        Flow flow1;
        flow1.flow_size = 1460 * (traffic[i]);
        // flow1.flow_size = 1460 * (i % 32 + 1) * 400;
        flow1.timestamp = poison_inter(flow_time_range);
        // flow1.timestamp = 0;
        flow1.tiimestamp_end = -1;
        flow_vec.emplace_back(flow1);
        /**
         * @brief: 80 was given by 1 / (bandwidth / 8.0 / 1000) * 1000000000
         */

        // cur_time += poison_inter(300);
        // cur_time += 1000;

        Flow flow2;
        flow2.flow_size = 1460 * (traffic[j]);
        // flow2.flow_size = 1460 * (j % 32 + 1) * 400;
        flow2.timestamp = poison_inter(flow_time_range);
        // flow2.timestamp = 0;
        flow2.tiimestamp_end = -1;
        flow_vec_reverse.emplace_back(flow2);

        // cur_time += poison_inter(300);
        // cur_time += 1000;

        // cur_time += gaps[traffic % traffic_pattern];
        m_flow_start_instants.emplace_back(flow1.timestamp);
        m_flow_start_instants.emplace_back(flow2.timestamp);

        // assign the traffic to the two nodes.
        // m_transmission_plan.insert(std::make_pair(i, std::make_pair(j, traffic[traffic_index++] * 1460)));
        // m_transmission_plan.insert(std::make_pair(j, std::make_pair(i, traffic[traffic_index++] * 1460)));
        tcp_con_num_per_node[i]++;
        tcp_con_num_per_node[j]++;
        auto *tcp_protocol = GetTCPProtocol(node1);
        // We assume that the TCP connection is already established and there is only 1 connection per node.
        GPUQueue<Flow> flow_queue;
        flow_queue.head = 0;
        flow_queue.size = flow_vec.size();
        flow_queue.queue_capacity = 100;
        cudaMalloc(&flow_queue.queue, sizeof(Flow) * 100);
        cudaMemcpy(flow_queue.queue, flow_vec.data(), sizeof(Flow) * flow_vec.size(), cudaMemcpyHostToDevice);

        // We assume only send to 1 node.
        cudaMemcpy(tcp_protocol->tcp_cons[0]->flows, &flow_queue, sizeof(GPUQueue<Flow>), cudaMemcpyHostToDevice);

        auto *tcp_protocol_reverse = GetTCPProtocol(node2);
        GPUQueue<Flow> flow_queue_reverse;
        flow_queue_reverse.head = 0;
        flow_queue_reverse.size = flow_vec_reverse.size();
        flow_queue_reverse.queue_capacity = 100;
        cudaMalloc(&flow_queue_reverse.queue, sizeof(Flow) * 100);
        cudaMemcpy(flow_queue_reverse.queue, flow_vec_reverse.data(), sizeof(Flow) * flow_vec_reverse.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(tcp_protocol_reverse->tcp_cons[0]->flows, &flow_queue_reverse, sizeof(GPUQueue<Flow>), cudaMemcpyHostToDevice);

        // remove the selected nodes from available nodes.
        std::swap(available_nodes[random_index], available_nodes.back());
        available_nodes.pop_back();
      }
    }

    /**
     * @deprecated: print the flow information.
     */
    // for (auto &[node, flows] : flows_map_)
    // {
    //   std::cout << "Node " << node << " flows: " << std::endl;
    //   for (auto &flow : flows)
    //   {
    //     std::cout << "start at: " << flow.timestamp << " packets: " << flow.flow_size / 1460 << std::endl;
    //   }
    //   std::cout << std::endl;
    // }
  }

  std::vector<int64_t> generatePoissonTraffic(double flow_lambda, int num_samples, int64_t total_sum, unsigned int seed)
  {
    flow_lambda *= 20000;
    std::vector<int64_t> traffic;
    std::mt19937 gen(seed); // 固定种子
    std::poisson_distribution<int64_t> dist(flow_lambda);

    for (int i = 0; i < num_samples; ++i)
    {
      int64_t value = dist(gen);
      traffic.push_back(value);
    }

    // // 计算生成流量的总和
    // double sum = std::accumulate(traffic.begin(), traffic.end(), 0.0);

    // // 计算归一化因子
    // double factor = total_sum / sum;

    // 归一化每个值
    // for (auto &value : traffic)
    // {
    //   value = static_cast<int64_t>(round(value * factor));
    // }

    return traffic;
  }

  std::vector<int64_t> generateAvgTraffic(double mu, double sigma, int num_samples, double total_sum, unsigned int seed)
  {
    std::vector<int64_t> traffic;

    std::mt19937 gen(seed);

    // 计算均匀采样的范围
    double lower = mu * (1.0 - sigma);
    double upper = mu * (1.0 + sigma);

    std::uniform_real_distribution<double> dist(lower, upper);

    // 生成数据，四舍五入取整
    for (int i = 0; i < num_samples; ++i)
    {
      double sampled = dist(gen);
      int64_t value = static_cast<int64_t>(round(sampled));
      traffic.push_back(value);
    }

    // 计算原始总和
    double sum = std::accumulate(traffic.begin(), traffic.end(), 0.0);

    // 归一化因子
    double factor = total_sum / sum;

    // // 调整每个值使总和为 total_sum
    // for (auto &value : traffic)
    // {
    //   value = static_cast<int64_t>(round(value * factor));
    // }

    return traffic;
  }

  std::vector<int64_t> generateTraffic(double mu, double sigma, int num_samples, double total_sum, unsigned int seed)
  {
    std::vector<int64_t> traffic;
    // use fixed seed for reproducibility.
    std::mt19937 gen(seed);
    std::normal_distribution<> dist(mu, sigma);

    // generate traffic and adjust to integer
    for (int i = 0; i < num_samples; ++i)
    {
      int64_t rounded_value = -1;

      // Ensure the generated value is non-negative
      while (rounded_value < 0)
      {
        double raw_value = dist(gen);
        rounded_value = static_cast<int64_t>(round(raw_value));
      }

      traffic.push_back(rounded_value);
    }

    // calculate the actual sum of traffic.
    double sum = std::accumulate(traffic.begin(), traffic.end(), 0.0);

    // calculate the factor to adjust the traffic to the total sum.
    double factor = total_sum / sum;

    // adjust each traffic flow to adhere to the total sum.
    for (auto &value : traffic)
    {
      value = static_cast<int>(round(value * factor));
    }

    return traffic;
  }

  void Topology::RecordFlowResults(std::string file_name)
  {
    m_tcp_controller.RecordFlowResults(file_name);

    // for (int i = 0; i < m_nodes.size(); i++)
    // {
    //   auto tcp = GetTCPProtocol(m_nodes[i]);
    //   std::cout << i << "," << tcp->tcp_cons[0]->src_ip << std::endl;
    // }
  }

} // namespace VDES

// namespace VDES
// {
//     void HostNodeCallback(void *user_data)
//     {
//         auto func = static_cast<std::function<void()> *>(user_data);
//         (*func)();
//     }

//     Topology::Topology()
//     {
//     }

//     Topology::~Topology()
//     {
//     }

//     void Topology::SetNodes(Node **nodes, int node_num)
//     {
//         m_nodes.insert(m_nodes.end(), nodes, nodes + node_num);
//     }

//     void Topology::SetSwitches(Switch **switches, int sw_num)
//     {
//         m_switches.insert(m_switches.end(), switches, switches + sw_num);
//     }

//     void Topology::SetFattreeProperties(int k, uint32_t ip_group_size,
//     uint32_t ip_base_addr, uint32_t ip_mask)
//     {
//         m_ft_k = k;
//         m_ip_group_size = ip_group_size;
//         m_ip_base_addr = ip_base_addr;
//         m_ip_mask = ip_mask;
//     }

//     void Topology::InitializeFtNodeAndSwitch()
//     {
//         int node_num = m_ft_k * m_ft_k * m_ft_k / 4;
//         int sw_num = m_ft_k * m_ft_k + m_ft_k * m_ft_k / 4;
//         int half_of_ft_k_sq = m_ft_k * m_ft_k / 2;

//         for (uint32_t i = 0; i < node_num; i++)
//         {
//             Node *node = CreateNode(1, NODE_DEFAULT_INGRESS_QUEUE_SIZE,
//             NODE_DEFAULT_EGRESS_QUEUE_SIZE); node->node_id = i;
//             memcpy(node->nics[0]->mac_addr, &i, 4);
//             m_nodes.push_back(node);
//         }

//         for (int i = 0; i < sw_num; i++)
//         {
//             Switch *sw = CreateSwitch(m_ft_k,
//             Switch_DEFAULT_INGRESS_QUEUE_SIZE,
//             Switch_DEFAULT_EGRESS_QUEUE_SIZE); m_switches.push_back(sw);
//         }

//         int sw_id = 0;
//         for (int i = 0; i < m_ft_k; i++)
//         {
//             for (int j = 0; j < m_ft_k / 2; j++)
//             {
//                 m_switches[sw_id]->sw_id = i * m_ft_k + j;
//                 m_switches[sw_id + half_of_ft_k_sq]->sw_id = i * m_ft_k + j +
//                 m_ft_k / 2; sw_id++;
//             }
//             /**
//              * @TODO: CHANGE THE WAY SW_ID.
//              */
//             // sw_id = i * m_ft_k;
//         }

//         for (int i = m_ft_k * m_ft_k; i < sw_num; i++)
//         {
//             m_switches[i]->sw_id = i;
//         }
//     }

//     void Topology::InstallIPv4ProtocolForAllNodes()
//     {
//         for (int i = 0; i < m_nodes.size(); i++)
//         {
//             InstallIPv4Protocol(m_nodes[i]);
//         }
//     }

//     void Topology::InstallTCPProtocolForAllNodes()
//     {
//         for (int i = 0; i < m_nodes.size(); i++)
//         {
//             InstallTCPProtocol(m_nodes[i]);
//         }
//     }

//     void Topology::AllocateIPAddrForAllNodes()
//     {
//         int group_num = m_ft_k * m_ft_k / 2;
//         for (int i = 0; i < group_num; i++)
//         {
//             AllocateIPAddr(m_nodes.data() + i * m_ft_k / 2, m_ft_k / 2,
//             m_ip_base_addr + i * m_ip_group_size, m_ip_mask);
//         }
//     }

//     void Topology::BuildTCPConnections()
//     {
//         int node_num = m_nodes.size();
//         for (int i = 0; i < node_num; i++)
//         {
//             for (int j = i; j < node_num; j++)
//             {
//                 if (m_transmission_plan[i * node_num + j] != 0 ||
//                 m_transmission_plan[j * node_num + i] != 0)
//                 {
//                     ConnectTCPConnection(m_nodes[i], m_transmission_plan[i *
//                     node_num + j], 0, m_nodes[j], m_transmission_plan[j *
//                     node_num + i], 0);
//                 }
//             }
//         }
//     }

//     void Topology::BuildIPv4RoutingTable()
//     {

//         for (int i = 0; i < m_nodes.size(); i++)
//         {
//             auto ipv4 = GetIPv4Protocol(m_nodes[i]);
//             GPUQueue<IPv4RoutingRule *> routing_table;
//             cudaMemcpy(&routing_table, ipv4->routing_table,
//             sizeof(GPUQueue<IPv4RoutingRule *>), cudaMemcpyDeviceToHost);

//             int nic_num = ipv4->nic_num;
//             for (int j = 0; j < nic_num; j++)
//             {
//                 IPv4RoutingRule rule_cpu;
//                 IPv4RoutingRule *rule_gpu;
//                 cudaMalloc(&rule_gpu, sizeof(IPv4RoutingRule));
//                 rule_cpu.dst = ipv4->ipv4_interfaces[j].ip;
//                 /**
//                  * TODO: UPDATE MASK
//                  */
//                 rule_cpu.mask = -1;
//                 rule_cpu.gw = 0;
//                 cudaMemcpy(rule_gpu, &rule_cpu, sizeof(IPv4RoutingRule),
//                 cudaMemcpyHostToDevice); cudaMemcpy(routing_table.queue + j,
//                 &rule_gpu, sizeof(IPv4RoutingRule *),
//                 cudaMemcpyHostToDevice);
//             }

//             // add default gateway
//             IPv4RoutingRule rule_cpu;
//             IPv4RoutingRule *rule_gpu;
//             cudaMalloc(&rule_gpu, sizeof(IPv4RoutingRule));
//             rule_cpu.dst = 0;
//             rule_cpu.mask = 0;
//             // next hop is the dst_ip
//             rule_cpu.gw = 1;
//             /**
//              * TODO: ADD INTERFACE ID.
//              */
//             rule_cpu.if_id = 0;
//             cudaMemcpy(rule_gpu, &rule_cpu, sizeof(IPv4RoutingRule),
//             cudaMemcpyHostToDevice); cudaMemcpy(routing_table.queue +
//             nic_num, &rule_gpu, sizeof(IPv4RoutingRule *),
//             cudaMemcpyHostToDevice); routing_table.size = nic_num + 1;
//             // cudaMemcpy(&routing_table.size, &size, sizeof(int),
//             cudaMemcpyHostToDevice); cudaMemcpy(ipv4->routing_table,
//             &routing_table, sizeof(GPUQueue<IPv4RoutingRule>),
//             cudaMemcpyHostToDevice);
//         }
//     }

//     void Topology::IntiializeControllers()
//     {
//         m_p2p_controller.SetIngressAndEgress(m_ch_frame_ingress_queues.data(),
//         m_ch_frame_egress_queues.data(), m_ch_popogation_delay.data(),
//         m_ch_tx_rate.data(), m_channels.size() * 2);
//         m_p2p_controller.SetStreams(m_ch_streams.data(),
//         m_ch_streams.size());
//         m_p2p_controller.SetBatchProperties(m_ch_batch_start_index.data(),
//         m_ch_batch_end_index.data(), m_ch_batch_start_index.size());
//         m_p2p_controller.SetTimeslot(m_timeslot_start_gpu,
//         m_timeslot_end_gpu); m_p2p_controller.InitializeKernelParams();
//         m_p2p_controller.BuildGraph();

//         m_frame_encapsulation_controller.SetStreams(m_node_streams.data(),
//         m_node_streams.size());
//         m_frame_encapsulation_controller.SetFrameProperties(m_node_frame_egress_queues.data(),
//         m_node_nic_mac.data(), m_nic_num_per_node.data(), m_nodes.size());
//         m_frame_encapsulation_controller.SetFatTreeArpProperties(m_ft_k,
//         m_ip_base_addr, m_ip_group_size);
//         m_frame_encapsulation_controller.SetPacketProperties((GPUQueue<void
//         *> **)m_node_ipv4_egress_queues.data(), m_nodes.size());
//         m_frame_encapsulation_controller.SetBatchProperties(m_node_batch_start_index.data(),
//         m_node_batch_end_index.data(), m_node_batch_start_index.size());
//         m_frame_encapsulation_controller.InitializeKernelParams();
//         m_frame_encapsulation_controller.BuildGraph();

//         m_frame_decapsulation_controller.SetStreams(m_node_streams.data(),
//         m_node_streams.size());
//         m_frame_decapsulation_controller.SetFrameIngress(m_node_frame_egress_queues.data(),
//         m_node_frame_egress_queues.size());
//         m_frame_decapsulation_controller.SetPacketIngress((GPUQueue<void *>
//         **)m_node_ipv4_egress_queues.data(),
//         m_node_ipv4_egress_queues.size());
//         m_frame_decapsulation_controller.SetNodeProperties(m_nic_num_per_node.data(),
//         m_nodes.size());
//         m_frame_decapsulation_controller.SetBatchProperties(m_node_batch_start_index.data(),
//         m_node_batch_end_index.data(), m_node_batch_start_index.size());
//         m_frame_decapsulation_controller.InitializeKernelParams();
//         m_frame_decapsulation_controller.BuildGraphs();

//         m_ipv4_controller.SetEgressQueues(m_node_ipv4_egress_queues.data(),
//         m_nic_num_per_node.data(), m_nodes.size());
//         m_ipv4_controller.SetIngressQueues(m_node_ipv4_ingress_queues.data(),
//         m_node_ipv4_ingress_queues.size());
//         m_ipv4_controller.SetErrorQueues(m_node_ipv4_error_queues.data(),
//         m_node_ipv4_error_queues.size());
//         m_ipv4_controller.SetLocalEgressQueues(m_node_ipv4_local_delivery_queues.data(),
//         m_node_ipv4_local_delivery_queues.size());
//         m_ipv4_controller.SetRoutingTables((GPUQueue<IPv4RoutingRule *>
//         **)m_node_ipv4_routing_table_queues.data(),
//         m_node_ipv4_routing_table_queues.size());
//         m_ipv4_controller.SetStreams(m_node_streams.data(),
//         m_node_streams.size());
//         m_ipv4_controller.SetBatchProperties(m_node_batch_start_index.data(),
//         m_node_batch_end_index.data(), m_node_batch_start_index.size());
//         m_ipv4_controller.SetEgressRemainingCapacity(m_node_ipv4_egress_remaing_capacity.data(),
//         m_node_ipv4_egress_remaing_capacity.size());
//         m_ipv4_controller.InitializeKernelParams();
//         m_ipv4_controller.BuildGraphs();

//         m_ipv4_decapsulation_controller.SetStreams(m_node_streams.data(),
//         m_node_streams.size());
//         m_ipv4_decapsulation_controller.SetIPv4Queues(m_node_ipv4_local_delivery_queues.data(),
//         m_nodes.size());
//         m_ipv4_decapsulation_controller.SetL4Queues((GPUQueue<uint8_t *>
//         **)m_node_tcp_ingress_queues.data(), m_nodes.size());
//         m_ipv4_decapsulation_controller.SetNICNum(m_nic_num_per_node.data(),
//         m_nodes.size());
//         m_ipv4_decapsulation_controller.SetBatchProperties(m_node_batch_start_index.data(),
//         m_node_batch_end_index.data(), m_node_batch_start_index.size());
//         m_ipv4_decapsulation_controller.InitalizeKernelParams();
//         m_ipv4_decapsulation_controller.BuildGraph();

//         m_ipv4_encapsulation_controller.SetStreams(m_node_streams.data(),
//         m_node_streams.size());
//         m_ipv4_encapsulation_controller.SetIPv4PacketQueue(m_node_ipv4_egress_queues.data(),
//         m_nodes.size());
//         m_ipv4_encapsulation_controller.SetL4PacketQueue((GPUQueue<uint8_t *>
//         **)m_node_tcp_egress_queues.data(), m_nodes.size());
//         m_ipv4_encapsulation_controller.SetNICNumPerNode(m_nic_num_per_node.data(),
//         m_nodes.size());
//         m_ipv4_encapsulation_controller.SetBatchProperties(m_node_batch_start_index.data(),
//         m_node_batch_end_index.data(), m_node_batch_start_index.size());
//         m_ipv4_encapsulation_controller.InitalizeKernelParams();
//         m_ipv4_encapsulation_controller.BuildGraph();

//         m_tcp_controller.SetStreams(m_node_streams.data(),
//         m_node_streams.size());
//         m_tcp_controller.SetNicNumPerNode(m_nic_num_per_node.data(),
//         m_nodes.size());
//         m_tcp_controller.SetBatchProperties(m_node_batch_start_index.data(),
//         m_node_batch_end_index.data(), m_node_batch_start_index.size());
//         m_tcp_controller.SetRecvQueues(m_node_tcp_ingress_queues.data(),
//         m_nodes.size());
//         m_tcp_controller.SetSendQueues(m_node_tcp_egress_queues.data(),
//         m_node_tcp_egress_queues.size());
//         m_tcp_controller.SetTCPConnections(m_tcp_connections.data(),
//         m_tcp_num_per_node.data(), m_nodes.size());
//         m_tcp_controller.SetRemainingCacheSizeArray(m_node_ipv4_egress_remaing_capacity.data(),
//         m_node_ipv4_egress_remaing_capacity.size());
//         m_tcp_controller.SetTimeslotInfo(m_timeslot_start_gpu,
//         m_timeslot_end_gpu); m_tcp_controller.InitKernelParams();
//         m_tcp_controller.BuildGraph();

//         m_sw_controller.SetStreams(m_sw_streams.data(), m_sw_streams.size());
//         m_sw_controller.SetFtProperties(m_ft_k);
//         m_sw_controller.SetIngresAndEgress(m_sw_frame_ingress_queues.data(),
//         m_sw_frame_egress_queues.data(), m_nic_num_per_sw.data(),
//         m_sw_id_per_node.data(), m_switches.size());
//         m_sw_controller.SetBatchproperties(m_sw_batch_start_index.data(),
//         m_sw_batch_end_index.data(), m_sw_batch_start_index.size());
//         m_sw_controller.InitalizeKernelParams();
//         m_sw_controller.BuildGraph();

// #if ENABLE_CACHE
//         m_cache_controller.SetStreams(m_cache_streams.data(),
//         m_cache_streams.size());
//         m_cache_controller.SetEgressProperties(m_cache_frame_queues.data(),
//         m_cache_frame_status.data(), m_cache_frame_num_per_node.data(),
//         m_nodes.size() + m_switches.size());
//         m_cache_controller.SetLookaheadTimeslotNum(1);
//         m_cache_controller.SetBatches(m_cache_batch_start_index.data(),
//         m_cache_batch_end_index.data(), m_cache_batch_start_index.size());
//         m_cache_controller.SetTimeSlot(m_timeslot_start_gpu,
//         m_timeslot_end_gpu); m_cache_controller.InitializeKernelParams();
//         m_cache_controller.BuildGraph();
// #endif
//     }

//     void Topology::ExractParamsForController()
//     {
//         for (int i = 0; i < m_nodes.size(); i++)
//         {
//             int nic_num = m_nodes[i]->nics.size();
//             auto nics = m_nodes[i]->nics;
//             auto ipv4 = GetIPv4Protocol(m_nodes[i]);
//             auto tcp = GetTCPProtocol(m_nodes[i]);
//             for (int j = 0; j < nic_num; j++)
//             {
//                 m_node_frame_ingress_queues.push_back(nics[j]->ingress);
//                 m_node_frame_egress_queues.push_back(nics[j]->egress);
//                 m_node_nic_mac.push_back(nics[j]->mac_addr);
//                 m_cache_frame_queues.push_back(nics[j]->egress);
//             }
//             m_nic_num_per_node.push_back(nic_num);
//             m_node_ipv4_ingress_queues.push_back(ipv4->ingress);
//             m_node_ipv4_egress_queues.insert(m_node_ipv4_egress_queues.end(),
//             ipv4->egresses, ipv4->egresses + nic_num);
//             m_node_ipv4_local_delivery_queues.push_back(ipv4->local_delivery);
//             m_node_ipv4_error_queues.push_back(ipv4->error_queue);
//             m_node_ipv4_routing_table_queues.push_back(ipv4->routing_table);
//             m_node_tcp_egress_queues.push_back(tcp->egress);
//             m_node_tcp_ingress_queues.push_back(tcp->ingress);
//             m_cache_frame_num_per_node.push_back(nic_num);

//             TCPConnection **tcp_cons;
//             for (int j = 0; j < MAX_TCP_CONNECTION_NUM; j++)
//             {
//                 if (j < tcp->tcp_cons_num)
//                 {
//                     TCPConnection *tcp_con;
//                     cudaMalloc(&tcp_con, sizeof(TCPConnection));
//                     cudaMemcpy(tcp_con, tcp->tcp_cons[j],
//                     sizeof(TCPConnection), cudaMemcpyHostToDevice);
//                     m_tcp_connections.push_back(tcp_con);
//                 }
//                 else
//                 {
//                     m_tcp_connections.push_back(NULL);
//                 }
//             }

//             m_tcp_num_per_node.push_back(tcp->tcp_cons_num);
//         }

//         for (int i = 0; i < m_switches.size(); i++)
//         {
//             int port_num = m_switches[i]->port_num;
//             auto nics = m_switches[i]->nics;
//             for (int j = 0; j < port_num; j++)
//             {
//                 m_sw_frame_egress_queues.push_back(nics[j]->egress);
//                 m_sw_frame_ingress_queues.push_back(nics[j]->ingress);
//                 m_cache_frame_queues.push_back(nics[j]->egress);
//             }
//             // m_nic_num_per_node.push_back(port_num);
//             m_nic_num_per_sw.push_back(port_num);
//             m_sw_id_per_node.push_back(m_switches[i]->sw_id);
//             m_cache_frame_num_per_node.push_back(port_num);
//         }

//         for (int i = 0; i < m_cache_frame_queues.size(); i++)
//         {
//             FrameQueueStatus *status_gpu;
//             cudaMalloc(&status_gpu, sizeof(FrameQueueStatus));
//             FrameQueueStatus status_cpu;
//             memset(&status_cpu, 0, sizeof(FrameQueueStatus));
//             cudaMalloc(&status_cpu.packet_status_in_cache_win,
//             sizeof(uint8_t) * MAX_TRANSMITTED_PACKET_NUM *
//             (m_look_ahead_timeslot * 2 + 2)); cudaMemcpy(status_gpu,
//             &status_cpu, sizeof(FrameQueueStatus), cudaMemcpyHostToDevice);
//             m_cache_frame_status.push_back(status_gpu);
//         }

//         // for (int i = 0; i < m_channels.size(); i++)
//         // {
//         // m_ch_frame_ingress_queues.push_back(m_channels[i]->nic1->egress);
//         // m_ch_frame_ingress_queues.push_back(m_channels[i]->nic2->egress);
//         // m_ch_frame_egress_queues.push_back(m_channels[i]->nic2->ingress);
//         // m_ch_frame_egress_queues.push_back(m_channels[i]->nic1->ingress);

//         //     m_ch_tx_rate.push_back(m_channels[i]->tx_rate);
//         //     m_ch_tx_rate.push_back(m_channels[i]->tx_rate);
//         // m_ch_popogation_delay.push_back(m_channels[i]->popogation_delay);
//         // m_ch_popogation_delay.push_back(m_channels[i]->popogation_delay);
//         // }

//         for (int i = 0; i < m_nodes.size(); i++)
//         {
//             m_ch_frame_ingress_queues.push_back(m_channels[i]->nic2->egress);
//             m_ch_frame_egress_queues.push_back(m_channels[i]->nic1->ingress);
//             m_ch_tx_rate.push_back(m_channels[i]->tx_rate);
//             m_ch_popogation_delay.push_back(m_channels[i]->popogation_delay);
//         }

//         for (int i = 0; i < m_nodes.size(); i++)
//         {
//             m_ch_frame_ingress_queues.push_back(m_channels[i]->nic1->egress);
//             m_ch_frame_egress_queues.push_back(m_channels[i]->nic2->ingress);
//             m_ch_tx_rate.push_back(m_channels[i]->tx_rate);
//             m_ch_popogation_delay.push_back(m_channels[i]->popogation_delay);
//         }

//         int node_ch_num = m_ft_k * m_ft_k * m_ft_k / 4;
//         for (int i = 0; i < m_ft_k * m_ft_k * m_ft_k / 4; i++)
//         {
//             m_ch_frame_ingress_queues.push_back(m_channels[node_ch_num +
//             i]->nic1->egress);
//             m_ch_frame_egress_queues.push_back(m_channels[node_ch_num +
//             i]->nic2->ingress); m_ch_tx_rate.push_back(m_channels[node_ch_num
//             + i]->tx_rate);
//             m_ch_popogation_delay.push_back(m_channels[node_ch_num +
//             i]->popogation_delay);
//         }

//         for (int i = 0; i < m_ft_k * m_ft_k * m_ft_k / 4; i++)
//         {
//             m_ch_frame_ingress_queues.push_back(m_channels[node_ch_num +
//             i]->nic2->egress);
//             m_ch_frame_egress_queues.push_back(m_channels[node_ch_num +
//             i]->nic1->ingress); m_ch_tx_rate.push_back(m_channels[node_ch_num
//             + i]->tx_rate);
//             m_ch_popogation_delay.push_back(m_channels[node_ch_num +
//             i]->popogation_delay);
//         }

//         int core_sw_ch_num = m_ft_k * m_ft_k * m_ft_k / 2;
//         for (int i = 0; i < m_ft_k * m_ft_k * m_ft_k / 4; i++)
//         {
//             m_ch_frame_ingress_queues.push_back(m_channels[core_sw_ch_num +
//             i]->nic1->egress);
//             m_ch_frame_egress_queues.push_back(m_channels[core_sw_ch_num +
//             i]->nic2->ingress);
//             m_ch_tx_rate.push_back(m_channels[core_sw_ch_num + i]->tx_rate);
//             m_ch_popogation_delay.push_back(m_channels[core_sw_ch_num +
//             i]->popogation_delay);
//         }

//         for (int i = 0; i < m_ft_k * m_ft_k * m_ft_k / 4; i++)
//         {
//             m_ch_frame_ingress_queues.push_back(m_channels[core_sw_ch_num +
//             i]->nic2->egress);
//             m_ch_frame_egress_queues.push_back(m_channels[core_sw_ch_num +
//             i]->nic1->ingress);
//             m_ch_tx_rate.push_back(m_channels[core_sw_ch_num + i]->tx_rate);
//             m_ch_popogation_delay.push_back(m_channels[core_sw_ch_num +
//             i]->popogation_delay);
//         }
//     }

//     void Topology::GenerateBatches()
//     {
//         m_node_batch_start_index.push_back(0);
//         m_node_batch_end_index.push_back(m_nodes.size());

//         int node_batch_num = m_node_batch_start_index.size();
//         for (int i = 0; i < node_batch_num; i++)
//         {
//             int *remaining_capacity;
//             cudaMalloc(&remaining_capacity, sizeof(int) *
//             (m_node_batch_end_index[i] - m_node_batch_start_index[i]));
//             m_node_ipv4_egress_remaing_capacity.push_back(remaining_capacity);
//             cudaStream_t stream;
//             cudaStreamCreate(&stream);
//             m_node_streams.push_back(stream);
//         }

//         // access layer
//         m_sw_batch_start_index.push_back(0);
//         m_sw_batch_end_index.push_back(m_ft_k * m_ft_k / 2);

//         // aggregation layer
//         m_sw_batch_start_index.push_back(m_ft_k * m_ft_k / 2);
//         m_sw_batch_end_index.push_back(m_ft_k * m_ft_k);

//         // core layer
//         m_sw_batch_start_index.push_back(m_ft_k * m_ft_k);
//         m_sw_batch_end_index.push_back(m_ft_k * m_ft_k + m_ft_k * m_ft_k /
//         4);

//         for (int i = 0; i < m_sw_batch_start_index.size(); i++)
//         {
//             cudaStream_t stream;
//             cudaStreamCreate(&stream);
//             m_sw_streams.push_back(stream);
//         }

//         // channel
//         // int ft_cube = m_ft_k * m_ft_k * m_ft_k;
//         // m_ch_batch_start_index.push_back(0);
//         // m_ch_batch_end_index.push_back(ft_cube / 4);
//         // m_ch_batch_start_index.push_back(ft_cube / 4);
//         // m_ch_batch_end_index.push_back(ft_cube * 3 / 4);
//         // m_ch_batch_start_index.push_back(ft_cube * 3 / 4);
//         // m_ch_batch_end_index.push_back(ft_cube * 5 / 4);
//         // m_ch_batch_start_index.push_back(ft_cube * 5 / 4);
//         // m_ch_batch_end_index.push_back(ft_cube * 6 / 4);

//         m_ch_batch_start_index.push_back(0);
//         m_ch_batch_end_index.push_back(m_ch_frame_ingress_queues.size());

//         for (int i = 0; i < m_ch_batch_start_index.size(); i++)
//         {
//             cudaStream_t stream;
//             cudaStreamCreate(&stream);
//             m_ch_streams.push_back(stream);
//         }

//         // cache
//         m_cache_batch_start_index.push_back(0);
//         m_cache_batch_end_index.push_back(m_cache_frame_num_per_node.size());
//         for (int i = 0; i < m_cache_batch_start_index.size(); i++)
//         {
//             cudaStream_t stream;
//             cudaStreamCreate(&stream);
//             m_cache_streams.push_back(stream);
//         }
//     }

//     void Topology::BuildTopology()
//     {
//         if (ENABLE_FATTREE_MODE)
//         {
//             // build fat-tree topology
//             InitializeFtNodeAndSwitch();

//             // networking between nodes and switches
//             int access_sw_num = m_ft_k * m_ft_k / 2;
//             for (int i = 0; i < access_sw_num; i++)
//             {
//                 for (int j = 0; j < m_ft_k / 2; j++)
//                 {
//                     P2PChanenl *ch = CreateP2PChannel(m_channel_tx_rate,
//                     m_channel_popogation_delay); ConnectDevices(m_nodes[i *
//                     m_ft_k / 2 + j], 0, m_switches[i], j, ch);
//                     m_channels.push_back(ch);
//                 }
//             }

//             for (int i = 0; i < m_ft_k; i++)
//             {
//                 for (int j = 0; j < m_ft_k / 2; j++)
//                 {
//                     for (int k = 0; k < m_ft_k / 2; k++)
//                     {
//                         P2PChanenl *ch = CreateP2PChannel(m_channel_tx_rate,
//                         m_channel_popogation_delay); bool is_success =
//                         ConnectDevices(m_switches[access_sw_num + i * m_ft_k
//                         / 2 + j], k, m_switches[i * m_ft_k / 2 + k], j +
//                         m_ft_k / 2, ch); if (!is_success)
//                         {
//                             std::cout << "ConnectDevices failed!" << i << j
//                             << k << std::endl; exit(1);
//                         }
//                         m_channels.push_back(ch);
//                     }
//                 }
//             }

//             int core_sw_offset = m_ft_k * m_ft_k;
//             int core_sw_num = m_ft_k * m_ft_k / 4;
//             for (int i = 0; i < m_ft_k / 2; i++)
//             {
//                 for (int j = 0; j < m_ft_k / 2; j++)
//                 {
//                     for (int k = 0; k < m_ft_k; k++)
//                     {
//                         P2PChanenl *ch = CreateP2PChannel(m_channel_tx_rate,
//                         m_channel_popogation_delay); bool is_success =
//                         ConnectDevices(m_switches[core_sw_offset + i * m_ft_k
//                         / 2 + j], k, m_switches[access_sw_num + k * m_ft_k /
//                         2 + i], j + m_ft_k / 2, ch); if (!is_success)
//                         {
//                             std::cout << "ConnectDevices failed!" << i << j
//                             << k << std::endl; exit(1);
//                         }
//                         m_channels.push_back(ch);
//                     }
//                 }
//             }

//             // intall protocols
//             GenerateTransmissionPlan();
//             InstallIPv4ProtocolForAllNodes();
//             InstallTCPProtocolForAllNodes();
//             AllocateIPAddrForAllNodes();
//             BuildTCPConnections();
//             BuildIPv4RoutingTable();
//             ExractParamsForController();
//             GenerateBatches();
//             IntiializeControllers();
//         }
//         else
//         {
//             // build network according to m_topolgoy
//         }
//     }

//     void Topology::Run()
//     {
//         int cache_batch_num = m_cache_batch_start_index.size();
// #if ENABLE_CACHE
//         for (int i = 0; i < cache_batch_num; i++)
//         {
//             // m_cache_controller.BackupFrameQueueInfo(i);
//             m_cache_controller.Run(i);
//         }
// #endif

//         int ch_batch_num = m_ch_batch_start_index.size();
//         for (int i = 0; i < ch_batch_num; i++)
//         {
//             m_p2p_controller.Run(i);
//         }

//         int node_batch_num = m_node_batch_start_index.size();
//         for (int i = 0; i < node_batch_num; i++)
//         {
//             m_frame_decapsulation_controller.Run(i);
//             m_ipv4_controller.Run(i);
//             m_ipv4_decapsulation_controller.Run(i);
//             m_tcp_controller.LaunchReceiveInstance(i);
//             m_tcp_controller.LaunchSendInstance(i);
//             m_ipv4_encapsulation_controller.Run(i);
//             m_ipv4_controller.Run(i);
//             m_frame_encapsulation_controller.Run(i);
//         }

//         int sw_batch_num = m_sw_batch_start_index.size();
//         for (int i = 0; i < sw_batch_num; i++)
//         {
//             m_sw_controller.Run(i);
//         }
//     }

//     void Topology::GenerateTransmissionPlan()
//     {
//         int node_num = m_nodes.size();
//         m_transmission_plan.resize(node_num * node_num);
//         std::vector<int> tcp_con_num_per_node(node_num, 0);
//         int planned_bytes = 1460 * 10000;

//         for (int i = 0; i < node_num; i++)
//         {
//             if (tcp_con_num_per_node[i] == 0)
//             {
//                 for (int j = i + 1; j < node_num; j++)
//                 {
//                     if (tcp_con_num_per_node[j] == 0)
//                     {
//                         m_transmission_plan[i * node_num + j] =
//                         planned_bytes; m_transmission_plan[j * node_num + i]
//                         = planned_bytes; tcp_con_num_per_node[i]++;
//                         tcp_con_num_per_node[j]++;
//                         break;
//                     }
//                 }
//             }
//         }
//     }

//     void Topology::SetChannelParams(int tx_rate, int popogation_delay)
//     {
//         m_channel_tx_rate = tx_rate;
//         m_channel_popogation_delay = popogation_delay;
//     }

//     /**
//      * TODO: Set and Initialize timeslot info.
//      */
//     void Topology::SetTimeslotInfo(int64_t timeslot_start, int64_t
//     timeslot_end)
//     {
//         m_timeslot_start_cpu = timeslot_start;
//         m_timeslot_end_cpu = timeslot_end;
//         cudaMalloc(&m_timeslot_start_gpu, sizeof(int64_t));
//         cudaMalloc(&m_timeslot_end_gpu, sizeof(int64_t));
//         cudaMemcpy(m_timeslot_start_gpu, &m_timeslot_start_cpu,
//         sizeof(int64_t), cudaMemcpyHostToDevice);
//         cudaMemcpy(m_timeslot_end_gpu, &m_timeslot_end_cpu, sizeof(int64_t),
//         cudaMemcpyHostToDevice);
//     }

//     /**
//      * @TODO: Set and Initialize look ahead timeslot.
//      */
//     void Topology::SetLookAheadTimeslot(int look_ahead_timeslot)
//     {
//         m_look_ahead_timeslot = look_ahead_timeslot;
//     }

// #if ENABLE_HUGE_GRAPH
//     void Topology::BuildHugeGraph()
//     {
//         cudaGraph_t main_graph;
//         cudaGraphCreate(&main_graph, 0);

//         // frame decapsulation nodes
//         int node_batch_num = m_node_batch_start_index.size();
//         std::vector<cudaGraphNode_t> frame_decapsulation_nodes;
//         for (int i = 0; i < node_batch_num; i++)
//         {
//             cudaGraph_t frame_decapsulation_graph =
//             m_frame_decapsulation_controller.GetGraph(i); cudaGraphNode_t
//             child_graph_node; cudaGraphAddChildGraphNode(&child_graph_node,
//             main_graph, NULL, 0, frame_decapsulation_graph);

//             auto &memcpy_param =
//             m_frame_decapsulation_controller.GetMemcpyParams(); auto
//             &host_param = m_frame_decapsulation_controller.GetHostParams();

//             std::vector<cudaGraphNode_t> mempcy_nodes(2);
//             cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph,
//             &child_graph_node, 1, &memcpy_param[0 + i * 2]);
//             cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph,
//             &child_graph_node, 1, &memcpy_param[1 + i * 2]);

//             cudaGraphNode_t host_node;
//             cudaGraphAddHostNode(&host_node, main_graph, mempcy_nodes.data(),
//             mempcy_nodes.size(), &host_param[i]);

//             frame_decapsulation_nodes.push_back(child_graph_node);
//         }

//         // ipv4 nodes
//         std::vector<cudaGraphNode_t> ipv4_nodes;
//         for (int i = 0; i < node_batch_num; i++)
//         {
//             cudaGraph_t ipv4_graph = m_ipv4_controller.GetGraph(i);
//             cudaGraphNode_t child_graph_node;
//             cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
//             &frame_decapsulation_nodes[i], 1, ipv4_graph);
//             ipv4_nodes.push_back(child_graph_node);
//         }

//         // ipv4 decapsulation nodes
//         std::vector<cudaGraphNode_t> ipv4_decapsulation_nodes;
//         for (int i = 0; i < node_batch_num; i++)
//         {
//             cudaGraph_t ipv4_decapsulation_graph =
//             m_ipv4_decapsulation_controller.GetGraph(i); cudaGraphNode_t
//             child_graph_node; cudaGraphAddChildGraphNode(&child_graph_node,
//             main_graph, &ipv4_nodes[i], 1, ipv4_decapsulation_graph);

//             auto memcpy_param =
//             m_ipv4_decapsulation_controller.GetMemcpyParams(); auto
//             host_param = m_ipv4_decapsulation_controller.GetHostParams();

//             std::vector<cudaGraphNode_t> mempcy_nodes(2);
//             cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph,
//             &child_graph_node, 1, &memcpy_param[0 + i * 2]);
//             cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph,
//             &child_graph_node, 1, &memcpy_param[1 + i * 2]);

//             cudaGraphNode_t host_node;
//             cudaGraphAddHostNode(&host_node, main_graph, mempcy_nodes.data(),
//             mempcy_nodes.size(), &host_param[i]);

//             ipv4_decapsulation_nodes.push_back(child_graph_node);
//         }

//         // tcp nodes
//         std::vector<cudaGraphNode_t> tcp_nodes;
//         for (int i = 0; i < node_batch_num; i++)
//         {

//             std::vector<cudaGraphNode_t> send_mempcy_nodes(2);
//             auto send_memcpy_param = m_tcp_controller.GetSendMemcpyParam();
//             auto send_host_param = m_tcp_controller.GetSendHostParam();
//             cudaGraphAddMemcpyNode(&send_mempcy_nodes[1], main_graph, NULL,
//             0, &send_memcpy_param[1 + i * 2]);

//             cudaGraph_t receive_graph = m_tcp_controller.GetReceiveGraph(i);
//             cudaGraphNode_t receive_node;
//             std::vector<cudaGraphNode_t> receive_dependencies;
//             // fixed bug here
//             receive_dependencies.push_back(ipv4_decapsulation_nodes[i]);
//             receive_dependencies.push_back(send_mempcy_nodes[1]);
//             cudaGraphAddChildGraphNode(&receive_node, main_graph,
//             receive_dependencies.data(), receive_dependencies.size(),
//             receive_graph);

//             auto memcpy_param = m_tcp_controller.GetReceiveMemcpyParam();
//             auto host_param = m_tcp_controller.GetReceiveHostParam();

//             std::vector<cudaGraphNode_t> receive_mempcy_nodes(2);
//             cudaGraphAddMemcpyNode(&receive_mempcy_nodes[0], main_graph,
//             &receive_node, 1, &memcpy_param[0 + i * 2]);
//             cudaGraphAddMemcpyNode(&receive_mempcy_nodes[1], main_graph,
//             &receive_node, 1, &memcpy_param[1 + i * 2]); cudaGraphNode_t
//             receive_host_node; cudaGraphAddHostNode(&receive_host_node,
//             main_graph, receive_mempcy_nodes.data(),
//             receive_mempcy_nodes.size(), &host_param[i]);

//             cudaGraph_t send_graph = m_tcp_controller.GetSendGraph(i);

//             cudaGraphNode_t send_node;
//             std::vector<cudaGraphNode_t> send_dependencies;
//             send_dependencies.push_back(receive_node);
//             send_dependencies.push_back(send_mempcy_nodes[1]);

//             cudaGraphAddChildGraphNode(&send_node, main_graph,
//             send_dependencies.data(), send_dependencies.size(), send_graph);

//             cudaGraphAddMemcpyNode(&send_mempcy_nodes[0], main_graph,
//             &send_node, 1, &send_memcpy_param[0 + i * 2]); cudaGraphNode_t
//             send_host_node; cudaGraphAddHostNode(&send_host_node, main_graph,
//             &send_mempcy_nodes[0], 1, &send_host_param[i]);

//             tcp_nodes.push_back(send_node);
//         }

//         // ipv4 encapsulation nodes
//         std::vector<cudaGraphNode_t> ipv4_encapsulation_nodes;
//         for (int i = 0; i < node_batch_num; i++)
//         {
//             auto memcpy_param =
//             m_ipv4_encapsulation_controller.GetMemcpyParams(); auto
//             host_param = m_ipv4_encapsulation_controller.GetHostParams();
//             std::vector<cudaGraphNode_t> mempcy_nodes(2);
//             cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph, NULL, 0,
//             &memcpy_param[1 + i * 2]);

//             cudaGraph_t ipv4_encapsulation_graph =
//             m_ipv4_encapsulation_controller.GetGraph(i); cudaGraphNode_t
//             child_graph_node; std::vector<cudaGraphNode_t> dependencies;
//             dependencies.push_back(tcp_nodes[i]);
//             dependencies.push_back(mempcy_nodes[1]);
//             cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
//             dependencies.data(), dependencies.size(),
//             ipv4_encapsulation_graph);

//             cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph,
//             &child_graph_node, 1, &memcpy_param[0 + i * 2]); cudaGraphNode_t
//             host_node; cudaGraphAddHostNode(&host_node, main_graph,
//             &mempcy_nodes[0], 1, &host_param[i]);

//             ipv4_encapsulation_nodes.push_back(child_graph_node);
//         }

//         // relaunch ipv4
//         std::vector<cudaGraphNode_t> relaunch_ipv4_nodes;
//         for (int i = 0; i < node_batch_num; i++)
//         {
//             cudaGraph_t relaunch_ipv4_graph = m_ipv4_controller.GetGraph(i);
//             cudaGraphNode_t child_graph_node;
//             cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
//             &ipv4_encapsulation_nodes[i], 1, relaunch_ipv4_graph);
//             relaunch_ipv4_nodes.push_back(child_graph_node);
//         }

//         // frame encapsulation nodes
//         std::vector<cudaGraphNode_t> frame_encapsulation_nodes;
//         for (int i = 0; i < node_batch_num; i++)
//         {
//             std::vector<cudaGraphNode_t> mempcy_nodes(2);
//             auto memcpy_param =
//             m_frame_encapsulation_controller.GetMemcpyParams(); auto
//             host_param = m_frame_encapsulation_controller.GetHostParams();
//             cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph, NULL, 0,
//             &memcpy_param[1 + i * 2]);

//             cudaGraph_t frame_encapsulation_graph =
//             m_frame_encapsulation_controller.GetGraph(i); cudaGraphNode_t
//             child_graph_node; std::vector<cudaGraphNode_t> dependencies;
//             dependencies.push_back(relaunch_ipv4_nodes[i]);
//             dependencies.push_back(mempcy_nodes[1]);
//             cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
//             dependencies.data(), dependencies.size(),
//             frame_encapsulation_graph);

//             cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph,
//             &child_graph_node, 1, &memcpy_param[0 + i * 2]); cudaGraphNode_t
//             host_node; cudaGraphAddHostNode(&host_node, main_graph,
//             &mempcy_nodes[0], 1, &host_param[i]);

//             frame_encapsulation_nodes.push_back(child_graph_node);
//         }

//         // sw nodes
//         int sw_batch_num = m_sw_batch_start_index.size();
//         std::vector<cudaGraphNode_t> sw_nodes;
//         for (int i = 0; i < sw_batch_num; i++)
//         {
//             cudaGraph_t sw_graph = m_sw_controller.GetGraph(i);
//             cudaGraphNode_t child_graph_node;
//             cudaGraphAddChildGraphNode(&child_graph_node, main_graph, NULL,
//             0, sw_graph); sw_nodes.push_back(child_graph_node);
//         }

//         // add p2p cuda graph nodes
//         int p2p_batch_num = m_ch_batch_start_index.size();
//         std::vector<cudaGraphNode_t> p2p_predecessor_nodes;
//         p2p_predecessor_nodes.insert(p2p_predecessor_nodes.end(),
//         frame_encapsulation_nodes.begin(), frame_encapsulation_nodes.end());
//         p2p_predecessor_nodes.insert(p2p_predecessor_nodes.end(),
//         sw_nodes.begin(), sw_nodes.end());

//         std::vector<cudaGraphNode_t> p2p_nodes;
//         for (int i = 0; i < p2p_batch_num; i++)
//         {
//             cudaGraph_t p2p_graph = m_p2p_controller.GetGraph(i);
//             cudaGraphNode_t p2p_child_node;

//             cudaGraphAddChildGraphNode(&p2p_child_node, main_graph,
//             p2p_predecessor_nodes.data(), p2p_predecessor_nodes.size(),
//             p2p_graph); p2p_nodes.push_back(p2p_child_node);
//         }

//         // add timer
//         cudaGraph_t timer_graph = create_timer_graph(m_timeslot_start_gpu,
//         m_timeslot_end_gpu); cudaGraphNode_t timer_node;
//         cudaGraphAddChildGraphNode(&timer_node, main_graph, p2p_nodes.data(),
//         p2p_nodes.size(), timer_graph);

//         m_graph = main_graph;
//         cudaGraphInstantiate(&m_graph_exec, m_graph, 0);

//         // exportGraph
//     }
//     // void Topology::BuildHugeGraph()
//     // {
//     //     cudaGraph_t main_graph;
//     //     cudaGraphCreate(&main_graph, 0);

//     //     // frame decapsulation nodes
//     //     int node_batch_num = m_node_batch_start_index.size();
//     //     std::vector<cudaGraphNode_t> frame_decapsulation_nodes;
//     //     for (int i = 0; i < node_batch_num; i++)
//     //     {
//     //         cudaGraph_t frame_decapsulation_graph =
//     m_frame_decapsulation_controller.GetGraph(i);
//     //         cudaGraphNode_t child_graph_node;
//     //         cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
//     NULL, 0, frame_decapsulation_graph);

//     //         auto &memcpy_param =
//     m_frame_decapsulation_controller.GetMemcpyParams();
//     //         auto &host_param =
//     m_frame_decapsulation_controller.GetHostParams();

//     //         std::vector<cudaGraphNode_t> mempcy_nodes(2);
//     //         cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph,
//     &child_graph_node, 1, &memcpy_param[0 + i * 2]);
//     //         cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph,
//     &child_graph_node, 1, &memcpy_param[1 + i * 2]);

//     //         cudaGraphNode_t host_node;
//     //         cudaGraphAddHostNode(&host_node, main_graph, &mempcy_nodes[0],
//     2, &host_param[i]);

//     //         frame_decapsulation_nodes.push_back(child_graph_node);
//     //     }

//     //     // ipv4 nodes
//     //     std::vector<cudaGraphNode_t> ipv4_nodes;
//     //     for (int i = 0; i < node_batch_num; i++)
//     //     {
//     //         cudaGraph_t ipv4_graph = m_ipv4_controller.GetGraph(i);
//     //         cudaGraphNode_t child_graph_node;
//     //         cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
//     &frame_decapsulation_nodes[i], 1, ipv4_graph);
//     //         ipv4_nodes.push_back(child_graph_node);
//     //     }

//     //     // ipv4 decapsulation nodes
//     //     std::vector<cudaGraphNode_t> ipv4_decapsulation_nodes;
//     //     for (int i = 0; i < node_batch_num; i++)
//     //     {
//     //         cudaGraph_t ipv4_decapsulation_graph =
//     m_ipv4_decapsulation_controller.GetGraph(i);
//     //         cudaGraphNode_t child_graph_node;
//     //         cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
//     &ipv4_nodes[i], 1, ipv4_decapsulation_graph);

//     //         auto memcpy_param =
//     m_ipv4_decapsulation_controller.GetMemcpyParams();
//     //         auto host_param =
//     m_ipv4_decapsulation_controller.GetHostParams();

//     //         std::vector<cudaGraphNode_t> mempcy_nodes(2);
//     //         cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph,
//     &child_graph_node, 1, &memcpy_param[0 + i * 2]);
//     //         cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph,
//     &child_graph_node, 1, &memcpy_param[1 + i * 2]);

//     //         cudaGraphNode_t host_node;
//     //         cudaGraphAddHostNode(&host_node, main_graph, &mempcy_nodes[0],
//     2, &host_param[i]);

//     //         ipv4_decapsulation_nodes.push_back(child_graph_node);
//     //     }

//     //     // tcp nodes
//     //     std::vector<cudaGraphNode_t> tcp_nodes;
//     //     for (int i = 0; i < node_batch_num; i++)
//     //     {
//     //         cudaGraph_t receive_graph =
//     m_tcp_controller.GetReceiveGraph(i);
//     //         cudaGraphNode_t receive_node;
//     //         cudaGraphAddChildGraphNode(&receive_node, main_graph,
//     &ipv4_decapsulation_nodes[i], 1, receive_graph);

//     //         auto memcpy_param = m_tcp_controller.GetReceiveMemcpyParam();
//     //         auto host_param = m_tcp_controller.GetReceiveHostParam();

//     //         std::vector<cudaGraphNode_t> receive_mempcy_nodes(2);
//     //         cudaGraphAddMemcpyNode(&receive_mempcy_nodes[0], main_graph,
//     &receive_node, 1, &memcpy_param[0 + i * 2]);
//     //         cudaGraphAddMemcpyNode(&receive_mempcy_nodes[1], main_graph,
//     &receive_node, 1, &memcpy_param[1 + i * 2]);
//     //         cudaGraphNode_t receive_host_node;
//     //         cudaGraphAddHostNode(&receive_host_node, main_graph,
//     &receive_mempcy_nodes[0], 2, &host_param[i]);

//     //         cudaGraph_t send_graph = m_tcp_controller.GetSendGraph(i);
//     //         cudaGraphNode_t send_node;
//     //         cudaGraphAddChildGraphNode(&send_node, main_graph,
//     &receive_node, 1, send_graph);
//     //         auto send_memcpy_param =
//     m_tcp_controller.GetSendMemcpyParam();
//     //         auto send_host_param = m_tcp_controller.GetSendHostParam();

//     //         std::vector<cudaGraphNode_t> send_mempcy_nodes(2);
//     //         cudaGraphAddMemcpyNode(&send_mempcy_nodes[0], main_graph,
//     &send_node, 1, &send_memcpy_param[0 + i * 2]);
//     //         cudaGraphNode_t send_host_node;
//     //         cudaGraphAddHostNode(&send_host_node, main_graph,
//     &send_mempcy_nodes[0], 1, &send_host_param[i]);
//     //         cudaGraphAddMemcpyNode(&send_mempcy_nodes[1], main_graph,
//     &send_host_node, 1, &send_memcpy_param[1 + i * 2]);
//     //         tcp_nodes.push_back(send_node);
//     //     }

//     //     // ipv4 encapsulation nodes
//     //     std::vector<cudaGraphNode_t> ipv4_encapsulation_nodes;
//     //     for (int i = 0; i < node_batch_num; i++)
//     //     {
//     //         cudaGraph_t ipv4_encapsulation_graph =
//     m_ipv4_encapsulation_controller.GetGraph(i);
//     //         cudaGraphNode_t child_graph_node;
//     //         cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
//     &tcp_nodes[i], 1, ipv4_encapsulation_graph);

//     //         auto memcpy_param =
//     m_ipv4_encapsulation_controller.GetMemcpyParams();
//     //         auto host_param =
//     m_ipv4_encapsulation_controller.GetHostParams();

//     //         std::vector<cudaGraphNode_t> mempcy_nodes(2);
//     //         cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph,
//     &child_graph_node, 1, &memcpy_param[0 + i * 2]);
//     //         cudaGraphNode_t host_node;
//     //         cudaGraphAddHostNode(&host_node, main_graph, &mempcy_nodes[0],
//     1, &host_param[i]);
//     //         cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph,
//     &host_node, 1, &memcpy_param[1 + i * 2]);

//     //         ipv4_encapsulation_nodes.push_back(child_graph_node);
//     //     }

//     //     // relaunch ipv4
//     //     std::vector<cudaGraphNode_t> relaunch_ipv4_nodes;
//     //     for (int i = 0; i < node_batch_num; i++)
//     //     {
//     //         cudaGraph_t relaunch_ipv4_graph =
//     m_ipv4_controller.GetGraph(i);
//     //         cudaGraphNode_t child_graph_node;
//     //         cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
//     &ipv4_encapsulation_nodes[i], 1, relaunch_ipv4_graph);
//     //         relaunch_ipv4_nodes.push_back(child_graph_node);
//     //     }

//     //     // frame encapsulation nodes
//     //     std::vector<cudaGraphNode_t> frame_encapsulation_nodes;
//     //     for (int i = 0; i < node_batch_num; i++)
//     //     {
//     //         cudaGraph_t frame_encapsulation_graph =
//     m_frame_encapsulation_controller.GetGraph(i);
//     //         cudaGraphNode_t child_graph_node;
//     //         cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
//     &relaunch_ipv4_nodes[i], 1, frame_encapsulation_graph);

//     //         auto memcpy_param =
//     m_frame_encapsulation_controller.GetMemcpyParams();
//     //         auto host_param =
//     m_frame_encapsulation_controller.GetHostParams();

//     //         std::vector<cudaGraphNode_t> mempcy_nodes(2);
//     //         cudaGraphAddMemcpyNode(&mempcy_nodes[0], main_graph,
//     &child_graph_node, 1, &memcpy_param[0 + i * 2]);
//     //         cudaGraphNode_t host_node;
//     //         cudaGraphAddHostNode(&host_node, main_graph, &mempcy_nodes[0],
//     1, &host_param[i]);
//     //         cudaGraphAddMemcpyNode(&mempcy_nodes[1], main_graph,
//     &host_node, 1, &memcpy_param[1 + i * 2]);
//     //     }

//     //     // sw nodes
//     //     int sw_batch_num = m_sw_batch_start_index.size();
//     //     std::vector<cudaGraphNode_t> sw_nodes;
//     //     for (int i = 0; i < sw_batch_num; i++)
//     //     {
//     //         cudaGraph_t sw_graph = m_sw_controller.GetGraph(i);
//     //         cudaGraphNode_t child_graph_node;
//     //         cudaGraphAddChildGraphNode(&child_graph_node, main_graph,
//     NULL, 0, sw_graph);
//     //         sw_nodes.push_back(child_graph_node);
//     //     }

//     //     // add p2p cuda graph nodes
//     //     int p2p_batch_num = m_ch_batch_start_index.size();
//     //     std::vector<cudaGraphNode_t> p2p_predecessor_nodes;
//     //     p2p_predecessor_nodes.insert(p2p_predecessor_nodes.end(),
//     frame_decapsulation_nodes.begin(), frame_decapsulation_nodes.end());
//     //     p2p_predecessor_nodes.insert(p2p_predecessor_nodes.end(),
//     sw_nodes.begin(), sw_nodes.end());

//     //     std::vector<cudaGraphNode_t> p2p_nodes;
//     //     for (int i = 0; i < p2p_batch_num; i++)
//     //     {
//     //         cudaGraph_t p2p_graph = m_p2p_controller.GetGraph(i);
//     //         cudaGraphNode_t p2p_child_node;

//     //         cudaGraphAddChildGraphNode(&p2p_child_node, main_graph,
//     p2p_predecessor_nodes.data(), p2p_predecessor_nodes.size(), p2p_graph);
//     //         p2p_nodes.push_back(p2p_child_node);
//     //     }

//     //     // add timer
//     //     cudaGraph_t timer_graph = create_timer_graph(m_timeslot_start_gpu,
//     m_timeslot_end_gpu);
//     //     cudaGraphNode_t timer_node;
//     //     cudaGraphAddChildGraphNode(&timer_node, main_graph,
//     p2p_nodes.data(), p2p_nodes.size(), timer_graph);

//     //     // cudaGraphNodeSet

//     //     m_graph = main_graph;
//     //     cudaGraphInstantiate(&m_graph_exec, m_graph, 0);
//     // }
// #endif
//     void Topology::RunGraph()
//     {
//         cudaGraphLaunch(m_graph_exec, m_graph_stream);
//         cudaStreamSynchronize(m_graph_stream);
//     }

// } // namespace VDES
