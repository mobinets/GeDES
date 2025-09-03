#include "ipv4_encapsulation.h"

#include <component.h>
#include <functional>

namespace VDES
{

  IPv4EncapsulationController::IPv4EncapsulationController() {}

  IPv4EncapsulationController::~IPv4EncapsulationController() {}

  void IPv4EncapsulationController::InitalizeKernelParams()
  {
    int batch_num = m_batch_start_index.size();
    m_packet_sizes.push_back(sizeof(TCPPacket));

    for (int i = 0; i < batch_num; i++)
    {
      int node_num = m_batch_end_index[i] - m_batch_start_index[i];
      cudaGraph_t graph;
      cudaGraphCreate(&graph, 0);
      m_graphs.push_back(graph);
      m_graph_execs.emplace_back();

      int nic_num =
          std::accumulate(m_nic_num_per_node.begin() + m_batch_start_index[i],
                          m_nic_num_per_node.begin() + m_batch_end_index[i], 0);
      int max_packet_num = nic_num * MAX_TRANSMITTED_PACKET_NUM +
                           node_num * MAX_GENERATED_PACKET_NUM;
      m_max_packet_num.push_back(max_packet_num);

      IPv4EncapsulationParams cpu_param;
      cudaMallocAsync(&cpu_param.l4_packet_queues,
                      sizeof(GPUQueue<uint8_t *>) * node_num *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.ipv4_packet_queues,
                      sizeof(GPUQueue<Ipv4Packet *>) * node_num, m_streams[i]);

#if ENABLE_CACHE
      cudaMallocAsync(&cpu_param.l4_cache_space,
                      sizeof(uint8_t *) *
                          TransportProtocolType::COUNT_TransportProtocolType *
                          max_packet_num,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.l4_swap_out_packets,
                      sizeof(uint8_t *) *
                          TransportProtocolType::COUNT_TransportProtocolType *
                          max_packet_num,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.l4_swap_out_packet_num,
                      sizeof(int) * node_num *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.l4_swap_out_offset, sizeof(int) * node_num,
                      m_streams[i]);
#endif
      cudaMallocAsync(&cpu_param.l4_packet_size,
                      sizeof(int) *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.l4_src_ip_offset,
                      sizeof(int) *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.l4_dst_ip_offset,
                      sizeof(int) *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.l4_timestamp_offset,
                      sizeof(int) *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.alloc_ipv4_packets,
                      sizeof(Ipv4Packet *) * max_packet_num, m_streams[i]);
      cudaMallocAsync(&cpu_param.packet_offset_per_node, sizeof(int) * node_num,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.used_packet_num_per_node, sizeof(int) * node_num,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.l4_len_offset,
                      sizeof(int) *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);

      // copy to device
      int node_offset = m_batch_start_index[i];
      for (int j = 0; j < TransportProtocolType::COUNT_TransportProtocolType;
           j++)
      {
        cudaMemcpyAsync(cpu_param.l4_packet_queues,
                        m_l4_packet_queues.data() + j * node_num + node_offset,
                        sizeof(GPUQueue<uint8_t *> *) * node_num,
                        cudaMemcpyHostToDevice, m_streams[i]);
      }
      cudaMemcpyAsync(cpu_param.ipv4_packet_queues,
                      m_ipv4_packet_queues.data() + node_offset,
                      sizeof(GPUQueue<Ipv4Packet *> *) * node_num,
                      cudaMemcpyHostToDevice, m_streams[i]);
      cpu_param.node_num = node_num;

      int *packet_offsets = new int[node_num];
      int offset = 0;
      for (int j = 0; j < node_num; j++)
      {
        packet_offsets[j] = offset;
        offset +=
            (m_nic_num_per_node[j + node_offset] * MAX_TRANSMITTED_PACKET_NUM +
             MAX_GENERATED_PACKET_NUM);
      }
      m_packet_offset_per_node.push_back(packet_offsets);

#if ENABLE_CACHE
      uint8_t *l4_cache_gpu;
      int cache_size = 0;

      for (int j = 0; j < TransportProtocolType::COUNT_TransportProtocolType;
           j++)
      {
        cache_size += (max_packet_num * m_packet_sizes[j]);
      }
      cudaMallocAsync(&l4_cache_gpu, cache_size, m_streams[i]);
      /**
       * TODO: Synchronize here to make sure all memory is allocated before
       * copying to device.
       */
      cudaStreamSynchronize(m_streams[i]);
      m_l4_cache_space_gpu.push_back(l4_cache_gpu);
      uint8_t *l4_cache_cpu = new uint8_t[cache_size];
      m_l4_cache_space_cpu.push_back(l4_cache_cpu);
      m_cache_sizes.push_back(cache_size);

      uint8_t **l4_cache_gpu_ptr = new uint8_t
          *[max_packet_num * TransportProtocolType::COUNT_TransportProtocolType];
      uint8_t **l4_cache_cpu_ptr = new uint8_t
          *[max_packet_num * TransportProtocolType::COUNT_TransportProtocolType];

      for (int j = 0; j < TransportProtocolType::COUNT_TransportProtocolType;
           j++)
      {
        for (int k = 0; k < max_packet_num; k++)
        {
          l4_cache_gpu_ptr[j * max_packet_num + k] =
              l4_cache_gpu + k * m_packet_sizes[j];
          l4_cache_cpu_ptr[j * max_packet_num + k] =
              l4_cache_cpu + k * m_packet_sizes[j];
        }
        l4_cache_cpu += (max_packet_num * m_packet_sizes[j]);
        l4_cache_gpu += (max_packet_num * m_packet_sizes[j]);
      }

      cudaMemcpyAsync(cpu_param.l4_cache_space, l4_cache_gpu_ptr,
                      sizeof(uint8_t *) *
                          TransportProtocolType::COUNT_TransportProtocolType *
                          max_packet_num,
                      cudaMemcpyHostToDevice, m_streams[i]);
      m_l4_cache_space_ptr_gpu.push_back(l4_cache_gpu_ptr);
      m_l4_cache_space_ptr_cpu.push_back(l4_cache_cpu_ptr);

      auto tcp_packets = tcp_packet_pool_cpu->allocate(max_packet_num);
      cudaMemcpyAsync(cpu_param.l4_swap_out_packets, tcp_packets.data(),
                      sizeof(uint8_t *) * max_packet_num, cudaMemcpyHostToDevice,
                      m_streams[i]);
      m_l4_swap_out_packets_ptr_gpu.push_back(cpu_param.l4_swap_out_packets);
      m_l4_swap_out_packets_ptr_cpu.push_back(
          new uint8_t *[max_packet_num *
                        TransportProtocolType::COUNT_TransportProtocolType]);
      /**
       * @warning: Maybe multiple sorts of packets need swap out packets ptr here
       * in the future.
       */
      memcpy(m_l4_swap_out_packets_ptr_cpu[i], tcp_packets.data(),
             sizeof(uint8_t *) * max_packet_num);
      cudaMemsetAsync(cpu_param.l4_swap_out_packet_num, 0,
                      sizeof(int) * node_num *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);
      m_l4_swap_out_packet_num_gpu.push_back(cpu_param.l4_swap_out_packet_num);
      m_l4_swap_out_packet_num_cpu.push_back(
          new int[node_num * TransportProtocolType::COUNT_TransportProtocolType]);

      // int *packet_offsets = new int[node_num];
      // int offset = 0;
      // for (int j = 0; j < node_num; j++)
      // {
      //     packet_offsets[j] = offset;
      //     offset += (m_nic_num_per_node[j + node_offset] *
      //     MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM);
      // }
      // m_packet_offset_per_node.push_back(packet_offsets);
      cudaMemcpyAsync(cpu_param.l4_swap_out_offset, packet_offsets,
                      sizeof(int) * node_num, cudaMemcpyHostToDevice,
                      m_streams[i]);
#endif

      int packet_size = sizeof(TCPPacket);
      int src_ip_offset = 38;
      int dst_ip_offset = 42;
      int timestamp_offset = 30;
      int len_offset = 28;
      cudaMemcpyAsync(cpu_param.l4_packet_size, &packet_size, sizeof(int),
                      cudaMemcpyHostToDevice, m_streams[i]);
      cudaMemcpyAsync(cpu_param.l4_src_ip_offset, &src_ip_offset, sizeof(int),
                      cudaMemcpyHostToDevice, m_streams[i]);
      cudaMemcpyAsync(cpu_param.l4_dst_ip_offset, &dst_ip_offset, sizeof(int),
                      cudaMemcpyHostToDevice, m_streams[i]);
      cudaMemcpyAsync(cpu_param.l4_timestamp_offset, &timestamp_offset,
                      sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
      cudaMemcpyAsync(cpu_param.l4_len_offset, &len_offset, sizeof(int),
                      cudaMemcpyHostToDevice, m_streams[i]);

      cpu_param.max_packet_num = max_packet_num;

#if ENABLE_GPU_MEM_POOL
      std::vector<Ipv4Packet *> alloc_ipv4_packets;
      for (int i = 0; i < max_packet_num; i++)
      {
        Ipv4Packet* packet;
        cudaMalloc(&packet,sizeof(Ipv4Packet));
        alloc_ipv4_packets.push_back(packet);
      }
#else 

      auto alloc_ipv4_packets = ipv4_packet_pool->allocate(max_packet_num);
#endif
      cudaMemcpyAsync(cpu_param.alloc_ipv4_packets, alloc_ipv4_packets.data(),
                      sizeof(Ipv4Packet *) * max_packet_num,
                      cudaMemcpyHostToDevice, m_streams[i]);
      m_alloc_ipv4_packets_cpu.push_back(new Ipv4Packet *[max_packet_num]);
      memcpy(m_alloc_ipv4_packets_cpu[i], alloc_ipv4_packets.data(),
             sizeof(Ipv4Packet *) * max_packet_num);
      m_alloc_ipv4_packets_gpu.push_back(cpu_param.alloc_ipv4_packets);

      cudaMemcpyAsync(cpu_param.packet_offset_per_node, packet_offsets,
                      sizeof(int) * node_num, cudaMemcpyHostToDevice,
                      m_streams[i]);
      cudaMemsetAsync(cpu_param.used_packet_num_per_node, 0,
                      sizeof(int) * node_num, m_streams[i]);

      m_used_packet_num_per_node_gpu.push_back(
          cpu_param.used_packet_num_per_node);
      m_used_packet_num_per_node_cpu.push_back(new int[node_num]);

      IPv4EncapsulationParams *gpu_param;
      cudaMallocAsync(&gpu_param, sizeof(IPv4EncapsulationParams), m_streams[i]);
      cudaStreamSynchronize(m_streams[i]);
      cudaMemcpyAsync(gpu_param, &cpu_param, sizeof(IPv4EncapsulationParams),
                      cudaMemcpyHostToDevice, m_streams[i]);
      m_kernel_params.push_back(gpu_param);
    }
  }

  void IPv4EncapsulationController::SetIPv4PacketQueue(
      GPUQueue<Ipv4Packet *> **queues, int node_num)
  {
    m_ipv4_packet_queues.insert(m_ipv4_packet_queues.end(), queues,
                                queues + node_num);
  }

  void IPv4EncapsulationController::SetL4PacketQueue(GPUQueue<uint8_t *> **queues,
                                                     int node_num)
  {
    m_l4_packet_queues.insert(
        m_l4_packet_queues.end(), queues,
        queues + node_num * TransportProtocolType::COUNT_TransportProtocolType);
  }

  void IPv4EncapsulationController::SetBatchProperties(int *batch_start_index,
                                                       int *batch_end_index,
                                                       int batch_num)
  {
    m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index,
                               batch_start_index + batch_num);
    m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index,
                             batch_end_index + batch_num);
  }

  void IPv4EncapsulationController::SetNICNumPerNode(int *nic_num_per_node,
                                                     int node_num)
  {
    m_nic_num_per_node.insert(m_nic_num_per_node.end(), nic_num_per_node,
                              nic_num_per_node + node_num);
  }

  void IPv4EncapsulationController::SetStreams(cudaStream_t *streams,
                                               int node_num)
  {
    m_streams.insert(m_streams.end(), streams, streams + node_num);
  }

#if ENABLE_CACHE
  void IPv4EncapsulationController::CacheOutL4Packets(int batch_id)
  {
    int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
    int max_packet_num = m_max_packet_num[batch_id];
    int cache_size = m_cache_sizes[batch_id];

    cudaMemcpyAsync(m_l4_cache_space_cpu[batch_id],
                    m_l4_cache_space_gpu[batch_id], cache_size,
                    cudaMemcpyDeviceToHost, m_streams[batch_id]);
    cudaMemcpyAsync(m_l4_swap_out_packet_num_cpu[batch_id],
                    m_l4_swap_out_packet_num_gpu[batch_id],
                    sizeof(int) * node_num *
                        TransportProtocolType::COUNT_TransportProtocolType,
                    cudaMemcpyDeviceToHost, m_streams[batch_id]);
    cudaStreamSynchronize(m_streams[batch_id]);

    int *swap_out_packet_num = m_l4_swap_out_packet_num_cpu[batch_id];
    int *packet_offsets = m_packet_offset_per_node[batch_id];

    for (int i = 0; i < TransportProtocolType::COUNT_TransportProtocolType; i++)
    {
      int packet_size = m_packet_sizes[i];
      uint8_t **dst_ptrs =
          m_l4_swap_out_packets_ptr_cpu[batch_id] + i * max_packet_num;
      uint8_t **src_ptrs =
          m_l4_cache_space_ptr_cpu[batch_id] + i * max_packet_num;

      std::vector<uint8_t *> alloc_l4_packets;
      if (i == TransportProtocolType::TCP)
      {
        int tcp_packet_num = std::accumulate(swap_out_packet_num,
                                             swap_out_packet_num + node_num, 0);
        auto l4_packets = tcp_packet_pool_cpu->allocate(tcp_packet_num);
        alloc_l4_packets.insert(alloc_l4_packets.end(),
                                (uint8_t **)l4_packets.data(),
                                (uint8_t **)(l4_packets.data() + tcp_packet_num));
      }

      int allocated_l4_packet_num = 0;
      for (int j = 0; j < node_num; j++)
      {
        int offset = packet_offsets[j];
        for (int k = 0; k < swap_out_packet_num[j]; k++)
        {
          int packet_index = offset + k;
          memcpy(dst_ptrs[packet_index], src_ptrs[packet_index], packet_size);
          dst_ptrs[packet_index] = alloc_l4_packets[allocated_l4_packet_num];
          allocated_l4_packet_num++;
        }
      }
      swap_out_packet_num = swap_out_packet_num + node_num;
    }
    /**
     * @TODO: CHANGE THE COPY SIZE FROM NODE_NUM TO MAX_PACKET_NUM
     */
    cudaMemcpyAsync(m_l4_swap_out_packets_ptr_gpu[batch_id],
                    m_l4_swap_out_packets_ptr_cpu[batch_id],
                    sizeof(uint8_t *) * max_packet_num *
                        TransportProtocolType::COUNT_TransportProtocolType,
                    cudaMemcpyHostToDevice, m_streams[batch_id]);
    // #if DEBUG_MODE
    //         cudaStreamSynchronize(m_streams[batch_id]);
    //         int *new_swap_out_packet_num =
    //         m_l4_swap_out_packet_num_cpu[batch_id]; for (int i = 0; i <
    //         TransportProtocolType::COUNT_TransportProtocolType; i++)
    //         {
    //             int packet_size = m_packet_sizes[i];
    //             uint8_t **dst_ptrs = m_l4_swap_out_packets_ptr_cpu[batch_id] +
    //             i * max_packet_num; for (int j = 0; j < node_num; j++)
    //             {
    //                 int offset = packet_offsets[j];
    //                 for (int k = 0; k < new_swap_out_packet_num[j]; k++)
    //                 {
    //                     int packet_index = offset + k;
    //                     VDES::TCPPacket *tcp_packet = (VDES::TCPPacket
    //                     *)(dst_ptrs[packet_index]); uint32_t src_ip; uint32_t
    //                     dst_ip; uint64_t timestamp; memcpy(&src_ip,
    //                     tcp_packet->src_ip, 4); memcpy(&dst_ip,
    //                     tcp_packet->dst_ip, 4); memcpy(&timestamp,
    //                     tcp_packet->send_timestamp, 8);
    //                     // LOG_INFO("src_ip: %u, dst_ip: %u, timestamp: %lu,
    //                     offset: %d", src_ip, dst_ip, timestamp, offset);

    //                     uint64_t validation_val;
    //                     memcpy(&validation_val, tcp_packet->payload,
    //                     sizeof(uint64_t));
    //                     // LOG_INFO("validation_val: %lx", validation_val);
    //                 }
    //             }
    //             new_swap_out_packet_num += node_num;
    //         }
    // #endif
  }
#endif

  void IPv4EncapsulationController::UpdateUsedIPv4Packets(int batch_id)
  {
    int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
#if !ENABLE_HUGE_GRAPH
    cudaMemcpy(m_used_packet_num_per_node_cpu[batch_id],
               m_used_packet_num_per_node_gpu[batch_id], sizeof(int) * node_num,
               cudaMemcpyDeviceToHost);
#endif
    int *used_packet_num_per_node = m_used_packet_num_per_node_cpu[batch_id];
    int total_used_packet_num = std::accumulate(
        used_packet_num_per_node, used_packet_num_per_node + node_num, 0);
    auto alloc_ipv4_packets = ipv4_packet_pool->allocate(total_used_packet_num);
    int max_packet_num = m_max_packet_num[batch_id];

    int *packet_offsets = m_packet_offset_per_node[batch_id];
    Ipv4Packet **alloc_ipv4_packets_cpu = m_alloc_ipv4_packets_cpu[batch_id];

    int allocated_ipv4_packet_num = 0;
    for (int i = 0; i < node_num; i++)
    {
      int offset = packet_offsets[i];
      memcpy(alloc_ipv4_packets_cpu + offset,
             alloc_ipv4_packets.data() + allocated_ipv4_packet_num,
             sizeof(Ipv4Packet *) * used_packet_num_per_node[i]);
      allocated_ipv4_packet_num += used_packet_num_per_node[i];
    }
#if !ENABLE_HUGE_GRAPH
    cudaMemcpy(m_alloc_ipv4_packets_gpu[batch_id],
               m_alloc_ipv4_packets_cpu[batch_id],
               sizeof(Ipv4Packet *) * max_packet_num, cudaMemcpyHostToDevice);
#endif
  }

  void IPv4EncapsulationController::BuildGraph(int batch_id)
  {
    int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

    dim3 block_dim(KERNEL_BLOCK_WIDTH);
    dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);

    cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
    LaunchIPv4EncapsulationKernel(grid_dim, block_dim, m_kernel_params[batch_id],
                                  m_streams[batch_id]);
    cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
    cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL,
                         0);

#if ENABLE_HUGE_GRAPH
    cudaGraphNode_t used_packet_num_memcpy_node;
    cudaMemcpy3DParms used_packet_num_memcpy_params = {0};
    used_packet_num_memcpy_params.srcPtr =
        make_cudaPitchedPtr(m_used_packet_num_per_node_gpu[batch_id],
                            sizeof(int) * node_num, node_num, 1);
    used_packet_num_memcpy_params.dstPtr =
        make_cudaPitchedPtr(m_used_packet_num_per_node_cpu[batch_id],
                            sizeof(int) * node_num, node_num, 1);
    used_packet_num_memcpy_params.extent =
        make_cudaExtent(sizeof(int) * node_num, 1, 1);
    used_packet_num_memcpy_params.kind = cudaMemcpyDeviceToHost;

    cudaHostNodeParams update_host_params = {0};
    auto update_func = std::bind(
        &IPv4EncapsulationController::UpdateUsedIPv4Packets, this, batch_id);
    auto update_func_ptr = new std::function<void()>(update_func);
    update_host_params.fn = VDES::HostNodeCallback;
    update_host_params.userData = update_func_ptr;

    cudaMemcpy3DParms alloc_packet_memcpy_params = {0};
    int max_packet_num = m_max_packet_num[batch_id];
    alloc_packet_memcpy_params.srcPtr = make_cudaPitchedPtr(
        m_alloc_ipv4_packets_cpu[batch_id], sizeof(Ipv4Packet *) * max_packet_num,
        max_packet_num, 1);
    alloc_packet_memcpy_params.dstPtr = make_cudaPitchedPtr(
        m_alloc_ipv4_packets_gpu[batch_id], sizeof(Ipv4Packet *) * max_packet_num,
        max_packet_num, 1);
    alloc_packet_memcpy_params.extent =
        make_cudaExtent(sizeof(Ipv4Packet *) * max_packet_num, 1, 1);
    alloc_packet_memcpy_params.kind = cudaMemcpyHostToDevice;

    m_memcpy_param.push_back(used_packet_num_memcpy_params);
    m_memcpy_param.push_back(alloc_packet_memcpy_params);
    m_host_param.push_back(update_host_params);
#endif
  }

  void IPv4EncapsulationController::BuildGraph()
  {
    int batch_num = m_batch_start_index.size();
    for (int i = 0; i < batch_num; i++)
    {
      BuildGraph(i);
    }
  }

  void IPv4EncapsulationController::LaunchInstance(int batch_id)
  {
    cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
  }

  void IPv4EncapsulationController::Run(int batch_id)
  {
    LaunchInstance(batch_id);
    cudaStreamSynchronize(m_streams[batch_id]);
#if ENABLE_CACHE
    CacheOutL4Packets(batch_id);
#endif
    UpdateUsedIPv4Packets(batch_id);
  }

  void IPv4EncapsulationController::Run() {}
  cudaGraph_t IPv4EncapsulationController::GetGraph(int batch_id)
  {
    return m_graphs[batch_id];
  }

#if ENABLE_HUGE_GRAPH
  std::vector<cudaMemcpy3DParms> &IPv4EncapsulationController::GetMemcpyParams()
  {
    return m_memcpy_param;
  }

  std::vector<cudaHostNodeParams> &IPv4EncapsulationController::GetHostParams()
  {
    return m_host_param;
  }
#endif

  std::vector<void *> IPv4EncapsulationController::GetAllocateInfo()
  {
    int batch_num = m_batch_start_index.size();
    std::vector<void *> res;

    for (int i = 0; i < batch_num; i++)
    {
      res.push_back(m_alloc_ipv4_packets_gpu[i]);
      res.push_back(m_used_packet_num_per_node_gpu[i]);
    }
    return res;
  }

} // namespace VDES

// namespace VDES
// {

//     IPv4EncapsulationController::IPv4EncapsulationController()
//     {
//     }

//     IPv4EncapsulationController::~IPv4EncapsulationController()
//     {
//     }

//     void IPv4EncapsulationController::InitalizeKernelParams()
//     {
//         int batch_num = m_batch_start_index.size();
//         m_packet_sizes.push_back(sizeof(TCPPacket));
//         // m_packet_sizes.push_back(sizeof(UDPPacket));
//         // m_packet_sizes.push_back(sizeof(ICMPPacket));

//         for (int i = 0; i < batch_num; i++)
//         {
//             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
//             cudaGraph_t graph;
//             cudaGraphCreate(&graph, 0);
//             m_graphs.push_back(graph);
//             m_graph_execs.emplace_back();

//             int nic_num = std::accumulate(m_nic_num_per_node.begin() +
//             m_batch_start_index[i], m_nic_num_per_node.begin() +
//             m_batch_end_index[i], 0); int max_packet_num = nic_num *
//             MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;
//             m_max_packet_num.push_back(max_packet_num);

//             IPv4EncapsulationParams cpu_param;
//             cudaMallocAsync(&cpu_param.l4_packet_queues,
//             sizeof(GPUQueue<uint8_t *>) * node_num *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.ipv4_packet_queues,
//             sizeof(GPUQueue<Ipv4Packet *>) * node_num, m_streams[i]);

// #if ENABLE_CACHE

//             cudaMallocAsync(&cpu_param.l4_cache_space, sizeof(uint8_t *) *
//             TransportProtocolType::COUNT_TransportProtocolType *
//             max_packet_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_swap_out_packets, sizeof(uint8_t *)
//             * TransportProtocolType::COUNT_TransportProtocolType *
//             max_packet_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_swap_out_packet_num, sizeof(int) *
//             node_num * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_swap_out_offset,
//             sizeof(int) * node_num, m_streams[i]);
// #endif

//             cudaMallocAsync(&cpu_param.l4_packet_size, sizeof(int) *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_src_ip_offset,
//             sizeof(int) * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_dst_ip_offset,
//             sizeof(int) * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_timestamp_offset,
//             sizeof(int) * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_len_offset,
//             sizeof(int) * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.alloc_ipv4_packets,
//             sizeof(Ipv4Packet *) * max_packet_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.packet_offset_per_node, sizeof(int) *
//             node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.used_packet_num_per_node, sizeof(int)
//             * node_num, m_streams[i]);

//             // copy to device
//             /**
//              *  @warning: Maybe add multiple
//              TransportProtocolType::COUNT_TransportProtocolType here.
//              */
//             int node_offset = m_batch_start_index[i];
//             for (int j = 0; j <
//             TransportProtocolType::COUNT_TransportProtocolType; j++)
//             {
//                 cudaMemcpyAsync(cpu_param.l4_packet_queues,
//                 m_l4_packet_queues.data() + j * node_num + node_offset,
//                 sizeof(GPUQueue<uint8_t *> *) * node_num,
//                 cudaMemcpyHostToDevice, m_streams[i]);
//             }
//             cudaMemcpyAsync(cpu_param.ipv4_packet_queues,
//             m_ipv4_packet_queues.data() + node_offset,
//             sizeof(GPUQueue<Ipv4Packet *> *) * node_num,
//             cudaMemcpyHostToDevice, m_streams[i]); cpu_param.node_num =
//             node_num;

//             int *packet_offsets = new int[node_num];
//             int offset = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 packet_offsets[j] = offset;
//                 offset += (m_nic_num_per_node[j + node_offset] *
//                 MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM);
//             }

//             m_packet_offset_per_node.push_back(packet_offsets);

// #if ENABLE_CACHE

//             uint8_t *l4_cache_gpu;
//             int cache_size = 0;

//             for (int j = 0; j <
//             TransportProtocolType::COUNT_TransportProtocolType; j++)
//             {
//                 cache_size += (max_packet_num * m_packet_sizes[j]);
//             }
//             cudaMallocAsync(&l4_cache_gpu, cache_size, m_streams[i]);
//             /**
//              * TODO: Synchronize here to make sure all memory is allocated
//              before copying to device.
//              */
//             cudaStreamSynchronize(m_streams[i]);
//             m_l4_cache_space_gpu.push_back(l4_cache_gpu);
//             uint8_t *l4_cache_cpu = new uint8_t[cache_size];
//             m_l4_cache_space_cpu.push_back(l4_cache_cpu);
//             m_cache_sizes.push_back(cache_size);

//             uint8_t **l4_cache_gpu_ptr = new uint8_t *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType]; uint8_t
//             **l4_cache_cpu_ptr = new uint8_t *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType];

//             for (int j = 0; j <
//             TransportProtocolType::COUNT_TransportProtocolType; j++)
//             {
//                 for (int k = 0; k < max_packet_num; k++)
//                 {
//                     l4_cache_gpu_ptr[j * max_packet_num + k] = l4_cache_gpu +
//                     k * m_packet_sizes[j]; l4_cache_cpu_ptr[j *
//                     max_packet_num + k] = l4_cache_cpu + k *
//                     m_packet_sizes[j];
//                 }
//                 l4_cache_cpu += (max_packet_num * m_packet_sizes[j]);
//                 l4_cache_gpu += (max_packet_num * m_packet_sizes[j]);
//             }

//             cudaMemcpyAsync(cpu_param.l4_cache_space, l4_cache_gpu_ptr,
//             sizeof(uint8_t *) *
//             TransportProtocolType::COUNT_TransportProtocolType *
//             max_packet_num, cudaMemcpyHostToDevice, m_streams[i]);
//             m_l4_cache_space_ptr_gpu.push_back(l4_cache_gpu_ptr);
//             m_l4_cache_space_ptr_cpu.push_back(l4_cache_cpu_ptr);

//             auto tcp_packets = tcp_packet_pool_cpu->allocate(max_packet_num);
//             cudaMemcpyAsync(cpu_param.l4_swap_out_packets,
//             tcp_packets.data(), sizeof(uint8_t *) * max_packet_num,
//             cudaMemcpyHostToDevice, m_streams[i]);
//             m_l4_swap_out_packets_ptr_gpu.push_back(cpu_param.l4_swap_out_packets);
//             m_l4_swap_out_packets_ptr_cpu.push_back(new uint8_t
//             *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);
//             /**
//              * @warning: Maybe multiple sorts of packets need swap out
//              packets ptr here in the future.
//              */
//             memcpy(m_l4_swap_out_packets_ptr_cpu[i], tcp_packets.data(),
//             sizeof(uint8_t *) * max_packet_num);
//             cudaMemsetAsync(cpu_param.l4_swap_out_packet_num, 0, sizeof(int)
//             * node_num * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]);
//             m_l4_swap_out_packet_num_gpu.push_back(cpu_param.l4_swap_out_packet_num);
//             m_l4_swap_out_packet_num_cpu.push_back(new int[node_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);

//             cudaMemcpyAsync(cpu_param.l4_swap_out_offset, packet_offsets,
//             sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
// #endif

//             int packet_size = sizeof(TCPPacket);
//             int src_ip_offset = 38;
//             int dst_ip_offset = 42;
//             int timestamp_offset = 30;
//             int len_offset = 28;
//             cudaMemcpyAsync(cpu_param.l4_packet_size, &packet_size,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.l4_src_ip_offset, &src_ip_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.l4_dst_ip_offset, &dst_ip_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.l4_timestamp_offset, &timestamp_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.l4_len_offset, &len_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);

//             cpu_param.max_packet_num = max_packet_num;

//             auto alloc_ipv4_packets =
//             ipv4_packet_pool->allocate(max_packet_num);
//             cudaMemcpyAsync(cpu_param.alloc_ipv4_packets,
//             alloc_ipv4_packets.data(), sizeof(Ipv4Packet *) * max_packet_num,
//             cudaMemcpyHostToDevice, m_streams[i]);
//             m_alloc_ipv4_packets_cpu.push_back(new Ipv4Packet
//             *[max_packet_num]); memcpy(m_alloc_ipv4_packets_cpu[i],
//             alloc_ipv4_packets.data(), sizeof(Ipv4Packet *) *
//             max_packet_num);
//             m_alloc_ipv4_packets_gpu.push_back(cpu_param.alloc_ipv4_packets);

//             cudaMemcpyAsync(cpu_param.packet_offset_per_node, packet_offsets,
//             sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemsetAsync(cpu_param.used_packet_num_per_node, 0,
//             sizeof(int) * node_num, m_streams[i]);
//             /**
//              * TODO: Prepare space for used_packet_num_per_node_gpu and
//              used_packet_num_per_node_cpu.
//              */
//             m_used_packet_num_per_node_gpu.push_back(cpu_param.used_packet_num_per_node);
//             m_used_packet_num_per_node_cpu.push_back(new int[node_num]);

//             IPv4EncapsulationParams *gpu_param;
//             cudaMallocAsync(&gpu_param, sizeof(IPv4EncapsulationParams),
//             m_streams[i]); cudaStreamSynchronize(m_streams[i]);
//             cudaMemcpyAsync(gpu_param, &cpu_param,
//             sizeof(IPv4EncapsulationParams), cudaMemcpyHostToDevice,
//             m_streams[i]); m_kernel_params.push_back(gpu_param);
//         }
//     }

//     void IPv4EncapsulationController::SetIPv4PacketQueue(GPUQueue<Ipv4Packet
//     *> **queues, int node_num)
//     {
//         m_ipv4_packet_queues.insert(m_ipv4_packet_queues.end(), queues,
//         queues + node_num);
//     }

//     void IPv4EncapsulationController::SetL4PacketQueue(GPUQueue<uint8_t *>
//     **queues, int node_num)
//     {
//         m_l4_packet_queues.insert(m_l4_packet_queues.end(), queues, queues +
//         node_num * TransportProtocolType::COUNT_TransportProtocolType);
//     }

//     void IPv4EncapsulationController::SetBatchProperties(int
//     *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(),
//         batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index,
//         batch_end_index + batch_num);
//     }

//     void IPv4EncapsulationController::SetNICNumPerNode(int *nic_num_per_node,
//     int node_num)
//     {
//         m_nic_num_per_node.insert(m_nic_num_per_node.end(), nic_num_per_node,
//         nic_num_per_node + node_num);
//     }

//     void IPv4EncapsulationController::SetStreams(cudaStream_t *streams, int
//     node_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + node_num);
//     }

// #if ENABLE_CACHE

//     void IPv4EncapsulationController::CacheOutL4Packets(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id]; int max_packet_num =
//         m_max_packet_num[batch_id]; int cache_size = m_cache_sizes[batch_id];

//         cudaMemcpyAsync(m_l4_cache_space_cpu[batch_id],
//         m_l4_cache_space_gpu[batch_id], cache_size, cudaMemcpyDeviceToHost,
//         m_streams[batch_id]);
//         cudaMemcpyAsync(m_l4_swap_out_packet_num_cpu[batch_id],
//         m_l4_swap_out_packet_num_gpu[batch_id], sizeof(int) * node_num *
//         TransportProtocolType::COUNT_TransportProtocolType,
//         cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);

//         int *swap_out_packet_num = m_l4_swap_out_packet_num_cpu[batch_id];
//         int *packet_offsets = m_packet_offset_per_node[batch_id];

//         for (int i = 0; i <
//         TransportProtocolType::COUNT_TransportProtocolType; i++)
//         {
//             int packet_size = m_packet_sizes[i];
//             uint8_t **dst_ptrs = m_l4_swap_out_packets_ptr_cpu[batch_id] + i
//             * max_packet_num; uint8_t **src_ptrs =
//             m_l4_cache_space_ptr_cpu[batch_id] + i * max_packet_num;

//             // uint8_t **alloc_l4_packets = NULL;
//             std::vector<uint8_t *> alloc_l4_packets;
//             if (i == TransportProtocolType::TCP)
//             {
//                 int tcp_packet_num = std::accumulate(swap_out_packet_num,
//                 swap_out_packet_num + node_num, 0); auto l4_packets =
//                 tcp_packet_pool_cpu->allocate(tcp_packet_num);
//                 /**
//                  * TODO: Use this instead of insert.
//                  */
//                 alloc_l4_packets.insert(alloc_l4_packets.end(), (uint8_t
//                 **)l4_packets.data(), (uint8_t **)(l4_packets.data() +
//                 tcp_packet_num));
//             }

//             int allocated_l4_packet_num = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 int offset = packet_offsets[j];
//                 for (int k = 0; k < swap_out_packet_num[j]; k++)
//                 {
//                     int packet_index = offset + k;
//                     memcpy(dst_ptrs[packet_index], src_ptrs[packet_index],
//                     packet_size); dst_ptrs[packet_index] =
//                     alloc_l4_packets[allocated_l4_packet_num];
//                     allocated_l4_packet_num++;
//                 }
//             }
//             swap_out_packet_num = swap_out_packet_num + node_num;
//         }
//         cudaMemcpyAsync(m_l4_swap_out_packets_ptr_gpu[batch_id],
//         m_l4_swap_out_packets_ptr_cpu[batch_id], sizeof(uint8_t **) *
//         max_packet_num * TransportProtocolType::COUNT_TransportProtocolType,
//         cudaMemcpyHostToDevice, m_streams[batch_id]);
//     }

// #endif

//     void IPv4EncapsulationController::UpdateUsedIPv4Packets(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id];

// #if !ENABLE_HUGE_GRAPH
//         cudaMemcpy(m_used_packet_num_per_node_cpu[batch_id],
//         m_used_packet_num_per_node_gpu[batch_id], sizeof(int) * node_num,
//         cudaMemcpyDeviceToHost);
// #endif
//         int *used_packet_num_per_node =
//         m_used_packet_num_per_node_cpu[batch_id]; int total_used_packet_num =
//         std::accumulate(used_packet_num_per_node, used_packet_num_per_node +
//         node_num, 0); auto alloc_ipv4_packets =
//         ipv4_packet_pool->allocate(total_used_packet_num); int max_packet_num
//         = m_max_packet_num[batch_id];

//         int *packet_offsets = m_packet_offset_per_node[batch_id];
//         Ipv4Packet **alloc_ipv4_packets_cpu =
//         m_alloc_ipv4_packets_cpu[batch_id];

//         int allocated_ipv4_packet_num = 0;
//         for (int i = 0; i < node_num; i++)
//         {
//             int offset = packet_offsets[i];
//             // for (int j = 0; j < used_packet_num_per_node[i]; j++)
//             // {
//             //     alloc_ipv4_packets_cpu[offset + j] =
//             alloc_ipv4_packets[allocated_ipv4_packet_num];
//             //     allocated_ipv4_packet_num++;
//             // }
//             memcpy(alloc_ipv4_packets_cpu + offset, alloc_ipv4_packets.data()
//             + allocated_ipv4_packet_num, sizeof(Ipv4Packet *) *
//             used_packet_num_per_node[i]); allocated_ipv4_packet_num +=
//             used_packet_num_per_node[i];
//         }
// #if !ENABLE_HUGE_GRAPH
//         cudaMemcpy(m_alloc_ipv4_packets_gpu[batch_id],
//         m_alloc_ipv4_packets_cpu[batch_id], sizeof(Ipv4Packet **) *
//         max_packet_num, cudaMemcpyHostToDevice);
// #endif
//     }

//     void IPv4EncapsulationController::BuildGraph(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id];

//         dim3 block_dim(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);

//         cudaStreamBeginCapture(m_streams[batch_id],
//         cudaStreamCaptureModeGlobal); LaunchIPv4EncapsulationKernel(grid_dim,
//         block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id],
//         NULL, NULL, 0);

// #if ENABLE_HUGE_GRAPH
//         // cudaGraphNode_t kernel_node;
//         // size_t num_nodes;
//         // cudaGraphGetNodes(m_graphs[batch_id], &kernel_node, &num_nodes);

//         cudaGraphNode_t used_packet_num_memcpy_node;
//         cudaMemcpy3DParms used_packet_num_memcpy_params = {0};
//         used_packet_num_memcpy_params.srcPtr =
//         make_cudaPitchedPtr(m_used_packet_num_per_node_gpu[batch_id],
//         sizeof(int) * node_num, node_num, 1);
//         used_packet_num_memcpy_params.dstPtr =
//         make_cudaPitchedPtr(m_used_packet_num_per_node_cpu[batch_id],
//         sizeof(int) * node_num, node_num, 1);
//         used_packet_num_memcpy_params.extent = make_cudaExtent(sizeof(int) *
//         node_num, 1, 1); used_packet_num_memcpy_params.kind =
//         cudaMemcpyDeviceToHost;
//         // cudaGraphAddMemcpyNode(&used_packet_num_memcpy_node,
//         m_graphs[batch_id], &kernel_node, 1, &used_packet_num_memcpy_params);

//         // cudaGraphNode_t update_host_node;
//         cudaHostNodeParams update_host_params = {0};
//         auto update_func =
//         std::bind(&IPv4EncapsulationController::UpdateUsedIPv4Packets, this,
//         batch_id); auto update_func_ptr = new
//         std::function<void()>(update_func); update_host_params.fn =
//         VDES::HostNodeCallback; update_host_params.userData =
//         update_func_ptr;
//         // cudaGraphAddHostNode(&update_host_node, m_graphs[batch_id],
//         &used_packet_num_memcpy_node, 1, &update_host_params);

//         // cudaGraphNode_t alloc_packet_memcpy_node;
//         cudaMemcpy3DParms alloc_packet_memcpy_params = {0};
//         int max_packet_num = m_max_packet_num[batch_id];
//         alloc_packet_memcpy_params.srcPtr =
//         make_cudaPitchedPtr(m_alloc_ipv4_packets_cpu[batch_id],
//         sizeof(Ipv4Packet *) * max_packet_num, max_packet_num, 1);
//         alloc_packet_memcpy_params.dstPtr =
//         make_cudaPitchedPtr(m_alloc_ipv4_packets_gpu[batch_id],
//         sizeof(Ipv4Packet *) * max_packet_num, max_packet_num, 1);
//         alloc_packet_memcpy_params.extent = make_cudaExtent(sizeof(Ipv4Packet
//         *) * max_packet_num, 1, 1); alloc_packet_memcpy_params.kind =
//         cudaMemcpyHostToDevice;
//         // cudaGraphAddMemcpyNode(&alloc_packet_memcpy_node,
//         m_graphs[batch_id], &update_host_node, 1,
//         &alloc_packet_memcpy_params);

//         m_memcpy_param.push_back(used_packet_num_memcpy_params);
//         m_memcpy_param.push_back(alloc_packet_memcpy_params);
//         m_host_param.push_back(update_host_params);
// #endif
//     }

//     void IPv4EncapsulationController::BuildGraph()
//     {
//         int batch_num = m_batch_start_index.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraph(i);
//         }
//     }

//     void IPv4EncapsulationController::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void IPv4EncapsulationController::Run(int batch_id)
//     {
//         LaunchInstance(batch_id);
//         cudaStreamSynchronize(m_streams[batch_id]);
// #if ENABLE_CACHE
//         CacheOutL4Packets(batch_id);
// #endif
//         UpdateUsedIPv4Packets(batch_id);
//         cudaStreamSynchronize(m_streams[batch_id]);
//     }

//     void IPv4EncapsulationController::Run()
//     {
//     }

//     cudaGraph_t IPv4EncapsulationController::GetGraph(int batch_id)
//     {
//         return m_graphs[batch_id];
//     }

// #if ENABLE_HUGE_GRAPH
//     std::vector<cudaMemcpy3DParms>
//     &IPv4EncapsulationController::GetMemcpyParams()
//     {
//         return m_memcpy_param;
//     }

//     std::vector<cudaHostNodeParams>
//     &IPv4EncapsulationController::GetHostParams()
//     {
//         return m_host_param;
//     }
// #endif

// }

// #include "ipv4_encapsulation.h"

// namespace VDES
// {

//     IPv4EncapsulationController::IPv4EncapsulationController()
//     {
//     }

//     IPv4EncapsulationController::~IPv4EncapsulationController()
//     {
//     }

//     void IPv4EncapsulationController::InitalizeKernelParams()
//     {
//         int batch_num = m_batch_start_index.size();
//         m_packet_sizes.push_back(sizeof(TCPPacket));

//         for (int i = 0; i < batch_num; i++)
//         {
//             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
//             cudaGraph_t graph;
//             cudaGraphCreate(&graph, 0);
//             m_graphs.push_back(graph);
//             m_graph_execs.emplace_back();

//             int nic_num = std::accumulate(m_nic_num_per_node.begin() +
//             m_batch_start_index[i], m_nic_num_per_node.begin() +
//             m_batch_end_index[i], 0); int max_packet_num = nic_num *
//             MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;
//             m_max_packet_num.push_back(max_packet_num);

//             IPv4EncapsulationParams cpu_param;
//             cudaMallocAsync(&cpu_param.l4_packet_queues,
//             sizeof(GPUQueue<uint8_t *>) * node_num *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.ipv4_packet_queues,
//             sizeof(GPUQueue<Ipv4Packet *>) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_cache_space, sizeof(uint8_t *) *
//             TransportProtocolType::COUNT_TransportProtocolType *
//             max_packet_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_swap_out_packets, sizeof(uint8_t *)
//             * TransportProtocolType::COUNT_TransportProtocolType *
//             max_packet_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_swap_out_packet_num, sizeof(int) *
//             node_num * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_swap_out_offset,
//             sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_packet_size, sizeof(int) *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_src_ip_offset,
//             sizeof(int) * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_dst_ip_offset,
//             sizeof(int) * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_timestamp_offset,
//             sizeof(int) * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.alloc_ipv4_packets,
//             sizeof(Ipv4Packet *) * max_packet_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.packet_offset_per_node, sizeof(int) *
//             node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.used_packet_num_per_node, sizeof(int)
//             * node_num, m_streams[i]);

//             // copy to device
//             int node_offset = m_batch_start_index[i];
//             for (int j = 0; j <
//             TransportProtocolType::COUNT_TransportProtocolType; j++)
//             {
//                 cudaMemcpyAsync(cpu_param.l4_packet_queues,
//                 m_l4_packet_queues.data() + j * node_num + node_offset,
//                 sizeof(GPUQueue<uint8_t *>) * node_num,
//                 cudaMemcpyHostToDevice, m_streams[i]);
//             }
//             cudaMemcpyAsync(cpu_param.ipv4_packet_queues,
//             m_ipv4_packet_queues.data() + node_offset,
//             sizeof(GPUQueue<Ipv4Packet *>) * node_num,
//             cudaMemcpyHostToDevice, m_streams[i]); cpu_param.node_num =
//             node_num;

//             uint8_t *l4_cache_gpu;
//             int cache_size = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 cache_size += (max_packet_num * m_packet_sizes[j]);
//             }
//             cudaMallocAsync(&l4_cache_gpu, cache_size, m_streams[i]);
//             m_l4_cache_space_gpu.push_back(l4_cache_gpu);
//             uint8_t *l4_cache_cpu = new uint8_t[cache_size];
//             m_l4_cache_space_cpu.push_back(l4_cache_cpu);
//             m_cache_sizes.push_back(cache_size);

//             uint8_t **l4_cache_gpu_ptr = new uint8_t *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType]; uint8_t
//             **l4_cache_cpu_ptr = new uint8_t *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType];

//             for (int j = 0; j <
//             TransportProtocolType::COUNT_TransportProtocolType; j++)
//             {
//                 for (int k = 0; k < max_packet_num; k++)
//                 {
//                     l4_cache_gpu_ptr[j * max_packet_num + k] = l4_cache_gpu +
//                     k * m_packet_sizes[j]; l4_cache_cpu_ptr[j *
//                     max_packet_num + k] = l4_cache_cpu + k *
//                     m_packet_sizes[j];
//                 }
//                 l4_cache_cpu += (max_packet_num * m_packet_sizes[j]);
//                 l4_cache_gpu += (max_packet_num * m_packet_sizes[j]);
//             }
//             cudaMemcpyAsync(cpu_param.l4_cache_space, l4_cache_gpu_ptr,
//             sizeof(uint8_t *) *
//             TransportProtocolType::COUNT_TransportProtocolType *
//             max_packet_num, cudaMemcpyHostToDevice, m_streams[i]);
//             m_l4_cache_space_ptr_gpu.push_back(l4_cache_gpu_ptr);
//             m_l4_cache_space_ptr_cpu.push_back(l4_cache_cpu_ptr);

//             auto tcp_packets = tcp_packet_pool_cpu->allocate(max_packet_num);
//             cudaMemcpyAsync(cpu_param.l4_swap_out_packets,
//             tcp_packets.data(), sizeof(uint8_t *) * max_packet_num,
//             cudaMemcpyHostToDevice, m_streams[i]);
//             m_l4_swap_out_packets_ptr_gpu.push_back(cpu_param.l4_swap_out_packets);
//             m_l4_swap_out_packets_ptr_cpu.push_back(new uint8_t
//             *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);
//             memcpy(m_l4_swap_out_packets_ptr_cpu[i], tcp_packets.data(),
//             sizeof(uint8_t *) * max_packet_num);
//             cudaMemsetAsync(cpu_param.l4_swap_out_packet_num, 0, sizeof(int)
//             * node_num * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]);
//             m_l4_swap_out_packet_num_gpu.push_back(cpu_param.l4_swap_out_packet_num);
//             m_l4_swap_out_packet_num_cpu.push_back(new int[node_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);

//             int *packet_offsets = new int[node_num];
//             int offset = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 packet_offsets[j] = offset;
//                 offset += (m_nic_num_per_node[j + node_offset] *
//                 MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM);
//             }
//             m_packet_offset_per_node.push_back(packet_offsets);
//             cudaMemcpyAsync(cpu_param.l4_swap_out_offset, packet_offsets,
//             sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

//             int packet_size = sizeof(TCPPacket);
//             int src_ip_offset = 38;
//             int dst_ip_offset = 42;
//             int timestamp_offset = 30;
//             cudaMemcpyAsync(cpu_param.l4_packet_size, &packet_size,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.l4_src_ip_offset, &src_ip_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.l4_dst_ip_offset, &dst_ip_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.l4_timestamp_offset, &timestamp_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);

//             cpu_param.max_packet_num = max_packet_num;

//             auto alloc_ipv4_packets =
//             ipv4_packet_pool->allocate(max_packet_num);
//             cudaMemcpyAsync(cpu_param.alloc_ipv4_packets,
//             alloc_ipv4_packets.data(), sizeof(Ipv4Packet *) * max_packet_num,
//             cudaMemcpyHostToDevice, m_streams[i]);
//             m_alloc_ipv4_packets_cpu.push_back(new Ipv4Packet
//             *[max_packet_num]); memcpy(m_alloc_ipv4_packets_cpu[i],
//             alloc_ipv4_packets.data(), sizeof(Ipv4Packet *) *
//             max_packet_num);
//             m_alloc_ipv4_packets_gpu.push_back(cpu_param.alloc_ipv4_packets);

//             cudaMemcpyAsync(cpu_param.packet_offset_per_node, packet_offsets,
//             sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemsetAsync(cpu_param.used_packet_num_per_node, 0,
//             sizeof(int) * node_num, m_streams[i]);

//             IPv4EncapsulationParams *gpu_param;
//             cudaMallocAsync(&gpu_param, sizeof(IPv4EncapsulationParams),
//             m_streams[i]); cudaStreamSynchronize(m_streams[i]);
//             cudaMemcpyAsync(gpu_param, &cpu_param,
//             sizeof(IPv4EncapsulationParams), cudaMemcpyHostToDevice,
//             m_streams[i]); m_kernel_params.push_back(gpu_param);
//         }
//     }

//     void IPv4EncapsulationController::SetIPv4PacketQueue(GPUQueue<Ipv4Packet
//     *> **queues, int node_num)
//     {
//         m_ipv4_packet_queues.insert(m_ipv4_packet_queues.end(), queues,
//         queues + node_num);
//     }

//     void IPv4EncapsulationController::SetL4PacketQueue(GPUQueue<uint8_t *>
//     **queues, int node_num)
//     {
//         m_l4_packet_queues.insert(m_l4_packet_queues.end(), queues, queues +
//         node_num * TransportProtocolType::COUNT_TransportProtocolType);
//     }

//     void IPv4EncapsulationController::SetBatchProperties(int
//     *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(),
//         batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index,
//         batch_end_index + batch_num);
//     }

//     void IPv4EncapsulationController::SetNICNumPerNode(int *nic_num_per_node,
//     int node_num)
//     {
//         m_nic_num_per_node.insert(m_nic_num_per_node.end(), nic_num_per_node,
//         nic_num_per_node + node_num);
//     }

//     void IPv4EncapsulationController::SetStreams(cudaStream_t *streams, int
//     node_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + node_num);
//     }

//     void IPv4EncapsulationController::CacheOutL4Packets(int batch_id)
//     {
//         int node_num = m_batch_start_index[batch_id] -
//         m_batch_end_index[batch_id]; int max_packet_num =
//         m_max_packet_num[batch_id]; int cache_size = m_cache_sizes[batch_id];

//         cudaMemcpyAsync(m_l4_cache_space_cpu[batch_id],
//         m_l4_cache_space_gpu[batch_id], cache_size, cudaMemcpyDeviceToHost,
//         m_streams[batch_id]);
//         cudaMemcpyAsync(m_l4_swap_out_packet_num_cpu[batch_id],
//         m_l4_swap_out_packet_num_gpu[batch_id], sizeof(int) * node_num *
//         TransportProtocolType::COUNT_TransportProtocolType,
//         cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);

//         int *swap_out_packet_num = m_l4_swap_out_packet_num_cpu[batch_id];
//         int *packet_offsets = m_packet_offset_per_node[batch_id];

//         for (int i = 0; i <
//         TransportProtocolType::COUNT_TransportProtocolType; i++)
//         {
//             int packet_size = m_packet_sizes[i];
//             uint8_t **dst_ptrs = m_l4_swap_out_packets_ptr_cpu[batch_id] + i
//             * max_packet_num; uint8_t **src_ptrs =
//             m_l4_cache_space_ptr_cpu[batch_id] + i * max_packet_num;

//             uint8_t **alloc_l4_packets = NULL;
//             if (i == TransportProtocolType::TCP)
//             {
//                 int tcp_packet_num = std::accumulate(swap_out_packet_num,
//                 swap_out_packet_num + node_num, 0); auto l4_packets =
//                 tcp_packet_pool_cpu->allocate(tcp_packet_num);
//                 alloc_l4_packets = (uint8_t **)l4_packets.data();
//             }

//             int allocated_l4_packet_num = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 int offset = packet_offsets[j];
//                 for (int k = 0; k < swap_out_packet_num[j]; k++)
//                 {
//                     int packet_index = offset + k;
//                     memcpy(dst_ptrs[packet_index], src_ptrs[packet_index],
//                     packet_size); src_ptrs[packet_index] =
//                     alloc_l4_packets[allocated_l4_packet_num];
//                     allocated_l4_packet_num++;
//                 }
//             }
//             swap_out_packet_num = swap_out_packet_num + node_num;
//         }
//         cudaMemcpyAsync(m_l4_swap_out_packets_ptr_gpu[batch_id],
//         m_l4_swap_out_packets_ptr_cpu[batch_id], sizeof(uint8_t **) *
//         node_num * TransportProtocolType::COUNT_TransportProtocolType,
//         cudaMemcpyHostToDevice, m_streams[batch_id]);
//     }

//     void IPv4EncapsulationController::UpdateUsedIPv4Packets(int batch_id)
//     {
//         int node_num = m_batch_start_index[batch_id] -
//         m_batch_end_index[batch_id];

//         cudaMemcpyAsync(m_used_packet_num_per_node_cpu[batch_id],
//         m_used_packet_num_per_node_gpu[batch_id], sizeof(int) * node_num,
//         cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);

//         int *used_packet_num_per_node =
//         m_used_packet_num_per_node_cpu[batch_id]; int total_used_packet_num =
//         std::accumulate(used_packet_num_per_node, used_packet_num_per_node +
//         node_num, 0); auto alloc_ipv4_packets =
//         ipv4_packet_pool->allocate(total_used_packet_num);

//         int *packet_offsets = m_packet_offset_per_node[batch_id];
//         Ipv4Packet **alloc_ipv4_packets_cpu =
//         m_alloc_ipv4_packets_cpu[batch_id];

//         int allocated_ipv4_packet_num = 0;
//         for (int i = 0; i < node_num; i++)
//         {
//             int offset = packet_offsets[i];
//             for (int j = 0; j < used_packet_num_per_node[i]; j++)
//             {
//                 alloc_ipv4_packets_cpu[offset + j] =
//                 alloc_ipv4_packets[allocated_ipv4_packet_num];
//                 allocated_ipv4_packet_num++;
//             }
//         }

//         cudaMemcpyAsync(m_alloc_ipv4_packets_gpu[batch_id],
//         m_alloc_ipv4_packets_cpu[batch_id], sizeof(Ipv4Packet **) *
//         total_used_packet_num, cudaMemcpyHostToDevice, m_streams[batch_id]);
//     }

//     void IPv4EncapsulationController::BuildGraph(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id];

//         dim3 block_dim(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);

//         cudaStreamBeginCapture(m_streams[batch_id],
//         cudaStreamCaptureModeGlobal); LaunchIPv4EncapsulationKernel(grid_dim,
//         block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id],
//         NULL, NULL, 0);
//     }

//     void IPv4EncapsulationController::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void IPv4EncapsulationController::Run(int batch_id)
//     {
//         LaunchInstance(batch_id);
//         cudaStreamSynchronize(m_streams[batch_id]);
//         CacheOutL4Packets(batch_id);
//         UpdateUsedIPv4Packets(batch_id);
//         cudaStreamSynchronize(m_streams[batch_id]);
//     }

//     void IPv4EncapsulationController::Run()
//     {
//     }

// }