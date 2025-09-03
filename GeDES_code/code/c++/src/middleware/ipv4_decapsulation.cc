#include "ipv4_decapsulation.h"
#include "component.h"
#include <cstring>
#include <functional>
#include <numeric>

namespace VDES
{
  IPv4DecapsulationController::IPv4DecapsulationController() {}

  IPv4DecapsulationController::~IPv4DecapsulationController() {}

  void IPv4DecapsulationController::InitalizeKernelParams()
  {
    int batch_num = m_batch_start_index.size();
    m_l4_packet_size.push_back(sizeof(TCPPacket));
    m_native_packet_size.push_back(38);

    for (int i = 0; i < batch_num; i++)
    {
      cudaGraph_t graph;
      cudaGraphCreate(&graph, 0);
      m_graphs.push_back(graph);
      m_graph_execs.emplace_back();

      int node_num = m_batch_end_index[i] - m_batch_start_index[i];
      int nic_num =
          std::accumulate(m_nic_num_per_node.begin() + m_batch_start_index[i],
                          m_nic_num_per_node.begin() + m_batch_end_index[i], 0);
      int max_packet_num = nic_num * MAX_TRANSMITTED_PACKET_NUM +
                           node_num * MAX_GENERATED_PACKET_NUM;

      IPv4DecapsulationParam cpu_param;
      cudaMallocAsync(&cpu_param.ipv4_queues,
                      sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);
      cudaMallocAsync(&cpu_param.l4_queues,
                      sizeof(GPUQueue<uint8_t *> *) * node_num *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.recycle_ipv4_packets,
                      sizeof(Ipv4Packet *) * max_packet_num, m_streams[i]);
      cudaMallocAsync(&cpu_param.recycle_offset_per_node, sizeof(int) * node_num,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.recycle_ipv4_packets_num, sizeof(int) * node_num,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.l4_timestamp_offset,
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

#if ENABLE_CACHE
      cudaMallocAsync(&cpu_param.l4_swap_in_packets,
                      sizeof(uint8_t *) * max_packet_num *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.l4_swap_in_offset_per_node,
                      sizeof(int) * node_num, m_streams[i]);
      cudaMallocAsync(&cpu_param.l4_swap_in_packets_num,
                      sizeof(int) * node_num *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);
      cpu_param.cache_packet_num = max_packet_num;
#endif

      cpu_param.node_num = node_num;

      int node_offset = m_batch_start_index[i];
      cudaMemcpyAsync(cpu_param.ipv4_queues, m_ipv4_queues.data() + node_offset,
                      sizeof(GPUQueue<Ipv4Packet *> *) * node_num,
                      cudaMemcpyHostToDevice, m_streams[i]);

      for (int j = 0; j < TransportProtocolType::COUNT_TransportProtocolType;
           j++)
      {
        cudaMemcpyAsync(cpu_param.l4_queues + j * node_num,
                        m_l4_queues.data() + j * m_nic_num_per_node.size() +
                            node_offset,
                        sizeof(GPUQueue<uint8_t *> *) * node_num,
                        cudaMemcpyHostToDevice, m_streams[i]);
      }

      cudaMemsetAsync(cpu_param.recycle_ipv4_packets, 0,
                      sizeof(Ipv4Packet *) * max_packet_num, m_streams[i]);

      int *packet_offsets = new int[node_num];
      int offset = 0;
      for (int i = 0; i < node_num; i++)
      {
        packet_offsets[i] = offset;
        offset += m_nic_num_per_node[i] * MAX_TRANSMITTED_PACKET_NUM +
                  MAX_GENERATED_PACKET_NUM;
      }
      cudaMemcpyAsync(cpu_param.recycle_offset_per_node, packet_offsets,
                      sizeof(int) * node_num, cudaMemcpyHostToDevice,
                      m_streams[i]);
      m_packet_offset_per_node_cpu.push_back(packet_offsets);

      cudaMemsetAsync(cpu_param.recycle_ipv4_packets_num, 0,
                      sizeof(int) * node_num, m_streams[i]);
      int timestamp_offset = 46;
      int src_ip_offset = 38;
      int dst_ip_offset = 42;
      cudaMemcpyAsync(cpu_param.l4_timestamp_offset, &timestamp_offset,
                      sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
      cudaMemcpyAsync(cpu_param.l4_src_ip_offset, &src_ip_offset, sizeof(int),
                      cudaMemcpyHostToDevice, m_streams[i]);
      cudaMemcpyAsync(cpu_param.l4_dst_ip_offset, &dst_ip_offset, sizeof(int),
                      cudaMemcpyHostToDevice, m_streams[i]);

#if ENABLE_CACHE
      uint8_t *cache_space_gpu;
      int cache_space_size = 0;
      for (int j = 0; j < TransportProtocolType::COUNT_TransportProtocolType;
           j++)
      {
        cache_space_size =
            cache_space_size + max_packet_num * m_l4_packet_size[j];
      }
      cudaMallocAsync(&cache_space_gpu, sizeof(uint8_t) * cache_space_size,
                      m_streams[i]);
      cudaStreamSynchronize(m_streams[i]);
      m_cache_space_gpu.push_back(cache_space_gpu);
      uint8_t *cache_space_cpu = new uint8_t[cache_space_size];
      m_cache_space_cpu.push_back(cache_space_cpu);
      m_cache_space_sizes.push_back(cache_space_size);

      std::vector<uint8_t *> swap_in_packets_gpu;
      std::vector<uint8_t *> swap_in_packets_cpu;
      for (int j = 0; j < TransportProtocolType::COUNT_TransportProtocolType;
           j++)
      {

        for (int k = 0; k < max_packet_num; k++)
        {
          uint8_t *gpu_ptr = cache_space_gpu + k * m_l4_packet_size[j];
          uint8_t *cpu_ptr = cache_space_cpu + k * m_l4_packet_size[j];
          swap_in_packets_cpu.push_back(cpu_ptr);
          swap_in_packets_gpu.push_back(gpu_ptr);
        }
        cache_space_gpu = cache_space_gpu + max_packet_num * m_l4_packet_size[j];
        cache_space_cpu = cache_space_cpu + max_packet_num * m_l4_packet_size[j];
      }

      cudaMemcpyAsync(cpu_param.l4_swap_in_packets, swap_in_packets_gpu.data(),
                      sizeof(uint8_t *) * max_packet_num *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      cudaMemcpyHostToDevice, m_streams[i]);
      m_l4_swap_in_packets_gpu.push_back(cpu_param.l4_swap_in_packets);
      m_l4_swap_in_packets_gpu_backup.push_back(
          new uint8_t *[max_packet_num *
                        TransportProtocolType::COUNT_TransportProtocolType]);
      memcpy(m_l4_swap_in_packets_gpu_backup[i], swap_in_packets_gpu.data(),
             sizeof(uint8_t *) * max_packet_num *
                 TransportProtocolType::COUNT_TransportProtocolType);
      m_l4_swap_in_packets_cpu.push_back(
          new uint8_t *[max_packet_num *
                        TransportProtocolType::COUNT_TransportProtocolType]);
      m_l4_swap_in_packets_cpu_backup.push_back(
          new uint8_t *[max_packet_num *
                        TransportProtocolType::COUNT_TransportProtocolType]);
      memcpy(m_l4_swap_in_packets_cpu_backup[i], swap_in_packets_cpu.data(),
             sizeof(uint8_t *) * max_packet_num *
                 TransportProtocolType::COUNT_TransportProtocolType);
      cudaMemcpyAsync(cpu_param.l4_swap_in_offset_per_node, packet_offsets,
                      sizeof(int) * node_num, cudaMemcpyHostToDevice,
                      m_streams[i]);
      cudaMemsetAsync(cpu_param.l4_swap_in_packets_num, 0,
                      sizeof(int) * node_num *
                          TransportProtocolType::COUNT_TransportProtocolType,
                      m_streams[i]);
      cudaStreamSynchronize(m_streams[i]);

      m_l4_swap_in_packet_num_gpu.push_back(cpu_param.l4_swap_in_packets_num);
      m_l4_swap_in_packet_num_cpu.push_back(
          new int[node_num * TransportProtocolType::COUNT_TransportProtocolType]);
#endif

      m_recycle_ipv4_packets_gpu.push_back(cpu_param.recycle_ipv4_packets);
      m_recycle_ipv4_packets_cpu.push_back(new Ipv4Packet *[max_packet_num]);
      m_recycle_ipv4_packets_num_gpu.push_back(
          cpu_param.recycle_ipv4_packets_num);
      m_recycle_ipv4_packets_num_cpu.push_back(new int[node_num]);
      m_max_packet_num_per_batch.push_back(max_packet_num);

      m_recycle_packets_tmp.push_back(new Ipv4Packet *[max_packet_num]);

      IPv4DecapsulationParam *gpu_param;
      cudaMallocAsync(&gpu_param, sizeof(IPv4DecapsulationParam), m_streams[i]);
      cudaMemcpyAsync(gpu_param, &cpu_param, sizeof(IPv4DecapsulationParam),
                      cudaMemcpyHostToDevice, m_streams[i]);
      m_kernel_params.push_back(gpu_param);
    }
  }

  void IPv4DecapsulationController::SetStreams(cudaStream_t *streams,
                                               int node_num)
  {
    m_streams.insert(m_streams.end(), streams, streams + node_num);
  }

  void IPv4DecapsulationController::SetIPv4Queues(
      GPUQueue<Ipv4Packet *> **ipv4_queues, int node_num)
  {
    m_ipv4_queues.insert(m_ipv4_queues.end(), ipv4_queues,
                         ipv4_queues + node_num);
  }

  void IPv4DecapsulationController::SetL4Queues(GPUQueue<uint8_t *> **l4_queues,
                                                int node_num)
  {
    m_l4_queues.insert(
        m_l4_queues.end(), l4_queues,
        l4_queues +
            node_num * TransportProtocolType::COUNT_TransportProtocolType);
  }

  void IPv4DecapsulationController::SetNICNum(int *nic_num_per_node,
                                              int node_num)
  {
    m_nic_num_per_node.insert(m_nic_num_per_node.end(), nic_num_per_node,
                              nic_num_per_node + node_num);
  }

  void IPv4DecapsulationController::SetBatchProperties(int *batch_start_index,
                                                       int *batch_end_index,
                                                       int batch_num)
  {
    m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index,
                               batch_start_index + batch_num);
    m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index,
                             batch_end_index + batch_num);
  }

#if ENABLE_CACHE
  void IPv4DecapsulationController::CacheInL4Packets(int batch_id)
  {
    int max_packet_num = m_max_packet_num_per_batch[batch_id];
    int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

    cudaMemcpyAsync(m_l4_swap_in_packets_cpu[batch_id],
                    m_l4_swap_in_packets_gpu[batch_id],
                    sizeof(uint8_t *) * max_packet_num *
                        TransportProtocolType::COUNT_TransportProtocolType,
                    cudaMemcpyDeviceToHost, m_streams[batch_id]);
    cudaMemcpyAsync(m_l4_swap_in_packet_num_cpu[batch_id],
                    m_l4_swap_in_packet_num_gpu[batch_id],
                    sizeof(int) * node_num *
                        TransportProtocolType::COUNT_TransportProtocolType,
                    cudaMemcpyDeviceToHost, m_streams[batch_id]);
    cudaMemcpyAsync(m_cache_space_cpu[batch_id], m_cache_space_gpu[batch_id],
                    sizeof(uint8_t) * m_cache_space_sizes[batch_id],
                    cudaMemcpyDeviceToHost, m_streams[batch_id]);

    cudaStreamSynchronize(m_streams[batch_id]);

    for (int i = 0; i < TransportProtocolType::COUNT_TransportProtocolType; i++)
    {
      int *swap_in_num = m_l4_swap_in_packet_num_cpu[batch_id] + i * node_num;
      int native_packet_size = m_native_packet_size[i];
      // int packet_size = m_l4_packet_size[i];

      uint8_t **origin_dst =
          m_l4_swap_in_packets_cpu_backup[batch_id] + i * max_packet_num;
      uint8_t **origin_src =
          m_l4_swap_in_packets_cpu[batch_id] + i * max_packet_num;

      std::vector<uint8_t *> recycle_packets;
      for (int j = 0; j < node_num; j++)
      {
        uint8_t **dst = origin_dst + m_packet_offset_per_node_cpu[batch_id][j];
        uint8_t **src = origin_src + m_packet_offset_per_node_cpu[batch_id][j];

        // copy discrete packets to cache space
        for (int k = 0; k < swap_in_num[j]; k++)
        {
          memcpy(dst[k], src[k], native_packet_size);
          recycle_packets.push_back(src[k]);
        }
      }

      if (i == TransportProtocolType::TCP)
      {
        // recyel tcp packets
        tcp_packet_pool_cpu->deallocate((TCPPacket **)recycle_packets.data(),
                                        recycle_packets.size());
      }
    }

    cudaMemcpyAsync(m_cache_space_gpu[batch_id], m_cache_space_cpu[batch_id],
                    sizeof(uint8_t) * m_cache_space_sizes[batch_id],
                    cudaMemcpyHostToDevice, m_streams[batch_id]);
    cudaMemcpyAsync(m_l4_swap_in_packets_gpu[batch_id],
                    m_l4_swap_in_packets_gpu_backup[batch_id],
                    sizeof(uint8_t *) * max_packet_num *
                        TransportProtocolType::COUNT_TransportProtocolType,
                    cudaMemcpyHostToDevice, m_streams[batch_id]);
  }
#endif

  void IPv4DecapsulationController::RecycleIPv4Packets(int batch_id)
  {
    int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
    int max_packet_num = m_max_packet_num_per_batch[batch_id];

#if !ENABLE_HUGE_GRAPH
    cudaMemcpy(m_recycle_ipv4_packets_cpu[batch_id],
               m_recycle_ipv4_packets_gpu[batch_id],
               sizeof(Ipv4Packet *) * max_packet_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_recycle_ipv4_packets_num_cpu[batch_id],
               m_recycle_ipv4_packets_num_gpu[batch_id], sizeof(int) * node_num,
               cudaMemcpyDeviceToHost);
#endif
    int *recycle_num = m_recycle_ipv4_packets_num_cpu[batch_id];
    int *packet_offsets = m_packet_offset_per_node_cpu[batch_id];
    Ipv4Packet **packets = m_recycle_ipv4_packets_cpu[batch_id];

    Ipv4Packet **dst_recycle = m_recycle_packets_tmp[batch_id];
    int offset = 0;
    for (int i = 0; i < node_num; i++)
    {
      Ipv4Packet **src = packets + packet_offsets[i];
      memcpy(dst_recycle + offset, src, sizeof(Ipv4Packet *) * recycle_num[i]);
      offset += recycle_num[i];
    }

    ipv4_packet_pool->deallocate(dst_recycle, offset);
  }

  void IPv4DecapsulationController::BuildGraph(int batch_id)
  {
    int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
    dim3 block_dim(KERNEL_BLOCK_WIDTH);
    dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);
    cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
    LaunchIPv4DecapsulationKernel(grid_dim, block_dim, m_kernel_params[batch_id],
                                  m_streams[batch_id]);
    cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
    cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL,
                         0);

#if ENABLE_HUGE_GRAPH
    int max_packet_num = m_max_packet_num_per_batch[batch_id];

    cudaMemcpy3DParms packet_memcpy_params = {0};
    packet_memcpy_params.srcPtr = make_cudaPitchedPtr(
        m_recycle_ipv4_packets_gpu[batch_id],
        sizeof(Ipv4Packet *) * max_packet_num, max_packet_num, 1);
    packet_memcpy_params.dstPtr = make_cudaPitchedPtr(
        m_recycle_ipv4_packets_cpu[batch_id],
        sizeof(Ipv4Packet *) * max_packet_num, max_packet_num, 1);
    packet_memcpy_params.extent =
        make_cudaExtent(sizeof(Ipv4Packet *) * max_packet_num, 1, 1);
    packet_memcpy_params.kind = cudaMemcpyDeviceToHost;

    cudaMemcpy3DParms packet_num_memcpy_params = {0};
    packet_num_memcpy_params.srcPtr =
        make_cudaPitchedPtr(m_recycle_ipv4_packets_num_gpu[batch_id],
                            sizeof(int) * node_num, node_num, 1);
    packet_num_memcpy_params.dstPtr =
        make_cudaPitchedPtr(m_recycle_ipv4_packets_num_cpu[batch_id],
                            sizeof(int) * node_num, node_num, 1);
    packet_num_memcpy_params.extent =
        make_cudaExtent(sizeof(int) * node_num, 1, 1);
    packet_num_memcpy_params.kind = cudaMemcpyDeviceToHost;

    // cudaGraphNode_t recycle_host_node;
    cudaHostNodeParams recycle_host_params = {0};
    auto recycle_host_func = std::bind(
        &IPv4DecapsulationController::RecycleIPv4Packets, this, batch_id);
    auto recycle_host_func_ptr = new std::function<void()>(recycle_host_func);
    recycle_host_params.fn = VDES::HostNodeCallback;
    recycle_host_params.userData = recycle_host_func_ptr;

    m_memcpy_param.push_back(packet_num_memcpy_params);
    m_memcpy_param.push_back(packet_memcpy_params);
    m_host_param.push_back(recycle_host_params);

#endif
  }

  void IPv4DecapsulationController::BuildGraph()
  {
    int batch_num = m_batch_start_index.size();
    for (int i = 0; i < batch_num; i++)
    {
      BuildGraph(i);
    }
  }

  void IPv4DecapsulationController::LaunchInstance(int batch_id)
  {
    cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
  }

  void IPv4DecapsulationController::Run(int batch_id)
  {
    cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
    cudaStreamSynchronize(m_streams[batch_id]);

#if ENABLE_CACHE
    CacheInL4Packets(batch_id);
#else

    RecycleIPv4Packets(batch_id);
#endif
  }

  void IPv4DecapsulationController::Run() {}

  cudaGraph_t IPv4DecapsulationController::GetGraph(int batch_id)
  {
    return m_graphs[batch_id];
  }

#if ENABLE_HUGE_GRAPH
  std::vector<cudaMemcpy3DParms> &IPv4DecapsulationController::GetMemcpyParams()
  {
    return m_memcpy_param;
  }

  std::vector<cudaHostNodeParams> &IPv4DecapsulationController::GetHostParams()
  {
    return m_host_param;
  }
#endif

  std::vector<void *> IPv4DecapsulationController::GetRecycleInfo()
  {
    std::vector<void *> res;
    int batch_num = m_batch_start_index.size();
    for (int i = 0; i < batch_num; i++)
    {
      res.push_back(m_recycle_ipv4_packets_gpu[i]);
      res.push_back(m_recycle_ipv4_packets_num_gpu[i]);
    }
    return res;
  }

} // namespace VDES

// namespace VDES
// {
//     IPv4DecapsulationController::IPv4DecapsulationController()
//     {
//     }

//     IPv4DecapsulationController::~IPv4DecapsulationController()
//     {
//     }

//     void IPv4DecapsulationController::InitalizeKernelParams()
//     {
//         int batch_num = m_batch_start_index.size();
//         m_l4_packet_size.push_back(sizeof(TCPPacket));
//         /**
//          * TODO: push the native packet into the queue.
//          */
//         m_native_packet_size.push_back(38);
//         // m_l4_packet_size.push_back(sizeof(UDPPacket) - 16);

//         for (int i = 0; i < batch_num; i++)
//         {
//             cudaGraph_t graph;
//             cudaGraphCreate(&graph, 0);
//             m_graphs.push_back(graph);
//             m_graph_execs.emplace_back();

//             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
//             int nic_num = std::accumulate(m_nic_num_per_node.begin() +
//             m_batch_start_index[i], m_nic_num_per_node.begin() +
//             m_batch_end_index[i], 0); int max_packet_num = nic_num *
//             MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;

//             IPv4DecapsulationParam cpu_param;
//             cudaMallocAsync(&cpu_param.ipv4_queues,
//             sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_queues, sizeof(GPUQueue<uint8_t *>
//             *) * node_num *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.recycle_ipv4_packets,
//             sizeof(Ipv4Packet *) * max_packet_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.recycle_offset_per_node, sizeof(int) *
//             node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.recycle_ipv4_packets_num, sizeof(int)
//             * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_timestamp_offset, sizeof(int) *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_src_ip_offset,
//             sizeof(int) * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_dst_ip_offset,
//             sizeof(int) * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]);

// #if ENABLE_CACHE

//             cudaMallocAsync(&cpu_param.l4_swap_in_packets, sizeof(uint8_t *)
//             * max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_swap_in_offset_per_node,
//             sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_swap_in_packets_num, sizeof(int) *
//             node_num * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cpu_param.cache_packet_num = max_packet_num;
// #endif

//             cpu_param.node_num = node_num;

//             int node_offset = m_batch_start_index[i];
//             cudaMemcpyAsync(cpu_param.ipv4_queues, m_ipv4_queues.data() +
//             node_offset, sizeof(GPUQueue<Ipv4Packet *> *) * node_num,
//             cudaMemcpyHostToDevice, m_streams[i]);

//             for (int j = 0; j <
//             TransportProtocolType::COUNT_TransportProtocolType; j++)
//             {
//                 cudaMemcpyAsync(cpu_param.l4_queues + j * node_num,
//                 m_l4_queues.data() + j * m_nic_num_per_node.size() +
//                 node_offset, sizeof(GPUQueue<uint8_t *> *) * node_num,
//                 cudaMemcpyHostToDevice, m_streams[i]);
//             }

//             cudaMemsetAsync(cpu_param.recycle_ipv4_packets, 0,
//             sizeof(Ipv4Packet *) * max_packet_num, m_streams[i]);

//             int *packet_offsets = new int[node_num];
//             int offset = 0;
//             for (int i = 0; i < node_num; i++)
//             {
//                 packet_offsets[i] = offset;
//                 offset += m_nic_num_per_node[i] * MAX_TRANSMITTED_PACKET_NUM
//                 + MAX_GENERATED_PACKET_NUM;
//             }
//             cudaMemcpyAsync(cpu_param.recycle_offset_per_node,
//             packet_offsets, sizeof(int) * node_num, cudaMemcpyHostToDevice,
//             m_streams[i]);
//             m_packet_offset_per_node_cpu.push_back(packet_offsets);

//             cudaMemsetAsync(cpu_param.recycle_ipv4_packets_num, 0,
//             sizeof(int) * node_num, m_streams[i]); int timestamp_offset = 46;
//             int src_ip_offset = 38;
//             int dst_ip_offset = 42;
//             cudaMemcpyAsync(cpu_param.l4_timestamp_offset, &timestamp_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.l4_src_ip_offset, &src_ip_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.l4_dst_ip_offset, &dst_ip_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);

// #if ENABLE_CACHE
//             uint8_t *cache_space_gpu;
//             int cache_space_size = 0;
//             for (int j = 0; j <
//             TransportProtocolType::COUNT_TransportProtocolType; j++)
//             {
//                 cache_space_size = cache_space_size + max_packet_num *
//                 m_l4_packet_size[j];
//             }
//             cudaMallocAsync(&cache_space_gpu, sizeof(uint8_t) *
//             cache_space_size, m_streams[i]);
//             cudaStreamSynchronize(m_streams[i]);
//             m_cache_space_gpu.push_back(cache_space_gpu);
//             uint8_t *cache_space_cpu = new uint8_t[cache_space_size];
//             m_cache_space_cpu.push_back(cache_space_cpu);
//             m_cache_space_sizes.push_back(cache_space_size);

//             std::vector<uint8_t *> swap_in_packets_gpu;
//             std::vector<uint8_t *> swap_in_packets_cpu;
//             for (int j = 0; j <
//             TransportProtocolType::COUNT_TransportProtocolType; j++)
//             {

//                 for (int k = 0; k < max_packet_num; k++)
//                 {
//                     uint8_t *gpu_ptr = cache_space_gpu + k *
//                     m_l4_packet_size[j]; uint8_t *cpu_ptr = cache_space_cpu +
//                     k * m_l4_packet_size[j];
//                     swap_in_packets_cpu.push_back(cpu_ptr);
//                     swap_in_packets_gpu.push_back(gpu_ptr);
//                 }
//                 cache_space_gpu = cache_space_gpu + max_packet_num *
//                 m_l4_packet_size[j]; cache_space_cpu = cache_space_cpu +
//                 max_packet_num * m_l4_packet_size[j];
//             }

//             cudaMemcpyAsync(cpu_param.l4_swap_in_packets,
//             swap_in_packets_gpu.data(), sizeof(uint8_t *) * max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             cudaMemcpyHostToDevice, m_streams[i]);
//             m_l4_swap_in_packets_gpu.push_back(cpu_param.l4_swap_in_packets);
//             m_l4_swap_in_packets_gpu_backup.push_back(new uint8_t
//             *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);
//             memcpy(m_l4_swap_in_packets_gpu_backup[i],
//             swap_in_packets_gpu.data(), sizeof(uint8_t *) * max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType);
//             m_l4_swap_in_packets_cpu.push_back(new uint8_t *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);
//             m_l4_swap_in_packets_cpu_backup.push_back(new uint8_t
//             *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);
//             memcpy(m_l4_swap_in_packets_cpu_backup[i],
//             swap_in_packets_cpu.data(), sizeof(uint8_t *) * max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType);
//             cudaMemcpyAsync(cpu_param.l4_swap_in_offset_per_node,
//             packet_offsets, sizeof(int) * node_num, cudaMemcpyHostToDevice,
//             m_streams[i]); cudaMemsetAsync(cpu_param.l4_swap_in_packets_num,
//             0, sizeof(int) * node_num *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaStreamSynchronize(m_streams[i]);
//             m_l4_swap_in_packet_num_gpu.push_back(cpu_param.l4_swap_in_packets_num);
//             m_l4_swap_in_packet_num_cpu.push_back(new int[node_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);
// #endif

//             m_recycle_ipv4_packets_gpu.push_back(cpu_param.recycle_ipv4_packets);
//             m_recycle_ipv4_packets_cpu.push_back(new Ipv4Packet
//             *[max_packet_num]);
//             m_recycle_ipv4_packets_num_gpu.push_back(cpu_param.recycle_ipv4_packets_num);
//             m_recycle_ipv4_packets_num_cpu.push_back(new int[node_num]);
//             m_max_packet_num_per_batch.push_back(max_packet_num);

//             m_recycle_packets_tmp.push_back(new Ipv4Packet
//             *[max_packet_num]);

//             IPv4DecapsulationParam *gpu_param;
//             cudaMallocAsync(&gpu_param, sizeof(IPv4DecapsulationParam),
//             m_streams[i]); cudaMemcpyAsync(gpu_param, &cpu_param,
//             sizeof(IPv4DecapsulationParam), cudaMemcpyHostToDevice,
//             m_streams[i]); m_kernel_params.push_back(gpu_param);
//         }
//     }

//     void IPv4DecapsulationController::SetStreams(cudaStream_t *streams, int
//     node_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + node_num);
//     }

//     void IPv4DecapsulationController::SetIPv4Queues(GPUQueue<Ipv4Packet *>
//     **ipv4_queues, int node_num)
//     {
//         m_ipv4_queues.insert(m_ipv4_queues.end(), ipv4_queues, ipv4_queues +
//         node_num);
//     }

//     void IPv4DecapsulationController::SetL4Queues(GPUQueue<uint8_t *>
//     **l4_queues, int node_num)
//     {
//         m_l4_queues.insert(m_l4_queues.end(), l4_queues, l4_queues + node_num
//         * TransportProtocolType::COUNT_TransportProtocolType);
//     }

//     void IPv4DecapsulationController::SetNICNum(int *nic_num_per_node, int
//     node_num)
//     {
//         m_nic_num_per_node.insert(m_nic_num_per_node.end(), nic_num_per_node,
//         nic_num_per_node + node_num);
//     }

//     void IPv4DecapsulationController::SetBatchProperties(int
//     *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(),
//         batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index,
//         batch_end_index + batch_num);
//     }

// #if ENABLE_CACHE
//     void IPv4DecapsulationController::CacheInL4Packets(int batch_id)
//     {
//         int max_packet_num = m_max_packet_num_per_batch[batch_id];
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id];

//         cudaMemcpyAsync(m_l4_swap_in_packets_cpu[batch_id],
//         m_l4_swap_in_packets_gpu[batch_id], sizeof(uint8_t *) *
//         max_packet_num * TransportProtocolType::COUNT_TransportProtocolType,
//         cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_l4_swap_in_packet_num_cpu[batch_id],
//         m_l4_swap_in_packet_num_gpu[batch_id], sizeof(int) * node_num *
//         TransportProtocolType::COUNT_TransportProtocolType,
//         cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_cache_space_cpu[batch_id],
//         m_cache_space_gpu[batch_id], sizeof(uint8_t) *
//         m_cache_space_sizes[batch_id], cudaMemcpyDeviceToHost,
//         m_streams[batch_id]);

//         cudaStreamSynchronize(m_streams[batch_id]);

//         for (int i = 0; i <
//         TransportProtocolType::COUNT_TransportProtocolType; i++)
//         {
//             int *swap_in_num = m_l4_swap_in_packet_num_cpu[batch_id] + i *
//             node_num; int native_packet_size = m_native_packet_size[i];
//             // int packet_size = m_l4_packet_size[i];

//             uint8_t **origin_dst = m_l4_swap_in_packets_cpu_backup[batch_id]
//             + i * max_packet_num; uint8_t **origin_src =
//             m_l4_swap_in_packets_cpu[batch_id] + i * max_packet_num;

//             std::vector<uint8_t *> recycle_packets;
//             for (int j = 0; j < node_num; j++)
//             {
//                 uint8_t **dst = origin_dst +
//                 m_packet_offset_per_node_cpu[batch_id][j]; uint8_t **src =
//                 origin_src + m_packet_offset_per_node_cpu[batch_id][j];

//                 // copy discrete packets to cache space
//                 for (int k = 0; k < swap_in_num[j]; k++)
//                 {
//                     memcpy(dst[k], src[k], native_packet_size);
//                     recycle_packets.push_back(src[k]);
//                 }
//             }

//             if (i == TransportProtocolType::TCP)
//             {
//                 // recyel tcp packets
//                 tcp_packet_pool_cpu->deallocate((TCPPacket
//                 **)recycle_packets.data(), recycle_packets.size());
//             }
//         }

//         cudaMemcpyAsync(m_cache_space_gpu[batch_id],
//         m_cache_space_cpu[batch_id], sizeof(uint8_t) *
//         m_cache_space_sizes[batch_id], cudaMemcpyHostToDevice,
//         m_streams[batch_id]);
//         cudaMemcpyAsync(m_l4_swap_in_packets_gpu[batch_id],
//         m_l4_swap_in_packets_gpu_backup[batch_id], sizeof(uint8_t *) *
//         max_packet_num * TransportProtocolType::COUNT_TransportProtocolType,
//         cudaMemcpyHostToDevice, m_streams[batch_id]);
//     }

// #endif

//     void IPv4DecapsulationController::RecycleIPv4Packets(int batch_id)
//     {
//         // LOG_INFO("size of m_recycle_ipv4_packets_cpu: %ld batch_id: %d",
//         m_recycle_ipv4_packets_num_cpu.size(), batch_id); int node_num =
//         m_batch_end_index[batch_id] - m_batch_start_index[batch_id]; int
//         max_packet_num = m_max_packet_num_per_batch[batch_id];

// #if !ENABLE_HUGE_GRAPH
//         cudaMemcpyAsync(m_recycle_ipv4_packets_cpu[batch_id],
//         m_recycle_ipv4_packets_gpu[batch_id], sizeof(Ipv4Packet *) *
//         max_packet_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_recycle_ipv4_packets_num_cpu[batch_id],
//         m_recycle_ipv4_packets_num_gpu[batch_id], sizeof(int) * node_num,
//         cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);
// #endif

//         int *recycle_num = m_recycle_ipv4_packets_num_cpu[batch_id];
//         int *packet_offsets = m_packet_offset_per_node_cpu[batch_id];
//         Ipv4Packet **packets = m_recycle_ipv4_packets_cpu[batch_id];

//         // std::vector<Ipv4Packet *> recycle_packets;
//         Ipv4Packet **dst_recycle = m_recycle_packets_tmp[batch_id];
//         int offset = 0;
//         for (int i = 0; i < node_num; i++)
//         {
//             Ipv4Packet **src = packets + packet_offsets[i];
//             // recycle_packets.insert(recycle_packets.end(), src, src +
//             recycle_num[i]); memcpy(dst_recycle + offset, src,
//             sizeof(Ipv4Packet *) * recycle_num[i]); offset += recycle_num[i];
//         }

//         ipv4_packet_pool->deallocate(dst_recycle, offset);
//     }

//     void IPv4DecapsulationController::BuildGraph(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id]; dim3 block_dim(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);
//         cudaStreamBeginCapture(m_streams[batch_id],
//         cudaStreamCaptureModeGlobal); LaunchIPv4DecapsulationKernel(grid_dim,
//         block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id],
//         NULL, NULL, 0);

// #if ENABLE_HUGE_GRAPH

//         // cudaGraphNode_t kernel_node;
//         // size_t num_nodes;
//         // cudaGraphGetNodes(m_graphs[batch_id], &kernel_node, &num_nodes);
//         int max_packet_num = m_max_packet_num_per_batch[batch_id];

//         // cudaGraphNode_t packet_memcpy_node;
//         cudaMemcpy3DParms packet_memcpy_params = {0};
//         packet_memcpy_params.srcPtr =
//         make_cudaPitchedPtr(m_recycle_ipv4_packets_gpu[batch_id],
//         sizeof(Ipv4Packet *) * max_packet_num, max_packet_num, 1);
//         packet_memcpy_params.dstPtr =
//         make_cudaPitchedPtr(m_recycle_ipv4_packets_cpu[batch_id],
//         sizeof(Ipv4Packet *) * max_packet_num, max_packet_num, 1);
//         packet_memcpy_params.extent = make_cudaExtent(sizeof(Ipv4Packet *) *
//         max_packet_num, 1, 1); packet_memcpy_params.kind =
//         cudaMemcpyDeviceToHost;
//         // cudaGraphAddMemcpyNode(&packet_memcpy_node, m_graphs[batch_id],
//         &kernel_node, 1, &packet_memcpy_params);

//         // cudaGraphNode_t packet_num_memcpy_node;
//         cudaMemcpy3DParms packet_num_memcpy_params = {0};
//         packet_num_memcpy_params.srcPtr =
//         make_cudaPitchedPtr(m_recycle_ipv4_packets_num_gpu[batch_id],
//         sizeof(int) * node_num, node_num, 1); packet_num_memcpy_params.dstPtr
//         = make_cudaPitchedPtr(m_recycle_ipv4_packets_num_cpu[batch_id],
//         sizeof(int) * node_num, node_num, 1); packet_num_memcpy_params.extent
//         = make_cudaExtent(sizeof(int) * node_num, 1, 1);
//         packet_num_memcpy_params.kind = cudaMemcpyDeviceToHost;
//         // cudaGraphAddMemcpyNode(&packet_num_memcpy_node,
//         m_graphs[batch_id], &kernel_node, 1, &packet_num_memcpy_params);

//         // std::vector<cudaGraphNode_t> memcpy_nodes;
//         // memcpy_nodes.push_back(packet_memcpy_node);
//         // memcpy_nodes.push_back(packet_num_memcpy_node);

//         // cudaGraphNode_t recycle_host_node;
//         cudaHostNodeParams recycle_host_params = {0};
//         auto recycle_host_func =
//         std::bind(&IPv4DecapsulationController::RecycleIPv4Packets, this,
//         batch_id); auto recycle_host_func_ptr = new
//         std::function<void()>(recycle_host_func); recycle_host_params.fn =
//         VDES::HostNodeCallback; recycle_host_params.userData =
//         recycle_host_func_ptr;
//         // cudaGraphAddHostNode(&recycle_host_node, m_graphs[batch_id],
//         memcpy_nodes.data(), memcpy_nodes.size(), &recycle_host_params);

//         m_memcpy_param.push_back(packet_num_memcpy_params);
//         m_memcpy_param.push_back(packet_memcpy_params);
//         m_host_param.push_back(recycle_host_params);

// #endif
//     }

//     void IPv4DecapsulationController::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void IPv4DecapsulationController::Run(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);
// #if ENABLE_CACHE

//         CacheInL4Packets(batch_id);
// #else
//         RecycleIPv4Packets(batch_id);
// #endif
//     }

//     void IPv4DecapsulationController::Run()
//     {
//     }

//     void IPv4DecapsulationController::BuildGraph()
//     {
//         int batch_num = m_batch_start_index.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraph(i);
//         }
//     }

//     cudaGraph_t IPv4DecapsulationController::GetGraph(int batch_id)
//     {
//         return m_graphs[batch_id];
//     }

// #if ENABLE_HUGE_GRAPH
//     std::vector<cudaMemcpy3DParms>
//     &IPv4DecapsulationController::GetMemcpyParams()
//     {
//         return m_memcpy_param;
//     }

//     std::vector<cudaHostNodeParams>
//     &IPv4DecapsulationController::GetHostParams()
//     {
//         return m_host_param;
//     }
// #endif

// }

// #include "ipv4_decapsulation.h"
// #include <numeric>
// #include <cstring>

// namespace VDES
// {
//     IPv4DecapsulationController::IPv4DecapsulationController()
//     {
//     }

//     IPv4DecapsulationController::~IPv4DecapsulationController()
//     {
//     }

//     void IPv4DecapsulationController::InitalizeKernelParams()
//     {
//         int batch_num = m_batch_start_index.size();
//         m_l4_packet_size.push_back(sizeof(TCPPacket));
//         /**
//          * TODO: push the native packet into the queue.
//          */
//         m_native_packet_size.push_back(sizeof(TCPPacket) - 16);
//         // m_l4_packet_size.push_back(sizeof(UDPPacket) - 16);

//         for (int i = 0; i < batch_num; i++)
//         {
//             cudaGraph_t graph;
//             cudaGraphCreate(&graph, 0);
//             m_graphs.push_back(graph);
//             m_graph_execs.emplace_back();

//             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
//             int nic_num = std::accumulate(m_nic_num_per_node.begin() +
//             m_batch_start_index[i], m_nic_num_per_node.begin() +
//             m_batch_end_index[i], 0); int max_packet_num = nic_num *
//             MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;

//             IPv4DecapsulationParam cpu_param;
//             cudaMallocAsync(&cpu_param.ipv4_queues,
//             sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_queues, sizeof(GPUQueue<uint8_t *>
//             *) * node_num *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.recycle_ipv4_packets,
//             sizeof(Ipv4Packet *) * max_packet_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.recycle_offset_per_node, sizeof(int) *
//             node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.recycle_ipv4_packets_num, sizeof(int)
//             * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_timestamp_offset, sizeof(int) *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_src_ip_offset,
//             sizeof(int) * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_dst_ip_offset,
//             sizeof(int) * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaMallocAsync(&cpu_param.l4_swap_in_packets,
//             sizeof(uint8_t *) * max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_swap_in_offset_per_node,
//             sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.l4_swap_in_packets_num, sizeof(int) *
//             node_num * TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cpu_param.cache_packet_num = max_packet_num;
//             cpu_param.node_num = node_num;

//             int node_offset = m_batch_start_index[i];
//             cudaMemcpyAsync(cpu_param.ipv4_queues, m_ipv4_queues.data() +
//             node_offset, sizeof(GPUQueue<Ipv4Packet *> *) * node_num,
//             cudaMemcpyHostToDevice, m_streams[i]);

//             for (int j = 0; j <
//             TransportProtocolType::COUNT_TransportProtocolType; j++)
//             {
//                 cudaMemcpyAsync(cpu_param.l4_queues + j * node_num,
//                 m_l4_queues.data() + j * m_nic_num_per_node.size() +
//                 node_offset, sizeof(GPUQueue<uint8_t *> *) * node_num,
//                 cudaMemcpyHostToDevice, m_streams[i]);
//             }

//             cudaMemsetAsync(cpu_param.recycle_ipv4_packets, 0,
//             sizeof(Ipv4Packet *) * max_packet_num, m_streams[i]);

//             int *packet_offsets = new int[node_num];
//             int offset = 0;
//             for (int i = 0; i < node_num; i++)
//             {
//                 packet_offsets[i] = offset;
//                 offset += m_nic_num_per_node[i] * MAX_TRANSMITTED_PACKET_NUM
//                 + MAX_GENERATED_PACKET_NUM;
//             }
//             cudaMemcpyAsync(cpu_param.recycle_offset_per_node,
//             packet_offsets, sizeof(int) * node_num, cudaMemcpyHostToDevice,
//             m_streams[i]);
//             m_packet_offset_per_node_cpu.push_back(packet_offsets);

//             cudaMemsetAsync(cpu_param.recycle_ipv4_packets_num, 0,
//             sizeof(int) * node_num, m_streams[i]); int timestamp_offset = 46;
//             int src_ip_offset = 38;
//             int dst_ip_offset = 42;
//             cudaMemcpyAsync(cpu_param.l4_timestamp_offset, &timestamp_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.l4_src_ip_offset, &src_ip_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.l4_dst_ip_offset, &dst_ip_offset,
//             sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);

//             uint8_t *cache_space_gpu;
//             int cache_space_size = 0;
//             for (int j = 0; j <
//             TransportProtocolType::COUNT_TransportProtocolType; j++)
//             {
//                 cache_space_size = cache_space_size + max_packet_num *
//                 m_l4_packet_size[j];
//             }
//             cudaMallocAsync(&cache_space_gpu, sizeof(uint8_t) *
//             cache_space_size, m_streams[i]);
//             cudaStreamSynchronize(m_streams[i]);
//             m_cache_space_gpu.push_back(cache_space_gpu);
//             uint8_t *cache_space_cpu = new uint8_t[cache_space_size];
//             m_cache_space_cpu.push_back(cache_space_cpu);
//             m_cache_space_sizes.push_back(cache_space_size);

//             std::vector<uint8_t *> swap_in_packets_gpu;
//             std::vector<uint8_t *> swap_in_packets_cpu;
//             for (int j = 0; j <
//             TransportProtocolType::COUNT_TransportProtocolType; j++)
//             {

//                 for (int k = 0; k < max_packet_num; k++)
//                 {
//                     uint8_t *gpu_ptr = cache_space_gpu + k *
//                     m_l4_packet_size[j]; uint8_t *cpu_ptr = cache_space_cpu +
//                     k * m_l4_packet_size[j];
//                     swap_in_packets_cpu.push_back(cpu_ptr);
//                     swap_in_packets_gpu.push_back(gpu_ptr);
//                 }
//                 cache_space_gpu = cache_space_gpu + max_packet_num *
//                 m_l4_packet_size[j]; cache_space_cpu = cache_space_cpu +
//                 max_packet_num * m_l4_packet_size[j];
//             }

//             cudaMemcpyAsync(cpu_param.l4_swap_in_packets,
//             swap_in_packets_gpu.data(), sizeof(uint8_t *) * max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             cudaMemcpyHostToDevice, m_streams[i]);
//             m_l4_swap_in_packets_gpu.push_back(cpu_param.l4_swap_in_packets);
//             m_l4_swap_in_packets_gpu_backup.push_back(new uint8_t
//             *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);
//             memcpy(m_l4_swap_in_packets_gpu_backup[i],
//             swap_in_packets_gpu.data(), sizeof(uint8_t *) * max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType);
//             m_l4_swap_in_packets_cpu.push_back(new uint8_t *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);
//             m_l4_swap_in_packets_cpu_backup.push_back(new uint8_t
//             *[max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);
//             memcpy(m_l4_swap_in_packets_cpu_backup[i],
//             swap_in_packets_cpu.data(), sizeof(uint8_t *) * max_packet_num *
//             TransportProtocolType::COUNT_TransportProtocolType);
//             cudaMemcpyAsync(cpu_param.l4_swap_in_offset_per_node,
//             packet_offsets, sizeof(int) * node_num, cudaMemcpyHostToDevice,
//             m_streams[i]); cudaMemsetAsync(cpu_param.l4_swap_in_packets_num,
//             0, sizeof(int) * node_num *
//             TransportProtocolType::COUNT_TransportProtocolType,
//             m_streams[i]); cudaStreamSynchronize(m_streams[i]);

//             m_recycle_ipv4_packets_gpu.push_back(cpu_param.recycle_ipv4_packets);
//             m_recycle_ipv4_packets_cpu.push_back(new Ipv4Packet
//             *[max_packet_num]);
//             m_recycle_ipv4_packets_num_gpu.push_back(cpu_param.recycle_ipv4_packets_num);
//             m_recycle_ipv4_packets_num_cpu.push_back(new int[node_num]);
//             m_l4_swap_in_packet_num_gpu.push_back(cpu_param.l4_swap_in_packets_num);
//             m_l4_swap_in_packet_num_cpu.push_back(new int[node_num *
//             TransportProtocolType::COUNT_TransportProtocolType]);
//             m_max_packet_num_per_batch.push_back(max_packet_num);

//             IPv4DecapsulationParam *gpu_param;
//             cudaMallocAsync(&gpu_param, sizeof(IPv4DecapsulationParam),
//             m_streams[i]); cudaMemcpyAsync(gpu_param, &cpu_param,
//             sizeof(IPv4DecapsulationParam), cudaMemcpyHostToDevice,
//             m_streams[i]); m_kernel_params.push_back(gpu_param);
//         }
//     }

//     void IPv4DecapsulationController::SetStreams(cudaStream_t *streams, int
//     node_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + node_num);
//     }

//     void IPv4DecapsulationController::SetIPv4Queues(GPUQueue<Ipv4Packet *>
//     **ipv4_queues, int node_num)
//     {
//         m_ipv4_queues.insert(m_ipv4_queues.end(), ipv4_queues, ipv4_queues +
//         node_num);
//     }

//     void IPv4DecapsulationController::SetL4Queues(GPUQueue<uint8_t *>
//     **l4_queues, int node_num)
//     {
//         m_l4_queues.insert(m_l4_queues.end(), l4_queues, l4_queues + node_num
//         * TransportProtocolType::COUNT_TransportProtocolType);
//     }

//     void IPv4DecapsulationController::SetNICNum(int *nic_num_per_node, int
//     node_num)
//     {
//         m_nic_num_per_node.insert(m_nic_num_per_node.end(), nic_num_per_node,
//         nic_num_per_node + node_num);
//     }

//     void IPv4DecapsulationController::SetBatchProperties(int
//     *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(),
//         batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index,
//         batch_end_index + batch_num);
//     }

//     void IPv4DecapsulationController::CacheInL4Packets(int batch_id)
//     {
//         int max_packet_num = m_max_packet_num_per_batch[batch_id];
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id];

//         cudaMemcpyAsync(m_l4_swap_in_packets_cpu[batch_id],
//         m_l4_swap_in_packets_gpu[batch_id], sizeof(uint8_t *) *
//         max_packet_num * TransportProtocolType::COUNT_TransportProtocolType,
//         cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_l4_swap_in_packet_num_cpu[batch_id],
//         m_l4_swap_in_packet_num_gpu[batch_id], sizeof(int) * node_num *
//         TransportProtocolType::COUNT_TransportProtocolType,
//         cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_cache_space_cpu[batch_id],
//         m_cache_space_gpu[batch_id], sizeof(uint8_t) *
//         m_cache_space_sizes[batch_id], cudaMemcpyDeviceToHost,
//         m_streams[batch_id]);

//         cudaStreamSynchronize(m_streams[batch_id]);

//         for (int i = 0; i <
//         TransportProtocolType::COUNT_TransportProtocolType; i++)
//         {
//             int *swap_in_num = m_l4_swap_in_packet_num_cpu[batch_id] + i *
//             node_num; int native_packet_size = m_native_packet_size[i];

//             uint8_t **origin_dst = m_l4_swap_in_packets_cpu_backup[batch_id]
//             + i * max_packet_num; uint8_t **origin_src =
//             m_l4_swap_in_packets_cpu[batch_id] + i * max_packet_num;

//             std::vector<uint8_t *> recycle_packets;
//             for (int j = 0; j < node_num; j++)
//             {
//                 uint8_t **dst = origin_dst +
//                 m_packet_offset_per_node_cpu[batch_id][j]; uint8_t **src =
//                 origin_src + m_packet_offset_per_node_cpu[batch_id][j];

//                 // copy discrete packets to cache space
//                 for (int k = 0; k < swap_in_num[j]; k++)
//                 {
//                     memcpy(dst[k], src[k], native_packet_size);
//                     recycle_packets.push_back(src[k]);
//                 }
//             }

//             if (i == TransportProtocolType::TCP)
//             {
//                 // recyel tcp packets
//                 tcp_packet_pool_cpu->deallocate((TCPPacket
//                 **)recycle_packets.data(), recycle_packets.size());
//             }
//         }

//         cudaMemcpyAsync(m_cache_space_gpu[batch_id],
//         m_cache_space_cpu[batch_id], sizeof(uint8_t) *
//         m_cache_space_sizes[batch_id], cudaMemcpyHostToDevice,
//         m_streams[batch_id]);
//         cudaMemcpyAsync(m_l4_swap_in_packets_gpu[batch_id],
//         m_l4_swap_in_packets_gpu_backup[batch_id], sizeof(uint8_t *) *
//         max_packet_num * TransportProtocolType::COUNT_TransportProtocolType,
//         cudaMemcpyHostToDevice, m_streams[batch_id]);
//     }

//     void IPv4DecapsulationController::RecycleIPv4Packets(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id]; int max_packet_num =
//         m_max_packet_num_per_batch[batch_id];

//         cudaMemcpyAsync(m_recycle_ipv4_packets_cpu[batch_id],
//         m_recycle_ipv4_packets_gpu[batch_id], sizeof(Ipv4Packet *) *
//         max_packet_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_recycle_ipv4_packets_num_cpu[batch_id],
//         m_recycle_ipv4_packets_num_gpu[batch_id], sizeof(int) * node_num,
//         cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);

//         int *recycle_num = m_recycle_ipv4_packets_num_cpu[batch_id];
//         int *packet_offsets = m_packet_offset_per_node_cpu[batch_id];
//         Ipv4Packet **packets = m_recycle_ipv4_packets_cpu[batch_id];

//         std::vector<Ipv4Packet *> recycle_packets;
//         for (int i = 0; i < node_num; i++)
//         {
//             Ipv4Packet **src = packets + packet_offsets[i];
//             recycle_packets.insert(recycle_packets.end(), src, src +
//             recycle_num[i]);
//         }

//         ipv4_packet_pool_cpu->deallocate(recycle_packets.data(),
//         recycle_packets.size());
//     }

//     void IPv4DecapsulationController::BuildGraph(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id]; dim3 block_dim(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);
//         cudaStreamBeginCapture(m_streams[batch_id],
//         cudaStreamCaptureModeGlobal); LaunchIPv4DecapsulationKernel(grid_dim,
//         block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id],
//         NULL, NULL, 0);
//     }

//     void IPv4DecapsulationController::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void IPv4DecapsulationController::Run(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);
//         CacheInL4Packets(batch_id);
//         RecycleIPv4Packets(batch_id);
//     }

//     void IPv4DecapsulationController::BuildGraph()
//     {
//         int batch_num = m_batch_start_index.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraph(i);
//         }
//     }

//     void IPv4DecapsulationController::Run()
//     {
//     }
// }

// // #include "ipv4_decapsulation.h"
// // #include <numeric>
// // #include <cstring>

// // namespace VDES
// // {
// //     IPv4DecapsulationController::IPv4DecapsulationController()
// //     {
// //     }

// //     IPv4DecapsulationController::~IPv4DecapsulationController()
// //     {
// //         int batch_num = m_batch_start_index.size();
// //         m_l4_packet_size.push_back(sizeof(TCPPacket));
// //         m_l4_packet_size.push_back(sizeof(UDPPacket) - 16);

// //         for (int i = 0; i < batch_num; i++)
// //         {
// //             cudaGraph_t graph;
// //             cudaGraphCreate(&graph, 0);
// //             m_graphs.push_back(graph);

// //             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
// //             int nic_num = std::accumulate(m_nic_num_per_node.begin() +
// m_batch_start_index[i], m_nic_num_per_node.begin() + m_batch_end_index[i],
// 0);
// //             int max_packet_num = nic_num * MAX_TRANSMITTED_PACKET_NUM +
// node_num * MAX_GENERATED_PACKET_NUM;

// //             IPv4DecapsulationParam cpu_param;
// //             cudaMallocAsync(&cpu_param.ipv4_queues,
// sizeof(GPUQueue<Ipv4Packet *> *) * node_num, m_streams[i]);
// //             cudaMallocAsync(&cpu_param.l4_queues, sizeof(GPUQueue<uint8_t
// *> *) * node_num * TransportProtocolType::COUNT_TransportProtocolType,
// m_streams[i]);
// //             cudaMallocAsync(&cpu_param.recycle_ipv4_packets,
// sizeof(Ipv4Packet *) * max_packet_num, m_streams[i]);
// //             cudaMallocAsync(&cpu_param.recycle_offset_per_node,
// sizeof(int) * node_num, m_streams[i]);
// //             cudaMallocAsync(&cpu_param.recycle_ipv4_packets_num,
// sizeof(int) * node_num, m_streams[i]);
// //             cudaMallocAsync(&cpu_param.l4_timestamp_offset, sizeof(int) *
// TransportProtocolType::COUNT_TransportProtocolType, m_streams[i]);
// //             cudaMallocAsync(&cpu_param.l4_src_ip_offset, sizeof(int) *
// TransportProtocolType::COUNT_TransportProtocolType, m_streams[i]);
// //             cudaMallocAsync(&cpu_param.l4_dst_ip_offset, sizeof(int) *
// TransportProtocolType::COUNT_TransportProtocolType, m_streams[i]);
// //             cudaMallocAsync(&cpu_param.l4_swap_in_packets, sizeof(uint8_t
// *) * max_packet_num * TransportProtocolType::COUNT_TransportProtocolType,
// m_streams[i]);
// //             cudaMallocAsync(&cpu_param.l4_swap_in_offset_per_node,
// sizeof(int) * node_num, m_streams[i]);
// //             cudaMallocAsync(&cpu_param.l4_swap_in_packets_num, sizeof(int)
// * node_num * TransportProtocolType::COUNT_TransportProtocolType,
// m_streams[i]);
// //             cpu_param.cache_packet_num = max_packet_num;
// //             cpu_param.node_num = node_num;

// //             int node_offset = m_batch_start_index[i];
// //             cudaMemcpyAsync(cpu_param.ipv4_queues, m_ipv4_queues.data() +
// node_offset, sizeof(GPUQueue<Ipv4Packet *> *) * node_num,
// cudaMemcpyHostToDevice, m_streams[i]);

// //             for (int j = 0; j <
// TransportProtocolType::COUNT_TransportProtocolType; j++)
// //             {
// //                 cudaMemcpyAsync(cpu_param.l4_queues + i * node_num,
// m_l4_queues.data() + i * node_num + node_offset, sizeof(GPUQueue<uint8_t *>
// *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
// //             }

// //             cudaMemsetAsync(cpu_param.recycle_ipv4_packets, 0,
// sizeof(Ipv4Packet *) * max_packet_num, m_streams[i]);

// //             int *packet_offsets = new int[node_num];
// //             int offset = 0;
// //             for (int i = 0; i < node_num; i++)
// //             {
// //                 packet_offsets[i] = offset;
// //                 offset += m_nic_num_per_node[i] *
// MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM;
// //             }
// //             cudaMemcpyAsync(cpu_param.recycle_offset_per_node,
// packet_offsets, sizeof(int) * node_num, cudaMemcpyHostToDevice,
// m_streams[i]);
// //             m_packet_offset_per_node_cpu.push_back(packet_offsets);

// //             cudaMemsetAsync(cpu_param.recycle_ipv4_packets_num, 0,
// sizeof(int) * node_num, m_streams[i]);
// //             int timestamp_offset = 46;
// //             int src_ip_offset = 38;
// //             int dst_ip_offset = 42;
// //             cudaMemcpyAsync(cpu_param.l4_timestamp_offset,
// &timestamp_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
// //             cudaMemcpyAsync(cpu_param.l4_src_ip_offset, &src_ip_offset,
// sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
// //             cudaMemcpyAsync(cpu_param.l4_dst_ip_offset, &dst_ip_offset,
// sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);

// //             uint8_t *cache_space_gpu;
// //             int cache_space_size = 0;
// //             for (int j = 0; j <
// TransportProtocolType::COUNT_TransportProtocolType; j++)
// //             {
// //                 cache_space_size = cache_space_size + max_packet_num *
// m_l4_packet_size[j];
// //             }
// //             cudaMallocAsync(&cache_space_gpu, sizeof(uint8_t) *
// cache_space_size, m_streams[i]);
// //             cudaStreamSynchronize(m_streams[i]);
// //             m_cache_space_gpu.push_back(cache_space_gpu);
// //             uint8_t *cache_space_cpu = new uint8_t[cache_space_size];
// //             m_cache_space_cpu.push_back(cache_space_cpu);
// //             m_cache_space_sizes.push_back(cache_space_size);

// //             std::vector<uint8_t *> swap_in_packets_gpu;
// //             std::vector<uint8_t *> swap_in_packets_cpu;
// //             for (int j = 0; j <
// TransportProtocolType::COUNT_TransportProtocolType; j++)
// //             {

// //                 for (int k = 0; k < max_packet_num; k++)
// //                 {
// //                     uint8_t *gpu_ptr = cache_space_gpu + k *
// m_l4_packet_size[j];
// //                     uint8_t *cpu_ptr = cache_space_cpu + k *
// m_l4_packet_size[j];
// //                     swap_in_packets_cpu.push_back(cpu_ptr);
// //                     swap_in_packets_gpu.push_back(gpu_ptr);
// //                 }
// //                 cache_space_gpu = cache_space_gpu + max_packet_num *
// m_l4_packet_size[j];
// //                 cache_space_cpu = cache_space_cpu + max_packet_num *
// m_l4_packet_size[j];
// //             }

// //             cudaMemcpyAsync(cpu_param.l4_swap_in_packets,
// swap_in_packets_gpu.data(), sizeof(uint8_t *) * max_packet_num *
// TransportProtocolType::COUNT_TransportProtocolType, cudaMemcpyHostToDevice,
// m_streams[i]);
// // m_l4_swap_in_packets_gpu.push_back(cpu_param.l4_swap_in_packets);
// //             m_l4_swap_in_packets_gpu_backup.push_back(new uint8_t
// *[max_packet_num * TransportProtocolType::COUNT_TransportProtocolType]);
// //             memcpy(m_l4_swap_in_packets_gpu_backup[i],
// swap_in_packets_gpu.data(), sizeof(uint8_t *) * max_packet_num *
// TransportProtocolType::COUNT_TransportProtocolType);
// //             m_l4_swap_in_packets_cpu.push_back(new uint8_t
// *[max_packet_num * TransportProtocolType::COUNT_TransportProtocolType]);
// //             m_l4_swap_in_packets_cpu_backup.push_back(new uint8_t
// *[max_packet_num * TransportProtocolType::COUNT_TransportProtocolType]);
// //             memcpy(m_l4_swap_in_packets_cpu_backup[i],
// swap_in_packets_cpu.data(), sizeof(uint8_t *) * max_packet_num *
// TransportProtocolType::COUNT_TransportProtocolType);
// //             cudaMemcpyAsync(cpu_param.l4_swap_in_offset_per_node,
// packet_offsets, sizeof(int) * node_num, cudaMemcpyHostToDevice,
// m_streams[i]);
// //             cudaMemsetAsync(cpu_param.l4_swap_in_packets_num, 0,
// sizeof(int) * node_num * TransportProtocolType::COUNT_TransportProtocolType,
// m_streams[i]);

// // m_recycle_ipv4_packets_gpu.push_back(cpu_param.recycle_ipv4_packets);
// //             m_recycle_ipv4_packets_cpu.push_back(new Ipv4Packet
// *[max_packet_num]);
// //
// m_recycle_ipv4_packets_num_gpu.push_back(cpu_param.recycle_ipv4_packets_num);
// //             m_recycle_ipv4_packets_num_cpu.push_back(new int[node_num]);
// // m_l4_swap_in_packet_num_gpu.push_back(cpu_param.l4_swap_in_packets_num);
// //             m_l4_swap_in_packet_num_cpu.push_back(new int[node_num *
// TransportProtocolType::COUNT_TransportProtocolType]);
// //             m_max_packet_num_per_batch.push_back(max_packet_num);

// //             IPv4DecapsulationParam* gpu_param;
// //             cudaMallocAsync(&gpu_param, sizeof(IPv4DecapsulationParam),
// m_streams[i]);
// //             cudaMemcpyAsync(gpu_param, &cpu_param,
// sizeof(IPv4DecapsulationParam), cudaMemcpyHostToDevice, m_streams[i]);
// //             m_kernel_params.push_back(gpu_param);
// //         }
// //     }

// //     void IPv4DecapsulationController::SetStreams(cudaStream_t *streams,
// int node_num)
// //     {
// //         m_streams.insert(m_streams.end(), streams, streams + node_num);
// //     }

// //     void IPv4DecapsulationController::SetIPv4Queues(GPUQueue<Ipv4Packet *>
// **ipv4_queues, int node_num)
// //     {
// //         m_ipv4_queues.insert(m_ipv4_queues.end(), ipv4_queues, ipv4_queues
// + node_num);
// //     }

// //     void IPv4DecapsulationController::SetL4Queues(GPUQueue<uint8_t *>
// **l4_queues, int node_num)
// //     {
// //         m_l4_queues.insert(m_l4_queues.end(), l4_queues, l4_queues +
// node_num * TransportProtocolType::COUNT_TransportProtocolType);
// //     }

// //     void IPv4DecapsulationController::SetNICNum(int *nic_num_per_node, int
// node_num)
// //     {
// //         m_nic_num_per_node.insert(m_nic_num_per_node.end(),
// nic_num_per_node, nic_num_per_node + node_num);
// //     }

// //     void IPv4DecapsulationController::SetBatchProperties(int
// *batch_start_index, int *batch_end_index, int batch_num)
// //     {
// //         m_batch_start_index.insert(m_batch_start_index.end(),
// batch_start_index, batch_start_index + batch_num);
// //         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index,
// batch_end_index + batch_num);
// //     }

// //     void IPv4DecapsulationController::CacheInL4Packets(int batch_id)
// //     {
// //         int max_packet_num = m_max_packet_num_per_batch[batch_id];
// //         int node_num = m_batch_end_index[batch_id] -
// m_batch_start_index[batch_id];

// //         cudaMemcpyAsync(m_l4_swap_in_packets_cpu[batch_id],
// m_l4_swap_in_packets_gpu[batch_id], sizeof(uint8_t *) * max_packet_num *
// TransportProtocolType::COUNT_TransportProtocolType, cudaMemcpyDeviceToHost,
// m_streams[batch_id]);
// //         cudaMemcpyAsync(m_l4_swap_in_packet_num_cpu[batch_id],
// m_l4_swap_in_packet_num_gpu[batch_id], sizeof(int) * node_num *
// TransportProtocolType::COUNT_TransportProtocolType, cudaMemcpyDeviceToHost,
// m_streams[batch_id]);
// //         cudaMemcpyAsync(m_cache_space_cpu[batch_id],
// m_cache_space_gpu[batch_id], sizeof(uint8_t) * m_cache_space_sizes[batch_id],
// cudaMemcpyDeviceToHost, m_streams[batch_id]);

// //         cudaStreamSynchronize(m_streams[batch_id]);

// //         for (int i = 0; i <
// TransportProtocolType::COUNT_TransportProtocolType; i++)
// //         {
// //             int *swap_in_num = m_l4_swap_in_packet_num_cpu[batch_id] + i *
// node_num;
// //             int native_packet_size = m_native_packet_size[i];
// //             // int packet_size = m_l4_packet_size[i];

// //             uint8_t **dst = m_l4_swap_in_packets_cpu_backup[batch_id] + i
// * max_packet_num;
// //             uint8_t **src = m_l4_swap_in_packets_cpu[batch_id] + i *
// max_packet_num;

// //             std::vector<uint8_t *> recycle_packets;
// //             for (int j = 0; j < node_num; j++)
// //             {
// //                 dst = dst + m_packet_offset_per_node_cpu[batch_id][j];
// //                 src = src + swap_in_num[j];

// //                 // copy discrete packets to cache space
// //                 for (int k = 0; k < swap_in_num[j]; k++)
// //                 {
// //                     memcpy(dst[k], src[k], native_packet_size);
// //                     recycle_packets.push_back(src[k]);
// //                 }
// //             }

// //             if (i == TransportProtocolType::TCP)
// //             {
// //                 // recyel tcp packets
// //                 tcp_packet_pool_cpu->deallocate((TCPPacket
// **)recycle_packets.data(), recycle_packets.size());
// //             }
// //         }

// //         cudaMemcpyAsync(m_cache_space_gpu[batch_id],
// m_cache_space_cpu[batch_id], sizeof(uint8_t) * m_cache_space_sizes[batch_id],
// cudaMemcpyHostToDevice, m_streams[batch_id]);
// //         cudaMemcpyAsync(m_l4_swap_in_packets_gpu[batch_id],
// m_l4_swap_in_packets_gpu_backup[batch_id], sizeof(uint8_t *) * max_packet_num
// * TransportProtocolType::COUNT_TransportProtocolType, cudaMemcpyHostToDevice,
// m_streams[batch_id]);
// //     }

// //     void IPv4DecapsulationController::RecycleIPv4Packets(int batch_id)
// //     {
// //         int node_num = m_batch_end_index[batch_id] -
// m_batch_start_index[batch_id];
// //         int max_packet_num = m_max_packet_num_per_batch[batch_id];

// //         cudaMemcpyAsync(m_recycle_ipv4_packets_cpu[batch_id],
// m_recycle_ipv4_packets_gpu[batch_id], sizeof(Ipv4Packet *) * node_num *
// max_packet_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);
// //         cudaMemcpyAsync(m_recycle_ipv4_packets_num_gpu[batch_id],
// m_recycle_ipv4_packets_num_cpu[batch_id], sizeof(int) * node_num,
// cudaMemcpyHostToDevice, m_streams[batch_id]);
// //         cudaStreamSynchronize(m_streams[batch_id]);

// //         int *recycle_num = m_recycle_ipv4_packets_num_cpu[batch_id];
// //         int *packet_offsets = m_packet_offset_per_node_cpu[batch_id];
// //         Ipv4Packet **packets = m_recycle_ipv4_packets_cpu[batch_id];

// //         std::vector<Ipv4Packet *> recycle_packets;
// //         for (int i = 0; i < node_num; i++)
// //         {
// //             Ipv4Packet **src = packets + packet_offsets[i];
// //             recycle_packets.insert(recycle_packets.end(), src, src +
// recycle_num[i]);
// //         }

// //         ipv4_packet_pool_cpu->deallocate(recycle_packets.data(),
// recycle_packets.size());
// //     }

// //     void IPv4DecapsulationController::BuildGraph(int batch_id)
// //     {
// //         int node_num = m_batch_end_index[batch_id] -
// m_batch_start_index[batch_id];
// //         dim3 block_dim(KERNEL_BLOCK_WIDTH);
// //         dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);
// //         cudaStreamBeginCapture(m_streams[batch_id],
// cudaStreamCaptureModeGlobal);
// //         LaunchIPv4DecapsulationKernel(grid_dim, block_dim,
// m_kernel_params[batch_id], m_streams[batch_id]);
// //         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
// //         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id],
// NULL, NULL, 0);
// //     }

// //     void IPv4DecapsulationController::LaunchInstance(int batch_id)
// //     {
// //         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
// //     }

// //     void IPv4DecapsulationController::Run(int batch_id)
// //     {
// //         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
// //         cudaStreamSynchronize(m_streams[batch_id]);
// //         CacheInL4Packets(batch_id);
// //         RecycleIPv4Packets(batch_id);
// //     }

// //     void IPv4DecapsulationController::Run()
// //     {
// //     }
// // }