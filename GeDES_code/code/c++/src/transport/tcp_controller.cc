#include "tcp_controller.h"
#include "component.h"
#include <cstring>
#include <functional>
#include <numeric>
#include <fstream>
#include <iostream>

namespace VDES
{
  TCPController::TCPController() {}

  TCPController::~TCPController() {}

  void TCPController::InitKernelParams()
  {
    int batch_num = m_batch_end_index.size();
    int total_node_num = m_batch_end_index[batch_num - 1] - m_batch_start_index[0];
    cudaMalloc(&m_packets_completed_gpu, total_node_num * sizeof(uint8_t));
    cudaMalloc(&m_packets_completed_temporary_gpu, total_node_num * sizeof(uint8_t));
    uint8_t *packets_comleted_gpu = m_packets_completed_gpu;
    uint8_t *packets_completed_temporary_gpu = m_packets_completed_temporary_gpu;

    for (int i = 0; i < batch_num; i++)
    {
      TCPParams cpu_param;
      int node_num = m_batch_end_index[i] - m_batch_start_index[i];
      int max_tcp_num = MAX_TCP_CONNECTION_NUM;

      cudaGraph_t graph_recv;
      cudaGraph_t graph_send;
      cudaGraphCreate(&graph_recv, 0);
      cudaGraphCreate(&graph_send, 0);

      m_receiver_graphs.emplace_back(graph_recv);
      m_sender_graphs.emplace_back(graph_send);
      m_receiver_instances.emplace_back();
      m_sender_instances.emplace_back();

      // assume each node at most has 10 TCP connections
      cpu_param.max_tcp_num = MAX_TCP_CONNECTION_NUM;
      cudaMallocAsync(&cpu_param.tcp_cons,
                      sizeof(TCPConnection *) * node_num * max_tcp_num,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.tcp_cons_num_per_node, sizeof(int) * node_num,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.recv_queues,
                      sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.send_queues,
                      sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num,
                      m_streams[i]);
      int nic_num =
          std::accumulate(m_nic_num_per_node.begin() + m_batch_start_index[i],
                          m_nic_num_per_node.begin() + m_batch_end_index[i], 0);
      int alloc_packet_num = nic_num * MAX_TRANSMITTED_PACKET_NUM +
                             node_num * MAX_GENERATED_PACKET_NUM;
      cudaMallocAsync(&cpu_param.alloc_packets,
                      sizeof(TCPPacket *) * alloc_packet_num, m_streams[i]);
      cudaMallocAsync(&cpu_param.packet_offset_per_node, sizeof(int) * node_num,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.used_packet_num_per_node, sizeof(int) * node_num,
                      m_streams[i]);

#if !ENABLE_CACHE
      cudaMallocAsync(&cpu_param.recycle_tcp_packet_num, sizeof(int) * node_num,
                      m_streams[i]);
      cudaMallocAsync(&cpu_param.recycle_packets,
                      sizeof(TCPPacket *) * alloc_packet_num, m_streams[i]);
#endif
      cpu_param.remaining_nic_cache_space_per_node =
          m_remainming_nic_cache_space_per_node[i];
      cpu_param.timeslot_start_time = m_timeslot_start_time;
      cpu_param.timeslot_end_time = m_timeslot_end_time;
      cpu_param.node_num = node_num;

      int node_offset = m_batch_start_index[i];
      cudaMemcpyAsync(cpu_param.tcp_cons,
                      m_tcp_cons.data() + node_offset * max_tcp_num,
                      sizeof(TCPConnection *) * node_num * max_tcp_num,
                      cudaMemcpyHostToDevice, m_streams[i]);
      cudaMemcpyAsync(cpu_param.tcp_cons_num_per_node,
                      m_tcp_num_per_node.data() + node_offset,
                      sizeof(int) * node_num, cudaMemcpyHostToDevice,
                      m_streams[i]);
      cudaMemcpyAsync(cpu_param.recv_queues, m_recv_queues.data() + node_offset,
                      sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num,
                      cudaMemcpyHostToDevice, m_streams[i]);
      cudaMemcpyAsync(cpu_param.send_queues, m_send_queues.data() + node_offset,
                      sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num,
                      cudaMemcpyHostToDevice, m_streams[i]);
      TCPPacket *packet_cache;

#if !ENABLE_CACHE
      cudaMemsetAsync(cpu_param.recycle_tcp_packet_num, 0, sizeof(int) * node_num,
                      m_streams[i]);
      cudaMemsetAsync(cpu_param.recycle_packets, 0,
                      sizeof(TCPPacket *) * alloc_packet_num, m_streams[i]);

      m_recycle_tcp_packet_num_gpu.push_back(cpu_param.recycle_tcp_packet_num);
      m_recycle_packets_gpu.push_back(cpu_param.recycle_packets);
      m_recycle_tcp_packet_num_cpu.push_back(new int[node_num]);
      m_recycle_packets_cpu.push_back(new TCPPacket *[alloc_packet_num]);

#if ENABLE_GPU_MEM_POOL
      std::vector<TCPPacket *> alloc_packets;
      for (int i = 0; i < alloc_packet_num; i++)
      {
        TCPPacket *packet;
        cudaMalloc(&packet, sizeof(TCPPacket));
        alloc_packets.push_back(packet);
      }
#else
      auto alloc_packets = tcp_packet_pool->allocate(alloc_packet_num);
#endif
      cudaMemcpyAsync(cpu_param.alloc_packets, alloc_packets.data(),
                      sizeof(TCPPacket *) * alloc_packet_num,
                      cudaMemcpyHostToDevice, m_streams[i]);
      m_alloc_packets_gpu.push_back(cpu_param.alloc_packets);
      m_alloc_packets_cpu.push_back(new TCPPacket *[alloc_packet_num]);
      memcpy(m_alloc_packets_cpu[i], alloc_packets.data(),
             sizeof(TCPPacket *) * alloc_packet_num);

      // memcpy(m_recycle_packets_cpu[i], alloc_packets.data(),
      //        sizeof(TCPPacket *) * alloc_packet_num);
      memset(m_recycle_packets_cpu[i], 0, sizeof(TCPPacket *) * alloc_packet_num);

      m_used_packet_num_per_node_gpu.push_back(
          cpu_param.used_packet_num_per_node);
      m_used_packet_num_per_node_cpu.push_back(new int[node_num]);

#else
      cudaMallocAsync(&packet_cache, sizeof(TCPPacket) * alloc_packet_num,
                      m_streams[i]);
      cudaStreamSynchronize(m_streams[i]);
      std::vector<TCPPacket *> alloc_packets;
      for (int j = 0; j < alloc_packet_num; j++)
      {
        alloc_packets.push_back(packet_cache + j);
      }
      m_packet_cache_space.push_back(packet_cache);
      cudaMemcpyAsync(cpu_param.alloc_packets, alloc_packets.data(),
                      sizeof(TCPPacket *) * alloc_packet_num,
                      cudaMemcpyHostToDevice, m_streams[i]);
#endif
      m_max_packet_num.push_back(alloc_packet_num);
      int *packet_offset = new int[node_num];

      int offset = 0;
      for (int j = 0; j < node_num; j++)
      {
        // packet_offset.push_back(offset);
        packet_offset[j] = offset;
        offset = offset + m_nic_num_per_node[node_offset + j] * MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM;
      }
      m_packet_offsets.push_back(packet_offset);

      m_recycle_tcp_tmp.push_back(new TCPPacket *[alloc_packet_num]);

      cudaMemcpyAsync(cpu_param.packet_offset_per_node, m_packet_offsets[i],
                      sizeof(int) * node_num, cudaMemcpyHostToDevice,
                      m_streams[i]);
      cudaMemsetAsync(cpu_param.used_packet_num_per_node, 0,
                      sizeof(int) * node_num, m_streams[i]);

      cpu_param.is_completed_traffic_plan = packets_comleted_gpu;
      cpu_param.is_completed_temporary_traffic_plan = packets_completed_temporary_gpu;
      packets_comleted_gpu += node_num;
      packets_completed_temporary_gpu += node_num;

      TCPParams *gpu_param;
      cudaMallocAsync(&gpu_param, sizeof(TCPParams), m_streams[i]);
      cudaMemcpyAsync(gpu_param, &cpu_param, sizeof(TCPParams),
                      cudaMemcpyHostToDevice, m_streams[i]);
      m_kernel_params.push_back(gpu_param);
    }
  }

  void TCPController::SetTCPConnections(TCPConnection **tcp_cons,
                                        int *tcp_cons_num_per_node,
                                        int node_num)
  {
    m_tcp_cons.insert(m_tcp_cons.end(), tcp_cons,
                      tcp_cons + node_num * MAX_TCP_CONNECTION_NUM);
    m_tcp_num_per_node.insert(m_tcp_num_per_node.end(), tcp_cons_num_per_node,
                              tcp_cons_num_per_node + node_num);
  }

  void TCPController::SetRecvQueues(GPUQueue<VDES::TCPPacket *> **recv_queues,
                                    int node_num)
  {
    m_recv_queues.insert(m_recv_queues.end(), recv_queues,
                         recv_queues + node_num);
  }

  void TCPController::SetSendQueues(GPUQueue<VDES::TCPPacket *> **send_queues,
                                    int node_num)
  {
    m_send_queues.insert(m_send_queues.end(), send_queues,
                         send_queues + node_num);
  }

  void TCPController::SetRemainingCacheSizeArray(
      int **remaining_nic_cache_space_per_node, int node_num)
  {
    m_remainming_nic_cache_space_per_node.insert(
        m_remainming_nic_cache_space_per_node.end(),
        remaining_nic_cache_space_per_node,
        remaining_nic_cache_space_per_node + node_num);
  }

  void TCPController::SetBatchProperties(int *batch_start_index,
                                         int *batch_end_index, int batch_num)
  {
    m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index,
                               batch_start_index + batch_num);
    m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index,
                             batch_end_index + batch_num);
  }

  void TCPController::SetStreams(cudaStream_t *streams, int batch_num)
  {
    m_streams.insert(m_streams.end(), streams, streams + batch_num);
  }

  void TCPController::SetTimeslotInfo(int64_t *timeslot_start_time,
                                      int64_t *timeslot_end_time)
  {
    m_timeslot_start_time = timeslot_start_time;
    m_timeslot_end_time = timeslot_end_time;
  }

  void TCPController::BuildGraph(int batch_id)
  {
    int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
    dim3 block_dim(KERNEL_BLOCK_WIDTH);
    dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);

    cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
    LaunchReceiveTCPPacketKernel(grid_dim, block_dim, m_kernel_params[batch_id],
                                 m_streams[batch_id]);
    cudaStreamEndCapture(m_streams[batch_id], &m_receiver_graphs[batch_id]);
    cudaGraphInstantiate(&m_receiver_instances[batch_id],
                         m_receiver_graphs[batch_id], NULL, NULL, 0);

    cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
    LaunchSendTCPPacketKernel(grid_dim, block_dim, m_kernel_params[batch_id],
                              m_streams[batch_id]);
    cudaStreamEndCapture(m_streams[batch_id], &m_sender_graphs[batch_id]);
    cudaGraphInstantiate(&m_sender_instances[batch_id], m_sender_graphs[batch_id],
                         NULL, NULL, 0);

#if ENABLE_HUGE_GRAPH
    int max_packet_num = m_max_packet_num[batch_id];

    cudaGraphNode_t recycle_packet_memcpy_node;
    cudaMemcpy3DParms recycle_packet_memcpy_params = {0};
    recycle_packet_memcpy_params.srcPtr = make_cudaPitchedPtr(
        m_recycle_packets_gpu[batch_id], sizeof(TCPPacket *) * max_packet_num,
        max_packet_num, 1);
    recycle_packet_memcpy_params.dstPtr = make_cudaPitchedPtr(
        m_recycle_packets_cpu[batch_id], sizeof(TCPPacket *) * max_packet_num,
        max_packet_num, 1);
    recycle_packet_memcpy_params.extent =
        make_cudaExtent(sizeof(TCPPacket *) * max_packet_num, 1, 1);
    recycle_packet_memcpy_params.kind = cudaMemcpyDeviceToHost;

    cudaMemcpy3DParms recycle_tcp_packet_num_memcpy_params = {0};
    recycle_tcp_packet_num_memcpy_params.srcPtr =
        make_cudaPitchedPtr(m_recycle_tcp_packet_num_gpu[batch_id],
                            sizeof(int) * node_num, node_num, 1);
    recycle_tcp_packet_num_memcpy_params.dstPtr =
        make_cudaPitchedPtr(m_recycle_tcp_packet_num_cpu[batch_id],
                            sizeof(int) * node_num, node_num, 1);
    recycle_tcp_packet_num_memcpy_params.extent =
        make_cudaExtent(sizeof(int) * node_num, 1, 1);
    recycle_tcp_packet_num_memcpy_params.kind = cudaMemcpyDeviceToHost;

    cudaHostNodeParams recycle_host_params = {0};
    auto recycle_func =
        std::bind(&VDES::TCPController::RecycleTCPPackets, this, batch_id);
    auto recycle_func_ptr = new std::function<void()>(recycle_func);
    recycle_host_params.fn = VDES::HostNodeCallback;
    recycle_host_params.userData = recycle_func_ptr;

    m_receive_memcpy_param.push_back(recycle_packet_memcpy_params);
    m_receive_memcpy_param.push_back(recycle_tcp_packet_num_memcpy_params);
    m_receive_host_param.push_back(recycle_host_params);

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

    cudaHostNodeParams update_used_packet_params = {0};
    auto update_used_func =
        std::bind(&VDES::TCPController::UpdateUsedTCPPackets, this, batch_id);
    auto update_used_func_ptr = new std::function<void()>(update_used_func);
    update_used_packet_params.fn = VDES::HostNodeCallback;
    update_used_packet_params.userData = update_used_func_ptr;

    cudaMemcpy3DParms alloc_packet_memcpy_params = {0};
    alloc_packet_memcpy_params.srcPtr = make_cudaPitchedPtr(
        m_alloc_packets_cpu[batch_id], sizeof(TCPPacket *) * max_packet_num,
        max_packet_num, 1);
    alloc_packet_memcpy_params.dstPtr = make_cudaPitchedPtr(
        m_alloc_packets_gpu[batch_id], sizeof(TCPPacket *) * max_packet_num,
        max_packet_num, 1);
    alloc_packet_memcpy_params.extent =
        make_cudaExtent(sizeof(TCPPacket *) * max_packet_num, 1, 1);
    alloc_packet_memcpy_params.kind = cudaMemcpyHostToDevice;

    m_send_memcpy_param.push_back(used_packet_num_memcpy_params);
    m_send_host_param.push_back(update_used_packet_params);
    m_send_memcpy_param.push_back(alloc_packet_memcpy_params);

#endif
  }
  void TCPController::BuildGraph()
  {
    int batch_num = m_batch_end_index.size();
    for (int i = 0; i < batch_num; i++)
    {
      BuildGraph(i);
    }
  }

  void TCPController::SetNicNumPerNode(int *nic_num_per_node, int node_num)
  {
    m_nic_num_per_node.insert(m_nic_num_per_node.end(), nic_num_per_node,
                              nic_num_per_node + node_num);
  }

  void TCPController::LaunchReceiveInstance(int batch_id)
  {
    cudaGraphLaunch(m_receiver_instances[batch_id], m_streams[batch_id]);
    cudaStreamSynchronize(m_streams[batch_id]);
    RecycleTCPPackets(batch_id);
  }

  void TCPController::LaunchSendInstance(int batch_id)
  {
    cudaGraphLaunch(m_sender_instances[batch_id], m_streams[batch_id]);
    cudaStreamSynchronize(m_streams[batch_id]);
    UpdateUsedTCPPackets(batch_id);
  }

  void TCPController::RecycleTCPPackets(int batch_id)
  {
    int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
    int max_packet_num = m_max_packet_num[batch_id];

#if !ENABLE_HUGE_GRAPH
    cudaMemcpyAsync(m_recycle_packets_cpu[batch_id],
                    m_recycle_packets_gpu[batch_id],
                    sizeof(TCPPacket *) * max_packet_num, cudaMemcpyDeviceToHost,
                    m_streams[batch_id]);
    cudaMemcpyAsync(m_recycle_tcp_packet_num_cpu[batch_id],
                    m_recycle_tcp_packet_num_gpu[batch_id],
                    sizeof(int) * node_num, cudaMemcpyDeviceToHost,
                    m_streams[batch_id]);
    cudaStreamSynchronize(m_streams[batch_id]);
#endif
    int total_tcp_packets =
        std::accumulate(m_recycle_tcp_packet_num_cpu[batch_id],
                        m_recycle_tcp_packet_num_cpu[batch_id] + node_num, 0);

    int *recycle_tcp_packet_num = m_recycle_tcp_packet_num_cpu[batch_id];
    TCPPacket **dst_recycle = m_recycle_tcp_tmp[batch_id];
    int packet_index = 0;
    for (int i = 0; i < node_num; i++)
    {
      TCPPacket **recycle_packets_src =
          m_recycle_packets_cpu[batch_id] + m_packet_offsets[batch_id][i];

      memcpy(dst_recycle + packet_index, recycle_packets_src,
             sizeof(TCPPacket *) * recycle_tcp_packet_num[i]);
      packet_index += recycle_tcp_packet_num[i];
    }
    tcp_packet_pool->deallocate(dst_recycle, total_tcp_packets);
  }

  void TCPController::UpdateUsedTCPPackets(int batch_id)
  {
    int max_packet_num = m_max_packet_num[batch_id];
    int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

#if !ENABLE_HUGE_GRAPH
    cudaMemcpy(m_used_packet_num_per_node_cpu[batch_id],
               m_used_packet_num_per_node_gpu[batch_id], sizeof(int) * node_num,
               cudaMemcpyDeviceToHost);
#endif
    int *used_packet_num = m_used_packet_num_per_node_cpu[batch_id];
    int total_used_packets =
        std::accumulate(used_packet_num, used_packet_num + node_num, 0);

    auto alloc_packets = tcp_packet_pool->allocate(total_used_packets);
    int alloc_index = 0;

    TCPPacket **alloc_packets_origin = m_alloc_packets_cpu[batch_id];
    for (int i = 0; i < node_num; i++)
    {
      TCPPacket **used_packets =
          alloc_packets_origin + m_packet_offsets[batch_id][i];
      memcpy(used_packets, alloc_packets.data() + alloc_index,
             sizeof(TCPPacket *) * used_packet_num[i]);
      alloc_index += used_packet_num[i];
    }
#if !ENABLE_HUGE_GRAPH
    cudaMemcpy(m_alloc_packets_gpu[batch_id], m_alloc_packets_cpu[batch_id],
               sizeof(TCPPacket *) * max_packet_num, cudaMemcpyHostToDevice);
#endif
  }

  cudaGraph_t TCPController::GetReceiveGraph(int batch_id)
  {
    return m_receiver_graphs[batch_id];
  }

  cudaGraph_t TCPController::GetSendGraph(int batch_id)
  {
    return m_sender_graphs[batch_id];
  }

#if ENABLE_HUGE_GRAPH

  std::vector<cudaMemcpy3DParms> &TCPController::GetReceiveMemcpyParam()
  {
    return m_receive_memcpy_param;
  }

  std::vector<cudaHostNodeParams> &TCPController::GetReceiveHostParam()
  {
    return m_receive_host_param;
  }

  std::vector<cudaMemcpy3DParms> &TCPController::GetSendMemcpyParam()
  {
    return m_send_memcpy_param;
  }

  std::vector<cudaHostNodeParams> &TCPController::GetSendHostParam()
  {
    return m_send_host_param;
  }
#endif

  std::vector<void *> TCPController::GetAllocInfo()
  {
    std::vector<void *> res;
    int batch_num = m_batch_start_index.size();
    for (int i = 0; i < batch_num; i++)
    {
      res.push_back(m_alloc_packets_gpu[i]);
      res.push_back(m_used_packet_num_per_node_gpu[i]);
    }
    return res;
  }

  std::vector<void *> TCPController::GetRecycleInfo()
  {
    std::vector<void *> res;
    int batch_num = m_batch_start_index.size();
    for (int i = 0; i < batch_num; i++)
    {
      res.push_back(m_recycle_packets_gpu[i]);
      res.push_back(m_recycle_tcp_packet_num_gpu[i]);
    }
    return res;
  }

  uint8_t *TCPController::GetCompletedArr()
  {
    return m_packets_completed_gpu;
  }

  uint8_t *TCPController::GetTemporaryCompletedArr()
  {
    return m_packets_completed_temporary_gpu;
  }

  void TCPController::RecordFlowResults(std::string file_name)
  {
    std::ofstream out_file(file_name);
    out_file << "source_ip" << "," << "dst_ip" << "," << "flow_size" << "," << "start_timestamp" << "," << "completed_timestamp" << "," << "flow_completion_time" << std::endl;

    TCPParams kernel_param_cpu;
    for (int i = 0; i < m_kernel_params.size(); i++)
    {
      cudaMemcpy(&kernel_param_cpu, m_kernel_params[i], sizeof(TCPParams), cudaMemcpyDeviceToHost);
      int *tcp_cons_num_per_node = new int[kernel_param_cpu.node_num];
      cudaMemcpy(tcp_cons_num_per_node, kernel_param_cpu.tcp_cons_num_per_node, sizeof(int) * kernel_param_cpu.node_num, cudaMemcpyDeviceToHost);
      int tcp_con_num = std::accumulate(tcp_cons_num_per_node, tcp_cons_num_per_node + kernel_param_cpu.node_num, 0);
      TCPConnection **cons = new TCPConnection *[tcp_con_num * MAX_TCP_CONNECTION_NUM];
      cudaMemcpy(cons, kernel_param_cpu.tcp_cons, sizeof(TCPConnection *) * tcp_con_num * MAX_TCP_CONNECTION_NUM, cudaMemcpyDeviceToHost);
      GPUQueue<Flow> flow_queue;
      TCPConnection con;
      // std::cout << "tcp_con_num:" << tcp_con_num << std::endl;
      for (int j = 0; j < tcp_con_num; j++)
      {
        cudaMemcpy(&con, cons[j * 10], sizeof(TCPConnection), cudaMemcpyDeviceToHost);
        cudaMemcpy(&flow_queue, con.flows, sizeof(GPUQueue<Flow>), cudaMemcpyDeviceToHost);
        // std::cout << "flow_queue.size:" << flow_queue.size + flow_queue.head << std::endl;
        int size = flow_queue.size + flow_queue.head;
        Flow *flows = new Flow[size];
        cudaMemcpy(flows, flow_queue.queue, sizeof(Flow) * size, cudaMemcpyDeviceToHost);
        // std::cout << j << "," << con.src_ip << "," << cons[j * MAX_TCP_CONNECTION_NUM] << std::endl;

        // std::cout << flow_queue.head << std::endl;
        for (int k = 0; k < size; k++)
        {
          if (k == 0)
          {
            // std::cout << j << "," << flows[k].flow_size << "," << flows[k].timestamp << "," << flows[k].tiimestamp_end << std::endl;
            out_file << con.src_ip << "," << con.dst_ip << "," << flows[k].flow_size << "," << flows[k].timestamp << "," << flows[k].tiimestamp_end << ","<<flows[k].tiimestamp_end-flows[k].timestamp<<std::endl;
          }
        }
        delete[] flows;
      }
      delete[] cons;
      delete[] tcp_cons_num_per_node;
    }
  }

} // namespace VDES

// namespace VDES
// {
//     TCPController::TCPController()
//     {
//     }

//     TCPController::~TCPController()
//     {
//     }

//     void TCPController::InitKernelParams()
//     {
//         int batch_num = m_batch_end_index.size();

//         for (int i = 0; i < batch_num; i++)
//         {
//             TCPParams cpu_param;
//             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
//             int max_tcp_num = 10;

//             cudaGraph_t graph_recv;
//             cudaGraph_t graph_send;
//             cudaGraphCreate(&graph_recv, 0);
//             cudaGraphCreate(&graph_send, 0);

//             m_receiver_graphs.emplace_back(graph_recv);
//             m_sender_graphs.emplace_back(graph_send);
//             m_receiver_instances.emplace_back();
//             m_sender_instances.emplace_back();

//             // assume each node at most has 10 TCP connections
//             cpu_param.max_tcp_num = 10;
//             cudaMallocAsync(&cpu_param.tcp_cons, sizeof(TCPConnection *) *
//             node_num * max_tcp_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.tcp_cons_num_per_node, sizeof(int) *
//             node_num, m_streams[i]); cudaMallocAsync(&cpu_param.recv_queues,
//             sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.send_queues,
//             sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num, m_streams[i]);
//             int nic_num = std::accumulate(m_nic_num_per_node.begin() +
//             m_batch_start_index[i], m_nic_num_per_node.begin() +
//             m_batch_end_index[i], 0); int alloc_packet_num = nic_num *
//             MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;
//             cudaMallocAsync(&cpu_param.alloc_packets, sizeof(TCPPacket *) *
//             alloc_packet_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.packet_offset_per_node, sizeof(int) *
//             node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.used_packet_num_per_node, sizeof(int)
//             * node_num, m_streams[i]);

// #if !ENABLE_CACHE
//             cudaMallocAsync(&cpu_param.recycle_tcp_packet_num, sizeof(int) *
//             node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.recycle_packets, sizeof(TCPPacket *) *
//             alloc_packet_num, m_streams[i]);
// #endif

//             /**
//              * @warning: should make sure that the remaining nic cache space
//              is equal to the size of the batch.
//              */
//             // cudaMallocAsync(&cpu_param.remaining_nic_cache_space_per_node,
//             sizeof(int) * node_num, m_streams[i]);
//             cpu_param.remaining_nic_cache_space_per_node =
//             m_remainming_nic_cache_space_per_node[i];
//             cpu_param.timeslot_start_time = m_timeslot_start_time;
//             cpu_param.timeslot_end_time = m_timeslot_end_time;
//             cpu_param.node_num = node_num;

//             int node_offset = m_batch_start_index[i];
//             cudaMemcpyAsync(cpu_param.tcp_cons, m_tcp_cons.data() +
//             node_offset * max_tcp_num, sizeof(TCPConnection *) * node_num *
//             max_tcp_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.tcp_cons_num_per_node,
//             m_tcp_num_per_node.data() + node_offset, sizeof(int) * node_num,
//             cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.recv_queues, m_recv_queues.data() +
//             node_offset, sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num,
//             cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.send_queues, m_send_queues.data() +
//             node_offset, sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num,
//             cudaMemcpyHostToDevice, m_streams[i]); TCPPacket *packet_cache;

// #if !ENABLE_CACHE
//             cudaMemsetAsync(cpu_param.recycle_tcp_packet_num, 0, sizeof(int)
//             * node_num, m_streams[i]);
//             cudaMemsetAsync(cpu_param.recycle_packets, 0, sizeof(TCPPacket *)
//             * alloc_packet_num, m_streams[i]);

//             m_recycle_tcp_packet_num_gpu.push_back(cpu_param.recycle_tcp_packet_num);
//             m_recycle_packets_gpu.push_back(cpu_param.recycle_packets);
//             m_recycle_tcp_packet_num_cpu.push_back(new int[node_num]);
//             m_recycle_packets_cpu.push_back(new TCPPacket
//             *[alloc_packet_num]); auto alloc_packets =
//             tcp_packet_pool->allocate(alloc_packet_num);

//             cudaMemcpyAsync(cpu_param.alloc_packets, alloc_packets.data(),
//             sizeof(TCPPacket *) * alloc_packet_num, cudaMemcpyHostToDevice,
//             m_streams[i]);
//             m_alloc_packets_gpu.push_back(cpu_param.alloc_packets);
//             m_alloc_packets_cpu.push_back(new TCPPacket *[alloc_packet_num]);
//             memcpy(m_alloc_packets_cpu[i], alloc_packets.data(),
//             sizeof(TCPPacket *) * alloc_packet_num);

//             memcpy(m_recycle_packets_cpu[i], alloc_packets.data(),
//             sizeof(TCPPacket *) * alloc_packet_num);

//             m_used_packet_num_per_node_gpu.push_back(cpu_param.used_packet_num_per_node);
//             m_used_packet_num_per_node_cpu.push_back(new int[node_num]);
// #else
//             cudaMallocAsync(&packet_cache, sizeof(TCPPacket) *
//             alloc_packet_num, m_streams[i]);
//             cudaStreamSynchronize(m_streams[i]);
//             std::vector<TCPPacket *> alloc_packets;
//             for (int j = 0; j < alloc_packet_num; j++)
//             {
//                 alloc_packets.push_back(packet_cache + j);
//             }
//             m_packet_cache_space.push_back(packet_cache);
//             cudaMemcpyAsync(cpu_param.alloc_packets, alloc_packets.data(),
//             sizeof(TCPPacket *) * alloc_packet_num, cudaMemcpyHostToDevice,
//             m_streams[i]);
// #endif

//             m_max_packet_num.push_back(alloc_packet_num);

//             int *packet_offset = new int[node_num];
//             /**
//              * @TODO: Calculate the right offset.
//              */
//             int offset = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 packet_offset[j] = offset;
//                 offset = m_nic_num_per_node[node_offset + j] *
//                 MAX_TRANSMITTED_PACKET_NUM + j * MAX_GENERATED_PACKET_NUM;
//             }
//             m_packet_offsets.push_back(packet_offset);

//             m_recycle_tcp_tmp.push_back(new TCPPacket *[alloc_packet_num]);

//             cudaMemcpyAsync(cpu_param.packet_offset_per_node, packet_offset,
//             sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemsetAsync(cpu_param.used_packet_num_per_node, 0,
//             sizeof(int) * node_num, m_streams[i]);

//             TCPParams *gpu_param;
//             cudaMallocAsync(&gpu_param, sizeof(TCPParams), m_streams[i]);
//             cudaMemcpyAsync(gpu_param, &cpu_param, sizeof(TCPParams),
//             cudaMemcpyHostToDevice, m_streams[i]);
//             m_kernel_params.push_back(gpu_param);
//         }
//     }

//     void TCPController::SetTCPConnections(TCPConnection **tcp_cons, int
//     *tcp_cons_num_per_node, int node_num)
//     {
//         /**
//          * TODO: COPY node_num * MAX_TCP_CONNECTION_NUM TO m_tcp_cons
//          */
//         m_tcp_cons.insert(m_tcp_cons.end(), tcp_cons, tcp_cons + node_num *
//         MAX_TCP_CONNECTION_NUM);
//         m_tcp_num_per_node.insert(m_tcp_num_per_node.end(),
//         tcp_cons_num_per_node, tcp_cons_num_per_node + node_num);
//     }

//     void TCPController::SetRecvQueues(GPUQueue<VDES::TCPPacket *>
//     **recv_queues, int node_num)
//     {
//         m_recv_queues.insert(m_recv_queues.end(), recv_queues, recv_queues +
//         node_num);
//     }

//     void TCPController::SetSendQueues(GPUQueue<VDES::TCPPacket *>
//     **send_queues, int node_num)
//     {
//         m_send_queues.insert(m_send_queues.end(), send_queues, send_queues +
//         node_num);
//     }

//     void TCPController::SetRemainingCacheSizeArray(int
//     **remaining_nic_cache_space_per_node, int node_num)
//     {
//         m_remainming_nic_cache_space_per_node.insert(m_remainming_nic_cache_space_per_node.end(),
//         remaining_nic_cache_space_per_node,
//         remaining_nic_cache_space_per_node + node_num);
//     }

//     void TCPController::SetBatchProperties(int *batch_start_index, int
//     *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(),
//         batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index,
//         batch_end_index + batch_num);
//     }

//     void TCPController::SetStreams(cudaStream_t *streams, int batch_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + batch_num);
//     }

//     void TCPController::SetTimeslotInfo(int64_t *timeslot_start_time, int64_t
//     *timeslot_end_time)
//     {
//         m_timeslot_start_time = timeslot_start_time;
//         m_timeslot_end_time = timeslot_end_time;
//     }

//     void TCPController::BuildGraph(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id]; dim3 block_dim(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);

//         cudaStreamBeginCapture(m_streams[batch_id],
//         cudaStreamCaptureModeGlobal); LaunchReceiveTCPPacketKernel(grid_dim,
//         block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id],
//         &m_receiver_graphs[batch_id]);
//         cudaGraphInstantiate(&m_receiver_instances[batch_id],
//         m_receiver_graphs[batch_id], NULL, NULL, 0);

//         cudaStreamBeginCapture(m_streams[batch_id],
//         cudaStreamCaptureModeGlobal); LaunchSendTCPPacketKernel(grid_dim,
//         block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id],
//         &m_sender_graphs[batch_id]);
//         cudaGraphInstantiate(&m_sender_instances[batch_id],
//         m_sender_graphs[batch_id], NULL, NULL, 0);

// #if ENABLE_HUGE_GRAPH
//         int max_packet_num = m_max_packet_num[batch_id];

//         cudaGraphNode_t recycle_packet_memcpy_node;
//         cudaMemcpy3DParms recycle_packet_memcpy_params = {0};
//         recycle_packet_memcpy_params.srcPtr =
//         make_cudaPitchedPtr(m_recycle_packets_gpu[batch_id], sizeof(TCPPacket
//         *) * max_packet_num, max_packet_num, 1);
//         recycle_packet_memcpy_params.dstPtr =
//         make_cudaPitchedPtr(m_recycle_packets_cpu[batch_id], sizeof(TCPPacket
//         *) * max_packet_num, max_packet_num, 1);
//         recycle_packet_memcpy_params.extent =
//         make_cudaExtent(sizeof(TCPPacket *) * max_packet_num, 1, 1);
//         recycle_packet_memcpy_params.kind = cudaMemcpyDeviceToHost;

//         cudaMemcpy3DParms recycle_tcp_packet_num_memcpy_params = {0};
//         recycle_tcp_packet_num_memcpy_params.srcPtr =
//         make_cudaPitchedPtr(m_recycle_tcp_packet_num_gpu[batch_id],
//         sizeof(int) * node_num, node_num, 1);
//         recycle_tcp_packet_num_memcpy_params.dstPtr =
//         make_cudaPitchedPtr(m_recycle_tcp_packet_num_cpu[batch_id],
//         sizeof(int) * node_num, node_num, 1);
//         recycle_tcp_packet_num_memcpy_params.extent =
//         make_cudaExtent(sizeof(int) * node_num, 1, 1);
//         recycle_tcp_packet_num_memcpy_params.kind = cudaMemcpyDeviceToHost;

//         cudaHostNodeParams recycle_host_params = {0};
//         auto recycle_func =
//         std::bind(&VDES::TCPController::RecycleTCPPackets, this, batch_id);
//         auto recycle_func_ptr = new std::function<void()>(recycle_func);
//         recycle_host_params.fn = VDES::HostNodeCallback;
//         recycle_host_params.userData = recycle_func_ptr;

//         m_receive_memcpy_param.push_back(recycle_packet_memcpy_params);
//         m_receive_memcpy_param.push_back(recycle_tcp_packet_num_memcpy_params);
//         m_receive_host_param.push_back(recycle_host_params);

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

//         cudaHostNodeParams update_used_packet_params = {0};
//         auto update_used_func =
//         std::bind(&VDES::TCPController::UpdateUsedTCPPackets, this,
//         batch_id); auto update_used_func_ptr = new
//         std::function<void()>(update_used_func); update_used_packet_params.fn
//         = VDES::HostNodeCallback; update_used_packet_params.userData =
//         update_used_func_ptr;

//         cudaMemcpy3DParms alloc_packet_memcpy_params = {0};
//         alloc_packet_memcpy_params.srcPtr =
//         make_cudaPitchedPtr(m_alloc_packets_cpu[batch_id], sizeof(TCPPacket
//         *) * max_packet_num, max_packet_num, 1);
//         alloc_packet_memcpy_params.dstPtr =
//         make_cudaPitchedPtr(m_alloc_packets_gpu[batch_id], sizeof(TCPPacket
//         *) * max_packet_num, max_packet_num, 1);
//         alloc_packet_memcpy_params.extent = make_cudaExtent(sizeof(TCPPacket
//         *) * max_packet_num, 1, 1); alloc_packet_memcpy_params.kind =
//         cudaMemcpyHostToDevice;

//         m_send_memcpy_param.push_back(used_packet_num_memcpy_params);
//         m_send_host_param.push_back(update_used_packet_params);
//         m_send_memcpy_param.push_back(alloc_packet_memcpy_params);

// #endif
//     }

//     void TCPController::BuildGraph()
//     {
//         int batch_num = m_batch_end_index.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraph(i);
//         }
//     }

//     void TCPController::SetNicNumPerNode(int *nic_num_per_node, int node_num)
//     {
//         m_nic_num_per_node.insert(m_nic_num_per_node.end(), nic_num_per_node,
//         nic_num_per_node + node_num);
//     }

//     void TCPController::LaunchReceiveInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_receiver_instances[batch_id], m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);
//         RecycleTCPPackets(batch_id);
//     }

//     void TCPController::LaunchSendInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_sender_instances[batch_id], m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);
//         UpdateUsedTCPPackets(batch_id);
//     }

//     void TCPController::RecycleTCPPackets(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id]; int max_packet_num =
//         m_max_packet_num[batch_id];

// #if !ENABLE_HUGE_GRAPH

//         cudaMemcpy(m_recycle_packets_cpu[batch_id],
//         m_recycle_packets_gpu[batch_id], sizeof(TCPPacket *) *
//         max_packet_num, cudaMemcpyDeviceToHost);
//         cudaMemcpy(m_recycle_tcp_packet_num_cpu[batch_id],
//         m_recycle_tcp_packet_num_gpu[batch_id], sizeof(int) * node_num,
//         cudaMemcpyDeviceToHost);

// #endif

//         int total_tcp_packets =
//         std::accumulate(m_recycle_tcp_packet_num_cpu[batch_id],
//         m_recycle_tcp_packet_num_cpu[batch_id] + node_num, 0);

//         int *recycle_tcp_packet_num = m_recycle_tcp_packet_num_cpu[batch_id];
//         // std::vector<TCPPacket *> recycle_packets(total_tcp_packets, NULL);
//         TCPPacket **dst_recycle = m_recycle_tcp_tmp[batch_id];
//         int packet_index = 0;
//         for (int i = 0; i < node_num; i++)
//         {
//             TCPPacket **recycle_packets_src = m_recycle_packets_cpu[batch_id]
//             + m_packet_offsets[batch_id][i];

//             // for (int j = 0; j < recycle_tcp_packet_num[i]; j++)
//             // {
//             //     recycle_packets[packet_index] = recycle_packets_src[j];
//             //     packet_index++;
//             // }
//             memcpy(dst_recycle + packet_index, recycle_packets_src,
//             sizeof(TCPPacket *) * recycle_tcp_packet_num[i]); packet_index +=
//             recycle_tcp_packet_num[i];
//         }

//         tcp_packet_pool->deallocate(dst_recycle, total_tcp_packets);
//     }

//     void TCPController::UpdateUsedTCPPackets(int batch_id)
//     {
//         int max_packet_num = m_max_packet_num[batch_id];
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id];

// #if !ENABLE_HUGE_GRAPH
//         cudaMemcpy(m_used_packet_num_per_node_cpu[batch_id],
//         m_used_packet_num_per_node_gpu[batch_id], sizeof(int) * node_num,
//         cudaMemcpyDeviceToHost);
// #endif
//         int *used_packet_num = m_used_packet_num_per_node_cpu[batch_id];
//         int total_used_packets = std::accumulate(used_packet_num,
//         used_packet_num + node_num, 0); auto alloc_packets =
//         tcp_packet_pool->allocate(total_used_packets); int alloc_index = 0;

//         TCPPacket **alloc_packets_origin = m_alloc_packets_cpu[batch_id];
//         for (int i = 0; i < node_num; i++)
//         {
//             TCPPacket **used_packets = alloc_packets_origin +
//             m_packet_offsets[batch_id][i];
//             // for (int j = 0; j < used_packet_num[i]; j++)
//             // {
//             //     used_packets[j] = alloc_packets[alloc_index];
//             //     alloc_index++;
//             // }
//             memcpy(used_packets, alloc_packets.data() + alloc_index,
//             sizeof(TCPPacket *) * used_packet_num[i]); alloc_index +=
//             used_packet_num[i];
//         }
// #if !ENABLE_HUGE_GRAPH
//         cudaMemcpy(m_alloc_packets_gpu[batch_id],
//         m_alloc_packets_cpu[batch_id], sizeof(TCPPacket *) * max_packet_num,
//         cudaMemcpyHostToDevice);
// #endif
//     }

//     cudaGraph_t TCPController::GetReceiveGraph(int batch_id)
//     {
//         return m_receiver_graphs[batch_id];
//     }

//     cudaGraph_t TCPController::GetSendGraph(int batch_id)
//     {
//         return m_sender_graphs[batch_id];
//     }

// #if ENABLE_HUGE_GRAPH

//     std::vector<cudaMemcpy3DParms> &TCPController::GetReceiveMemcpyParam()
//     {
//         return m_receive_memcpy_param;
//     }

//     std::vector<cudaHostNodeParams> &TCPController::GetReceiveHostParam()
//     {
//         return m_receive_host_param;
//     }

//     std::vector<cudaMemcpy3DParms> &TCPController::GetSendMemcpyParam()
//     {
//         return m_send_memcpy_param;
//     }

//     std::vector<cudaHostNodeParams> &TCPController::GetSendHostParam()
//     {
//         return m_send_host_param;
//     }

// #endif
// }
// #include "tcp_controller.h"
// #include <numeric>

// namespace VDES
// {
//     TCPController::TCPController()
//     {
//     }

//     TCPController::~TCPController()
//     {
//     }

//     void TCPController::InitKernelParams()
//     {
//         int batch_num = m_batch_end_index.size();

//         for (int i = 0; i < batch_num; i++)
//         {
//             TCPParams cpu_param;
//             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
//             int max_tcp_num = 10;

//             // assume each node at most has 10 TCP connections
//             cpu_param.max_tcp_num = 10;
//             cudaMallocAsync(&cpu_param.tcp_cons, sizeof(TCPConnection *) *
//             node_num * max_tcp_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.tcp_cons_num_per_node, sizeof(int) *
//             node_num, m_streams[i]); cudaMallocAsync(&cpu_param.recv_queues,
//             sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.send_queues,
//             sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num, m_streams[i]);
//             int nic_num = std::accumulate(m_nic_num_per_node.begin() +
//             m_batch_start_index[i], m_nic_num_per_node.begin() +
//             m_batch_end_index[i], 0); int alloc_packet_num = nic_num *
//             MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;
//             cudaMallocAsync(&cpu_param.alloc_packets, sizeof(TCPPacket *) *
//             alloc_packet_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.packet_offset_per_node, sizeof(int) *
//             node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.used_packet_num_per_node, sizeof(int)
//             * node_num, m_streams[i]);
//             cpu_param.remaining_nic_cache_space_per_node =
//             m_remainming_nic_cache_space_per_node[i];
//             cpu_param.timeslot_start_time = m_timeslot_start_time;
//             cpu_param.timeslot_end_time = m_timeslot_end_time;
//             cpu_param.node_num = node_num;

//             int node_offset = m_batch_start_index[i];
//             cudaMemcpyAsync(cpu_param.tcp_cons, m_tcp_cons.data() +
//             node_offset * max_tcp_num, sizeof(TCPConnection *) * node_num *
//             max_tcp_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.tcp_cons_num_per_node,
//             m_tcp_num_per_node.data() + node_offset, sizeof(int) * node_num,
//             cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.recv_queues, m_recv_queues.data() +
//             node_offset, sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num,
//             cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.send_queues, m_send_queues.data() +
//             node_offset, sizeof(GPUQueue<VDES::TCPPacket *> *) * node_num,
//             cudaMemcpyHostToDevice, m_streams[i]); TCPPacket *packet_cache;
//             cudaMallocAsync(&packet_cache, sizeof(TCPPacket) *
//             alloc_packet_num, m_streams[i]);
//             cudaStreamSynchronize(m_streams[i]);
//             std::vector<TCPPacket *> alloc_packets;
//             for (int j = 0; j < alloc_packet_num; j++)
//             {
//                 alloc_packets.push_back(packet_cache + j);
//             }
//             m_packet_cache_space.push_back(packet_cache);
//             cudaMemcpyAsync(cpu_param.alloc_packets, alloc_packets.data(),
//             sizeof(TCPPacket *) * alloc_packet_num, cudaMemcpyHostToDevice,
//             m_streams[i]);

//             std::vector<int> packet_offset;
//             for (int j = 0; j < node_num; j++)
//             {
//                 packet_offset.push_back(m_nic_num_per_node[node_offset + j] *
//                 MAX_TRANSMITTED_PACKET_NUM + j * MAX_GENERATED_PACKET_NUM);
//             }

//             cudaMemcpyAsync(cpu_param.packet_offset_per_node,
//             packet_offset.data(), sizeof(int) * node_num,
//             cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemsetAsync(cpu_param.used_packet_num_per_node, 0,
//             sizeof(int) * node_num, m_streams[i]);
//         }
//     }

//     void TCPController::SetTCPConnections(TCPConnection **tcp_cons, int
//     *tcp_cons_num_per_node, int node_num)
//     {
//         m_tcp_cons.insert(m_tcp_cons.end(), tcp_cons, tcp_cons + node_num);
//         m_tcp_num_per_node.insert(m_tcp_num_per_node.end(),
//         tcp_cons_num_per_node, tcp_cons_num_per_node + node_num);
//     }

//     void TCPController::SetRecvQueues(GPUQueue<VDES::TCPPacket *>
//     **recv_queues, int node_num)
//     {
//         m_recv_queues.insert(m_recv_queues.end(), recv_queues, recv_queues +
//         node_num);
//     }

//     void TCPController::SetSendQueues(GPUQueue<VDES::TCPPacket *>
//     **send_queues, int node_num)
//     {
//         m_send_queues.insert(m_send_queues.end(), send_queues, send_queues +
//         node_num);
//     }

//     void TCPController::SetRemainingCacheSizeArray(int
//     **remaining_nic_cache_space_per_node, int node_num)
//     {
//         m_remainming_nic_cache_space_per_node.insert(m_remainming_nic_cache_space_per_node.end(),
//         remaining_nic_cache_space_per_node,
//         remaining_nic_cache_space_per_node + node_num);
//     }

//     void TCPController::SetBatchProperties(int *batch_start_index, int
//     *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(),
//         batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index,
//         batch_end_index + batch_num);
//     }

//     void TCPController::SetStreams(cudaStream_t *streams, int batch_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + batch_num);
//     }

//     void TCPController::SetTimeslotInfo(int64_t *timeslot_start_time, int64_t
//     *timeslot_end_time)
//     {
//         m_timeslot_start_time = timeslot_start_time;
//         m_timeslot_end_time = timeslot_end_time;
//     }

//     void TCPController::BuildGraph(int batch_id)
//     {
//         int node_num = m_batch_end_index[batch_id] -
//         m_batch_start_index[batch_id]; dim3 block_dim(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);

//         cudaStreamBeginCapture(m_streams[batch_id],
//         cudaStreamCaptureModeGlobal); LaunchReceiveTCPPacketKernel(grid_dim,
//         block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id],
//         &m_receiver_graphs[batch_id]);
//         cudaGraphInstantiate(&m_receiver_instances[batch_id],
//         m_receiver_graphs[batch_id], NULL, NULL, 0);

//         cudaStreamBeginCapture(m_streams[batch_id],
//         cudaStreamCaptureModeGlobal); LaunchSendTCPPacketKernel(grid_dim,
//         block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id],
//         &m_sender_graphs[batch_id]);
//         cudaGraphInstantiate(&m_sender_instances[batch_id],
//         m_sender_graphs[batch_id], NULL, NULL, 0);
//     }

//     void TCPController::SetNicNumPerNode(int *nic_num_per_node, int node_num)
//     {
//         m_nic_num_per_node.insert(m_nic_num_per_node.end(), nic_num_per_node,
//         nic_num_per_node + node_num);
//     }

//     void TCPController::LaunchReceiveInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_receiver_instances[batch_id], m_streams[batch_id]);
//     }

//     void TCPController::LaunchSendInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_sender_instances[batch_id], m_streams[batch_id]);
//     }

// }