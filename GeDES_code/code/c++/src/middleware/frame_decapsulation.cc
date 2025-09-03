#include "frame_decapsulation.h"
#include "component.h"

namespace VDES
{
  FrameDecapsulationConatroller::FrameDecapsulationConatroller() {}

  FrameDecapsulationConatroller::~FrameDecapsulationConatroller() {}

  void FrameDecapsulationConatroller::InitializeKernelParams()
  {
    int batch_num = m_batch_start.size();
    m_graph_execs.resize(batch_num);

    m_packet_size.push_back(sizeof(Ipv4Packet));
    m_native_packet_size.push_back(35);

    int total_node_num = m_batch_end[batch_num - 1];
    for (int i = 0; i < batch_num; i++)
    {
      int node_num = m_batch_end[i] - m_batch_start[i];
      int frame_queue_num =
          std::accumulate(m_frame_queue_num_per_node.begin() + m_batch_start[i],
                          m_frame_queue_num_per_node.begin() + m_batch_end[i], 0);
      m_frame_ingress_num.push_back(frame_queue_num);

      cudaStream_t stream;
      cudaStreamCreate(&stream);
      m_streams.push_back(stream);
      cudaStreamCreate(&stream);
      m_cache_streams.push_back(stream);

      cudaGraph_t graph;
      cudaGraphCreate(&graph, 0);
      m_graphs.push_back(graph);

      FrameDecapsulationParams cpu_params;
      int packet_queue_num =
          node_num * NetworkProtocolType::COUNT_NetworkProtocolType;
      int swap_packet_size = frame_queue_num *
                             NetworkProtocolType::COUNT_NetworkProtocolType *
                             MAX_TRANSMITTED_PACKET_NUM;

      cudaMallocAsync(&cpu_params.frame_ingresses,
                      sizeof(GPUQueue<Frame *> *) * frame_queue_num, stream);
      cudaMallocAsync(&cpu_params.frame_queue_num_per_node,
                      sizeof(int) * node_num, stream);
      cudaMallocAsync(&cpu_params.frame_queue_offset_per_node,
                      sizeof(int) * node_num, stream);
      cudaMallocAsync(&cpu_params.packet_ingresses,
                      sizeof(GPUQueue<uint8_t *> *) * packet_queue_num, stream);

#if ENABLE_CACHE
      cudaMallocAsync(&cpu_params.swap_in_packet_buffer,
                      sizeof(uint8_t *) * swap_packet_size, stream);
      cudaMallocAsync(&cpu_params.swap_packet_num,
                      sizeof(int) *
                          NetworkProtocolType::COUNT_NetworkProtocolType *
                          frame_queue_num,
                      stream);
#endif
      cudaMallocAsync(
          &cpu_params.recycle_frames,
          sizeof(Frame *) * frame_queue_num * MAX_TRANSMITTED_PACKET_NUM, stream);
      cudaMallocAsync(&cpu_params.recycle_num_per_frame_queue,
                      sizeof(int) * frame_queue_num, stream);
      cudaMallocAsync(
          &cpu_params.l3_timestamp_offset,
          sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, stream);
      cudaStreamSynchronize(stream);

      int frame_queue_offset = std::accumulate(
          m_frame_queue_num_per_node.begin(),
          m_frame_queue_num_per_node.begin() + m_batch_start[i], 0);
      cudaMemcpyAsync(cpu_params.frame_ingresses,
                      m_frame_ingresses.data() + frame_queue_offset,
                      sizeof(GPUQueue<Frame *> *) * frame_queue_num,
                      cudaMemcpyHostToDevice, stream);

      cpu_params.queue_num = frame_queue_num;
      cudaMemcpyAsync(cpu_params.frame_queue_num_per_node,
                      m_frame_queue_num_per_node.data() + m_batch_start[i],
                      sizeof(int) * node_num, cudaMemcpyHostToDevice, stream);

      std::vector<int> frame_queue_offset_per_node;
      int offset = 0;
      for (int j = 0; j < node_num; j++)
      {
        frame_queue_offset_per_node.push_back(offset);
        offset += m_frame_queue_num_per_node[m_batch_start[i] + j];
      }

      cudaMemcpyAsync(cpu_params.frame_queue_offset_per_node,
                      frame_queue_offset_per_node.data(), sizeof(int) * node_num,
                      cudaMemcpyHostToDevice, stream);

      cpu_params.node_num = node_num;

      for (size_t j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType;
           j++)
      {
        cudaMemcpyAsync(cpu_params.packet_ingresses + j * node_num,
                        m_packet_ingresses.data() + j * total_node_num +
                            m_batch_start[i],
                        sizeof(GPUQueue<void *> *) * node_num,
                        cudaMemcpyHostToDevice, stream);
      }
      int timestamp_offset = 35;
      cudaMemcpyAsync(cpu_params.l3_timestamp_offset, &timestamp_offset,
                      sizeof(int), cudaMemcpyHostToDevice, stream);

#if ENABLE_CACHE
      // initialize swap packet buffer
      int cache_space = 0;
      for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
      {
        cache_space +=
            frame_queue_num * MAX_TRANSMITTED_PACKET_NUM * m_packet_size[j];
      }
      m_cache_space_size.push_back(cache_space);
      uint8_t *cache_space_gpu;
      cudaMalloc(&cache_space_gpu, cache_space);
      m_cache_space_gpu.push_back(cache_space_gpu);
      uint8_t *cache_space_cpu = new uint8_t[cache_space];
      m_cache_space_cpu.push_back(cache_space_cpu);

      uint8_t **swap_in_packet_buffer_cpu = new uint8_t *[swap_packet_size];
      uint8_t **swap_in_packet_buffer_cpu_backup =
          new uint8_t *[swap_packet_size];
      uint8_t **swap_in_packet_buffer_gpu_backup =
          new uint8_t *[swap_packet_size];

      uint8_t *cache_ptr_gpu = cache_space_gpu;
      uint8_t *cache_ptr_cpu = cache_space_cpu;
      for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
      {
        int swap_offset = j * frame_queue_num * MAX_TRANSMITTED_PACKET_NUM;

        for (int k = 0; k < frame_queue_num * MAX_TRANSMITTED_PACKET_NUM; k++)
        {
          swap_in_packet_buffer_gpu_backup[swap_offset + k] =
              cache_ptr_gpu + k * m_packet_size[j];
          swap_in_packet_buffer_cpu_backup[swap_offset + k] =
              cache_ptr_cpu + k * m_packet_size[j];
        }
        cache_ptr_gpu +=
            frame_queue_num * MAX_TRANSMITTED_PACKET_NUM * m_packet_size[j];
        cache_ptr_cpu +=
            frame_queue_num * MAX_TRANSMITTED_PACKET_NUM * m_packet_size[j];
      }

      cudaMemcpyAsync(
          cpu_params.swap_in_packet_buffer, swap_in_packet_buffer_gpu_backup,
          sizeof(uint8_t *) * swap_packet_size, cudaMemcpyHostToDevice, stream);
      cudaMemsetAsync(cpu_params.swap_packet_num, 0,
                      sizeof(int) *
                          NetworkProtocolType::COUNT_NetworkProtocolType *
                          frame_queue_num,
                      stream);

      m_swap_in_packet_buffer_cpu.push_back(swap_in_packet_buffer_cpu);
      m_swap_in_packet_buffer_cpu_backup.push_back(
          swap_in_packet_buffer_cpu_backup);
      m_swap_in_packet_buffer_gpu_backup.push_back(
          swap_in_packet_buffer_gpu_backup);
      m_swap_in_packet_buffer_gpu.push_back(cpu_params.swap_in_packet_buffer);
      m_swap_packet_num_gpu.push_back(cpu_params.swap_packet_num);
      m_swap_packet_num_cpu.push_back(
          new int[frame_queue_num *
                  NetworkProtocolType::COUNT_NetworkProtocolType]);
#endif

      cudaMemsetAsync(
          cpu_params.recycle_frames, 0,
          sizeof(Frame *) * frame_queue_num * MAX_TRANSMITTED_PACKET_NUM, stream);
      cudaMemsetAsync(cpu_params.recycle_num_per_frame_queue, 0,
                      sizeof(int) * frame_queue_num, stream);

      m_recycle_frames_cpu.push_back(
          new Frame *[frame_queue_num * MAX_TRANSMITTED_PACKET_NUM]);
      m_recycle_num_per_frame_queue_cpu.push_back(new int[frame_queue_num]);
      m_recycle_frames_gpu.push_back(cpu_params.recycle_frames);
      m_recycle_num_per_frame_queue_gpu.push_back(
          cpu_params.recycle_num_per_frame_queue);

      m_recycle_frame_tmp.push_back(
          new Frame *[frame_queue_num * MAX_TRANSMITTED_PACKET_NUM]);

      m_recycle_packets.emplace_back(std::vector<void *>());

      FrameDecapsulationParams *gpu_params;
      cudaMallocAsync(&gpu_params, sizeof(FrameDecapsulationParams), stream);
      cudaMemcpyAsync(gpu_params, &cpu_params, sizeof(FrameDecapsulationParams),
                      cudaMemcpyHostToDevice, stream);
      cudaStreamSynchronize(stream);
      FrameDecapsulationParams cpu_params_2;
      cudaMemcpy(&cpu_params_2, gpu_params, sizeof(FrameDecapsulationParams),
                 cudaMemcpyDeviceToHost);
      m_kernel_params.push_back(gpu_params);
    }
  }

  void FrameDecapsulationConatroller::SetFrameIngress(
      GPUQueue<Frame *> **frame_ingresses, int frame_queue_num)
  {
    m_frame_ingresses.insert(m_frame_ingresses.end(), frame_ingresses,
                             frame_ingresses + frame_queue_num);
  }

  void FrameDecapsulationConatroller::SetPacketIngress(
      GPUQueue<void *> **packet_ingresses, int node_num)
  {
    m_packet_ingresses.insert(
        m_packet_ingresses.end(), (GPUQueue<uint8_t *> **)packet_ingresses,
        (GPUQueue<uint8_t *> **)(packet_ingresses +
                                 NetworkProtocolType::COUNT_NetworkProtocolType *
                                     node_num));
  }

  void FrameDecapsulationConatroller::SetNodeProperties(
      int *frame_queue_num_per_node, int node_num)
  {
    m_frame_queue_num_per_node.insert(m_frame_queue_num_per_node.end(),
                                      frame_queue_num_per_node,
                                      frame_queue_num_per_node + node_num);
  }

  void FrameDecapsulationConatroller::SetBatchProperties(int *batch_start,
                                                         int *batch_end,
                                                         int batch_num)
  {
    m_batch_start.insert(m_batch_start.end(), batch_start,
                         batch_start + batch_num);
    m_batch_end.insert(m_batch_end.end(), batch_end, batch_end + batch_num);
  }

  void FrameDecapsulationConatroller::SetStreams(cudaStream_t *streams,
                                                 int stream_num)
  {
    m_streams.insert(m_streams.end(), streams, streams + stream_num);
  }

  void FrameDecapsulationConatroller::BuildGraphs()
  {
    int batch_num = m_batch_start.size();
    for (int i = 0; i < batch_num; i++)
    {
      BuildGraph(i);
    }
  }

  void FrameDecapsulationConatroller::BuildGraph(int batch_id)
  {
    int node_num = m_batch_end[batch_id] - m_batch_start[batch_id];

    int frame_queue_num = m_frame_ingress_num[batch_id];

    dim3 block_dim(KERNEL_BLOCK_WIDTH);
    dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);

    std::vector<cudaGraphNode_t> nodes(1);

    cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
    LaunchFrameDecapsulationKernel(grid_dim, block_dim, m_kernel_params[batch_id],
                                   m_streams[batch_id]);
    cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
    cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL,
                         0);

#if ENABLE_HUGE_GRAPH

    // cudaGraphNode_t recycle_packet_num_node;
    cudaMemcpy3DParms recycle_packet_num_params = {0};
    recycle_packet_num_params.srcPtr =
        make_cudaPitchedPtr(m_recycle_num_per_frame_queue_gpu[batch_id],
                            frame_queue_num * sizeof(int), frame_queue_num, 1);
    recycle_packet_num_params.dstPtr =
        make_cudaPitchedPtr(m_recycle_num_per_frame_queue_cpu[batch_id],
                            frame_queue_num * sizeof(int), frame_queue_num, 1);
    recycle_packet_num_params.extent =
        make_cudaExtent(frame_queue_num * sizeof(int), 1, 1);
    recycle_packet_num_params.kind = cudaMemcpyDeviceToHost;

    // cudaGraphNode_t recycle_frame_node;
    cudaMemcpy3DParms recycle_frame_params = {0};
    recycle_frame_params.srcPtr = make_cudaPitchedPtr(
        m_recycle_frames_gpu[batch_id],
        sizeof(Frame *) * MAX_TRANSMITTED_PACKET_NUM * frame_queue_num,
        frame_queue_num * MAX_TRANSMITTED_PACKET_NUM, 1);
    recycle_frame_params.dstPtr = make_cudaPitchedPtr(
        m_recycle_frames_cpu[batch_id],
        sizeof(Frame *) * MAX_TRANSMITTED_PACKET_NUM * frame_queue_num,
        frame_queue_num * MAX_TRANSMITTED_PACKET_NUM, 1);
    recycle_frame_params.extent = make_cudaExtent(
        sizeof(Frame *) * frame_queue_num * MAX_TRANSMITTED_PACKET_NUM, 1, 1);
    recycle_frame_params.kind = cudaMemcpyDeviceToHost;

    // cudaGraphNode_t recycle_node;
    cudaHostNodeParams recycle_params = {0};
    auto recycle_func =
        std::bind(&FrameDecapsulationConatroller::RecycleFrames, this, batch_id);
    auto recycle_func_ptr = new std::function<void()>(recycle_func);
    recycle_params.fn = VDES::HostNodeCallback;
    recycle_params.userData = recycle_func_ptr;

    m_memcpy_param.push_back(recycle_packet_num_params);
    m_memcpy_param.push_back(recycle_frame_params);
    m_host_param.push_back(recycle_params);

#endif
  }

#if ENBALE_CACHE
  void FrameDecapsulationConatroller::CacheInPackets(int batch_id)
  {
    cudaStreamSynchronize(m_streams[batch_id]);

    // cache in packets
    cudaMemcpyAsync(m_swap_packet_num_cpu[batch_id],
                    m_swap_packet_num_gpu[batch_id],
                    sizeof(int) * m_frame_ingress_num[batch_id] *
                        NetworkProtocolType::COUNT_NetworkProtocolType,
                    cudaMemcpyDeviceToHost, m_cache_streams[batch_id]);
    int swap_buffer_size = m_frame_ingress_num[batch_id] *
                           MAX_TRANSMITTED_PACKET_NUM *
                           NetworkProtocolType::COUNT_NetworkProtocolType;
    cudaMemcpyAsync(m_swap_in_packet_buffer_cpu[batch_id],
                    m_swap_in_packet_buffer_gpu[batch_id],
                    sizeof(uint8_t *) * swap_buffer_size, cudaMemcpyDeviceToHost,
                    m_cache_streams[batch_id]);
    cudaMemcpyAsync(m_cache_space_cpu[batch_id], m_cache_space_gpu[batch_id],
                    m_cache_space_size[batch_id], cudaMemcpyDeviceToHost,
                    m_cache_streams[batch_id]);

    // copy packets to cache space

    uint8_t **dst = m_swap_in_packet_buffer_cpu_backup[batch_id];
    uint8_t **src = m_swap_in_packet_buffer_cpu[batch_id];

    for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
    {
      for (int j = 0; j < m_frame_ingress_num[batch_id]; j++)
      {

        int cache_queue_id = i * m_frame_ingress_num[batch_id] + j;
        int cache_packet_num = m_swap_packet_num_cpu[batch_id][cache_queue_id];
        int offset = cache_queue_id * MAX_TRANSMITTED_PACKET_NUM;

        for (int k = 0; k < cache_packet_num; k++)
        {
          memcpy(dst[offset + k], src[offset + k], m_native_packet_size[i]);
          m_recycle_packets[batch_id].push_back(src[offset + k]);
        }
      }
    }

    cudaMemcpyAsync(m_cache_space_gpu[batch_id], m_cache_space_cpu[batch_id],
                    m_cache_space_size[batch_id], cudaMemcpyHostToDevice,
                    m_cache_streams[batch_id]);
  }

  void FrameDecapsulationConatroller::RecyclePackets(int batch_id)
  {
    // recycle ipv4 packets
    int start = std::accumulate(m_swap_packet_num_cpu[batch_id],
                                m_swap_packet_num_cpu[batch_id] +
                                    NetworkProtocolType::IPv4 *
                                        m_frame_ingress_num[batch_id],
                                0);
    int end = std::accumulate(m_swap_packet_num_cpu[batch_id],
                              m_swap_packet_num_cpu[batch_id] +
                                  (NetworkProtocolType::IPv4 + 1) *
                                      m_frame_ingress_num[batch_id],
                              0);
    ipv4_packet_pool_cpu->deallocate(
        (Ipv4Packet **)m_recycle_packets[batch_id].data() + start, end - start);

    m_recycle_packets[batch_id].clear();

    // reset swap_in_packet_buffer
    cudaMemcpy(m_swap_in_packet_buffer_gpu[batch_id],
               m_swap_in_packet_buffer_gpu_backup[batch_id],
               sizeof(void *) * m_frame_ingress_num[batch_id] *
                   MAX_TRANSMITTED_PACKET_NUM *
                   NetworkProtocolType::COUNT_NetworkProtocolType,
               cudaMemcpyHostToDevice);
  }

#endif

  void FrameDecapsulationConatroller::RecycleFrames(int batch_id)
  {
#if !ENABLE_HUGE_GRAPH
    // recycle frames
    cudaMemcpy(m_recycle_num_per_frame_queue_cpu[batch_id],
               m_recycle_num_per_frame_queue_gpu[batch_id],
               sizeof(int) * m_frame_ingress_num[batch_id],
               cudaMemcpyDeviceToHost);
    cudaMemcpy(m_recycle_frames_cpu[batch_id], m_recycle_frames_gpu[batch_id],
               sizeof(Frame *) * m_frame_ingress_num[batch_id] *
                   MAX_TRANSMITTED_PACKET_NUM,
               cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(m_streams[batch_id]);
#endif

    int insert_index = 0;
    Frame **dst_recycle = m_recycle_frame_tmp[batch_id];
    Frame **recycle_frames = m_recycle_frames_cpu[batch_id];
    int *recycle_num_per_frame_queue =
        m_recycle_num_per_frame_queue_cpu[batch_id];

    for (int i = 0; i < m_frame_ingress_num[batch_id]; i++)
    {
      memcpy(dst_recycle + insert_index,
             recycle_frames + i * MAX_TRANSMITTED_PACKET_NUM,
             sizeof(Frame *) * recycle_num_per_frame_queue[i]);

      insert_index += recycle_num_per_frame_queue[i];
    }

    frame_pool->deallocate(dst_recycle, insert_index);
    memset(dst_recycle, 0,
           sizeof(Frame *) * MAX_TRANSMITTED_PACKET_NUM *
               m_frame_ingress_num[batch_id]);
  }

  void FrameDecapsulationConatroller::LaunchInstance(int batch_id)
  {
    cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
  }

  void FrameDecapsulationConatroller::SynchronizeCache(int batch_id)
  {
    cudaStreamSynchronize(m_cache_streams[batch_id]);
  }

  void FrameDecapsulationConatroller::Run(int batch_id)
  {
    cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);

#if ENABLE_CACHE
    CacheInPackets(batch_id);

    RecyclePackets(batch_id);
#endif

    RecycleFrames(batch_id);
  }

  cudaGraph_t FrameDecapsulationConatroller::GetGraph(int batch_id)
  {
    return m_graphs[batch_id];
  }

#if ENABLE_HUGE_GRAPH
  std::vector<cudaMemcpy3DParms> &
  FrameDecapsulationConatroller::GetMemcpyParams()
  {
    return m_memcpy_param;
  }

  std::vector<cudaHostNodeParams> &
  FrameDecapsulationConatroller::GetHostParams()
  {
    return m_host_param;
  }
#endif

  std::vector<void *> FrameDecapsulationConatroller::GetRecycleInfo()
  {
    int batch_num = m_batch_start.size();
    std::vector<void *> res;
    for (int i = 0; i < batch_num; i++)
    {
      res.push_back(m_recycle_frames_gpu[i]);
      res.push_back(m_recycle_num_per_frame_queue_gpu[i]);
      FrameDecapsulationParams cpu_param;
      cudaMemcpy(&cpu_param, m_kernel_params[i], sizeof(FrameDecapsulationParams), cudaMemcpyDeviceToHost);
      res.push_back(cpu_param.frame_queue_offset_per_node);
    }
    return res;
  }

} // namespace VDES

// namespace VDES
// {
//     FrameDecapsulationConatroller::FrameDecapsulationConatroller()
//     {
//     }

//     FrameDecapsulationConatroller::~FrameDecapsulationConatroller()
//     {
//     }

//     void FrameDecapsulationConatroller::InitializeKernelParams()
//     {
//         int batch_num = m_batch_start.size();
//         m_graph_execs.resize(batch_num);

//         m_packet_size.push_back(sizeof(Ipv4Packet));
//         m_native_packet_size.push_back(35);

//         int total_node_num = m_batch_end[batch_num - 1];
//         for (int i = 0; i < batch_num; i++)
//         {
//             int node_num = m_batch_end[i] - m_batch_start[i];
//             int frame_queue_num =
//             std::accumulate(m_frame_queue_num_per_node.begin() +
//             m_batch_start[i], m_frame_queue_num_per_node.begin() +
//             m_batch_end[i], 0);
//             m_frame_ingress_num.push_back(frame_queue_num);

//             cudaStream_t stream;
//             cudaStreamCreate(&stream);
//             m_cache_streams.push_back(stream);

//             cudaGraph_t graph;
//             cudaGraphCreate(&graph, 0);
//             m_graphs.push_back(graph);

//             FrameDecapsulationParams cpu_params;
//             int packet_queue_num = node_num *
//             NetworkProtocolType::COUNT_NetworkProtocolType; int
//             swap_packet_size = frame_queue_num *
//             NetworkProtocolType::COUNT_NetworkProtocolType *
//             MAX_TRANSMITTED_PACKET_NUM;

//             cudaMallocAsync(&cpu_params.frame_ingresses,
//             sizeof(GPUQueue<Frame *> *) * frame_queue_num, stream);
//             cudaMallocAsync(&cpu_params.frame_queue_num_per_node, sizeof(int)
//             * node_num, stream);
//             cudaMallocAsync(&cpu_params.frame_queue_offset_per_node,
//             sizeof(int) * node_num, stream);
//             cudaMallocAsync(&cpu_params.packet_ingresses,
//             sizeof(GPUQueue<void *> *) * packet_queue_num, stream);

// #if ENABLE_CACHE
//             cudaMallocAsync(&cpu_params.swap_in_packet_buffer, sizeof(void *)
//             * swap_packet_size, stream);
//             cudaMallocAsync(&cpu_params.swap_packet_num, sizeof(int) *
//             NetworkProtocolType::COUNT_NetworkProtocolType * frame_queue_num,
//             stream);
// #endif
//             cudaMallocAsync(&cpu_params.recycle_frames, sizeof(Frame *) *
//             frame_queue_num * MAX_TRANSMITTED_PACKET_NUM, stream);
//             cudaMallocAsync(&cpu_params.recycle_num_per_frame_queue,
//             sizeof(int) * frame_queue_num, stream);
//             cudaMallocAsync(&cpu_params.l3_timestamp_offset, sizeof(int) *
//             NetworkProtocolType::COUNT_NetworkProtocolType, stream);
//             cudaStreamSynchronize(stream);

//             int frame_queue_offset =
//             std::accumulate(m_frame_queue_num_per_node.begin(),
//             m_frame_queue_num_per_node.begin() + m_batch_start[i], 0);
//             cudaMemcpyAsync(cpu_params.frame_ingresses,
//             m_frame_ingresses.data() + frame_queue_offset,
//             sizeof(GPUQueue<Frame *> *) * frame_queue_num,
//             cudaMemcpyHostToDevice, stream); cpu_params.queue_num =
//             frame_queue_num;
//             cudaMemcpyAsync(cpu_params.frame_queue_num_per_node,
//             m_frame_queue_num_per_node.data() + m_batch_start[i], sizeof(int)
//             * node_num, cudaMemcpyHostToDevice, stream);

//             std::vector<int> frame_queue_offset_per_node;
//             int offset = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 frame_queue_offset_per_node.push_back(offset);
//                 offset += m_frame_queue_num_per_node[m_batch_start[i] + j];
//             }

//             cudaMemcpyAsync(cpu_params.frame_queue_offset_per_node,
//             frame_queue_offset_per_node.data(), sizeof(int) * node_num,
//             cudaMemcpyHostToDevice, stream);

//             cpu_params.node_num = node_num;

//             for (size_t j = 0; j <
//             NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 cudaMemcpyAsync(cpu_params.packet_ingresses + j * node_num,
//                 m_packet_ingresses.data() + j * total_node_num +
//                 m_batch_start[i], sizeof(GPUQueue<void *> *) * node_num,
//                 cudaMemcpyHostToDevice, stream);
//             }
//             int timestamp_offset = 35;
//             cudaMemcpyAsync(cpu_params.l3_timestamp_offset,
//             &timestamp_offset, sizeof(int), cudaMemcpyHostToDevice, stream);

// #if ENABLE_CACHE
//             // initialize swap packet buffer
//             int cache_space = 0;
//             for (int j = 0; j <
//             NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 cache_space += frame_queue_num * MAX_TRANSMITTED_PACKET_NUM *
//                 m_packet_size[j];
//             }
//             m_cache_space_size.push_back(cache_space);
//             uint8_t *cache_space_gpu;
//             cudaMalloc(&cache_space_gpu, cache_space);
//             m_cache_space_gpu.push_back(cache_space_gpu);
//             uint8_t *cache_space_cpu = new uint8_t[cache_space];
//             m_cache_space_cpu.push_back(cache_space_cpu);

//             uint8_t **swap_in_packet_buffer_cpu = new uint8_t
//             *[swap_packet_size]; uint8_t **swap_in_packet_buffer_cpu_backup =
//             new uint8_t *[swap_packet_size]; uint8_t
//             **swap_in_packet_buffer_gpu_backup = new uint8_t
//             *[swap_packet_size];

//             uint8_t *cache_ptr_gpu = cache_space_gpu;
//             uint8_t *cache_ptr_cpu = cache_space_cpu;
//             for (int j = 0; j <
//             NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 int swap_offset = j * frame_queue_num *
//                 MAX_TRANSMITTED_PACKET_NUM;

//                 for (int k = 0; k < frame_queue_num *
//                 MAX_TRANSMITTED_PACKET_NUM; k++)
//                 {
//                     swap_in_packet_buffer_gpu_backup[swap_offset + k] =
//                     cache_ptr_gpu + k * m_packet_size[j];
//                     swap_in_packet_buffer_cpu_backup[swap_offset + k] =
//                     cache_ptr_cpu + k * m_packet_size[j];
//                 }
//                 cache_ptr_gpu += frame_queue_num * MAX_TRANSMITTED_PACKET_NUM
//                 * m_packet_size[j]; cache_ptr_cpu += frame_queue_num *
//                 MAX_TRANSMITTED_PACKET_NUM * m_packet_size[j];
//             }

//             cudaMemcpyAsync(cpu_params.swap_in_packet_buffer,
//             swap_in_packet_buffer_gpu_backup, sizeof(void *) *
//             swap_packet_size, cudaMemcpyHostToDevice, stream);
//             cudaMemsetAsync(cpu_params.swap_packet_num, 0, sizeof(int) *
//             NetworkProtocolType::COUNT_NetworkProtocolType * frame_queue_num,
//             stream);

//             m_swap_in_packet_buffer_cpu.push_back(swap_in_packet_buffer_cpu);
//             m_swap_in_packet_buffer_cpu_backup.push_back(swap_in_packet_buffer_cpu_backup);
//             m_swap_in_packet_buffer_gpu_backup.push_back(swap_in_packet_buffer_gpu_backup);
//             m_swap_in_packet_buffer_gpu.push_back(cpu_params.swap_in_packet_buffer);
//             m_swap_packet_num_gpu.push_back(cpu_params.swap_packet_num);
//             m_swap_packet_num_cpu.push_back(new int[frame_queue_num *
//             NetworkProtocolType::COUNT_NetworkProtocolType]);

// #endif

//             cudaMemsetAsync(cpu_params.recycle_frames, 0, sizeof(Frame *) *
//             frame_queue_num * MAX_TRANSMITTED_PACKET_NUM, stream);
//             cudaMemsetAsync(cpu_params.recycle_num_per_frame_queue, 0,
//             sizeof(int) * frame_queue_num, stream);

//             m_recycle_frames_cpu.push_back(new Frame *[frame_queue_num *
//             MAX_TRANSMITTED_PACKET_NUM]);
//             m_recycle_num_per_frame_queue_cpu.push_back(new
//             int[frame_queue_num]);
//             m_recycle_frames_gpu.push_back(cpu_params.recycle_frames);
//             m_recycle_num_per_frame_queue_gpu.push_back(cpu_params.recycle_num_per_frame_queue);

//             m_recycle_frame_tmp.push_back(new Frame *[frame_queue_num *
//             MAX_TRANSMITTED_PACKET_NUM]);

//             m_recycle_packets.emplace_back(std::vector<void *>());

//             FrameDecapsulationParams *gpu_params;
//             cudaMallocAsync(&gpu_params, sizeof(FrameDecapsulationParams),
//             stream); cudaMemcpyAsync(gpu_params, &cpu_params,
//             sizeof(FrameDecapsulationParams), cudaMemcpyHostToDevice,
//             stream); cudaStreamSynchronize(stream); FrameDecapsulationParams
//             cpu_params_2; cudaMemcpy(&cpu_params_2, gpu_params,
//             sizeof(FrameDecapsulationParams), cudaMemcpyDeviceToHost);
//             m_kernel_params.push_back(gpu_params);
//         }
//     }

//     void FrameDecapsulationConatroller::SetFrameIngress(GPUQueue<Frame *>
//     **frame_ingresses, int frame_queue_num)
//     {
//         m_frame_ingresses.insert(m_frame_ingresses.end(), frame_ingresses,
//         frame_ingresses + frame_queue_num);
//         // m_frame_ingresses_ts.insert(m_frame_ingresses_ts.end(),
//         frame_ingresses_ts, frame_ingresses_ts + frame_queue_num);
//     }

//     void FrameDecapsulationConatroller::SetPacketIngress(GPUQueue<void *>
//     **packet_ingresses, int node_num)
//     {
//         m_packet_ingresses.insert(m_packet_ingresses.end(), (GPUQueue<uint8_t
//         *> **)packet_ingresses, (GPUQueue<uint8_t *> **)(packet_ingresses +
//         NetworkProtocolType::COUNT_NetworkProtocolType * node_num));
//         // m_packet_ingresses_ts.insert(m_packet_ingresses_ts.end(),
//         packet_ingresses_ts, packet_ingresses_ts +
//         NetworkProtocolType::COUNT_NetworkProtocolType * node_num);
//     }

//     void FrameDecapsulationConatroller::SetNodeProperties(int
//     *frame_queue_num_per_node, int node_num)
//     {
//         m_frame_queue_num_per_node.insert(m_frame_queue_num_per_node.end(),
//         frame_queue_num_per_node, frame_queue_num_per_node + node_num);
//     }

//     void FrameDecapsulationConatroller::SetBatchProperties(int *batch_start,
//     int *batch_end, int batch_num)
//     {
//         m_batch_start.insert(m_batch_start.end(), batch_start, batch_start +
//         batch_num); m_batch_end.insert(m_batch_end.end(), batch_end,
//         batch_end + batch_num);
//     }

//     void FrameDecapsulationConatroller::SetStreams(cudaStream_t *streams, int
//     stream_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + stream_num);
//     }

//     void FrameDecapsulationConatroller::BuildGraphs()
//     {
//         int batch_num = m_batch_start.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraph(i);
//         }
//     }

//     void FrameDecapsulationConatroller::BuildGraph(int batch_id)
//     {
//         int node_num = m_batch_end[batch_id] - m_batch_start[batch_id];

//         int frame_queue_num = m_frame_ingress_num[batch_id];

//         dim3 block_dim(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);

//         std::vector<cudaGraphNode_t> nodes(1);

//         cudaStreamBeginCapture(m_streams[batch_id],
//         cudaStreamCaptureModeGlobal);
//         LaunchFrameDecapsulationKernel(grid_dim, block_dim,
//         m_kernel_params[batch_id], m_streams[batch_id]);
//         // LaunchSortPacketKernel(grid_dim, block_dim,
//         m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id],
//         NULL, NULL, 0);

// #if ENABLE_HUGE_GRAPH

//         // cudaGraphNode_t recycle_packet_num_node;
//         cudaMemcpy3DParms recycle_packet_num_params = {0};
//         recycle_packet_num_params.srcPtr =
//         make_cudaPitchedPtr(m_recycle_num_per_frame_queue_gpu[batch_id],
//         frame_queue_num * sizeof(int), frame_queue_num, 1);
//         recycle_packet_num_params.dstPtr =
//         make_cudaPitchedPtr(m_recycle_num_per_frame_queue_cpu[batch_id],
//         frame_queue_num * sizeof(int), frame_queue_num, 1);
//         recycle_packet_num_params.extent = make_cudaExtent(frame_queue_num *
//         sizeof(int), 1, 1); recycle_packet_num_params.kind =
//         cudaMemcpyDeviceToHost;
//         // cudaGraphAddMemcpyNode(&recycle_packet_num_node,
//         m_graphs[batch_id], nodes.data(), 1, &recycle_packet_num_params);

//         // cudaGraphNode_t recycle_frame_node;
//         cudaMemcpy3DParms recycle_frame_params = {0};
//         recycle_frame_params.srcPtr =
//         make_cudaPitchedPtr(m_recycle_frames_gpu[batch_id], sizeof(Frame *) *
//         MAX_TRANSMITTED_PACKET_NUM * frame_queue_num, frame_queue_num *
//         MAX_TRANSMITTED_PACKET_NUM, 1); recycle_frame_params.dstPtr =
//         make_cudaPitchedPtr(m_recycle_frames_cpu[batch_id], sizeof(Frame *) *
//         MAX_TRANSMITTED_PACKET_NUM * frame_queue_num, frame_queue_num *
//         MAX_TRANSMITTED_PACKET_NUM, 1); recycle_frame_params.extent =
//         make_cudaExtent(sizeof(Frame *) * frame_queue_num *
//         MAX_TRANSMITTED_PACKET_NUM, 1, 1); recycle_frame_params.kind =
//         cudaMemcpyDeviceToHost;
//         // cudaGraphAddMemcpyNode(&recycle_frame_node, m_graphs[batch_id],
//         nodes.data(), 1, &recycle_frame_params);

//         // std::vector<cudaGraphNode_t> memcpy_nodes;
//         // memcpy_nodes.push_back(recycle_packet_num_node);
//         // memcpy_nodes.push_back(recycle_frame_node);

//         // cudaGraphNode_t recycle_node;
//         cudaHostNodeParams recycle_params = {0};
//         auto recycle_func =
//         std::bind(&FrameDecapsulationConatroller::RecycleFrames, this,
//         batch_id); auto recycle_func_ptr = new
//         std::function<void()>(recycle_func); recycle_params.fn =
//         VDES::HostNodeCallback; recycle_params.userData = recycle_func_ptr;
//         // cudaGraphAddHostNode(&recycle_node, m_graphs[batch_id],
//         memcpy_nodes.data(), 2, &recycle_params);

//         m_memcpy_param.push_back(recycle_packet_num_params);
//         m_memcpy_param.push_back(recycle_frame_params);
//         m_host_param.push_back(recycle_params);

// #endif
//     }

// #if ENABLE_CACHE
//     void FrameDecapsulationConatroller::CacheInPackets(int batch_id)
//     {
//         /**
//          * @TODO: Update the event sycchronization.
//          */
//         // cudaStreamSynchronize(m_streams[batch_id]);

//         // cache in packets
//         cudaMemcpy(m_swap_packet_num_cpu[batch_id],
//         m_swap_packet_num_gpu[batch_id], sizeof(int) *
//         m_frame_ingress_num[batch_id] *
//         NetworkProtocolType::COUNT_NetworkProtocolType,
//         cudaMemcpyDeviceToHost); int swap_buffer_size =
//         m_frame_ingress_num[batch_id] * MAX_TRANSMITTED_PACKET_NUM *
//         NetworkProtocolType::COUNT_NetworkProtocolType;
//         cudaMemcpy(m_swap_in_packet_buffer_cpu[batch_id],
//         m_swap_in_packet_buffer_gpu[batch_id], sizeof(uint8_t *) *
//         swap_buffer_size, cudaMemcpyDeviceToHost);
//         cudaMemcpy(m_cache_space_cpu[batch_id], m_cache_space_gpu[batch_id],
//         m_cache_space_size[batch_id], cudaMemcpyDeviceToHost);
//         // cudaStreamSynchronize(m_cache_streams[batch_id]);

//         // copy packets to cache space

//         uint8_t **dst = m_swap_in_packet_buffer_cpu_backup[batch_id];
//         uint8_t **src = m_swap_in_packet_buffer_cpu[batch_id];

//         for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType;
//         i++)
//         {
//             for (int j = 0; j < m_frame_ingress_num[batch_id]; j++)
//             {

//                 int cache_queue_id = i * m_frame_ingress_num[batch_id] + j;
//                 int cache_packet_num =
//                 m_swap_packet_num_cpu[batch_id][cache_queue_id]; int offset =
//                 cache_queue_id * MAX_TRANSMITTED_PACKET_NUM;

//                 for (int k = 0; k < cache_packet_num; k++)
//                 {
//                     memcpy(dst[offset + k], src[offset + k],
//                     m_native_packet_size[i]);
//                     m_recycle_packets[batch_id].push_back(src[offset + k]);
//                 }
//             }
//         }

//         // modified
//         cudaMemcpy(m_cache_space_gpu[batch_id], m_cache_space_cpu[batch_id],
//         m_cache_space_size[batch_id], cudaMemcpyHostToDevice);
//     }

//     void FrameDecapsulationConatroller::RecyclePackets(int batch_id)
//     {
//         // recycle ipv4 packets
//         int start = std::accumulate(m_swap_packet_num_cpu[batch_id],
//         m_swap_packet_num_cpu[batch_id] + NetworkProtocolType::IPv4 *
//         m_frame_ingress_num[batch_id], 0); int end =
//         std::accumulate(m_swap_packet_num_cpu[batch_id],
//         m_swap_packet_num_cpu[batch_id] + (NetworkProtocolType::IPv4 + 1) *
//         m_frame_ingress_num[batch_id], 0);
//         ipv4_packet_pool_cpu->deallocate((Ipv4Packet
//         **)m_recycle_packets[batch_id].data() + start, end - start);

//         m_recycle_packets[batch_id].clear();

//         // reset swap_in_packet_buffer
//         cudaMemcpy(m_swap_in_packet_buffer_gpu[batch_id],
//         m_swap_in_packet_buffer_gpu_backup[batch_id], sizeof(void *) *
//         m_frame_ingress_num[batch_id] * MAX_TRANSMITTED_PACKET_NUM *
//         NetworkProtocolType::COUNT_NetworkProtocolType,
//         cudaMemcpyHostToDevice);
//     }

// #endif

//     void FrameDecapsulationConatroller::RecycleFrames(int batch_id)
//     {
// #if !ENABLE_HUGE_GRAPH
//         // recycle frames
//         cudaMemcpy(m_recycle_num_per_frame_queue_cpu[batch_id],
//         m_recycle_num_per_frame_queue_gpu[batch_id], sizeof(int) *
//         m_frame_ingress_num[batch_id], cudaMemcpyDeviceToHost);
//         cudaMemcpy(m_recycle_frames_cpu[batch_id],
//         m_recycle_frames_gpu[batch_id], sizeof(Frame *) *
//         m_frame_ingress_num[batch_id] * MAX_TRANSMITTED_PACKET_NUM,
//         cudaMemcpyDeviceToHost); cudaStreamSynchronize(m_streams[batch_id]);
// #endif

//         int insert_index = 0;
//         Frame **dst_recycle = m_recycle_frame_tmp[batch_id];
//         Frame **recycle_frames = m_recycle_frames_cpu[batch_id];
//         int *recycle_num_per_frame_queue =
//         m_recycle_num_per_frame_queue_cpu[batch_id];

//         for (int i = 0; i < m_frame_ingress_num[batch_id]; i++)
//         {
//             memcpy(dst_recycle + insert_index, recycle_frames + i *
//             MAX_TRANSMITTED_PACKET_NUM, sizeof(Frame *) *
//             recycle_num_per_frame_queue[i]);

//             insert_index += recycle_num_per_frame_queue[i];
//         }

//         frame_pool->deallocate(dst_recycle, insert_index);
//         memset(dst_recycle, 0, sizeof(Frame *) * MAX_TRANSMITTED_PACKET_NUM *
//         m_frame_ingress_num[batch_id]);
//     }

//     void FrameDecapsulationConatroller::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void FrameDecapsulationConatroller::SynchronizeCache(int batch_id)
//     {
//         cudaStreamSynchronize(m_cache_streams[batch_id]);
//     }

//     void FrameDecapsulationConatroller::Run(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);

// #if ENABLE_CACHE

//         CacheInPackets(batch_id);
//         RecyclePackets(batch_id);
// #endif

//         RecycleFrames(batch_id);
//     }

//     // void FrameDecapsulationConatroller::HandleCacheAndRecycle(int
//     batch_id)
//     // {
//     //     CacheInPackets(batch_id);
//     //     RecyclePackets(batch_id);
//     //     RecycleFrames(batch_id);
//     // }

//     cudaGraph_t FrameDecapsulationConatroller::GetGraph(int batch_id)
//     {
//         return m_graphs[batch_id];
//     }

// #if ENABLE_HUGE_GRAPH
//     std::vector<cudaMemcpy3DParms>
//     &FrameDecapsulationConatroller::GetMemcpyParams()
//     {
//         return m_memcpy_param;
//     }

//     std::vector<cudaHostNodeParams>
//     &FrameDecapsulationConatroller::GetHostParams()
//     {
//         return m_host_param;
//     }
// #endif

// }

// #include "frame_decapsulation.h"

// namespace VDES
// {
//     FrameDecapsulationConatroller::FrameDecapsulationConatroller()
//     {
//     }

//     FrameDecapsulationConatroller::~FrameDecapsulationConatroller()
//     {
//     }

//     void FrameDecapsulationConatroller::InitializeKernelParams()
//     {
//         int batch_num = m_batch_start.size();
//         m_graph_execs.resize(batch_num);

//         m_packet_size.push_back(sizeof(Ipv4Packet));

//         for (int i = 0; i < batch_num; i++)
//         {
//             int node_num = m_batch_end[i] - m_batch_start[i];
//             int frame_queue_num =
//             std::accumulate(m_frame_queue_num_per_node.begin() +
//             m_batch_start[i], m_frame_queue_num_per_node.begin() +
//             m_batch_end[i], 0);
//             m_frame_ingress_num.push_back(frame_queue_num);

//             cudaStream_t stream;
//             cudaStreamCreate(&stream);
//             m_streams.push_back(stream);
//             cudaStreamCreate(&stream);
//             m_cache_streams.push_back(stream);

//             cudaEvent_t event;
//             cudaEventCreate(&event);
//             m_events.push_back(event);

//             cudaGraph_t graph;
//             cudaGraphCreate(&graph, 0);
//             m_graphs.push_back(graph);

//             FrameDecapsulationParams cpu_params;
//             int packet_queue_num = frame_queue_num *
//             NetworkProtocolType::COUNT_NetworkProtocolType; int
//             swap_packet_size = packet_queue_num * MAX_TRANSMITTED_PACKET_NUM;

//             cudaMallocAsync(&cpu_params.frame_ingresses,
//             sizeof(GPUQueue<Frame *> *) * frame_queue_num, stream);
//             cudaMallocAsync(&cpu_params.frame_ingresses_ts,
//             sizeof(GPUQueue<int64_t> *) * frame_queue_num, stream);
//             cudaMallocAsync(&cpu_params.frame_queue_num_per_node, sizeof(int)
//             * node_num, stream);
//             cudaMallocAsync(&cpu_params.frame_queue_offset_per_node,
//             sizeof(int) * node_num, stream);
//             cudaMallocAsync(&cpu_params.packet_ingresses,
//             sizeof(GPUQueue<void *> *) * packet_queue_num, stream);
//             cudaMallocAsync(&cpu_params.packet_ingresses_ts,
//             sizeof(GPUQueue<int64_t> *) * packet_queue_num, stream);
//             cudaMallocAsync(&cpu_params.swap_in_packet_buffer, sizeof(void *)
//             * swap_packet_size, stream);
//             cudaMallocAsync(&cpu_params.swap_packet_num, sizeof(int) *
//             packet_queue_num, stream);
//             cudaMallocAsync(&cpu_params.recycle_frames, sizeof(Frame *) *
//             frame_queue_num * MAX_TRANSMITTED_PACKET_NUM, stream);
//             cudaMallocAsync(&cpu_params.recycle_num_per_frame_queue,
//             sizeof(int) * frame_queue_num, stream);
//             // cudaStreamSynchronize(stream);

//             int frame_queue_offset =
//             std::accumulate(m_frame_queue_num_per_node.begin(),
//             m_frame_queue_num_per_node.begin() + m_batch_start[i], 0);
//             cudaMemcpyAsync(cpu_params.frame_ingresses,
//             m_frame_ingresses.data() + frame_queue_offset,
//             sizeof(GPUQueue<Frame *> *) * frame_queue_num,
//             cudaMemcpyHostToDevice, stream);
//             cudaMemcpyAsync(cpu_params.frame_ingresses_ts,
//             m_frame_ingresses_ts.data() + frame_queue_offset,
//             sizeof(GPUQueue<int64_t> *) * frame_queue_num,
//             cudaMemcpyHostToDevice, stream); cpu_params.queue_num =
//             frame_queue_num;
//             cudaMemcpyAsync(cpu_params.frame_queue_num_per_node,
//             m_frame_queue_num_per_node.data() + m_batch_start[i], sizeof(int)
//             * node_num, cudaMemcpyHostToDevice, stream);

//             std::vector<int> frame_queue_offset_per_node;
//             int offset = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 frame_queue_offset_per_node.push_back(offset);
//                 offset += m_frame_queue_num_per_node[m_batch_start[i] + j];
//             }

//             cudaMemcpyAsync(cpu_params.frame_queue_offset_per_node,
//             frame_queue_offset_per_node.data(), sizeof(int) * node_num,
//             cudaMemcpyHostToDevice, stream);

//             // initialize swap packet buffer
//             int cache_space = 0;
//             for (int j = 0; j <
//             NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 cache_space += frame_queue_num * MAX_TRANSMITTED_PACKET_NUM *
//                 m_packet_size[j];
//             }
//             m_cache_space_size.push_back(cache_space);
//             char *cache_space_gpu;
//             cudaMalloc(&cache_space_gpu, cache_space);
//             m_cache_space_gpu.push_back(cache_space_gpu);
//             char *cache_space_cpu = new char[cache_space];
//             m_cache_space_cpu.push_back(cache_space_cpu);

//             void **swap_in_packet_buffer_cpu = new void *[swap_packet_size];
//             void **swap_in_packet_buffer_cpu_backup = new void
//             *[swap_packet_size]; void **swap_in_packet_buffer_gpu_backup =
//             new void *[swap_packet_size];

//             char *cache_ptr_gpu = cache_space_gpu;
//             char *cache_ptr_cpu = cache_space_cpu;
//             for (int j = 0; j <
//             NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 int swap_offset = j * frame_queue_num *
//                 MAX_TRANSMITTED_PACKET_NUM;

//                 for (int k = 0; k < frame_queue_num *
//                 MAX_TRANSMITTED_PACKET_NUM; k++)
//                 {
//                     swap_in_packet_buffer_gpu_backup[swap_offset + k] =
//                     cache_ptr_gpu + k * m_packet_size[j];
//                     swap_in_packet_buffer_cpu_backup[swap_offset + k] =
//                     cache_ptr_cpu + k * m_packet_size[j];
//                 }
//                 cache_ptr_gpu += frame_queue_num * MAX_TRANSMITTED_PACKET_NUM
//                 * m_packet_size[j]; cache_ptr_cpu += frame_queue_num *
//                 MAX_TRANSMITTED_PACKET_NUM * m_packet_size[j];
//             }

//             cudaMemcpyAsync(cpu_params.swap_in_packet_buffer,
//             swap_in_packet_buffer_gpu_backup, sizeof(void *) *
//             swap_packet_size, cudaMemcpyHostToDevice, stream);
//             cudaMemsetAsync(cpu_params.swap_packet_num, 0, sizeof(int) *
//             packet_queue_num, stream);

//             m_swap_in_packet_buffer_cpu.push_back(swap_in_packet_buffer_cpu);
//             m_swap_in_packet_buffer_cpu_backup.push_back(swap_in_packet_buffer_cpu_backup);
//             m_swap_in_packet_buffer_gpu_backup.push_back(swap_in_packet_buffer_gpu_backup);
//             m_swap_in_packet_buffer_gpu.push_back(cpu_params.swap_in_packet_buffer);

//             cudaMemsetAsync(cpu_params.recycle_frames, 0, sizeof(Frame *) *
//             frame_queue_num * MAX_TRANSMITTED_PACKET_NUM, stream);
//             cudaMemsetAsync(cpu_params.recycle_num_per_frame_queue, 0,
//             sizeof(int) * frame_queue_num, stream);

//             m_recycle_frames_cpu.push_back(new Frame *[frame_queue_num *
//             MAX_TRANSMITTED_PACKET_NUM]);
//             m_recycle_num_per_frame_queue_cpu.push_back(new
//             int[frame_queue_num]);
//             m_recycle_frames_gpu.push_back(cpu_params.recycle_frames);
//             m_recycle_num_per_frame_queue_gpu.push_back(cpu_params.recycle_num_per_frame_queue);
//             m_swap_packet_num_gpu.push_back(cpu_params.swap_packet_num);
//             m_swap_packet_num_cpu.push_back(new int[packet_queue_num]);
//             m_recycle_packets.emplace_back();

//             FrameDecapsulationParams *gpu_params;
//             cudaMallocAsync(&gpu_params, sizeof(FrameDecapsulationParams),
//             stream); cudaMemcpyAsync(gpu_params, &cpu_params,
//             sizeof(FrameDecapsulationParams), cudaMemcpyHostToDevice,
//             stream); cudaStreamSynchronize(stream);
//             m_kernel_params.push_back(gpu_params);
//         }
//     }

//     void FrameDecapsulationConatroller::SetFrameIngress(GPUQueue<Frame *>
//     **frame_ingresses, GPUQueue<int64_t> **frame_ingresses_ts, int
//     frame_queue_num)
//     {
//         m_frame_ingresses.insert(m_frame_ingresses.end(), frame_ingresses,
//         frame_ingresses + frame_queue_num);
//         m_frame_ingresses_ts.insert(m_frame_ingresses_ts.end(),
//         frame_ingresses_ts, frame_ingresses_ts + frame_queue_num);
//     }

//     void FrameDecapsulationConatroller::SetNodeProperties(int
//     *frame_queue_num_per_node, int node_num)
//     {
//         m_frame_queue_num_per_node.insert(m_frame_queue_num_per_node.end(),
//         frame_queue_num_per_node, frame_queue_num_per_node + node_num);
//     }

//     void FrameDecapsulationConatroller::SetBatchProperties(int *batch_start,
//     int *batch_end, int batch_num)
//     {
//         m_batch_start.insert(m_batch_start.end(), batch_start, batch_start +
//         batch_num); m_batch_end.insert(m_batch_end.end(), batch_end,
//         batch_end + batch_num);
//     }

//     void FrameDecapsulationConatroller::SetStreams(cudaStream_t *streams, int
//     stream_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + stream_num);
//     }

//     void FrameDecapsulationConatroller::BuildGraphs()
//     {
//         int batch_num = m_batch_start.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             int node_num = m_batch_end[i] - m_batch_start[i];

//             dim3 block_dim(KERNEL_BLOCK_WIDTH);
//             dim3 grid_dim((node_num + block_dim.x - 1) / block_dim.x);

//             cudaStreamBeginCapture(m_streams[i],
//             cudaStreamCaptureModeGlobal);
//             LaunchFrameDecapsulationKernel(grid_dim, block_dim,
//             m_kernel_params[i], m_streams[i]); cudaEventRecord(m_events[i],
//             m_streams[i]); cudaStreamEndCapture(m_streams[i], &m_graphs[i]);
//             cudaGraphInstantiate(&m_graph_execs[i], m_graphs[i], NULL, NULL,
//             0);
//         }
//     }

//     void FrameDecapsulationConatroller::CacheInPackets(int batch_id)
//     {
//         cudaEventSynchronize(m_events[batch_id]);

//         // cache in packets
//         cudaMemcpyAsync(m_swap_packet_num_cpu[batch_id],
//         m_swap_packet_num_gpu[batch_id], sizeof(int) *
//         m_frame_ingress_num[batch_id] *
//         NetworkProtocolType::COUNT_NetworkProtocolType,
//         cudaMemcpyDeviceToHost, m_cache_streams[batch_id]); int
//         swap_buffer_size = m_frame_ingress_num[batch_id] *
//         MAX_TRANSMITTED_PACKET_NUM *
//         NetworkProtocolType::COUNT_NetworkProtocolType;
//         cudaMemcpyAsync(m_swap_in_packet_buffer_cpu[batch_id],
//         m_swap_in_packet_buffer_gpu[batch_id], sizeof(void *) *
//         swap_buffer_size, cudaMemcpyDeviceToHost, m_cache_streams[batch_id]);
//         cudaStreamSynchronize(m_cache_streams[batch_id]);

//         // copy packets to cache space
//         // int total_packet_num =
//         std::accumulate(m_swap_packet_num_cpu[batch_id],
//         m_swap_packet_num_cpu[batch_id] + m_frame_ingress_num[batch_id] *
//         NetworkProtocolType::COUNT_NetworkProtocolType, 0);
//         // auto &recycle_packets = m_recycle_packets[batch_id];

//         void **dst = m_swap_in_packet_buffer_cpu_backup[batch_id];
//         void **src = m_swap_in_packet_buffer_cpu[batch_id];

//         for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType;
//         i++)
//         {
//             for (int j = 0; j < m_frame_ingress_num[batch_id]; j++)
//             {

//                 int cache_queue_id = i * m_frame_ingress_num[batch_id] + j;
//                 int cache_packet_num =
//                 m_swap_packet_num_cpu[batch_id][cache_queue_id]; int offset =
//                 cache_queue_id * MAX_TRANSMITTED_PACKET_NUM;

//                 for (int k = 0; k < cache_packet_num; k++)
//                 {
//                     memcpy(dst[offset + k], src[offset + k],
//                     m_packet_size[i]);
//                     m_recycle_packets[batch_id].push_back(src[offset + k]);
//                 }
//             }
//         }

//         cudaMemcpyAsync(m_cache_space_gpu[batch_id],
//         m_cache_space_cpu[batch_id], m_cache_space_size[batch_id],
//         cudaMemcpyHostToDevice, m_cache_streams[batch_id]);
//         // cudaEventRecord(m_events[batch_id], m_cache_streams[batch_id]);
//     }

//     void FrameDecapsulationConatroller::RecycleFrames(int batch_id)
//     {
//         // recycle frames
//         cudaMemcpy(m_recycle_num_per_frame_queue_cpu[batch_id],
//         m_recycle_num_per_frame_queue_gpu[batch_id], sizeof(int) *
//         m_frame_ingress_num[batch_id], cudaMemcpyDeviceToHost);
//         cudaMemcpy(m_recycle_frames_cpu[batch_id],
//         m_recycle_frames_gpu[batch_id], sizeof(Frame *) *
//         m_frame_ingress_num[batch_id] * MAX_TRANSMITTED_PACKET_NUM,
//         cudaMemcpyDeviceToHost);

//         int insert_index = 0;
//         Frame **recycle_frames = m_recycle_frames_cpu[batch_id];
//         int *recycle_num_per_frame_queue =
//         m_recycle_num_per_frame_queue_cpu[batch_id]; for (int i = 0; i <
//         m_frame_ingress_num[batch_id]; i++)
//         {
//             memcpy(recycle_frames + insert_index, recycle_frames + i *
//             MAX_TRANSMITTED_PACKET_NUM, sizeof(Frame *) *
//             recycle_num_per_frame_queue[i]); insert_index +=
//             recycle_num_per_frame_queue[i];
//         }

//         frame_pool->deallocate(recycle_frames, insert_index);
//     }

//     void FrameDecapsulationConatroller::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void FrameDecapsulationConatroller::SynchronizeCache(int batch_id)
//     {
//         cudaStreamSynchronize(m_cache_streams[batch_id]);
//     }

//     void FrameDecapsulationConatroller::Run(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//         CacheInPackets(batch_id);
//         RecycleFrames(batch_id);
//         RecycleFrames(batch_id);
//     }

//     void FrameDecapsulationConatroller::HandleCacheAndRecycle(int batch_id)
//     {
//         CacheInPackets(batch_id);
//         RecycleFrames(batch_id);
//         RecycleFrames(batch_id);
//     }

//     void FrameDecapsulationConatroller::RecyclePackets(int batch_id)
//     {
//         // recycle ipv4 packets
//         int start = std::accumulate(m_swap_packet_num_cpu[batch_id],
//         m_swap_packet_num_cpu[batch_id] + NetworkProtocolType::IPv4 *
//         m_frame_ingress_num[batch_id], 0); int end =
//         std::accumulate(m_swap_packet_num_cpu[batch_id],
//         m_swap_packet_num_cpu[batch_id] + (NetworkProtocolType::IPv4 + 1) *
//         m_frame_ingress_num[batch_id], 0);
//         ipv4_packet_pool_cpu->deallocate((Ipv4Packet
//         **)m_recycle_packets[batch_id].data() + start, end - start);

//         m_recycle_packets[batch_id].clear();

//         // reset swap_in_packet_buffer
//         cudaMemcpy(m_swap_in_packet_buffer_gpu[batch_id],
//         m_swap_in_packet_buffer_gpu_backup[batch_id], sizeof(void *) *
//         m_frame_ingress_num[batch_id] * MAX_TRANSMITTED_PACKET_NUM *
//         NetworkProtocolType::COUNT_NetworkProtocolType,
//         cudaMemcpyHostToDevice);
//     }

// }