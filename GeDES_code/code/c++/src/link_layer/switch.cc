#include "switch.h"
#include <numeric>

namespace VDES
{
    SwitchController::SwitchController()
    {
    }

    SwitchController::~SwitchController()
    {
    }

    void SwitchController::SetIngresAndEgress(GPUQueue<Frame *> **ingresses, GPUQueue<Frame *> **egresses, int *queue_num_per_node, int *sw_id_per_node, int node_num)
    {
        int queue_num = std::accumulate(queue_num_per_node, queue_num_per_node + node_num, 0);
        m_ingresses.insert(m_ingresses.end(), ingresses, ingresses + queue_num);
        m_egresses.insert(m_egresses.end(), egresses, egresses + queue_num);
        m_sw_id_per_node.insert(m_sw_id_per_node.end(), sw_id_per_node, sw_id_per_node + node_num);
        m_queue_num_per_node.insert(m_queue_num_per_node.end(), queue_num_per_node, queue_num_per_node + node_num);
    }

    void SwitchController::SetMacForwardingTable(GPUQueue<MacForwardingRule *> **mac_forwarding_table, int node_num)
    {
        m_mac_forwarding_table.insert(m_mac_forwarding_table.end(), mac_forwarding_table, mac_forwarding_table + node_num);
    }

    void SwitchController::SetBatchproperties(int *batch_start_index, int *batch_end_index, int batch_num)
    {
        m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
        m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
    }

    void SwitchController::SetFtProperties(int ft_k)
    {
        m_ft_k = ft_k;
        m_ft_k_sq_quarter = (m_ft_k * m_ft_k) / 4;
    }

    void SwitchController::SetStreams(cudaStream_t *streams, int num)
    {
        m_streams.insert(m_streams.end(), streams, streams + num);
    }

    void SwitchController::InitalizeKernelParams()
    {
        int batch_num = m_batch_start_index.size();
        m_graphs.resize(batch_num);
        m_graph_execs.resize(batch_num);

        int total_sw_num = m_batch_end_index[batch_num - 1] - m_batch_start_index[0];
        cudaMalloc(&m_transmission_completed, total_sw_num * sizeof(uint8_t));
        uint8_t *transmission_completed = m_transmission_completed;

        for (int i = 0; i < batch_num; i++)
        {
            int node_num = m_batch_end_index[i] - m_batch_start_index[i];
            int queue_num = std::accumulate(m_queue_num_per_node.begin() + m_batch_start_index[i], m_queue_num_per_node.begin() + m_batch_end_index[i], 0);

            SwitchParams cpu_param;
            cudaMallocAsync(&cpu_param.ingresses, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param.egresses, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param.queue_num_per_node, sizeof(int) * node_num, m_streams[i]);
            cudaMallocAsync(&cpu_param.queue_offset_per_node, sizeof(int) * node_num, m_streams[i]);
            cudaMallocAsync(&cpu_param.node_offset_per_queue, sizeof(int) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param.sw_id_per_node, sizeof(int) * node_num, m_streams[i]);
            cudaMallocAsync(&cpu_param.received_packets_per_queue, sizeof(int *) * node_num, m_streams[i]);

#if ENABLE_FATTREE_MODE
            cpu_param.ft_k = m_ft_k;
            cpu_param.ft_k_sq_quarter = m_ft_k_sq_quarter;
            cudaMallocAsync(&cpu_param.ft_current_port_up_forward, sizeof(uint8_t) * node_num, m_streams[i]);
#else
            cudaMallocAsync(&cpu_param.mac_forwarding_table, sizeof(GPUQueue<MacForwardingRule *> *) * node_num, m_streams[i]);
#endif

            cudaMallocAsync(&cpu_param.drop_frames, sizeof(Frame *) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
            cudaMallocAsync(&cpu_param.drop_frame_num, sizeof(int) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param.drop_cache, sizeof(RecycleFramePayload) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
            cpu_param.node_num = node_num;
            cpu_param.queue_num = queue_num;
            m_queue_num_per_batch.push_back(queue_num);

            // Copy data to GPU
            int queue_offset = std::accumulate(m_queue_num_per_node.begin(), m_queue_num_per_node.begin() + m_batch_start_index[i], 0);
            int node_offset = m_batch_start_index[i];
            cudaMemcpyAsync(cpu_param.ingresses, m_ingresses.data() + queue_offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param.egresses, m_egresses.data() + queue_offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param.queue_num_per_node, m_queue_num_per_node.data() + node_offset, sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

            std::vector<int> queue_offset_per_node;
            std::vector<int> node_offset_per_queue;
            int offset = 0;
            for (int j = 0; j < node_num; j++)
            {
                for (int k = 0; k < m_queue_num_per_node[j + node_offset]; k++)
                {
                    node_offset_per_queue.push_back(j);
                }
                queue_offset_per_node.push_back(offset);
                offset += m_queue_num_per_node[j + node_offset];
            }
            cudaStreamSynchronize(m_streams[i]);
            cudaMemcpyAsync(cpu_param.queue_offset_per_node, queue_offset_per_node.data(), sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param.node_offset_per_queue, node_offset_per_queue.data(), sizeof(int) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param.sw_id_per_node, m_sw_id_per_node.data() + node_offset, sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

            std::vector<int *> received_packets_per_queue;
            for (int j = 0; j < node_num; j++)
            {
                int arr_size = m_queue_num_per_node[j + node_offset] * (m_queue_num_per_node[j + node_offset] + 1);
                int *received_packets;
                cudaMalloc(&received_packets, sizeof(int) * arr_size);
                received_packets_per_queue.push_back(received_packets);
            }
            cudaMemcpyAsync(cpu_param.received_packets_per_queue, received_packets_per_queue.data(), sizeof(int *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

#if ENABLE_FATTREE_MODE

            std::vector<uint8_t> ft_port_up_forward;
            std::vector<uint8_t> ft_port_down_forward;
            std::vector<uint8_t> ft_port_num_per_direction;

            for (int j = 0; j < node_num; j++)
            {

                for (int k = 0; k < m_queue_num_per_node[j + node_offset] / 2; k++)
                {
                    ft_port_up_forward.push_back(k);
                    ft_port_down_forward.push_back(k);
                }
                ft_port_num_per_direction.push_back(m_queue_num_per_node[j + node_offset] / 2);
            }
            cudaMemcpyAsync(cpu_param.ft_current_port_up_forward, ft_port_up_forward.data(), sizeof(uint8_t) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

#else

            cudaMemcpyAsync(cpu_param.mac_forwarding_table, m_mac_forwarding_table.data() + node_offset, sizeof(GPUQueue<MacForwardingRule *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
#endif

            cudaMemsetAsync(cpu_param.drop_frame_num, 0, sizeof(int) * queue_num, m_streams[i]);

            m_drop_frames_gpu.push_back(cpu_param.drop_frames);
            m_drop_frames_cpu.push_back(new Frame *[queue_num * MAX_TRANSMITTED_PACKET_NUM]);
            m_drop_frame_num_gpu.push_back(cpu_param.drop_frame_num);
            m_drop_frame_num_cpu.push_back(new int[queue_num]);
            m_drop_cache_gpu.push_back(cpu_param.drop_cache);
            m_drop_cache_cpu.push_back(new RecycleFramePayload[queue_num * MAX_TRANSMITTED_PACKET_NUM]);

            cudaStreamSynchronize(m_streams[i]);

            cpu_param.is_completed = transmission_completed;
            transmission_completed += node_num;

            SwitchParams *gpu_param;
            cudaMalloc(&gpu_param, sizeof(SwitchParams));
            cudaMemcpy(gpu_param, &cpu_param, sizeof(SwitchParams), cudaMemcpyHostToDevice);
            m_kernel_params.push_back(gpu_param);
        }
    }

    void SwitchController::RecycleDropFrames(int batch_id)
    {
        int queue_num = m_queue_num_per_batch[batch_id];
        cudaMemcpyAsync(m_drop_frame_num_cpu[batch_id], m_drop_frame_num_gpu[batch_id], sizeof(int) * queue_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);
        cudaMemcpyAsync(m_drop_frames_cpu[batch_id], m_drop_frames_gpu[batch_id], sizeof(Frame *) * queue_num * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyDeviceToHost, m_streams[batch_id]);
        cudaMemcpyAsync(m_drop_cache_cpu[batch_id], m_drop_cache_gpu[batch_id], sizeof(RecycleFramePayload) * queue_num * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyDeviceToHost, m_streams[batch_id]);
        cudaStreamSynchronize(m_streams[batch_id]);

        std::vector<Frame *> recycle_frames;
        std::vector<Ipv4Packet *> recycle_ipv4_packets;
        std::vector<TCPPacket *> recycle_tcp_packets;

        int *drop_frame_num = m_drop_frame_num_cpu[batch_id];
        VDES::Frame **drop_frames = m_drop_frames_cpu[batch_id];
        VDES::RecycleFramePayload *drop_cache = m_drop_cache_cpu[batch_id];

        for (int i = 0; i < queue_num; i++)
        {
            for (int j = 0; j < drop_frame_num[i]; j++)
            {
                recycle_frames.push_back(drop_frames[j]);
                Ipv4Packet *ipv4_packet = (Ipv4Packet *)drop_cache[j].payload;
                recycle_ipv4_packets.push_back(ipv4_packet);
                recycle_tcp_packets.push_back(*(TCPPacket **)ipv4_packet->payload);
            }
            drop_frames += MAX_TRANSMITTED_PACKET_NUM;
            drop_cache += MAX_TRANSMITTED_PACKET_NUM;
        }

        ipv4_packet_pool_cpu->deallocate(recycle_ipv4_packets.data(), recycle_ipv4_packets.size());
        frame_pool->deallocate(recycle_frames.data(), recycle_frames.size());
        tcp_packet_pool_cpu->deallocate(recycle_tcp_packets.data(), recycle_tcp_packets.size());
    }

    void SwitchController::BuildGraph(int batch_id)
    {
        int queue_num = m_queue_num_per_batch[batch_id];
        int block_width = std::min(std::lcm(KERNEL_BLOCK_HEIGHT, m_ft_k), 1024);
        dim3 block_dim(block_width);
        dim3 grid_dim((queue_num + block_dim.x - 1) / block_dim.x);
        cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
        LaunchForwardFramesKernel(grid_dim, block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
        cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
        cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);
    }

    void SwitchController::BuildGraph()
    {
        int batch_num = m_batch_start_index.size();
        for (int i = 0; i < batch_num; i++)
        {
            BuildGraph(i);
        }
    }

    void SwitchController::LaunchInstance(int batch_id)
    {
        cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
    }

    void SwitchController::Run(int batch_id)
    {
        LaunchInstance(batch_id);
        cudaStreamSynchronize(m_streams[batch_id]);
    }

    void SwitchController::Run()
    {
    }
    cudaGraph_t SwitchController::GetGraph(int batch_id)
    {
        return m_graphs[batch_id];
    }

    uint8_t *SwitchController::GetTransmissionCompletedArr()
    {
        return m_transmission_completed;
    }
}

// #include "switch.h"

// namespace VDES
// {
//     SwitchController::SwitchController()
//     {
//     }

//     SwitchController::~SwitchController()
//     {
//     }

//     void SwitchController::SetIngresAndEgress(GPUQueue<Frame *> **ingresses, GPUQueue<Frame *> **egresses, int *queue_num_per_node, int *sw_id_per_node, int node_num)
//     {
//         int queue_num = std::accumulate(queue_num_per_node, queue_num_per_node + node_num, 0);
//         m_ingresses.insert(m_ingresses.end(), ingresses, ingresses + queue_num);
//         m_egresses.insert(m_egresses.end(), egresses, egresses + queue_num);
//         m_sw_id_per_node.insert(m_sw_id_per_node.end(), sw_id_per_node, sw_id_per_node + queue_num);
//         m_queue_num_per_node.insert(m_queue_num_per_node.end(), queue_num_per_node, queue_num_per_node + node_num);
//     }

//     void SwitchController::SetMacForwardingTable(GPUQueue<MacForwardingRule *> **mac_forwarding_table, int node_num)
//     {
//         m_mac_forwarding_table.insert(m_mac_forwarding_table.end(), mac_forwarding_table, mac_forwarding_table + node_num);
//     }

//     void SwitchController::SetBatchproperties(int *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
//     }

//     void SwitchController::SetFtProperties(int ft_k)
//     {
//         m_ft_k = ft_k;
//         m_ft_k_sq_quarter = (m_ft_k * m_ft_k) / 4;
//     }

//     void SwitchController::SetStreams(cudaStream_t *streams, int num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + num);
//     }

//     void SwitchController::InitalizeKernelParams()
//     {
//         int batch_num = m_batch_start_index.size();
//         m_graphs.resize(batch_num);
//         m_graph_execs.resize(batch_num);

//         for (int i = 0; i < batch_num; i++)
//         {
//             int node_num = m_queue_num_per_node[i];
//             int queue_num = std::accumulate(m_queue_num_per_node.begin() + m_batch_start_index[i], m_queue_num_per_node.begin() + m_batch_end_index[i], 0);

//             SwitchParams cpu_param;
//             cudaMallocAsync(&cpu_param.ingresses, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.egresses, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.queue_num_per_node, sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.queue_offset_per_node, sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.node_offset_per_queue, sizeof(int) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.sw_id_per_node, sizeof(int) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.received_packets_per_queue, sizeof(int *) * node_num, m_streams[i]);

//             if (ENABLE_FATTREE_MODE)
//             {
//                 cpu_param.ft_k = m_ft_k;
//                 cpu_param.ft_k_sq_quarter = m_ft_k_sq_quarter;
//                 cudaMallocAsync(&cpu_param.ft_current_port_up_forward, sizeof(uint8_t) * node_num, m_streams[i]);
//                 cudaMallocAsync(&cpu_param.ft_current_port_down_forward, sizeof(uint8_t) * node_num, m_streams[i]);
//                 cudaMallocAsync(&cpu_param.ft_port_num_per_direction, sizeof(uint8_t) * node_num, m_streams[i]);
//             }
//             else
//             {
//                 cudaMallocAsync(&cpu_param.mac_forwarding_table, sizeof(GPUQueue<MacForwardingRule *> *) * node_num, m_streams[i]);
//             }

//             cudaMallocAsync(&cpu_param.drop_frames, sizeof(Frame *) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
//             cudaMallocAsync(&cpu_param.drop_frame_num, sizeof(int) * queue_num, m_streams[i]);
//             cpu_param.node_num = node_num;
//             cpu_param.queue_num = queue_num;
//             m_queue_num_per_batch.push_back(queue_num);

//             // Copy data to GPU
//             int queue_offset = std::accumulate(m_queue_num_per_node.begin(), m_queue_num_per_node.begin() + m_batch_start_index[i], 0);
//             int node_offset = m_batch_start_index[i];
//             cudaMemcpyAsync(cpu_param.ingresses, m_ingresses.data() + queue_offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.egresses, m_egresses.data() + queue_offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.queue_num_per_node, m_queue_num_per_node.data() + node_offset, sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

//             std::vector<int> queue_offset_per_node;
//             std::vector<int> node_offset_per_queue;
//             int offset = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 for (int k = 0; k < m_queue_num_per_node[j + node_offset]; k++)
//                 {
//                     queue_offset_per_node.push_back(j);
//                 }
//                 node_offset_per_queue.push_back(offset);
//                 offset += m_queue_num_per_node[j + node_offset];
//             }
//             cudaMemcpyAsync(cpu_param.queue_offset_per_node, queue_offset_per_node.data(), sizeof(int) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.node_offset_per_queue, node_offset_per_queue.data(), sizeof(int) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.sw_id_per_node, m_sw_id_per_node.data() + queue_offset, sizeof(int) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);

//             std::vector<int *> received_packets_per_queue;
//             for (int j = 0; j < node_num; j++)
//             {
//                 int arr_size = m_queue_num_per_node[j + node_offset] * (m_queue_num_per_node[j + node_offset] + 1);
//                 int *received_packets;
//                 cudaMalloc(&received_packets, sizeof(int) * arr_size);
//                 received_packets_per_queue.push_back(received_packets);
//             }
//             cudaMemcpyAsync(cpu_param.received_packets_per_queue, received_packets_per_queue.data(), sizeof(int *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

//             if (ENABLE_FATTREE_MODE)
//             {
//                 std::vector<uint8_t> ft_port_up_forward;
//                 std::vector<uint8_t> ft_port_down_forward;
//                 std::vector<uint8_t> ft_port_num_per_direction;

//                 for (int j = 0; j < node_num; j++)
//                 {

//                     for (int k = 0; k < m_queue_num_per_node[j + node_offset]; k++)
//                     {
//                         ft_port_up_forward.push_back(k);
//                         ft_port_down_forward.push_back(k);
//                     }
//                     ft_port_num_per_direction.push_back(m_queue_num_per_node[j + node_offset] / 2);
//                 }
//                 cudaMemcpyAsync(cpu_param.ft_current_port_up_forward, ft_port_up_forward.data(), sizeof(uint8_t) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//                 cudaMemcpyAsync(cpu_param.ft_current_port_down_forward, ft_port_down_forward.data(), sizeof(uint8_t) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//                 cudaMemcpyAsync(cpu_param.ft_port_num_per_direction, ft_port_num_per_direction.data(), sizeof(uint8_t) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             }
//             else
//             {
//                 cudaMemcpyAsync(cpu_param.mac_forwarding_table, m_mac_forwarding_table.data() + node_offset, sizeof(GPUQueue<MacForwardingRule *> *) * node_num, cudaMemcpyHostToDevice, m_streams[i]);
//             }
//             cudaMemsetAsync(cpu_param.drop_frame_num, 0, sizeof(int) * queue_num, m_streams[i]);

//             m_drop_frames_gpu.push_back(cpu_param.drop_frames);
//             m_drop_frames_cpu.push_back(new Frame *[queue_num * MAX_TRANSMITTED_PACKET_NUM]);
//             m_drop_frame_num_gpu.push_back(cpu_param.drop_frame_num);
//             m_drop_frame_num_cpu.push_back(new int[queue_num]);

//             cudaStreamSynchronize(m_streams[i]);

//             SwitchParams *gpu_param;
//             cudaMalloc(&gpu_param, sizeof(SwitchParams));
//             cudaMemcpy(gpu_param, &cpu_param, sizeof(SwitchParams), cudaMemcpyHostToDevice);
//             m_kernel_params.push_back(gpu_param);
//         }
//     }

//     void SwitchController::RecycleDropFrames(int batch_id)
//     {
//         int queue_num = m_queue_num_per_batch[batch_id];
//         cudaMemcpyAsync(m_drop_frame_num_cpu[batch_id], m_drop_frame_num_gpu[batch_id], sizeof(int) * queue_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_drop_frames_gpu[batch_id], m_drop_frames_cpu[batch_id], sizeof(Frame *) * queue_num * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyHostToDevice, m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);

//         std::vector<Frame *> recycle_frames;
//         std::vector<Ipv4Packet *> recycle_ipv4_packets;
//         int *drop_frame_num = m_drop_frame_num_cpu[batch_id];
//         VDES::Frame **drop_frames = m_drop_frames_cpu[batch_id];
//         for (int i = 0; i < queue_num; i++)
//         {
//             for (int j = 0; j < drop_frame_num[i]; j++)
//             {
//                 Frame *frame = drop_frames[j];
//                 recycle_frames.push_back(frame);
//                 recycle_ipv4_packets.push_back(*((Ipv4Packet **)frame->data));
//             }
//             drop_frames += (i * MAX_TRANSMITTED_PACKET_NUM);
//         }

//         ipv4_packet_pool_cpu->deallocate(recycle_ipv4_packets.data(), recycle_ipv4_packets.size());
//         frame_pool->deallocate(recycle_frames.data(), recycle_frames.size());
//     }

//     void SwitchController::BuildGraph(int batch_id)
//     {
//         int queue_num = m_queue_num_per_batch[batch_id];
//         dim3 block_dim(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim((queue_num + block_dim.x - 1) / block_dim.x);
//         cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
//         LaunchForwardFramesKernel(grid_dim, block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);
//     }

//     void SwitchController::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void SwitchController::Run(int batch_id)
//     {
//         LaunchInstance(batch_id);
//         cudaStreamSynchronize(m_streams[batch_id]);
//         RecycleDropFrames(batch_id);
//     }

//     void SwitchController::Run()
//     {
//     }
// }