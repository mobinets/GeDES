#include "frame_encapsulation.h"
#include <functional>
#include "component.h"

namespace VDES
{

    FrameEncapsulationController::FrameEncapsulationController()
    {
    }

    FrameEncapsulationController::~FrameEncapsulationController()
    {
    }

    void FrameEncapsulationController::InitializeKernelParams()
    {
        int batch_num = m_batch_end_index.size();

        m_packet_sizes.push_back(sizeof(Ipv4Packet));

        m_graph_execs.resize(batch_num);

        for (int i = 0; i < batch_num; i++)
        {
            cudaGraph_t graph;
            cudaGraphCreate(&graph, 0);
            m_graphs.push_back(graph);

            int node_num = m_batch_end_index[i] - m_batch_start_index[i];
            int frame_queue_num = std::accumulate(m_frame_num_per_node.begin() + m_batch_start_index[i], m_frame_num_per_node.begin() + m_batch_end_index[i], 0);
            int max_frame_num = frame_queue_num * MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;

            FrameEncapsulationParams *cpu_param = new FrameEncapsulationParams;
            cudaMallocAsync(&cpu_param->packets_egresses, sizeof(GPUQueue<void *> *) * frame_queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param->frame_queues, sizeof(GPUQueue<Frame *> *) * frame_queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param->frame_queue_macs, sizeof(uint8_t *) * frame_queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param->frame_queue_offset_per_node, sizeof(int) * node_num, m_streams[i]);
            cudaMallocAsync(&cpu_param->node_id_per_frame_queue, sizeof(int) * frame_queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param->l3_packet_len_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
            cudaMallocAsync(&cpu_param->l3_packet_timestamp_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
            cudaMallocAsync(&cpu_param->l3_dst_ip_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
            cudaMallocAsync(&cpu_param->l3_packet_size, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);

#if ENABLE_FATTREE_MODE
            cudaMallocAsync(&cpu_param->mac_addr_ft, sizeof(uint8_t *) * frame_queue_num, m_streams[i]);
            cpu_param->k = m_ft_k;
            cpu_param->base_ip = m_ft_base_ip;
            cpu_param->ip_group_size = m_ft_ip_group_size;
#else
            cudaMallocAsync(&cpu_param->arp_tables, sizeof(GPUQueue<ARPRule *> *) * frame_queue_num, m_streams[i]);
#endif

            cudaMallocAsync(&cpu_param->alloc_frames, sizeof(Frame *) * max_frame_num, m_streams[i]);
            cudaMallocAsync(&cpu_param->alloc_num_per_frame_queue, sizeof(int) * frame_queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param->alloc_offset_per_node, sizeof(int) * node_num, m_streams[i]);

#if ENBALE_CACHE
            cudaMallocAsync(&cpu_param->swap_out_l3_packets, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
            /**
             * @TODO: Use max size as the copy size
             */
            cudaMallocAsync(&cpu_param->swap_out_l3_packets_num, sizeof(int) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
            cudaMallocAsync(&cpu_param->l3_cache_ptr, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
#endif

            // initialize
            int node_offset = m_batch_start_index[i];
            int frame_offset = std::accumulate(m_frame_num_per_node.begin(), m_frame_num_per_node.begin() + m_batch_start_index[i], 0);
            cudaMemcpyAsync(cpu_param->packets_egresses, m_packets_egresses.data() + frame_offset, sizeof(GPUQueue<void *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param->frame_queues, m_frame_queues.data() + frame_offset, sizeof(GPUQueue<Frame *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);

            int len_offset = 2;
            int timestamp_offset = 35;
            int dst_ip_offset = 16;
            int packet_size = sizeof(Ipv4Packet);
            cudaMemcpyAsync(cpu_param->l3_packet_len_offset, &len_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param->l3_packet_timestamp_offset, &timestamp_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param->l3_dst_ip_offset, &dst_ip_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param->l3_packet_size, &packet_size, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);

            for (int j = 0; j < 6; j++)
            {
                cpu_param->null_mac[j] = 0;
            }
            uint8_t *frame_queue_macs_gpu;
            cudaMallocAsync(&frame_queue_macs_gpu, sizeof(uint8_t) * 6 * frame_queue_num, m_streams[i]);
            m_frame_queue_macs_gpu.push_back(frame_queue_macs_gpu);
            std::vector<uint8_t *> frame_queue_macs_cpu;
            for (int j = 0; j < frame_queue_num; j++)
            {
                cudaMemcpyAsync(frame_queue_macs_gpu + 6 * j, m_frame_queue_macs_cpu[node_offset + j], sizeof(uint8_t) * 6, cudaMemcpyHostToDevice, m_streams[i]);
                frame_queue_macs_cpu.push_back(frame_queue_macs_gpu + 6 * j);
            }
            cudaMemcpyAsync(cpu_param->frame_queue_macs, frame_queue_macs_cpu.data(), sizeof(uint8_t *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);

#if ENABLE_FATTREE_MODE
            uint8_t *temp_gpu_macs;
            cudaMalloc(&temp_gpu_macs, sizeof(uint8_t) * 6 * frame_queue_num);
            std::vector<uint8_t *> mac_addr_ft;
            for (int j = 0; j < frame_queue_num; j++)
            {
                mac_addr_ft.push_back(temp_gpu_macs + 6 * j);
            }
            cudaMemcpyAsync(cpu_param->mac_addr_ft, mac_addr_ft.data(), sizeof(uint8_t *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
#else
            cudaMemcpyAsync(cpu_param->arp_tables, m_arp_tables.data() + frame_offset, sizeof(GPUQueue<ARPRule *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
#endif

#if ENABLE_GPU_MEM_POOL
            std::vector<Frame *> alloc_frame;
            for (int i = 0; i < max_frame_num; i++)
            {
                Frame *frame;
                cudaMalloc(&frame, sizeof(Frame));
                alloc_frame.push_back(frame);
            }
#else
            auto alloc_frame = frame_pool->allocate(max_frame_num);
#endif
            cudaMemcpyAsync(cpu_param->alloc_frames, alloc_frame.data(), sizeof(Frame *) * max_frame_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemsetAsync(cpu_param->alloc_num_per_frame_queue, 0, sizeof(int) * frame_queue_num, m_streams[i]);

            int *alloc_offset = new int[node_num];
            int offset = 0;
            for (int j = 0; j < node_num; j++)
            {
                alloc_offset[j] = offset;
                offset += (m_frame_num_per_node[node_offset + j] * MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM);
            }
            cudaMemcpyAsync(cpu_param->alloc_offset_per_node, alloc_offset, sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

            std::vector<int> node_id_per_frame;
            std::vector<int> frame_offset_per_node;
            int frame_offset_temp = 0;
            for (int j = 0; j < node_num; j++)
            {
                for (int k = 0; k < m_frame_num_per_node[node_offset + j]; k++)
                {
                    node_id_per_frame.push_back(j);
                    // frame_queue_id
                }
                frame_offset_per_node.push_back(frame_offset_temp);
                frame_offset_temp += m_frame_num_per_node[node_offset + j];
            }
            cudaMemcpyAsync(cpu_param->node_id_per_frame_queue, node_id_per_frame.data(), sizeof(int) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param->frame_queue_offset_per_node, frame_offset_per_node.data(), sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

#if ENABLE_CACHE
            int cache_size = 0;
            for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
            {
                cache_size += (max_frame_num * m_packet_sizes[j]);
            }
            uint8_t *l3_cache_gpu;
            cudaMalloc(&l3_cache_gpu, cache_size);
            uint8_t *l3_cache_cpu = new uint8_t[cache_size];
            m_l3_cache_gpu.push_back(l3_cache_gpu);
            m_l3_cache_cpu.push_back(l3_cache_cpu);

            uint8_t **l3_cache_ptr_gpu = new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType];
            uint8_t **l3_cache_ptr_cpu = new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType];
            for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
            {
                for (int k = 0; k < max_frame_num; k++)
                {
                    l3_cache_ptr_gpu[j * max_frame_num + k] = l3_cache_gpu + k * m_packet_sizes[j];
                    l3_cache_ptr_cpu[j * max_frame_num + k] = l3_cache_cpu + k * m_packet_sizes[j];
                }
                l3_cache_gpu += max_frame_num * m_packet_sizes[j];
                l3_cache_cpu += max_frame_num * m_packet_sizes[j];
            }
            cudaMemcpyAsync(cpu_param->l3_cache_ptr, l3_cache_ptr_gpu, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyHostToDevice, m_streams[i]);
            m_l3_cache_ptr_cpu.push_back(l3_cache_ptr_cpu);
            m_l3_cache_ptr_gpu.push_back(l3_cache_ptr_gpu);

            cudaStreamSynchronize(m_streams[i]);
            m_l3_swap_out_packet_gpu.push_back(cpu_param->swap_out_l3_packets);
            m_l3_swap_out_packet_cpu.push_back(new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType]);
            for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
            {
                if (j == NetworkProtocolType::IPv4)
                {
                    /**
                     * TODO: Allocate cpu memory on for the alloc_packets cpu.
                     */
                    auto alloc_packets = ipv4_packet_pool_cpu->allocate(max_frame_num);
                    cudaMemcpyAsync(cpu_param->swap_out_l3_packets, (uint8_t **)alloc_packets.data(), sizeof(uint8_t *) * max_frame_num, cudaMemcpyHostToDevice, m_streams[i]);
                    memcpy(m_l3_swap_out_packet_cpu[i], (uint8_t **)alloc_packets.data(), sizeof(uint8_t *) * max_frame_num);
                    memcpy(m_l3_swap_out_packet_cpu_backup[i], (uint8_t **)alloc_packets.data(), sizeof(uint8_t *) * max_frame_num);
                }
            }
            cudaMemsetAsync(cpu_param->swap_out_l3_packets_num, 0, sizeof(int) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
            m_l3_swap_out_packet_num_gpu.push_back(cpu_param->swap_out_l3_packets_num);
            m_l3_swap_out_packet_num_cpu.push_back(new int[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType]);
            m_cache_sizes.push_back(cache_size);
#endif

            cpu_param->max_packet_num = max_frame_num;
            cpu_param->queue_num = frame_queue_num;
            cpu_param->node_num = node_num;

            dim3 block_dim(KERNEL_BLOCK_WIDTH);
            dim3 grid_dim((frame_queue_num + block_dim.x - 1) / block_dim.x);

            m_grid_dim.push_back(grid_dim);
            m_block_dim.push_back(block_dim);

            m_frame_num_per_batch.push_back(frame_queue_num);
            m_alloc_frames_cpu.push_back(new Frame *[max_frame_num]);
            m_alloc_frames_gpu.push_back(cpu_param->alloc_frames);
            memcpy(m_alloc_frames_cpu[i], alloc_frame.data(), sizeof(Frame *) * max_frame_num);
            m_alloc_num_per_frame_queue_gpu.push_back(cpu_param->alloc_num_per_frame_queue);
            m_alloc_num_per_frame_queue_cpu.push_back(new int[frame_queue_num]);
            m_alloc_offset_per_node_gpu.push_back(cpu_param->alloc_offset_per_node);
            m_alloc_offset_per_node_cpu.push_back(alloc_offset);
            m_total_alloc_frame_num.push_back(max_frame_num);

            m_max_packet_num.push_back(max_frame_num);

            FrameEncapsulationParams *gpu_param;
            cudaMallocAsync(&gpu_param, sizeof(FrameEncapsulationParams), m_streams[i]);
            cudaMemcpyAsync(gpu_param, cpu_param, sizeof(FrameEncapsulationParams), cudaMemcpyHostToDevice, m_streams[i]);
            cudaStreamSynchronize(m_streams[i]);
            m_kernel_params.push_back(gpu_param);
            delete cpu_param;
        }
    }

    void FrameEncapsulationController::SetStreams(cudaStream_t *streams, int stream_num)
    {
        m_streams.insert(m_streams.end(), streams, streams + stream_num);
    }

    void FrameEncapsulationController::SetPacketProperties(GPUQueue<void *> **packet_queues, int queue_num)
    {
        m_packets_egresses.insert(m_packets_egresses.end(), packet_queues, packet_queues + queue_num);
    }

    void FrameEncapsulationController::SetFrameProperties(GPUQueue<Frame *> **frame_queues, uint8_t **frame_queue_macs, int *frame_num_per_node, int node_num)
    {
        int frame_queue = std::accumulate(frame_num_per_node, frame_num_per_node + node_num, 0);
        m_frame_queues.insert(m_frame_queues.end(), frame_queues, frame_queues + frame_queue);

        m_frame_queue_macs_cpu.insert(m_frame_queue_macs_cpu.end(), frame_queue_macs, frame_queue_macs + frame_queue);
        m_frame_num_per_node.insert(m_frame_num_per_node.end(), frame_num_per_node, frame_num_per_node + node_num);
    }

    void FrameEncapsulationController::SetArpProperties(GPUQueue<ARPRule *> **arp_tables, int node_num)
    {
        m_arp_tables.insert(m_arp_tables.end(), arp_tables, arp_tables + node_num);
    }

    void FrameEncapsulationController::SetFatTreeArpProperties(uint16_t k, uint32_t base_ip, uint32_t ip_group_size)
    {
        m_ft_k = k;
        m_ft_base_ip = base_ip;
        m_ft_ip_group_size = ip_group_size;
    }

#if ENABLE_CACHE
    void FrameEncapsulationController::CacheOutL3Packets(int batch_id)
    {
        int max_packet_num = m_max_packet_num[batch_id];
        int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

        cudaMemcpyAsync(m_l3_swap_out_packet_num_cpu[batch_id], m_l3_swap_out_packet_num_gpu[batch_id], sizeof(int) * max_packet_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyDeviceToHost, m_streams[batch_id]);
        cudaMemcpyAsync(m_l3_cache_cpu[batch_id], m_l3_cache_gpu[batch_id], m_cache_sizes[batch_id], cudaMemcpyDeviceToHost, m_streams[batch_id]);
        cudaMemcpyAsync(m_alloc_num_per_frame_queue_cpu[batch_id], m_alloc_num_per_frame_queue_gpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], cudaMemcpyDeviceToHost, m_streams[batch_id]);
        cudaMemcpyAsync(m_l3_swap_out_packet_cpu[batch_id], m_l3_swap_out_packet_gpu[batch_id], sizeof(uint8_t *) * max_packet_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);
        cudaStreamSynchronize(m_streams[batch_id]);

        int *packet_offsets = m_alloc_offset_per_node_cpu[batch_id];
        int node_offset = m_batch_start_index[batch_id];
        int *swap_out_num = m_l3_swap_out_packet_num_cpu[batch_id];

        for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
        {

            int queue_id = 0;
            uint8_t **origin_dst = m_l3_swap_out_packet_cpu_backup[batch_id] + i * max_packet_num;
            uint8_t **origin_src = m_l3_cache_ptr_cpu[batch_id] + i * max_packet_num;
            uint8_t **origin_recycle_packet = m_l3_swap_out_packet_cpu[batch_id] + i * max_packet_num;

            int total_packet_num = std::accumulate(swap_out_num, swap_out_num + max_packet_num, 0);
            std::vector<uint8_t *> alloc_packets;
            if (i == NetworkProtocolType::IPv4)
            {
                auto ipv4_packets = ipv4_packet_pool_cpu->allocate(total_packet_num);
                alloc_packets.insert(alloc_packets.end(), (uint8_t **)ipv4_packets.data(), (uint8_t **)(ipv4_packets.data() + total_packet_num));
            }

            // copy data from cache to discrete packets
            int alloc_offset = 0;
            std::vector<uint8_t *> recycle_l3_packets;
            for (int j = 0; j < node_num; j++)
            {
                uint8_t **dst = origin_dst + packet_offsets[j];
                uint8_t **src = origin_src + packet_offsets[j];
                uint8_t **recycle_packets = origin_recycle_packet + packet_offsets[j];

                int offset = 0;

                for (int k = 0; k < m_frame_num_per_node[node_offset + j]; k++)
                {
                    for (int m = 0; m < swap_out_num[queue_id]; m++)
                    {
                        // LOG_INFO("j: %d, m : %d", j, m);
                        // if (j == 950 && m == 6)
                        // {
                        //     VDES::Ipv4Packet tmp_packet;
                        //     VDES::Ipv4Packet *tmp_dst = (VDES::Ipv4Packet *)(dst[offset + m]);
                        //     memcpy(&tmp_packet, tmp_dst, sizeof(VDES::Ipv4Packet));
                        //     VDES::Ipv4Packet *tmp_src = (VDES::Ipv4Packet *)(src[offset + m]);
                        //     memcpy(&tmp_packet, tmp_src, sizeof(VDES::Ipv4Packet));
                        //     j = 950;
                        // }
                        memcpy(dst[offset + m], src[offset + m], m_packet_sizes[i]);
                        // update used packet
                        dst[offset + m] = alloc_packets[alloc_offset];
                        recycle_l3_packets.push_back(recycle_packets[offset + m]);
                        alloc_offset++;
                    }
                    offset += swap_out_num[queue_id];
                    queue_id++;
                }
            }

            if (i == NetworkProtocolType::IPv4)
            {
                ipv4_packet_pool->deallocate((Ipv4Packet **)recycle_l3_packets.data(), recycle_l3_packets.size());
            }

            recycle_l3_packets.clear();
        }
        cudaMemcpyAsync(m_l3_swap_out_packet_gpu[batch_id], m_l3_swap_out_packet_cpu_backup[batch_id], sizeof(uint8_t *) * max_packet_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyHostToDevice, m_streams[batch_id]);
    }
#endif

    void FrameEncapsulationController::LaunchInstance(int batch_id)
    {
        LaunchEncapsulateFrameKernel(m_grid_dim[batch_id], m_block_dim[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
    }

    void FrameEncapsulationController::Run(int batch_id)
    {
        LaunchInstance(batch_id);
        cudaStreamSynchronize(m_streams[batch_id]);
#if ENABLE_CACHE
        CacheOutL3Packets(batch_id);
#endif
        UpdateComsumedFrames(batch_id);
    }

    void FrameEncapsulationController::Run()
    {
    }

    void FrameEncapsulationController::UpdateComsumedFrames(int batch_id)
    {
#if !ENABLE_HUGE_GRAPH
        cudaMemcpy(m_alloc_num_per_frame_queue_cpu[batch_id], m_alloc_num_per_frame_queue_gpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], cudaMemcpyDeviceToHost);
#endif

        int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
        int *consumed_frame_num = m_alloc_num_per_frame_queue_cpu[batch_id];
        VDES::Frame **frames = m_alloc_frames_cpu[batch_id];

        int total_frame_num = std::accumulate(consumed_frame_num, consumed_frame_num + m_frame_num_per_batch[batch_id], 0);
        auto alloc_frames = frame_pool->allocate(total_frame_num);
        int node_offset = m_batch_start_index[batch_id];

        int frame_index = 0;
        int queue_id = 0;

        for (int i = 0; i < node_num; i++)
        {
            int offset = m_alloc_offset_per_node_cpu[batch_id][i];
            int packet_id = 0;
            for (int j = 0; j < m_frame_num_per_node[m_batch_start_index[batch_id] + i]; j++)
            {
                if (consumed_frame_num[queue_id] > 0)
                {
                    memcpy(frames + offset, alloc_frames.data() + frame_index, 8 * consumed_frame_num[queue_id]);
                    frame_index += consumed_frame_num[queue_id];
                    offset += consumed_frame_num[queue_id];
                }
                queue_id++;
            }
        }

#if !ENABLE_HUGE_GRAPH
        cudaMemcpy(m_alloc_frames_gpu[batch_id], m_alloc_frames_cpu[batch_id], sizeof(Frame *) * m_total_alloc_frame_num[batch_id], cudaMemcpyHostToDevice);
#endif
    }

    void FrameEncapsulationController::SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num)
    {
        m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
        m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
    }

    void FrameEncapsulationController::BuildGraph(int batch_id)
    {
        cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
        LaunchEncapsulateFrameKernel(m_grid_dim[batch_id], m_block_dim[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
        cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
        cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);

#if ENABLE_HUGE_GRAPH
        cudaMemcpy3DParms alloc_num_memcpy_params = {0};
        alloc_num_memcpy_params.srcPtr = make_cudaPitchedPtr(m_alloc_num_per_frame_queue_gpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], m_frame_num_per_batch[batch_id], 1);
        alloc_num_memcpy_params.dstPtr = make_cudaPitchedPtr(m_alloc_num_per_frame_queue_cpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], m_frame_num_per_batch[batch_id], 1);
        alloc_num_memcpy_params.extent = make_cudaExtent(sizeof(int) * m_frame_num_per_batch[batch_id], 1, 1);
        alloc_num_memcpy_params.kind = cudaMemcpyDeviceToHost;

        cudaHostNodeParams update_host_params = {0};
        auto update_host_func = std::bind(&FrameEncapsulationController::UpdateComsumedFrames, this, batch_id);
        auto update_host_func_ptr = new std::function<void()>(update_host_func);
        update_host_params.fn = VDES::HostNodeCallback;
        update_host_params.userData = update_host_func_ptr;

        cudaMemcpy3DParms alloc_frame_memcpy_params = {0};
        alloc_frame_memcpy_params.srcPtr = make_cudaPitchedPtr(m_alloc_frames_cpu[batch_id], sizeof(Frame *) * m_total_alloc_frame_num[batch_id], m_total_alloc_frame_num[batch_id], 1);
        alloc_frame_memcpy_params.dstPtr = make_cudaPitchedPtr(m_alloc_frames_gpu[batch_id], sizeof(Frame *) * m_total_alloc_frame_num[batch_id], m_total_alloc_frame_num[batch_id], 1);
        alloc_frame_memcpy_params.extent = make_cudaExtent(sizeof(Frame *) * m_total_alloc_frame_num[batch_id], 1, 1);
        alloc_frame_memcpy_params.kind = cudaMemcpyHostToDevice;

        m_memcpy_param.push_back(alloc_num_memcpy_params);
        m_memcpy_param.push_back(alloc_frame_memcpy_params);
        m_host_param.push_back(update_host_params);

#endif
    }

    void FrameEncapsulationController::BuildGraph()
    {
        int batch_num = m_batch_start_index.size();
        for (int i = 0; i < batch_num; i++)
        {
            BuildGraph(i);
        }
    }

    cudaGraph_t FrameEncapsulationController::GetGraph(int batch_id)
    {
        return m_graphs[batch_id];
    }

#if ENABLE_HUGE_GRAPH

    std::vector<cudaMemcpy3DParms> &FrameEncapsulationController::GetMemcpyParams()
    {
        return m_memcpy_param;
    }

    std::vector<cudaHostNodeParams> &FrameEncapsulationController::GetHostParams()
    {
        return m_host_param;
    }

    std::vector<void *> FrameEncapsulationController::GetAllocateInfo()
    {
        int batch_num = m_batch_start_index.size();
        std::vector<void *> res;
        for (int i = 0; i < batch_num; i++)
        {
            res.push_back(m_alloc_frames_gpu[i]);
            res.push_back(m_alloc_num_per_frame_queue_gpu[i]);
            FrameEncapsulationParams cpu_param;
            cudaMemcpy(&cpu_param, m_kernel_params[i], sizeof(VDES::FrameEncapsulationParams),cudaMemcpyDeviceToHost);
            res.push_back(cpu_param.alloc_offset_per_node);
        }
        return res;
    }

#endif
}

// namespace VDES
// {

//     FrameEncapsulationController::FrameEncapsulationController()
//     {
//     }

//     FrameEncapsulationController::~FrameEncapsulationController()
//     {
//     }

//     void FrameEncapsulationController::InitializeKernelParams()
//     {
//         int batch_num = m_batch_end_index.size();

//         m_packet_sizes.push_back(sizeof(Ipv4Packet));

//         m_graph_execs.resize(batch_num);

//         for (int i = 0; i < batch_num; i++)
//         {
//             cudaGraph_t graph;
//             cudaGraphCreate(&graph, 0);
//             m_graphs.push_back(graph);

//             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
//             int frame_queue_num = std::accumulate(m_frame_num_per_node.begin() + m_batch_start_index[i], m_frame_num_per_node.begin() + m_batch_end_index[i], 0);
//             int max_frame_num = frame_queue_num * MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;

//             FrameEncapsulationParams *cpu_param = new FrameEncapsulationParams;
//             cudaMallocAsync(&cpu_param->packets_egresses, sizeof(GPUQueue<void *> *) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->frame_queues, sizeof(GPUQueue<Frame *> *) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->frame_queue_macs, sizeof(uint8_t *) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->frame_queue_offset_per_node, sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->node_id_per_frame_queue, sizeof(int) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_packet_len_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_packet_timestamp_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_dst_ip_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_packet_size, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);

//             if (ENABLE_FATTREE_MODE)
//             {
//                 cudaMallocAsync(&cpu_param->mac_addr_ft, sizeof(uint8_t *) * frame_queue_num, m_streams[i]);
//                 cpu_param->k = m_ft_k;
//                 cpu_param->base_ip = m_ft_base_ip;
//                 cpu_param->ip_group_size = m_ft_ip_group_size;
//             }
//             else
//             {
//                 cudaMallocAsync(&cpu_param->arp_tables, sizeof(GPUQueue<ARPRule *> *) * frame_queue_num, m_streams[i]);
//             }

//             cudaMallocAsync(&cpu_param->alloc_frames, sizeof(Frame *) * max_frame_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->alloc_num_per_frame_queue, sizeof(int) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->alloc_offset_per_node, sizeof(int) * node_num, m_streams[i]);

// #if ENBALE_CACHE
//             cudaMallocAsync(&cpu_param->swap_out_l3_packets, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             /**
//              * @TODO: Use max size as the copy size
//              */
//             cudaMallocAsync(&cpu_param->swap_out_l3_packets_num, sizeof(int) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_cache_ptr, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
// #endif

//             // initialize
//             /**
//              * TODO: Copy the right number of the queues and the right start index of the queues.
//              */
//             int node_offset = m_batch_start_index[i];
//             int frame_offset = std::accumulate(m_frame_num_per_node.begin(), m_frame_num_per_node.begin() + m_batch_start_index[i], 0);
//             cudaMemcpyAsync(cpu_param->packets_egresses, m_packets_egresses.data() + frame_offset, sizeof(GPUQueue<void *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->frame_queues, m_frame_queues.data() + frame_offset, sizeof(GPUQueue<Frame *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);

//             int len_offset = 2;
//             int timestamp_offset = 35;
//             int dst_ip_offset = 16;
//             int packet_size = sizeof(Ipv4Packet);
//             cudaMemcpyAsync(cpu_param->l3_packet_len_offset, &len_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->l3_packet_timestamp_offset, &timestamp_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->l3_dst_ip_offset, &dst_ip_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->l3_packet_size, &packet_size, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);

//             for (int j = 0; j < 6; j++)
//             {
//                 cpu_param->null_mac[j] = 0;
//             }
//             uint8_t *frame_queue_macs_gpu;
//             cudaMallocAsync(&frame_queue_macs_gpu, sizeof(uint8_t) * 6 * frame_queue_num, m_streams[i]);
//             m_frame_queue_macs_gpu.push_back(frame_queue_macs_gpu);
//             std::vector<uint8_t *> frame_queue_macs_cpu;
//             for (int j = 0; j < frame_queue_num; j++)
//             {
//                 cudaMemcpyAsync(frame_queue_macs_gpu + 6 * j, m_frame_queue_macs_cpu[node_offset + j], sizeof(uint8_t) * 6, cudaMemcpyHostToDevice, m_streams[i]);
//                 frame_queue_macs_cpu.push_back(frame_queue_macs_gpu + 6 * j);
//             }
//             cudaMemcpyAsync(cpu_param->frame_queue_macs, frame_queue_macs_cpu.data(), sizeof(uint8_t *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);

// #if ENABLE_FATTREE_MODE
//             uint8_t *temp_gpu_macs;
//             cudaMalloc(&temp_gpu_macs, sizeof(uint8_t) * 6 * frame_queue_num);
//             std::vector<uint8_t *> mac_addr_ft;
//             for (int j = 0; j < frame_queue_num; j++)
//             {
//                 mac_addr_ft.push_back(temp_gpu_macs + 6 * j);
//             }
//             cudaMemcpyAsync(cpu_param->mac_addr_ft, mac_addr_ft.data(), sizeof(uint8_t *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
// #else
//             cudaMemcpyAsync(cpu_param->arp_tables, m_arp_tables.data() + frame_offset, sizeof(GPUQueue<ARPRule *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
// #endif

//             auto alloc_frame = frame_pool->allocate(max_frame_num);
//             cudaMemcpyAsync(cpu_param->alloc_frames, alloc_frame.data(), sizeof(Frame *) * max_frame_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemsetAsync(cpu_param->alloc_num_per_frame_queue, 0, sizeof(int) * frame_queue_num, m_streams[i]);

//             int *alloc_offset = new int[node_num];
//             int offset = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 alloc_offset[j] = offset;
//                 offset += (m_frame_num_per_node[node_offset + j] * MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM);
//             }
//             cudaMemcpyAsync(cpu_param->alloc_offset_per_node, alloc_offset, sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

//             std::vector<int> node_id_per_frame;
//             std::vector<int> frame_offset_per_node;
//             int frame_offset_temp = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 for (int k = 0; k < m_frame_num_per_node[node_offset + j]; k++)
//                 {
//                     node_id_per_frame.push_back(j);
//                     // frame_queue_id
//                 }
//                 frame_offset_per_node.push_back(frame_offset_temp);
//                 frame_offset_temp += m_frame_num_per_node[node_offset + j];
//             }
//             cudaMemcpyAsync(cpu_param->node_id_per_frame_queue, node_id_per_frame.data(), sizeof(int) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->frame_queue_offset_per_node, frame_offset_per_node.data(), sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

// #if ENABLE_CACHE
//             int cache_size = 0;
//             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 cache_size += (max_frame_num * m_packet_sizes[j]);
//             }
//             uint8_t *l3_cache_gpu;
//             cudaMalloc(&l3_cache_gpu, cache_size);
//             uint8_t *l3_cache_cpu = new uint8_t[cache_size];
//             m_l3_cache_gpu.push_back(l3_cache_gpu);
//             m_l3_cache_cpu.push_back(l3_cache_cpu);

//             uint8_t **l3_cache_ptr_gpu = new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType];
//             uint8_t **l3_cache_ptr_cpu = new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType];
//             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 for (int k = 0; k < max_frame_num; k++)
//                 {
//                     l3_cache_ptr_gpu[j * max_frame_num + k] = l3_cache_gpu + k * m_packet_sizes[j];
//                     l3_cache_ptr_cpu[j * max_frame_num + k] = l3_cache_cpu + k * m_packet_sizes[j];
//                 }
//                 l3_cache_gpu += max_frame_num * m_packet_sizes[j];
//                 l3_cache_cpu += max_frame_num * m_packet_sizes[j];
//             }
//             cudaMemcpyAsync(cpu_param->l3_cache_ptr, l3_cache_ptr_gpu, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyHostToDevice, m_streams[i]);
//             m_l3_cache_ptr_cpu.push_back(l3_cache_ptr_cpu);
//             m_l3_cache_ptr_gpu.push_back(l3_cache_ptr_gpu);

//             cudaStreamSynchronize(m_streams[i]);
//             m_l3_swap_out_packet_gpu.push_back(cpu_param->swap_out_l3_packets);
//             m_l3_swap_out_packet_cpu.push_back(new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType]);
//             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 if (j == NetworkProtocolType::IPv4)
//                 {
//                     /**
//                      * TODO: Allocate cpu memory on for the alloc_packets cpu.
//                      */
//                     auto alloc_packets = ipv4_packet_pool_cpu->allocate(max_frame_num);
//                     cudaMemcpyAsync(cpu_param->swap_out_l3_packets, (uint8_t **)alloc_packets.data(), sizeof(uint8_t *) * max_frame_num, cudaMemcpyHostToDevice, m_streams[i]);
//                     memcpy(m_l3_swap_out_packet_cpu[i], (uint8_t **)alloc_packets.data(), sizeof(uint8_t *) * max_frame_num);
//                     memcpy(m_l3_swap_out_packet_cpu_backup[i], (uint8_t **)alloc_packets.data(), sizeof(uint8_t *) * max_frame_num);
//                 }
//             }
//             cudaMemsetAsync(cpu_param->swap_out_l3_packets_num, 0, sizeof(int) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             m_l3_swap_out_packet_num_gpu.push_back(cpu_param->swap_out_l3_packets_num);
//             m_l3_swap_out_packet_num_cpu.push_back(new int[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType]);
//             m_cache_sizes.push_back(cache_size);
// #endif

//             cpu_param->max_packet_num = max_frame_num;
//             cpu_param->queue_num = frame_queue_num;
//             cpu_param->node_num = node_num;

//             /**
//              * @warning:Initialize the grid dim and block dim
//              */
//             dim3 block_dim(KERNEL_BLOCK_WIDTH);
//             dim3 grid_dim((frame_queue_num + block_dim.x - 1) / block_dim.x);

//             m_grid_dim.push_back(grid_dim);
//             m_block_dim.push_back(block_dim);

//             /**
//              * @warning:Initialize the total alloc frame num
//              */
//             m_frame_num_per_batch.push_back(frame_queue_num);
//             // cudaStreamSynchronize(m_streams[i]);
//             m_alloc_frames_cpu.push_back(new Frame *[max_frame_num]);
//             m_alloc_frames_gpu.push_back(cpu_param->alloc_frames);
//             memcpy(m_alloc_frames_cpu[i], alloc_frame.data(), sizeof(Frame *) * max_frame_num);
//             m_alloc_num_per_frame_queue_gpu.push_back(cpu_param->alloc_num_per_frame_queue);
//             m_alloc_num_per_frame_queue_cpu.push_back(new int[frame_queue_num]);
//             m_alloc_offset_per_node_gpu.push_back(cpu_param->alloc_offset_per_node);
//             m_alloc_offset_per_node_cpu.push_back(alloc_offset);
//             m_total_alloc_frame_num.push_back(max_frame_num);

//             m_max_packet_num.push_back(max_frame_num);

//             FrameEncapsulationParams *gpu_param;
//             cudaMallocAsync(&gpu_param, sizeof(FrameEncapsulationParams), m_streams[i]);
//             cudaMemcpyAsync(gpu_param, cpu_param, sizeof(FrameEncapsulationParams), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaStreamSynchronize(m_streams[i]);
//             m_kernel_params.push_back(gpu_param);
//             delete cpu_param;
//         }
//     }

//     void FrameEncapsulationController::SetStreams(cudaStream_t *streams, int stream_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + stream_num);
//     }

//     void FrameEncapsulationController::SetPacketProperties(GPUQueue<void *> **packet_queues, int queue_num)
//     {
//         m_packets_egresses.insert(m_packets_egresses.end(), packet_queues, packet_queues + queue_num);
//     }

//     void FrameEncapsulationController::SetFrameProperties(GPUQueue<Frame *> **frame_queues, uint8_t **frame_queue_macs, int *frame_num_per_node, int node_num)
//     {
//         int frame_queue = std::accumulate(frame_num_per_node, frame_num_per_node + node_num, 0);
//         m_frame_queues.insert(m_frame_queues.end(), frame_queues, frame_queues + frame_queue);

//         m_frame_queue_macs_cpu.insert(m_frame_queue_macs_cpu.end(), frame_queue_macs, frame_queue_macs + frame_queue);
//         m_frame_num_per_node.insert(m_frame_num_per_node.end(), frame_num_per_node, frame_num_per_node + node_num);
//     }

//     void FrameEncapsulationController::SetArpProperties(GPUQueue<ARPRule *> **arp_tables, int node_num)
//     {
//         m_arp_tables.insert(m_arp_tables.end(), arp_tables, arp_tables + node_num);
//     }

//     void FrameEncapsulationController::SetFatTreeArpProperties(uint16_t k, uint32_t base_ip, uint32_t ip_group_size)
//     {
//         m_ft_k = k;
//         m_ft_base_ip = base_ip;
//         m_ft_ip_group_size = ip_group_size;
//     }

// #if ENABLE_CACHE
//     void FrameEncapsulationController::CacheOutL3Packets(int batch_id)
//     {
//         int max_packet_num = m_max_packet_num[batch_id];
//         int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

//         cudaMemcpyAsync(m_l3_swap_out_packet_num_cpu[batch_id], m_l3_swap_out_packet_num_gpu[batch_id], sizeof(int) * max_packet_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_l3_cache_cpu[batch_id], m_l3_cache_gpu[batch_id], m_cache_sizes[batch_id], cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_alloc_num_per_frame_queue_cpu[batch_id], m_alloc_num_per_frame_queue_gpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_l3_swap_out_packet_cpu[batch_id], m_l3_swap_out_packet_gpu[batch_id], sizeof(uint8_t *) * max_packet_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);

//         // for (size_t i = 0; i < m_cache_sizes[batch_id]; i++)
//         // {
//         //     Ipv4Packet *tmp_packet = (Ipv4Packet *)(m_l3_cache_cpu[batch_id] + (i * m_packet_sizes[0]));
//         //     uint32_t dst_ip;
//         //     memcpy(&dst_ip, tmp_packet->dst_ip, sizeof(uint32_t));
//         //     LOG_INFO("dst_ip: %u", dst_ip);
//         // }

//         int *packet_offsets = m_alloc_offset_per_node_cpu[batch_id];
//         int node_offset = m_batch_start_index[batch_id];
//         int *swap_out_num = m_l3_swap_out_packet_num_cpu[batch_id];

//         for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
//         {

//             int queue_id = 0;
//             uint8_t **origin_dst = m_l3_swap_out_packet_cpu_backup[batch_id] + i * max_packet_num;
//             uint8_t **origin_src = m_l3_cache_ptr_cpu[batch_id] + i * max_packet_num;
//             uint8_t **origin_recycle_packet = m_l3_swap_out_packet_cpu[batch_id] + i * max_packet_num;

//             int total_packet_num = std::accumulate(swap_out_num, swap_out_num + max_packet_num, 0);
//             std::vector<uint8_t *> alloc_packets;
//             if (i == NetworkProtocolType::IPv4)
//             {
//                 auto ipv4_packets = ipv4_packet_pool_cpu->allocate(total_packet_num);
//                 alloc_packets.insert(alloc_packets.end(), (uint8_t **)ipv4_packets.data(), (uint8_t **)(ipv4_packets.data() + total_packet_num));
//             }

//             // copy data from cache to discrete packets
//             int alloc_offset = 0;
//             std::vector<uint8_t *> recycle_l3_packets;
//             for (int j = 0; j < node_num; j++)
//             {
//                 uint8_t **dst = origin_dst + packet_offsets[j];
//                 uint8_t **src = origin_src + packet_offsets[j];
//                 uint8_t **recycle_packets = origin_recycle_packet + packet_offsets[j];

//                 int offset = 0;

//                 for (int k = 0; k < m_frame_num_per_node[node_offset + j]; k++)
//                 {
//                     for (int m = 0; m < swap_out_num[queue_id]; m++)
//                     {
//                         // LOG_INFO("j: %d, m : %d", j, m);
//                         // if (j == 950 && m == 6)
//                         // {
//                         //     VDES::Ipv4Packet tmp_packet;
//                         //     VDES::Ipv4Packet *tmp_dst = (VDES::Ipv4Packet *)(dst[offset + m]);
//                         //     memcpy(&tmp_packet, tmp_dst, sizeof(VDES::Ipv4Packet));
//                         //     VDES::Ipv4Packet *tmp_src = (VDES::Ipv4Packet *)(src[offset + m]);
//                         //     memcpy(&tmp_packet, tmp_src, sizeof(VDES::Ipv4Packet));
//                         //     j = 950;
//                         // }
//                         memcpy(dst[offset + m], src[offset + m], m_packet_sizes[i]);
//                         // update used packet
//                         dst[offset + m] = alloc_packets[alloc_offset];
//                         recycle_l3_packets.push_back(recycle_packets[offset + m]);
//                         alloc_offset++;
//                     }
//                     offset += swap_out_num[queue_id];
//                     queue_id++;
//                 }
//             }

//             if (i == NetworkProtocolType::IPv4)
//             {
//                 ipv4_packet_pool->deallocate((Ipv4Packet **)recycle_l3_packets.data(), recycle_l3_packets.size());
//             }

//             recycle_l3_packets.clear();
//         }
//         cudaMemcpyAsync(m_l3_swap_out_packet_gpu[batch_id], m_l3_swap_out_packet_cpu_backup[batch_id], sizeof(uint8_t *) * max_packet_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyHostToDevice, m_streams[batch_id]);
//     }
// #endif

//     void FrameEncapsulationController::LaunchInstance(int batch_id)
//     {
//         LaunchEncapsulateFrameKernel(m_grid_dim[batch_id], m_block_dim[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
//     }

//     void FrameEncapsulationController::Run(int batch_id)
//     {
//         // std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
//         LaunchInstance(batch_id);
//         cudaStreamSynchronize(m_streams[batch_id]);
//         // std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
//         // std::chrono::duration<double, std::milli> duration = end_time - start_time;
//         // LOG_INFO("Frame Encap Time: %f ms", duration.count());
// #if ENABLE_CACHE
//         CacheOutL3Packets(batch_id);
// #endif
//         // std::chrono::high_resolution_clock::time_point end_time_2 = std::chrono::high_resolution_clock::now();
//         UpdateComsumedFrames(batch_id);
//         // std::chrono::duration<double, std::milli> duration_2 = end_time_2 - end_time;
//         // LOG_INFO("Frame Encap Time: %f ms, Update Comsumed Frames Time: %f ms", duration.count(), duration_2.count());
//     }

//     void FrameEncapsulationController::Run()
//     {
//     }

//     void FrameEncapsulationController::UpdateComsumedFrames(int batch_id)
//     {
// #if !ENABLE_HUGE_GRAPH
//         cudaMemcpy(m_alloc_num_per_frame_queue_cpu[batch_id], m_alloc_num_per_frame_queue_gpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], cudaMemcpyDeviceToHost);
// #endif

//         int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
//         int *consumed_frame_num = m_alloc_num_per_frame_queue_cpu[batch_id];
//         VDES::Frame **frames = m_alloc_frames_cpu[batch_id];
//         /**
//          * TODO: The calculation of total_frame_num is not correct.
//          */
//         int total_frame_num = std::accumulate(consumed_frame_num, consumed_frame_num + m_frame_num_per_batch[batch_id], 0);
//         auto alloc_frames = frame_pool->allocate(total_frame_num);
//         int node_offset = m_batch_start_index[batch_id];

//         int frame_index = 0;
//         int queue_id = 0;

//         for (int i = 0; i < node_num; i++)
//         {
//             int offset = m_alloc_offset_per_node_cpu[batch_id][i];
//             int packet_id = 0;
//             /**
//              * @TODO: DO NOT ADD NODE_NUM, ADD m_batch_start_index[batch_id] instead.
//              */
//             for (int j = 0; j < m_frame_num_per_node[m_batch_start_index[batch_id] + i]; j++)
//             {
//                 if (consumed_frame_num[queue_id] > 0)
//                 {
//                     memcpy(frames + offset, alloc_frames.data() + frame_index, 8 * consumed_frame_num[queue_id]);
//                     frame_index += consumed_frame_num[queue_id];
//                     offset += consumed_frame_num[queue_id];
//                 }
//                 queue_id++;
//             }
//         }

//         // std::this_thread::sleep_for(std::chrono::milliseconds(10));

// #if !ENABLE_HUGE_GRAPH
//         cudaMemcpy(m_alloc_frames_gpu[batch_id], m_alloc_frames_cpu[batch_id], sizeof(Frame *) * m_total_alloc_frame_num[batch_id], cudaMemcpyHostToDevice);
// #endif
//     }

//     void FrameEncapsulationController::SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
//     }

//     void FrameEncapsulationController::BuildGraph(int batch_id)
//     {
//         cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
//         LaunchEncapsulateFrameKernel(m_grid_dim[batch_id], m_block_dim[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);

// #if ENABLE_HUGE_GRAPH

//         // cudaGraphNode_t kernel_node;
//         // size_t num_nodes;
//         // cudaGraphGetNodes(m_graphs[batch_id], &kernel_node, &num_nodes);

//         // cudaGraphNode_t alloc_num_memcpy_node;
//         cudaMemcpy3DParms alloc_num_memcpy_params = {0};
//         alloc_num_memcpy_params.srcPtr = make_cudaPitchedPtr(m_alloc_num_per_frame_queue_gpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], m_frame_num_per_batch[batch_id], 1);
//         alloc_num_memcpy_params.dstPtr = make_cudaPitchedPtr(m_alloc_num_per_frame_queue_cpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], m_frame_num_per_batch[batch_id], 1);
//         alloc_num_memcpy_params.extent = make_cudaExtent(sizeof(int) * m_frame_num_per_batch[batch_id], 1, 1);
//         alloc_num_memcpy_params.kind = cudaMemcpyDeviceToHost;
//         // cudaGraphAddMemcpyNode(&alloc_num_memcpy_node, m_graphs[batch_id], &kernel_node,1, &alloc_num_memcpy_params);

//         // cudaGraphNode_t update_host_node;
//         cudaHostNodeParams update_host_params = {0};
//         auto update_host_func = std::bind(&FrameEncapsulationController::UpdateComsumedFrames, this, batch_id);
//         auto update_host_func_ptr = new std::function<void()>(update_host_func);
//         update_host_params.fn = VDES::HostNodeCallback;
//         update_host_params.userData = update_host_func_ptr;
//         // cudaGraphAddHostNode(&update_host_node, m_graphs[batch_id], &alloc_num_memcpy_node, 1, &update_host_params);

//         // cudaGraphNode_t alloc_frame_memcpy_node;
//         cudaMemcpy3DParms alloc_frame_memcpy_params = {0};
//         alloc_frame_memcpy_params.srcPtr = make_cudaPitchedPtr(m_alloc_frames_cpu[batch_id], sizeof(Frame *) * m_total_alloc_frame_num[batch_id], m_total_alloc_frame_num[batch_id], 1);
//         alloc_frame_memcpy_params.dstPtr = make_cudaPitchedPtr(m_alloc_frames_gpu[batch_id], sizeof(Frame *) * m_total_alloc_frame_num[batch_id], m_total_alloc_frame_num[batch_id], 1);
//         alloc_frame_memcpy_params.extent = make_cudaExtent(sizeof(Frame *) * m_total_alloc_frame_num[batch_id], 1, 1);
//         alloc_frame_memcpy_params.kind = cudaMemcpyHostToDevice;
//         // cudaGraphAddMemcpyNode(&alloc_frame_memcpy_node, m_graphs[batch_id], &update_host_node,1, &alloc_frame_memcpy_params);

//         m_memcpy_param.push_back(alloc_num_memcpy_params);
//         m_memcpy_param.push_back(alloc_frame_memcpy_params);
//         m_host_param.push_back(update_host_params);

// #endif
//     }

//     void FrameEncapsulationController::BuildGraph()
//     {
//         int batch_num = m_batch_start_index.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraph(i);
//         }
//     }

//     cudaGraph_t FrameEncapsulationController::GetGraph(int batch_id)
//     {
//         return m_graphs[batch_id];
//     }

// #if ENABLE_HUGE_GRAPH

//     std::vector<cudaMemcpy3DParms> &FrameEncapsulationController::GetMemcpyParams()
//     {
//         return m_memcpy_param;
//     }

//     std::vector<cudaHostNodeParams> &FrameEncapsulationController::GetHostParams()
//     {
//         return m_host_param;
//     }

// #endif
// }

// namespace VDES
// {

//     FrameEncapsulationController::FrameEncapsulationController()
//     {
//     }

//     FrameEncapsulationController::~FrameEncapsulationController()
//     {
//     }

//     void FrameEncapsulationController::InitializeKernelParams()
//     {
//         int batch_num = m_batch_end_index.size();

//         m_packet_sizes.push_back(sizeof(Ipv4Packet));

//         m_graph_execs.resize(batch_num);

//         for (int i = 0; i < batch_num; i++)
//         {
//             cudaGraph_t graph;
//             cudaGraphCreate(&graph, 0);
//             m_graphs.push_back(graph);

//             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
//             int frame_queue_num = std::accumulate(m_frame_num_per_node.begin() + m_batch_start_index[i], m_frame_num_per_node.begin() + m_batch_end_index[i], 0);
//             int max_frame_num = frame_queue_num * MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;

//             FrameEncapsulationParams *cpu_param = new FrameEncapsulationParams;
//             cudaMallocAsync(&cpu_param->packets_egresses, sizeof(GPUQueue<void *> *) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->frame_queues, sizeof(GPUQueue<Frame *> *) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->frame_queue_macs, sizeof(uint8_t *) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->frame_queue_offset_per_node, sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->node_id_per_frame_queue, sizeof(int) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_packet_len_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_packet_timestamp_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_dst_ip_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_packet_size, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);

// #if ENABLE_FATTREE_MODE
//             cudaMallocAsync(&cpu_param->mac_addr_ft, sizeof(uint8_t *) * frame_queue_num, m_streams[i]);
//             cpu_param->k = m_ft_k;
//             cpu_param->base_ip = m_ft_base_ip;
//             cpu_param->ip_group_size = m_ft_ip_group_size;

// #else
//             cudaMallocAsync(&cpu_param->arp_tables, sizeof(GPUQueue<ARPRule *> *) * frame_queue_num, m_streams[i]);
// #endif

//             cudaMallocAsync(&cpu_param->alloc_frames, sizeof(Frame *) * max_frame_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->alloc_num_per_frame_queue, sizeof(int) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->alloc_offset_per_node, sizeof(int) * node_num, m_streams[i]);

// #if ENABLE_CACHE

//             cudaMallocAsync(&cpu_param->swap_out_l3_packets, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             /**
//              * @TODO: Use max size as the copy size
//              */
//             cudaMallocAsync(&cpu_param->swap_out_l3_packets_num, sizeof(int) * frame_queue_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_cache_ptr, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);

// #endif

//             // initialize
//             /**
//              * TODO: Copy the right number of the queues and the right start index of the queues.
//              */
//             int node_offset = m_batch_start_index[i];
//             int frame_offset = std::accumulate(m_frame_num_per_node.begin(), m_frame_num_per_node.begin() + m_batch_start_index[i], 0);
//             cudaMemcpyAsync(cpu_param->packets_egresses, m_packets_egresses.data() + frame_offset, sizeof(GPUQueue<void *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->frame_queues, m_frame_queues.data() + frame_offset, sizeof(GPUQueue<Frame *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);

//             int len_offset = 2;
//             int timestamp_offset = 35;
//             int dst_ip_offset = 16;
//             int packet_size = sizeof(Ipv4Packet);
//             cudaMemcpyAsync(cpu_param->l3_packet_len_offset, &len_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->l3_packet_timestamp_offset, &timestamp_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->l3_dst_ip_offset, &dst_ip_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->l3_packet_size, &packet_size, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);

//             for (int j = 0; j < 6; j++)
//             {
//                 cpu_param->null_mac[j] = 0;
//             }
//             uint8_t *frame_queue_macs_gpu;
//             cudaMallocAsync(&frame_queue_macs_gpu, sizeof(uint8_t) * 6 * frame_queue_num, m_streams[i]);
//             m_frame_queue_macs_gpu.push_back(frame_queue_macs_gpu);
//             std::vector<uint8_t *> frame_queue_macs_cpu;
//             for (int j = 0; j < frame_queue_num; j++)
//             {
//                 cudaMemcpyAsync(frame_queue_macs_gpu + 6 * j, m_frame_queue_macs_cpu[node_offset + j], sizeof(uint8_t) * 6, cudaMemcpyHostToDevice, m_streams[i]);
//                 frame_queue_macs_cpu.push_back(frame_queue_macs_gpu + 6 * j);
//             }
//             cudaMemcpyAsync(cpu_param->frame_queue_macs, frame_queue_macs_cpu.data(), sizeof(uint8_t *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);

// #if ENABLE_FATTREE_MODE
//             uint8_t *temp_gpu_macs;
//             cudaMalloc(&temp_gpu_macs, sizeof(uint8_t) * 6 * frame_queue_num);
//             std::vector<uint8_t *> mac_addr_ft;
//             for (int j = 0; j < frame_queue_num; j++)
//             {
//                 mac_addr_ft.push_back(temp_gpu_macs + 6 * j);
//             }
//             cudaMemcpyAsync(cpu_param->mac_addr_ft, mac_addr_ft.data(), sizeof(uint8_t *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);

// #else
//             cudaMemcpyAsync(cpu_param->arp_tables, m_arp_tables.data() + frame_offset, sizeof(GPUQueue<ARPRule *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
// #endif

//             auto alloc_frame = frame_pool->allocate(max_frame_num);
//             cudaMemcpyAsync(cpu_param->alloc_frames, alloc_frame.data(), sizeof(Frame *) * max_frame_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemsetAsync(cpu_param->alloc_num_per_frame_queue, 0, sizeof(int) * frame_queue_num, m_streams[i]);

//             int *alloc_offset = new int[node_num];
//             int offset = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 alloc_offset[j] = offset;
//                 offset += (m_frame_num_per_node[node_offset + j] * MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM);
//             }
//             cudaMemcpyAsync(cpu_param->alloc_offset_per_node, alloc_offset, sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

//             std::vector<int> node_id_per_frame;
//             std::vector<int> frame_offset_per_node;
//             int frame_offset_temp = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 for (int k = 0; k < m_frame_num_per_node[node_offset + j]; k++)
//                 {
//                     node_id_per_frame.push_back(j);
//                     // frame_queue_id
//                 }
//                 frame_offset_per_node.push_back(frame_offset_temp);
//                 frame_offset_temp += m_frame_num_per_node[node_offset + j];
//             }
//             cudaMemcpyAsync(cpu_param->node_id_per_frame_queue, node_id_per_frame.data(), sizeof(int) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->frame_queue_offset_per_node, frame_offset_per_node.data(), sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

// #if ENABLE_CACHE

//             int cache_size = 0;
//             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 cache_size += (max_frame_num * m_packet_sizes[j]);
//             }
//             uint8_t *l3_cache_gpu;
//             cudaMalloc(&l3_cache_gpu, cache_size);
//             uint8_t *l3_cache_cpu = new uint8_t[cache_size];
//             m_l3_cache_gpu.push_back(l3_cache_gpu);
//             m_l3_cache_cpu.push_back(l3_cache_cpu);

//             uint8_t **l3_cache_ptr_gpu = new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType];
//             uint8_t **l3_cache_ptr_cpu = new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType];
//             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 for (int k = 0; k < max_frame_num; k++)
//                 {
//                     l3_cache_ptr_gpu[j * max_frame_num + k] = l3_cache_gpu + k * m_packet_sizes[j];
//                     l3_cache_ptr_cpu[j * max_frame_num + k] = l3_cache_cpu + k * m_packet_sizes[j];
//                 }
//                 l3_cache_gpu += max_frame_num * m_packet_sizes[j];
//                 l3_cache_cpu += max_frame_num * m_packet_sizes[j];
//             }
//             cudaMemcpyAsync(cpu_param->l3_cache_ptr, l3_cache_ptr_gpu, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyHostToDevice, m_streams[i]);
//             m_l3_cache_ptr_cpu.push_back(l3_cache_ptr_cpu);
//             m_l3_cache_ptr_gpu.push_back(l3_cache_ptr_gpu);

//             cudaStreamSynchronize(m_streams[i]);
//             m_l3_swap_out_packet_gpu.push_back(cpu_param->swap_out_l3_packets);
//             m_l3_swap_out_packet_cpu.push_back(new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType]);
//             m_l3_swap_out_packet_cpu_backup.push_back(new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType]);
//             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 if (j == NetworkProtocolType::IPv4)
//                 {
//                     /**
//                      * TODO: Allocate cpu memory on for the alloc_packets cpu.
//                      */
//                     auto alloc_packets = ipv4_packet_pool_cpu->allocate(max_frame_num);
//                     cudaMemcpyAsync(cpu_param->swap_out_l3_packets, (uint8_t **)alloc_packets.data(), sizeof(uint8_t *) * max_frame_num, cudaMemcpyHostToDevice, m_streams[i]);
//                     memcpy(m_l3_swap_out_packet_cpu[i], (uint8_t **)alloc_packets.data(), sizeof(uint8_t *) * max_frame_num);
//                     memcpy(m_l3_swap_out_packet_cpu_backup[i], (uint8_t **)alloc_packets.data(), sizeof(uint8_t *) * max_frame_num);
//                 }
//             }
//             cudaMemsetAsync(cpu_param->swap_out_l3_packets_num, 0, sizeof(int) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             m_l3_swap_out_packet_num_gpu.push_back(cpu_param->swap_out_l3_packets_num);
//             m_l3_swap_out_packet_num_cpu.push_back(new int[frame_queue_num * NetworkProtocolType::COUNT_NetworkProtocolType]);
//             m_cache_sizes.push_back(cache_size);
// #endif

//             cpu_param->max_packet_num = max_frame_num;
//             cpu_param->queue_num = frame_queue_num;
//             cpu_param->node_num = node_num;

//             /**
//              * @warning:Initialize the grid dim and block dim
//              */
//             dim3 block_dim(KERNEL_BLOCK_WIDTH);
//             dim3 grid_dim((frame_queue_num + block_dim.x - 1) / block_dim.x);

//             m_grid_dim.push_back(grid_dim);
//             m_block_dim.push_back(block_dim);

//             /**
//              * @warning:Initialize the total alloc frame num
//              */
//             m_frame_num_per_batch.push_back(frame_queue_num);
//             // cudaStreamSynchronize(m_streams[i]);
//             m_alloc_frames_cpu.push_back(new Frame *[max_frame_num]);
//             m_alloc_frames_gpu.push_back(cpu_param->alloc_frames);
//             m_alloc_num_per_frame_queue_gpu.push_back(cpu_param->alloc_num_per_frame_queue);
//             m_alloc_num_per_frame_queue_cpu.push_back(new int[frame_queue_num]);
//             m_alloc_offset_per_node_gpu.push_back(cpu_param->alloc_offset_per_node);
//             m_alloc_offset_per_node_cpu.push_back(alloc_offset);
//             m_total_alloc_frame_num.push_back(max_frame_num);

//             m_max_packet_num.push_back(max_frame_num);

//             FrameEncapsulationParams *gpu_param;
//             cudaMallocAsync(&gpu_param, sizeof(FrameEncapsulationParams), m_streams[i]);
//             cudaMemcpyAsync(gpu_param, cpu_param, sizeof(FrameEncapsulationParams), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaStreamSynchronize(m_streams[i]);
//             m_kernel_params.push_back(gpu_param);
//             delete cpu_param;
//             /**
//              * @deprecated: TEST FRAMES.
//              */
//             cudaStreamSynchronize(m_streams[i]);
//             Frame **tmp_frames = new Frame *[max_frame_num];
//             cudaMemcpy(tmp_frames, m_alloc_frames_gpu[i], sizeof(Frame *) * max_frame_num, cudaMemcpyDeviceToHost);
//             for (int j = 0; j < max_frame_num; j++)
//             {
//                 Frame tmp_frame;
//                 cudaMemcpy(&tmp_frame, tmp_frames[j], sizeof(Frame), cudaMemcpyDeviceToHost);
//                 // printf("addr of frame: %p\n", tmp_frames[j]);
//                 int m = max_frame_num;
//             }
//         }
//     }

//     void FrameEncapsulationController::SetStreams(cudaStream_t *streams, int stream_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + stream_num);
//     }

//     void FrameEncapsulationController::SetPacketProperties(GPUQueue<void *> **packet_queues, int queue_num)
//     {
//         m_packets_egresses.insert(m_packets_egresses.end(), packet_queues, packet_queues + queue_num);
//     }

//     void FrameEncapsulationController::SetFrameProperties(GPUQueue<Frame *> **frame_queues, uint8_t **frame_queue_macs, int *frame_num_per_node, int node_num)
//     {
//         int frame_queue = std::accumulate(frame_num_per_node, frame_num_per_node + node_num, 0);
//         m_frame_queues.insert(m_frame_queues.end(), frame_queues, frame_queues + frame_queue);

//         m_frame_queue_macs_cpu.insert(m_frame_queue_macs_cpu.end(), frame_queue_macs, frame_queue_macs + frame_queue);
//         m_frame_num_per_node.insert(m_frame_num_per_node.end(), frame_num_per_node, frame_num_per_node + node_num);
//     }

//     void FrameEncapsulationController::SetArpProperties(GPUQueue<ARPRule *> **arp_tables, int node_num)
//     {
//         m_arp_tables.insert(m_arp_tables.end(), arp_tables, arp_tables + node_num);
//     }

//     void FrameEncapsulationController::SetFatTreeArpProperties(uint16_t k, uint32_t base_ip, uint32_t ip_group_size)
//     {
//         m_ft_k = k;
//         m_ft_base_ip = base_ip;
//         m_ft_ip_group_size = ip_group_size;
//     }

// #if ENABLE_CACHE

//     void FrameEncapsulationController::CacheOutL3Packets(int batch_id)
//     {
//         int max_packet_num = m_max_packet_num[batch_id];
//         int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

//         cudaMemcpyAsync(m_l3_swap_out_packet_num_cpu[batch_id], m_l3_swap_out_packet_num_gpu[batch_id], sizeof(int) * max_packet_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_l3_cache_cpu[batch_id], m_l3_cache_gpu[batch_id], m_cache_sizes[batch_id], cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_alloc_num_per_frame_queue_cpu[batch_id], m_alloc_num_per_frame_queue_gpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_l3_swap_out_packet_cpu[batch_id], m_l3_swap_out_packet_gpu[batch_id], sizeof(uint8_t *) * max_packet_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);

//         int *packet_offsets = m_alloc_offset_per_node_cpu[batch_id];
//         int node_offset = m_batch_start_index[batch_id];
//         int *swap_out_num = m_l3_swap_out_packet_num_cpu[batch_id];

//         for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
//         {

//             int queue_id = 0;
//             uint8_t **origin_dst = m_l3_swap_out_packet_cpu_backup[batch_id] + i * max_packet_num;
//             uint8_t **origin_src = m_l3_cache_ptr_cpu[batch_id] + i * max_packet_num;
//             uint8_t **origin_recycle_packet = m_l3_swap_out_packet_cpu[batch_id] + i * max_packet_num;

//             int total_packet_num = std::accumulate(swap_out_num, swap_out_num + max_packet_num, 0);
//             std::vector<uint8_t *> alloc_packets;
//             if (i == NetworkProtocolType::IPv4)
//             {
//                 auto ipv4_packets = ipv4_packet_pool_cpu->allocate(total_packet_num);
//                 alloc_packets.insert(alloc_packets.end(), (uint8_t **)ipv4_packets.data(), (uint8_t **)(ipv4_packets.data() + total_packet_num));
//             }

//             // copy data from cache to discrete packets
//             int alloc_offset = 0;
//             std::vector<uint8_t *> recycle_l3_packets;
//             for (int j = 0; j < node_num; j++)
//             {
//                 uint8_t **dst = origin_dst + packet_offsets[j];
//                 uint8_t **src = origin_src + packet_offsets[j];
//                 uint8_t **recycle_packets = origin_recycle_packet + packet_offsets[j];

//                 int offset = 0;

//                 for (int k = 0; k < m_frame_num_per_node[node_offset + j]; k++)
//                 {
//                     for (int m = 0; m < swap_out_num[queue_id]; m++)
//                     {
//                         memcpy(dst[offset + m], src[offset + m], m_packet_sizes[i]);
//                         // update used packet
//                         dst[offset + m] = alloc_packets[alloc_offset];
//                         recycle_l3_packets.push_back(recycle_packets[offset + m]);
//                         alloc_offset++;
//                     }
//                     offset += swap_out_num[queue_id];
//                     queue_id++;
//                 }
//             }

//             if (i == NetworkProtocolType::IPv4)
//             {
//                 ipv4_packet_pool->deallocate((Ipv4Packet **)recycle_l3_packets.data(), recycle_l3_packets.size());
//             }

//             recycle_l3_packets.clear();
//         }
//         cudaMemcpyAsync(m_l3_swap_out_packet_gpu[batch_id], m_l3_swap_out_packet_cpu_backup[batch_id], sizeof(uint8_t *) * max_packet_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyHostToDevice, m_streams[batch_id]);
//     }

// #endif

//     void FrameEncapsulationController::LaunchInstance(int batch_id)
//     {
//         LaunchEncapsulateFrameKernel(m_grid_dim[batch_id], m_block_dim[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
//     }

//     void FrameEncapsulationController::Run(int batch_id)
//     {
//         LaunchInstance(batch_id);
//         cudaStreamSynchronize(m_streams[batch_id]);

// #if ENABLE_CACHE
//         CacheOutL3Packets(batch_id);
// #endif
//         UpdateComsumedFrames(batch_id);
//     }

//     void FrameEncapsulationController::Run()
//     {
//     }

//     void FrameEncapsulationController::UpdateComsumedFrames(int batch_id)
//     {

// #if !ENABLE_HUGE_GRAPH
//         cudaMemcpy(m_alloc_num_per_frame_queue_cpu[batch_id], m_alloc_num_per_frame_queue_gpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], cudaMemcpyDeviceToHost);
// #endif

//         int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
//         int *consumed_frame_num = m_alloc_num_per_frame_queue_cpu[batch_id];
//         VDES::Frame **frames = m_alloc_frames_cpu[batch_id];

//         int total_frame_num = std::accumulate(consumed_frame_num, consumed_frame_num + m_frame_num_per_batch[batch_id], 0);
//         auto alloc_frames = frame_pool->allocate(total_frame_num);
//         int node_offset = m_batch_start_index[batch_id];

//         int frame_index = 0;
//         int queue_id = 0;
//         for (int i = 0; i < node_num; i++)
//         {
//             int offset = m_alloc_offset_per_node_cpu[batch_id][i];
//             int packet_id = 0;
//             /**
//              * @TODO: DO NOT ADD NODE_NUM, ADD m_batch_start_index[batch_id] instead.
//              */
//             for (int j = 0; j < m_frame_num_per_node[m_batch_start_index[batch_id] + i]; j++)
//             {
//                 if (consumed_frame_num[queue_id] > 0)
//                 {
//                     memcpy(frames + offset, alloc_frames.data() + frame_index, 8 * consumed_frame_num[queue_id]);
//                     frame_index += consumed_frame_num[queue_id];
//                     offset += consumed_frame_num[queue_id];
//                 }
//                 queue_id++;
//             }
//         }

// #if !ENABLE_HUGE_GRAPH
//         cudaMemcpy(m_alloc_frames_gpu[batch_id], m_alloc_frames_cpu[batch_id], sizeof(Frame *) * m_total_alloc_frame_num[batch_id], cudaMemcpyHostToDevice);
// #endif
//     }

//     void FrameEncapsulationController::SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
//     }

//     void FrameEncapsulationController::BuildGraph(int batch_id)
//     {
//         cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
//         LaunchEncapsulateFrameKernel(m_grid_dim[batch_id], m_block_dim[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);

// #if ENABLE_HUGE_GRAPH

//         cudaMemcpy3DParms alloc_num_memcpy_params = {0};
//         alloc_num_memcpy_params.srcPtr = make_cudaPitchedPtr(m_alloc_num_per_frame_queue_gpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], m_frame_num_per_batch[batch_id], 1);
//         alloc_num_memcpy_params.dstPtr = make_cudaPitchedPtr(m_alloc_num_per_frame_queue_cpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], m_frame_num_per_batch[batch_id], 1);
//         alloc_num_memcpy_params.extent = make_cudaExtent(sizeof(int) * m_frame_num_per_batch[batch_id], 1, 1);
//         alloc_num_memcpy_params.kind = cudaMemcpyDeviceToHost;

//         cudaHostNodeParams update_host_params = {0};
//         auto update_host_func = std::bind(&FrameEncapsulationController::UpdateComsumedFrames, this, batch_id);
//         auto update_host_func_ptr = new std::function<void()>(update_host_func);
//         update_host_params.fn = VDES::HostNodeCallback;
//         update_host_params.userData = update_host_func_ptr;

//         cudaMemcpy3DParms alloc_frame_memcpy_params = {0};
//         alloc_frame_memcpy_params.srcPtr = make_cudaPitchedPtr(m_alloc_frames_cpu[batch_id], sizeof(Frame *) * m_total_alloc_frame_num[batch_id], m_total_alloc_frame_num[batch_id], 1);
//         alloc_frame_memcpy_params.dstPtr = make_cudaPitchedPtr(m_alloc_frames_gpu[batch_id], sizeof(Frame *) * m_total_alloc_frame_num[batch_id], m_total_alloc_frame_num[batch_id], 1);
//         alloc_frame_memcpy_params.extent = make_cudaExtent(sizeof(Frame *) * m_total_alloc_frame_num[batch_id], 1, 1);
//         alloc_frame_memcpy_params.kind = cudaMemcpyHostToDevice;

//         m_memcpy_param.push_back(alloc_num_memcpy_params);
//         m_memcpy_param.push_back(alloc_frame_memcpy_params);
//         m_host_param.push_back(update_host_params);

// #endif
//     }

//     void FrameEncapsulationController::BuildGraph()
//     {
//         int batch_num = m_batch_start_index.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraph(i);
//         }
//     }

//     cudaGraph_t FrameEncapsulationController::GetGraph(int batch_id)
//     {
//         return m_graphs[batch_id];
//     }

// #if ENABLE_HUGE_GRAPH

//     std::vector<cudaMemcpy3DParms> &FrameEncapsulationController::GetMemcpyParams()
//     {
//         return m_memcpy_param;
//     }

//     std::vector<cudaHostNodeParams> &FrameEncapsulationController::GetHostParams()
//     {
//         return m_host_param;
//     }

// #endif

// }

// #include "frame_encapsulation.h"

// namespace VDES
// {

//     FrameEncapsulationController::FrameEncapsulationController()
//     {
//     }

//     FrameEncapsulationController::~FrameEncapsulationController()
//     {
//     }

//     void FrameEncapsulationController::InitializeKernelParams()
//     {
//         int batch_num = m_batch_end_index.size();

//         m_packet_sizes.push_back(sizeof(Ipv4Packet));
//         m_graphs.resize(batch_num);
//         m_graph_execs.resize(batch_num);

//         for (int i = 0; i < batch_num; i++)
//         {
//             int node_num = m_batch_end_index[i] - m_batch_start_index[i];
//             int frame_queue_num = std::accumulate(m_frame_num_per_node.begin() + m_batch_start_index[i], m_frame_num_per_node.begin() + m_batch_end_index[i], 0);
//             int max_frame_num = frame_queue_num * MAX_TRANSMITTED_PACKET_NUM + node_num * MAX_GENERATED_PACKET_NUM;

//             FrameEncapsulationParams *cpu_param = new FrameEncapsulationParams;
//             cudaMallocAsync(&cpu_param->packets_egresses, sizeof(GPUQueue<void *> *) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->frame_queues, sizeof(GPUQueue<Frame *> *) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->frame_queue_macs, sizeof(uint8_t *) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->frame_queue_offset_per_node, sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->node_id_per_frame_queue, sizeof(int) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_packet_len_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_packet_timestamp_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_dst_ip_offset, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_packet_size, sizeof(int) * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);

//             if (ENABLE_FATTREE_MODE)
//             {
//                 cudaMallocAsync(&cpu_param->mac_addr_ft, sizeof(uint8_t *) * frame_queue_num, m_streams[i]);
//                 cpu_param->k = m_ft_k;
//                 cpu_param->base_ip = m_ft_base_ip;
//                 cpu_param->ip_group_size = m_ft_ip_group_size;
//             }
//             else
//             {
//                 cudaMallocAsync(&cpu_param->arp_tables, sizeof(GPUQueue<ARPRule *> *) * frame_queue_num, m_streams[i]);
//             }

//             cudaMallocAsync(&cpu_param->alloc_frames, sizeof(Frame *) * max_frame_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->alloc_num_per_frame_queue, sizeof(int) * frame_queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->alloc_offset_per_node, sizeof(int) * node_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param->swap_out_l3_packets, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->swap_out_l3_packets_num, sizeof(int) * frame_queue_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             cudaMallocAsync(&cpu_param->l3_cache_ptr, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);

//             // initialize
//             /**
//              * TODO: Copy the right number of the queues and the write start index of the queues.
//              */
//             int node_offset = m_batch_start_index[i];
//             int frame_offset = std::accumulate(m_frame_num_per_node.begin(), m_frame_num_per_node.begin() + m_batch_start_index[i], 0);
//             cudaMemcpyAsync(cpu_param->packets_egresses, m_packets_egresses.data() + frame_offset, sizeof(GPUQueue<void *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->frame_queues, m_frame_queues.data() + frame_offset, sizeof(GPUQueue<Frame *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);

//             int len_offset = 2;
//             int timestamp_offset = 35;
//             int dst_ip_offset = 16;
//             int packet_size = sizeof(Ipv4Packet);
//             cudaMemcpyAsync(cpu_param->l3_packet_len_offset, &len_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->l3_packet_timestamp_offset, &timestamp_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->l3_dst_ip_offset, &dst_ip_offset, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->l3_packet_size, &packet_size, sizeof(int), cudaMemcpyHostToDevice, m_streams[i]);

//             for (int j = 0; j < 6; j++)
//             {
//                 cpu_param->null_mac[j] = 0;
//             }
//             uint8_t *frame_queue_macs_gpu;
//             cudaMallocAsync(&frame_queue_macs_gpu, sizeof(uint8_t) * 6 * frame_queue_num, m_streams[i]);
//             m_frame_queue_macs_gpu.push_back(frame_queue_macs_gpu);
//             std::vector<uint8_t *> frame_queue_macs_cpu;
//             for (int j = 0; j < frame_queue_num; j++)
//             {
//                 cudaMemcpyAsync(frame_queue_macs_gpu + 6 * j, m_frame_queue_macs_cpu[node_offset + j], sizeof(uint8_t) * 6, cudaMemcpyHostToDevice, m_streams[i]);
//                 frame_queue_macs_cpu.push_back(frame_queue_macs_gpu + 6 * j);
//             }
//             cudaMemcpyAsync(cpu_param->frame_queue_macs, frame_queue_macs_cpu.data(), sizeof(uint8_t *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);

//             if (ENABLE_FATTREE_MODE)
//             {
//                 uint8_t *temp_gpu_macs;
//                 cudaMallocAsync(&temp_gpu_macs, sizeof(uint8_t) * 6 * frame_queue_num, m_streams[i]);
//                 std::vector<uint8_t *> mac_addr_ft;
//                 for (int j = 0; j < frame_queue_num; j++)
//                 {
//                     mac_addr_ft.push_back(temp_gpu_macs + 6 * j);
//                 }
//                 cudaMemcpyAsync(cpu_param->mac_addr_ft, mac_addr_ft.data(), sizeof(uint8_t *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             }
//             else
//             {
//                 cudaMemcpyAsync(cpu_param->arp_tables, m_arp_tables.data() + frame_offset, sizeof(GPUQueue<ARPRule *> *) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             }

//             auto alloc_frame = frame_pool->allocate(max_frame_num);
//             cudaMemcpyAsync(cpu_param->alloc_frames, alloc_frame.data(), sizeof(Frame *) * max_frame_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemsetAsync(cpu_param->alloc_num_per_frame_queue, 0, sizeof(int) * frame_queue_num, m_streams[i]);

//             int *alloc_offset = new int[node_num];
//             int offset = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 alloc_offset[j] = offset;
//                 offset += (m_frame_num_per_node[node_offset + j] * MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM);
//             }
//             cudaMemcpyAsync(cpu_param->alloc_offset_per_node, alloc_offset, sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

//             std::vector<int> node_id_per_frame;
//             std::vector<int> frame_offset_per_node;
//             int frame_offset_temp = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 for (int k = 0; k < m_frame_num_per_node[node_offset + j]; k++)
//                 {
//                     node_id_per_frame.push_back(j);
//                     // frame_queue_id
//                 }
//                 frame_offset_per_node.push_back(frame_offset_temp);
//                 frame_offset_temp += m_frame_num_per_node[node_offset + j];
//             }
//             cudaMemcpyAsync(cpu_param->node_id_per_frame_queue, node_id_per_frame.data(), sizeof(int) * frame_queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param->frame_queue_offset_per_node, frame_offset_per_node.data(), sizeof(int) * node_num, cudaMemcpyHostToDevice, m_streams[i]);

//             int cache_size = 0;
//             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 cache_size += (max_frame_num * m_packet_sizes[j]);
//             }
//             uint8_t *l3_cache_gpu;
//             cudaMalloc(&l3_cache_gpu, cache_size);
//             uint8_t *l3_cache_cpu = new uint8_t[cache_size];
//             m_l3_cache_gpu.push_back(l3_cache_gpu);
//             m_l3_cache_cpu.push_back(l3_cache_cpu);

//             uint8_t **l3_cache_ptr_gpu = new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType];
//             uint8_t **l3_cache_ptr_cpu = new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType];
//             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 for (int k = 0; k < max_frame_num; k++)
//                 {
//                     l3_cache_ptr_gpu[j * max_frame_num + k] = l3_cache_gpu + k * m_packet_sizes[j];
//                     l3_cache_ptr_cpu[j * max_frame_num + k] = l3_cache_cpu + k * m_packet_sizes[j];
//                 }
//                 l3_cache_gpu += max_frame_num * m_packet_sizes[j];
//                 l3_cache_cpu += max_frame_num * m_packet_sizes[j];
//             }
//             cudaMemcpyAsync(cpu_param->l3_cache_ptr, l3_cache_ptr_gpu, sizeof(uint8_t *) * max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyHostToDevice, m_streams[i]);
//             m_l3_cache_ptr_cpu.push_back(l3_cache_ptr_cpu);
//             m_l3_cache_ptr_gpu.push_back(l3_cache_ptr_gpu);

//             cudaStreamSynchronize(m_streams[i]);
//             m_l3_swap_out_packet_gpu.push_back(cpu_param->swap_out_l3_packets);
//             m_l3_swap_out_packet_cpu.push_back(new uint8_t *[max_frame_num * NetworkProtocolType::COUNT_NetworkProtocolType]);
//             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 if (j == NetworkProtocolType::IPv4)
//                 {
//                     auto alloc_packets = ipv4_packet_pool->allocate(max_frame_num);
//                     cudaMemcpyAsync(cpu_param->swap_out_l3_packets, (uint8_t **)alloc_packets.data(), sizeof(uint8_t *) * max_frame_num, cudaMemcpyHostToDevice, m_streams[i]);
//                     memcpy(m_l3_swap_out_packet_cpu[i], (uint8_t **)alloc_packets.data(), sizeof(uint8_t *) * max_frame_num);
//                 }
//             }
//             cudaMemsetAsync(cpu_param->swap_out_l3_packets_num, 0, sizeof(int) * frame_queue_num * NetworkProtocolType::COUNT_NetworkProtocolType, m_streams[i]);
//             m_l3_swap_out_packet_num_gpu.push_back(cpu_param->swap_out_l3_packets_num);
//             m_l3_swap_out_packet_num_cpu.push_back(new int[frame_queue_num * NetworkProtocolType::COUNT_NetworkProtocolType]);

//             cpu_param->max_packet_num = max_frame_num;
//             cpu_param->queue_num = frame_queue_num;
//             cpu_param->node_num = node_num;

//             /**
//              * @warning:Initialize the grid dim and block dim
//              */
//             dim3 block_dim(KERNEL_BLOCK_WIDTH);
//             dim3 grid_dim((frame_queue_num + block_dim.x - 1) / block_dim.x);

//             m_grid_dim.push_back(grid_dim);
//             m_block_dim.push_back(block_dim);

//             /**
//              * @warning:Initialize the total alloc frame num
//              */
//             m_frame_num_per_batch.push_back(frame_queue_num);
//             // cudaStreamSynchronize(m_streams[i]);
//             m_alloc_frames_cpu.push_back(new Frame *[max_frame_num]);
//             m_alloc_frames_gpu.push_back(cpu_param->alloc_frames);
//             m_alloc_num_per_frame_queue_gpu.push_back(cpu_param->alloc_num_per_frame_queue);
//             m_alloc_num_per_frame_queue_cpu.push_back(new int[frame_queue_num]);
//             m_alloc_offset_per_node_gpu.push_back(cpu_param->alloc_offset_per_node);
//             m_alloc_offset_per_node_cpu.push_back(alloc_offset);
//             m_total_alloc_frame_num.push_back(max_frame_num);

//             m_cache_sizes.push_back(cache_size);
//             m_max_packet_num.push_back(max_frame_num);

//             FrameEncapsulationParams *gpu_param;
//             cudaMallocAsync(&gpu_param, sizeof(FrameEncapsulationParams), m_streams[i]);
//             cudaMemcpyAsync(gpu_param, cpu_param, sizeof(FrameEncapsulationParams), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaStreamSynchronize(m_streams[i]);
//             m_kernel_params.push_back(gpu_param);
//             delete cpu_param;
//         }
//     }

//     void FrameEncapsulationController::SetStreams(cudaStream_t *streams, int stream_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + stream_num);
//     }

//     void FrameEncapsulationController::SetPacketProperties(GPUQueue<void *> **packet_queues, int queue_num)
//     {
//         m_packets_egresses.insert(m_packets_egresses.end(), packet_queues, packet_queues + queue_num);
//     }

//     void FrameEncapsulationController::SetFrameProperties(GPUQueue<Frame *> **frame_queues, uint8_t **frame_queue_macs, int *frame_num_per_node, int node_num)
//     {
//         int frame_queue = std::accumulate(frame_num_per_node, frame_num_per_node + node_num, 0);
//         m_frame_queues.insert(m_frame_queues.end(), frame_queues, frame_queues + frame_queue);

//         m_frame_queue_macs_cpu.insert(m_frame_queue_macs_cpu.end(), frame_queue_macs, frame_queue_macs + frame_queue);
//         m_frame_num_per_node.insert(m_frame_num_per_node.end(), frame_num_per_node, frame_num_per_node + node_num);
//     }

//     void FrameEncapsulationController::SetArpProperties(GPUQueue<ARPRule *> **arp_tables, int node_num)
//     {
//         m_arp_tables.insert(m_arp_tables.end(), arp_tables, arp_tables + node_num);
//     }

//     void FrameEncapsulationController::SetFatTreeArpProperties(uint16_t k, uint32_t base_ip, uint32_t ip_group_size)
//     {
//         m_ft_k = k;
//         m_ft_base_ip = base_ip;
//         m_ft_ip_group_size = ip_group_size;
//     }

//     void FrameEncapsulationController::CacheOutL3Packets(int batch_id)
//     {
//         int max_packet_num = m_max_packet_num[batch_id];
//         int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];

//         cudaMemcpyAsync(m_l3_swap_out_packet_num_cpu[batch_id], m_l3_swap_out_packet_num_gpu[batch_id], sizeof(int) * max_packet_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_l3_cache_cpu[batch_id], m_l3_cache_gpu[batch_id], m_cache_sizes[batch_id], cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_alloc_num_per_frame_queue_cpu[batch_id], m_alloc_num_per_frame_queue_gpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);

//         int *packet_offsets = m_alloc_offset_per_node_cpu[batch_id];
//         int node_offset = m_batch_start_index[batch_id];
//         int *swap_out_num = m_l3_swap_out_packet_num_cpu[batch_id];

//         for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
//         {

//             int queue_id = 0;
//             uint8_t **origin_dst = m_l3_swap_out_packet_cpu[batch_id] + i * max_packet_num;
//             uint8_t **origin_src = m_l3_cache_ptr_cpu[batch_id] + i * max_packet_num;

//             int total_packet_num = std::accumulate(swap_out_num, swap_out_num + max_packet_num, 0);
//             std::vector<uint8_t *> alloc_packets;
//             if (i == NetworkProtocolType::IPv4)
//             {
//                 auto ipv4_packets = ipv4_packet_pool->allocate(total_packet_num);
//                 alloc_packets.insert(alloc_packets.end(), (uint8_t **)ipv4_packets.data(), (uint8_t **)(ipv4_packets.data() + total_packet_num));
//             }

//             // copy data from cache to discrete packets
//             int alloc_offset = 0;
//             for (int j = 0; j < node_num; j++)
//             {
//                 uint8_t **dst = origin_dst + packet_offsets[j];
//                 uint8_t **src = origin_src + packet_offsets[j];

//                 int offset = 0;

//                 for (int k = 0; k < m_frame_num_per_node[node_offset + j]; k++)
//                 {
//                     for (int m = 0; m < swap_out_num[queue_id]; m++)
//                     {
//                         memcpy(dst[offset + m], src[offset + m], m_packet_sizes[i]);
//                         // update used packet
//                         dst[offset + m] = alloc_packets[alloc_offset];
//                         alloc_offset++;
//                     }
//                     offset += swap_out_num[queue_id];
//                     queue_id++;
//                 }
//             }
//         }
//         cudaMemcpyAsync(m_l3_swap_out_packet_gpu[batch_id], m_l3_swap_out_packet_cpu[batch_id], sizeof(uint8_t *) * max_packet_num * NetworkProtocolType::COUNT_NetworkProtocolType, cudaMemcpyHostToDevice, m_streams[batch_id]);
//     }

//     void FrameEncapsulationController::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void FrameEncapsulationController::Run(int batch_id)
//     {
//         LaunchInstance(batch_id);
//         cudaStreamSynchronize(m_streams[batch_id]);
//         CacheOutL3Packets(batch_id);
//         UpdateComsumedFrames(batch_id);
//     }

//     void FrameEncapsulationController::Run()
//     {
//     }

//     void FrameEncapsulationController::UpdateComsumedFrames(int batch_id)
//     {
//         cudaMemcpy(m_alloc_num_per_frame_queue_cpu[batch_id], m_alloc_num_per_frame_queue_gpu[batch_id], sizeof(int) * m_frame_num_per_batch[batch_id], cudaMemcpyDeviceToHost);

//         int node_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
//         int *consumed_frame_num = m_alloc_num_per_frame_queue_cpu[batch_id];
//         VDES::Frame **frames = m_alloc_frames_cpu[batch_id];

//         int total_frame_num = std::accumulate(consumed_frame_num, consumed_frame_num + m_frame_num_per_batch[batch_id], 0);
//         auto alloc_frames = frame_pool->allocate(total_frame_num);
//         int node_offset = m_batch_start_index[batch_id];

//         int frame_index = 0;
//         int queue_id = 0;
//         for (int i = 0; i < node_num; i++)
//         {
//             int offset = m_alloc_offset_per_node_cpu[batch_id][i];
//             int packet_id = 0;
//             for (int j = 0; j < m_frame_num_per_node[node_num + i]; j++)
//             {
//                 for (int k = 0; k < consumed_frame_num[queue_id]; k++)
//                 {
//                     frames[offset + packet_id] = alloc_frames[frame_index++];
//                     packet_id++;
//                 }
//                 queue_id++;
//             }
//         }

//         cudaMemcpy(m_alloc_frames_gpu[batch_id], m_alloc_frames_cpu[batch_id], sizeof(Frame *) * m_total_alloc_frame_num[batch_id], cudaMemcpyHostToDevice);
//     }

//     void FrameEncapsulationController::SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
//     }

//     void FrameEncapsulationController::BuildGraph(int batch_id)
//     {
//         cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
//         LaunchEncapsulateFrameKernel(m_grid_dim[batch_id], m_block_dim[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);
//     }

//     void FrameEncapsulationController::BuildGraph()
//     {
//         int batch_num = m_batch_start_index.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraph(i);
//         }
//     }
// }