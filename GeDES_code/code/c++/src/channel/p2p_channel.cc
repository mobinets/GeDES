#include "p2p_channel.h"

namespace VDES
{
    P2PChannelController::P2PChannelController()
    {
    }

    P2PChannelController::~P2PChannelController()
    {
    }

    void P2PChannelController::SetIngressAndEgress(GPUQueue<Frame *> **ingresses, GPUQueue<Frame *> **egresses, int *propogation_delay_per_ch, int *link_transmission_rate, int ch_num)
    {
        m_ingress_queues.insert(m_ingress_queues.end(), ingresses, ingresses + ch_num);
        m_egress_queues.insert(m_egress_queues.end(), egresses, egresses + ch_num);
        m_propogation_delay_per_ch.insert(m_propogation_delay_per_ch.end(), propogation_delay_per_ch, propogation_delay_per_ch + ch_num);
        m_link_transmission_rate.insert(m_link_transmission_rate.end(), link_transmission_rate, link_transmission_rate + ch_num);
    }

    void P2PChannelController::SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num)
    {
        m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
        m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
    }

    void P2PChannelController::SetTimeslot(int64_t *timeslot_start_time, int64_t *timeslot_end_time)
    {
        m_timeslot_start_time = timeslot_start_time;
        m_timeslot_end_time = timeslot_end_time;
    }

    void P2PChannelController::SetStreams(cudaStream_t *streams, int stream_num)
    {
        m_streams.insert(m_streams.end(), streams, streams + stream_num);
    }

    void P2PChannelController::InitializeKernelParams()
    {
        int batch_num = m_batch_start_index.size();
        m_graph_execs.resize(batch_num);
        m_kernel_params.reserve(batch_num);

        m_total_block_num = 0;

        /**
         * @brief: Allocate the memory for the m_transmission_completed, which is uesed for checking status of transmission in channel.
         */
        for (int i = 0; i < batch_num; i++)
        {
            int queue_num = m_batch_end_index[i] - m_batch_start_index[i];
            m_total_block_num += (queue_num + KERNEL_BLOCK_WIDTH - 1) / KERNEL_BLOCK_WIDTH;
        }

        cudaMalloc(&m_transmission_completed, sizeof(uint8_t) * m_total_block_num);
        cudaMemsetAsync(m_transmission_completed, 0, sizeof(uint8_t) * m_total_block_num);
        uint8_t *transmission_completed = m_transmission_completed;

        for (int i = 0; i < batch_num; i++)
        {
            cudaGraph_t graph;
            cudaGraphCreate(&graph, 0);
            m_graphs.push_back(graph);

            int queue_num = m_batch_end_index[i] - m_batch_start_index[i];

            P2PParams cpu_param;
            cudaMallocAsync(&cpu_param.ingresses, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param.egresses, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
            cpu_param.queue_num = queue_num;
            cudaMallocAsync(&cpu_param.propogation_delay_per_ch, sizeof(int) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param.link_transmission_rate, sizeof(int) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param.last_tx_end_time, sizeof(int64_t) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_param.remainder_of_packet_len, sizeof(uint16_t) * queue_num, m_streams[i]);
            cpu_param.timeslot_start_time = m_timeslot_start_time;
            cpu_param.timeslot_end_time = m_timeslot_end_time;

            cpu_param.is_completed = transmission_completed;
            /**
             * @warning: Must make sure that queue_num can be divided by KERNEL_BLOCK_WIDTH.
             */
            transmission_completed += (queue_num + KERNEL_BLOCK_WIDTH - 1) / KERNEL_BLOCK_WIDTH;

            // Copy the pointers to the GPU
            int queue_offset = m_batch_start_index[i];
            cudaMemcpyAsync(cpu_param.ingresses, m_ingress_queues.data() + queue_offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param.egresses, m_egress_queues.data() + queue_offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param.propogation_delay_per_ch, m_propogation_delay_per_ch.data() + queue_offset, sizeof(int) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_param.link_transmission_rate, m_link_transmission_rate.data() + queue_offset, sizeof(int) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemsetAsync(cpu_param.last_tx_end_time, 0, sizeof(int64_t) * queue_num, m_streams[i]);
            cudaMemsetAsync(cpu_param.remainder_of_packet_len, 0, sizeof(uint16_t) * queue_num, m_streams[i]);

            P2PParams *gpu_param;
            cudaMallocAsync(&gpu_param, sizeof(P2PParams), m_streams[i]);
            cudaMemcpyAsync(gpu_param, &cpu_param, sizeof(P2PParams), cudaMemcpyHostToDevice, m_streams[i]);
            cudaStreamSynchronize(m_streams[i]);
            m_kernel_params.push_back(gpu_param);
        }
    }

    void P2PChannelController::BuildGraph(int batch_id)
    {
        int queue_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
        dim3 block_dim(KERNEL_BLOCK_WIDTH);
        dim3 grid_dim((queue_num + block_dim.x - 1) / block_dim.x);
        cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
        LuanchP2PTransmiFramesKernel(grid_dim, block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
        cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
        cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);
    }

    void P2PChannelController::BuildGraph()
    {
        int batch_num = m_batch_start_index.size();
        for (int i = 0; i < batch_num; i++)
        {
            BuildGraph(i);
        }
    }

    void P2PChannelController::LaunchInstance(int batch_id)
    {
        cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
    }

    void P2PChannelController::Run(int batch_id)
    {
        cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
        cudaStreamSynchronize(m_streams[batch_id]);
    }

    void P2PChannelController::Run()
    {
    }

    cudaGraph_t P2PChannelController::GetGraph(int batch_id)
    {
        return m_graphs[batch_id];
    }
    uint8_t *P2PChannelController::GetTransmissionCompletedAddr()
    {
        return m_transmission_completed;
    }

    uint32_t P2PChannelController::GetTotalBlockNum()
    {
        return m_total_block_num;
    }
}

// namespace VDES
// {
//     P2PChannelController::P2PChannelController()
//     {
//     }

//     P2PChannelController::~P2PChannelController()
//     {
//     }

//     void P2PChannelController::SetIngressAndEgress(GPUQueue<Frame *> **ingresses, GPUQueue<Frame *> **egresses, int *propogation_delay_per_ch, int *link_transmission_rate, int ch_num)
//     {
//         m_ingress_queues.insert(m_ingress_queues.end(), ingresses, ingresses + ch_num);
//         m_egress_queues.insert(m_egress_queues.end(), egresses, egresses + ch_num);
//         m_propogation_delay_per_ch.insert(m_propogation_delay_per_ch.end(), propogation_delay_per_ch, propogation_delay_per_ch + ch_num);
//         m_link_transmission_rate.insert(m_link_transmission_rate.end(), link_transmission_rate, link_transmission_rate + ch_num);
//     }

//     void P2PChannelController::SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
//         m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
//     }

//     void P2PChannelController::SetTimeslot(int64_t *timeslot_start_time, int64_t *timeslot_end_time)
//     {
//         m_timeslot_start_time = timeslot_start_time;
//         m_timeslot_end_time = timeslot_end_time;
//     }

//     void P2PChannelController::SetStreams(cudaStream_t *streams, int stream_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + stream_num);
//     }

//     void P2PChannelController::InitializeKernelParams()
//     {
//         int batch_num = m_batch_start_index.size();
//         m_graph_execs.resize(batch_num);
//         /**
//          * TODO: Reserve instead of resize.
//          */
//         m_kernel_params.reserve(batch_num);

//         for (int i = 0; i < batch_num; i++)
//         {
//             /**
//              * TODO: Create cuda Graph
//              */
//             cudaGraph_t graph;
//             cudaGraphCreate(&graph, 0);
//             m_graphs.push_back(graph);

//             int queue_num = m_batch_end_index[i] - m_batch_start_index[i];

//             P2PParams cpu_param;
//             cudaMallocAsync(&cpu_param.ingresses, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.egresses, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
//             cpu_param.queue_num = queue_num;
//             cudaMallocAsync(&cpu_param.propogation_delay_per_ch, sizeof(int) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.link_transmission_rate, sizeof(int) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.last_tx_end_time, sizeof(int64_t) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.remainder_of_packet_len, sizeof(uint16_t) * queue_num, m_streams[i]);
//             cpu_param.timeslot_start_time = m_timeslot_start_time;
//             cpu_param.timeslot_end_time = m_timeslot_end_time;

//             // Copy the pointers to the GPU
//             int queue_offset = m_batch_start_index[i];
//             cudaMemcpyAsync(cpu_param.ingresses, m_ingress_queues.data() + queue_offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.egresses, m_egress_queues.data() + queue_offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.propogation_delay_per_ch, m_propogation_delay_per_ch.data() + queue_offset, sizeof(int) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.link_transmission_rate, m_link_transmission_rate.data() + queue_offset, sizeof(int) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemsetAsync(cpu_param.last_tx_end_time, 0, sizeof(int64_t) * queue_num, m_streams[i]);
//             cudaMemsetAsync(cpu_param.remainder_of_packet_len, 0, sizeof(uint16_t) * queue_num, m_streams[i]);

//             P2PParams *gpu_param;
//             cudaMallocAsync(&gpu_param, sizeof(P2PParams), m_streams[i]);
//             cudaMemcpyAsync(gpu_param, &cpu_param, sizeof(P2PParams), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaStreamSynchronize(m_streams[i]);
//             m_kernel_params.push_back(gpu_param);
//         }
//     }

//     void P2PChannelController::BuildGraph(int batch_id)
//     {
//         int queue_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
//         dim3 block_dim(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim((queue_num + block_dim.x - 1) / block_dim.x);
//         cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
//         LuanchP2PTransmiFramesKernel(grid_dim, block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);
//     }

//     void P2PChannelController::BuildGraph()
//     {
//         int batch_num = m_batch_start_index.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraph(i);
//         }
//     }

//     void P2PChannelController::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void P2PChannelController::Run(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);
//     }

//     void P2PChannelController::Run()
//     {
//     }

//     cudaGraph_t P2PChannelController::GetGraph(int batch_id)
//     {
//         return m_graphs[batch_id];
//     }

// }

// #include "p2p_channel.h"

// namespace VDES
// {
//     P2PChannelController::P2PChannelController()
//     {
//     }

//     P2PChannelController::~P2PChannelController()
//     {
//     }

//     void P2PChannelController::SetIngressAndEgress(GPUQueue<Frame *> **ingresses, GPUQueue<Frame *> **egresses, int *propogation_delay_per_ch, int *link_transmission_rate, int ch_num)
//     {
//         m_ingress_queues.insert(m_ingress_queues.end(), ingresses, ingresses + ch_num);
//         m_egress_queues.insert(m_egress_queues.end(), egresses, egresses + ch_num);
//         m_propogation_delay_per_ch.insert(m_propogation_delay_per_ch.end(), propogation_delay_per_ch, propogation_delay_per_ch + ch_num);
//         m_link_transmission_rate.insert(m_link_transmission_rate.end(), link_transmission_rate, link_transmission_rate + ch_num);
//     }

//     void P2PChannelController::SetBatchProperties(int *batch_start_index, int *batch_end_index, int batch_num)
//     {
//         m_batch_end_index.insert(m_batch_end_index.end(), batch_end_index, batch_end_index + batch_num);
//         m_batch_start_index.insert(m_batch_start_index.end(), batch_start_index, batch_start_index + batch_num);
//     }

//     void P2PChannelController::SetTimeslot(int64_t *timeslot_start_time, int64_t *timeslot_end_time)
//     {
//         m_timeslot_start_time = timeslot_start_time;
//         m_timeslot_end_time = timeslot_end_time;
//     }

//     void P2PChannelController::SetStreams(cudaStream_t *streams, int stream_num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + stream_num);
//     }

//     void P2PChannelController::InitializeKernelParams()
//     {
//         int batch_num = m_batch_start_index.size();
//         m_graph_execs.resize(batch_num);
//         m_kernel_params.resize(batch_num);

//         for (int i = 0; i < batch_num; i++)
//         {
//             int queue_num = m_batch_end_index[i] - m_batch_start_index[i];

//             P2PParams cpu_param;
//             cudaMallocAsync(&cpu_param.ingresses, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.egresses, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
//             cpu_param.queue_num = queue_num;
//             cudaMallocAsync(&cpu_param.propogation_delay_per_ch, sizeof(int) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.link_transmission_rate, sizeof(int) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_param.last_tx_end_time, sizeof(int64_t) * queue_num, m_streams[i]);
//             cpu_param.timeslot_start_time = m_timeslot_start_time;
//             cpu_param.timeslot_end_time = m_timeslot_end_time;

//             // Copy the pointers to the GPU
//             int queue_offset = m_batch_start_index[i];
//             cudaMemcpyAsync(cpu_param.ingresses, m_ingress_queues.data() + queue_offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.egresses, m_egress_queues.data() + queue_offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.propogation_delay_per_ch, m_propogation_delay_per_ch.data() + queue_offset, sizeof(int) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_param.link_transmission_rate, m_link_transmission_rate.data() + queue_offset, sizeof(int) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemsetAsync(cpu_param.last_tx_end_time, 0, sizeof(int64_t) * queue_num, m_streams[i]);

//             P2PParams *gpu_param;
//             cudaMallocAsync(&gpu_param, sizeof(P2PParams), m_streams[i]);
//             cudaMemcpyAsync(gpu_param, &cpu_param, sizeof(P2PParams), cudaMemcpyHostToDevice, m_streams[i]);
//             cudaStreamSynchronize(m_streams[i]);
//             m_kernel_params.push_back(gpu_param);
//         }
//     }

//     void P2PChannelController::BuildGraph(int batch_id)
//     {
//         int queue_num = m_batch_end_index[batch_id] - m_batch_start_index[batch_id];
//         dim3 block_dim(KERNEL_BLOCK_WIDTH);
//         dim3 grid_dim((queue_num + block_dim.x - 1) / block_dim.x);
//         cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
//         LuanchP2PTransmiFramesKernel(grid_dim, block_dim, m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
//         cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);
//     }

//     void P2PChannelController::BuildGraph()
//     {
//         int batch_num = m_batch_start_index.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraph(i);
//         }
//     }

//     void P2PChannelController::LaunchInstance(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//     }

//     void P2PChannelController::Run(int batch_id)
//     {
//         cudaGraphLaunch(m_graph_execs[batch_id], m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);
//     }

//     void P2PChannelController::Run()
//     {
//     }

// }