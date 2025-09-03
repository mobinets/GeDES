#include "packet_cache.h"
#include "thread_pool.h"
#include "conf.h"
#include <algorithm>
#include <numeric>

namespace VDES
{
    CacheController::CacheController()
    {
    }

    CacheController::~CacheController()
    {
    }

    void CacheController::InitializeKernelParams()
    {
        int batch_num = m_batch_start_index.size();
        m_graphs.resize(batch_num);
        m_graph_execs.resize(batch_num);
        for (int i = 0; i < batch_num; i++)
        {
            int queue_num = std::accumulate(m_frame_queue_num_per_node.begin() + m_batch_start_index[i], m_frame_queue_num_per_node.begin() + m_batch_end_index[i], 0);
            m_frame_queue_num_per_batch.push_back(queue_num);

            dim3 block_dim(KERNEL_BLOCK_WIDTH);
            dim3 grid_dim((queue_num + block_dim.x - 1) / block_dim.x);
            m_grid_dims.push_back(grid_dim);
            m_block_dims.push_back(block_dim);

            // allocate gpu memory
            CacheParam cpu_params;
            cudaMallocAsync(&cpu_params.egress_queues, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_params.egress_status, sizeof(FrameQueueStatus *) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_params.swap_in_frame_num, sizeof(int) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_params.swap_out_frame_num, sizeof(int) * queue_num, m_streams[i]);
            cudaMallocAsync(&cpu_params.swap_in_frame_egress, sizeof(Frame **) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
            cudaMallocAsync(&cpu_params.swap_out_frame_egress, sizeof(Frame **) * queue_num * MAX_SWAP_FRAME_NUM, m_streams[i]);
            cudaMallocAsync(&cpu_params.swap_in_cache_space, sizeof(VDES::Frame) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
            cudaMallocAsync(&cpu_params.swap_out_cache_space, sizeof(VDES::Frame) * queue_num * MAX_SWAP_FRAME_NUM, m_streams[i]);

            m_swap_in_frame_num_cpu.push_back(new int[queue_num]);
            m_swap_out_frame_num_cpu.push_back(new int[queue_num]);
            m_swap_in_frame_num_gpu.push_back(cpu_params.swap_in_frame_num);
            m_swap_out_frame_num_gpu.push_back(cpu_params.swap_out_frame_num);

            // initialize params
            int offset = std::accumulate(m_frame_queue_num_per_node.begin(), m_frame_queue_num_per_node.begin() + m_batch_start_index[i], 0);
            cudaMemcpyAsync(cpu_params.egress_queues, m_egress_queues.data() + offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemcpyAsync(cpu_params.egress_status, m_egress_status.data() + offset, sizeof(FrameQueueStatus *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
            cudaMemsetAsync(cpu_params.swap_in_frame_num, 0, sizeof(int) * queue_num, m_streams[i]);
            cudaMemsetAsync(cpu_params.swap_out_frame_num, 0, sizeof(int) * queue_num, m_streams[i]);

            // initialize swap in/out frame egress
            m_swap_in_frame_egress_cpu.push_back(new Frame *[queue_num * MAX_TRANSMITTED_PACKET_NUM]);
            m_swap_in_frame_egress_gpu.push_back(cpu_params.swap_in_frame_egress);
            m_swap_in_frame_egress_gpu_backup.push_back(new Frame *[queue_num * MAX_TRANSMITTED_PACKET_NUM]);
            m_swap_out_frame_egress_cpu.push_back(new Frame *[queue_num * MAX_SWAP_FRAME_NUM]);
            m_swap_out_frame_egress_gpu.push_back(cpu_params.swap_out_frame_egress);
            m_swap_out_frame_egress_gpu_backup.push_back(new Frame *[queue_num * MAX_SWAP_FRAME_NUM]);
            InitializeSwapFramePtr(i);

            // initialize swap in/out cache space
            m_swap_in_cache_space_cpu.push_back(new VDES::Frame[queue_num * MAX_TRANSMITTED_PACKET_NUM]);
            m_swap_out_cache_space_cpu.push_back(new VDES::Frame[queue_num * MAX_SWAP_FRAME_NUM]);
            m_swap_in_cache_space_gpu.push_back(cpu_params.swap_in_cache_space);
            m_swap_out_cache_space_gpu.push_back(cpu_params.swap_out_cache_space);

            cudaMemsetAsync(cpu_params.swap_in_cache_space, 0, sizeof(VDES::Frame) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
            cudaMemsetAsync(cpu_params.swap_out_cache_space, 0, sizeof(VDES::Frame) * queue_num * MAX_SWAP_FRAME_NUM, m_streams[i]);

            cpu_params.timeslot_start = m_timeslot_start;
            cpu_params.timeslot_end = m_timeslot_end;
            cpu_params.lookahead_timeslot_num = m_lookahead_timeslot_num;
            cpu_params.queue_num = queue_num;

            CacheParam *gpu_params;
            cudaMallocAsync(&gpu_params, sizeof(CacheParam), m_streams[i]);
            cudaMemcpyAsync(gpu_params, &cpu_params, sizeof(CacheParam), cudaMemcpyHostToDevice, m_streams[i]);

            m_kernel_params.push_back(gpu_params);
            cudaStreamSynchronize(m_streams[i]);
        }
    }

    void CacheController::InitializeSwapFramePtr(int batch_id)
    {
        int queue_num = m_frame_queue_num_per_batch[batch_id];
        int total_swap_in_num = queue_num * MAX_TRANSMITTED_PACKET_NUM;
        int total_swap_out_num = queue_num * MAX_SWAP_FRAME_NUM;

        // initialize swap in frame egress, gpu ptr
        auto gpu_frames = frame_pool->allocate(total_swap_in_num);
        cudaMemcpyAsync(m_swap_in_frame_egress_gpu[batch_id], gpu_frames.data(), sizeof(Frame *) * total_swap_in_num, cudaMemcpyHostToDevice, m_streams[batch_id]);

        // initialize swap out frame egress, cpu ptr
        auto cpu_frames = frame_pool_cpu->allocate(total_swap_out_num);
        cudaMemcpyAsync(m_swap_out_frame_egress_gpu[batch_id], cpu_frames.data(), sizeof(Frame *) * total_swap_out_num, cudaMemcpyHostToDevice, m_streams[batch_id]);

        // backup frame ptrs
        memcpy(m_swap_in_frame_egress_cpu[batch_id], gpu_frames.data(), sizeof(Frame *) * total_swap_in_num);
        memcpy(m_swap_in_frame_egress_gpu_backup[batch_id], gpu_frames.data(), sizeof(Frame *) * total_swap_in_num);
        memcpy(m_swap_out_frame_egress_cpu[batch_id], cpu_frames.data(), sizeof(Frame *) * total_swap_out_num);
        memcpy(m_swap_out_frame_egress_gpu_backup[batch_id], cpu_frames.data(), sizeof(Frame *) * total_swap_out_num);
    }

    void CacheController::SetBatches(int *start_index, int *end_index, int batch_num)
    {
        m_batch_start_index.insert(m_batch_start_index.end(), start_index, start_index + batch_num);
        m_batch_end_index.insert(m_batch_end_index.end(), end_index, end_index + batch_num);
    }

    void CacheController::SetEgressProperties(GPUQueue<Frame *> **egress_queues, FrameQueueStatus **egress_status, int *frame_queue_num_per_node, int node_num)
    {
        int queue_num = std::accumulate(frame_queue_num_per_node, frame_queue_num_per_node + node_num, 0);
        m_egress_queues.insert(m_egress_queues.end(), egress_queues, egress_queues + queue_num);
        m_egress_status.insert(m_egress_status.end(), egress_status, egress_status + queue_num);
        m_frame_queue_num_per_node.insert(m_frame_queue_num_per_node.end(), frame_queue_num_per_node, frame_queue_num_per_node + node_num);
    }

    void CacheController::SetLookaheadTimeslotNum(int lookahead_timeslot_num)
    {
        m_lookahead_timeslot_num = lookahead_timeslot_num;
    }

    void CacheController::SetTimeSlot(int64_t *timeslot_start, int64_t *timeslot_end)
    {
        m_timeslot_start = timeslot_start;
        m_timeslot_end = timeslot_end;
    }

    void CacheController::CacheInFrames(int batch_id)
    {
        int batch_size = m_frame_queue_num_per_batch[batch_id];
        int batch_num = m_batch_start_index.size();
        int queue_num = std::accumulate(m_frame_queue_num_per_node.begin() + m_batch_start_index[batch_id], m_frame_queue_num_per_node.begin() + m_batch_end_index[batch_id], 0);

        cudaMemcpyAsync(m_swap_in_frame_num_cpu[batch_id], m_swap_in_frame_num_gpu[batch_id], sizeof(int) * queue_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);
        cudaMemcpyAsync(m_swap_in_frame_egress_cpu[batch_id], m_swap_in_frame_egress_gpu[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyDeviceToHost, m_streams[batch_id]);

        cudaMemcpyAsync(m_swap_in_frame_egress_gpu[batch_id], m_swap_in_frame_egress_gpu_backup[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyHostToDevice, m_streams[batch_id]);

        // copy frame to cache space
        int *swap_in_num = m_swap_in_frame_num_cpu[batch_id];
        int total_used_frame_num = std::accumulate(swap_in_num, swap_in_num + batch_size, 0);
        auto alloc_frames = frame_pool->allocate(total_used_frame_num);
        std::vector<VDES::Frame *> recycle_frames;
        int frame_index = 0;

        for (int i = 0; i < queue_num; i++)
        {
            int offset = i * MAX_TRANSMITTED_PACKET_NUM;
            VDES::Frame *dst = m_swap_in_cache_space_cpu[batch_id] + offset;
            VDES::Frame **src = m_swap_in_frame_egress_cpu[batch_id] + offset;

            for (int j = 0; j < swap_in_num[i]; j++)
            {
                // copy frame to cache space
                VDES::Frame tmp_frame;
                memcpy(dst + j, src[j], sizeof(VDES::Frame));
                // recycle cpu frames
                recycle_frames.push_back(src[j]);
                // reset comsumed frames using gpu ptr
                src[j] = alloc_frames[frame_index++];
            }
        }
        cudaStreamSynchronize(m_streams[batch_id]);
        cudaMemcpyAsync(m_swap_in_cache_space_gpu[batch_id], m_swap_in_cache_space_cpu[batch_id], sizeof(VDES::Frame) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyHostToDevice, m_streams[batch_id]);
        LaunchCopyFrameCacheSpaceKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
        cudaMemcpyAsync(m_swap_in_frame_egress_gpu[batch_id], m_swap_in_frame_egress_cpu[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyHostToDevice, m_streams[batch_id]);
        memcpy(m_swap_in_frame_egress_gpu_backup[batch_id], m_swap_in_frame_egress_cpu[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM);
        frame_pool_cpu->deallocate(recycle_frames.data(), total_used_frame_num);
    }

    void CacheController::CacheOutFrames(int batch_id)
    {
        int batch_size = m_frame_queue_num_per_batch[batch_id];

        cudaMemcpyAsync(m_swap_out_cache_space_cpu[batch_id], m_swap_out_cache_space_gpu[batch_id], sizeof(VDES::Frame) * MAX_SWAP_FRAME_NUM * batch_size, cudaMemcpyDeviceToHost, m_streams[batch_id]);
        cudaMemcpyAsync(m_swap_out_frame_egress_cpu[batch_id], m_swap_out_frame_egress_gpu[batch_id], sizeof(Frame *) * MAX_SWAP_FRAME_NUM * batch_size, cudaMemcpyDeviceToHost, m_streams[batch_id]);
        cudaMemcpyAsync(m_swap_out_frame_num_cpu[batch_id], m_swap_out_frame_num_gpu[batch_id], sizeof(int) * batch_size, cudaMemcpyDeviceToHost, m_streams[batch_id]);
        cudaStreamSynchronize(m_streams[batch_id]);

        // copy cache space to frame
        int *swap_out_num = m_swap_out_frame_num_cpu[batch_id];
        int total_used_frame_num = std::accumulate(swap_out_num, swap_out_num + batch_size, 0);
        auto alloc_frames = frame_pool_cpu->allocate(total_used_frame_num);
        std::vector<VDES::Frame *> recycle_frames;
        int frame_index = 0;

        for (int i = 0; i < batch_size; i++)
        {
            int offset = i * MAX_SWAP_FRAME_NUM;
            VDES::Frame *src = m_swap_out_cache_space_cpu[batch_id] + offset;
            VDES::Frame **dst = m_swap_out_frame_egress_gpu_backup[batch_id] + offset;
            VDES::Frame **gpu_frames = m_swap_out_frame_egress_cpu[batch_id] + offset;

            for (int j = 0; j < swap_out_num[i]; j++)
            {
                // copy cache space to frame
                memcpy(dst[j], src + j, sizeof(VDES::Frame));
                // recycle gpu frames
                recycle_frames.push_back(gpu_frames[j]);
                // reset comsumed frames using cpu ptr
                gpu_frames[j] = alloc_frames[frame_index++];
            }
        }

        // reset used frame ptr
        cudaMemcpyAsync(m_swap_out_frame_egress_gpu[batch_id], m_swap_out_frame_egress_cpu[batch_id], sizeof(Frame *) * MAX_SWAP_FRAME_NUM * batch_size, cudaMemcpyHostToDevice, m_streams[batch_id]);
        memcpy(m_swap_out_frame_egress_gpu_backup[batch_id], m_swap_out_frame_egress_cpu[batch_id], sizeof(Frame *) * MAX_SWAP_FRAME_NUM * batch_size);

        frame_pool->deallocate(recycle_frames.data(), total_used_frame_num);
    }

    void CacheController::Run(int batch_id)
    {
        BackupFrameQueueInfo(batch_id);
        CacheFrame(batch_id);
        cudaStreamSynchronize(m_streams[batch_id]);
        CacheInFrames(batch_id);
        CacheOutFrames(batch_id);
    }

    void CacheController::Run()
    {
    }

    void CacheController::BackupFrameQueueInfo(int batch_id)
    {
        LaunchBackupFrameQueueInfoKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
    }

    void CacheController::CacheFrame(int batch_id)
    {
        LaunchCacheFrameKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
    }

    void CacheController::CopyFrameCacheSpace(int batch_id)
    {
        LaunchCopyFrameCacheSpaceKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
    }

    void CacheController::SetStreams(cudaStream_t *streams, int num)
    {
        m_streams.insert(m_streams.end(), streams, streams + num);
    }

    void CacheController::BuildGraph(int batch_id)
    {
        cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
        LaunchCacheFrameKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
        cudaStreamEndCapture(m_streams[batch_id], &m_graphs[batch_id]);
        cudaGraphInstantiate(&m_graph_execs[batch_id], m_graphs[batch_id], NULL, NULL, 0);
    }

    void CacheController::BuildGraph()
    {
        int batch_num = m_batch_start_index.size();
        for (int i = 0; i < batch_num; i++)
        {
            BuildGraph(i);
        }
    }
}

// #include "packet_cache.h"
// #include "thread_pool.h"
// #include "conf.h"
// #include <algorithm>
// #include <numeric>

// namespace VDES
// {
//     CacheController::CacheController()
//     {
//     }

//     CacheController::~CacheController()
//     {
//     }

//     void CacheController::InitializeKernelParams()
//     {
//         int batch_num = m_batch_start_index.size();
//         m_cache_graphs.resize(batch_num);
//         m_cache_graph_execs.resize(batch_num);
//         m_backup_graphs.resize(batch_num);
//         m_backup_graph_execs.resize(batch_num);

//         for (int i = 0; i < batch_num; i++)
//         {
//             // cudaStream_t stream;
//             // cudaStreamCreate(&stream);
//             // m_streams.push_back(stream);

//             int queue_num = std::accumulate(m_frame_queue_num_per_node.begin() + m_batch_start_index[i], m_frame_queue_num_per_node.begin() + m_batch_end_index[i], 0);
//             m_frame_queue_num_per_batch.push_back(queue_num);

//             dim3 block_dim(KERNEL_BLOCK_WIDTH);
//             dim3 grid_dim((queue_num + block_dim.x - 1) / block_dim.x);
//             m_grid_dims.push_back(grid_dim);
//             m_block_dims.push_back(block_dim);

//             // allocate gpu memory
//             CacheParam cpu_params;
//             cudaMallocAsync(&cpu_params.egress_queues, sizeof(GPUQueue<Frame *> *) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.egress_status, sizeof(FrameQueueStatus *) * queue_num, m_streams[i]);
//             // cudaMallocAsync(&cpu_params.egress_tss, sizeof(GPUQueue<int64_t> *) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.swap_in_frame_num, sizeof(int) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.swap_out_frame_num, sizeof(int) * queue_num, m_streams[i]);
//             cudaMallocAsync(&cpu_params.swap_in_frame_egress, sizeof(Frame **) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
//             cudaMallocAsync(&cpu_params.swap_out_frame_egress, sizeof(Frame **) * queue_num * MAX_SWAP_FRAME_NUM, m_streams[i]);
//             cudaMallocAsync(&cpu_params.swap_in_cache_space, sizeof(VDES::Frame) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
//             cudaMallocAsync(&cpu_params.swap_out_cache_space, sizeof(VDES::Frame) * queue_num * MAX_SWAP_FRAME_NUM, m_streams[i]);

//             /**
//              * TODO: Allocate memory for m_swap_in_frame_num_cpu and m_swap_out_frame_num_cpu, m_swap_in_frame_num_gpu, m_swap_out_frame_num_gpu
//              */
//             m_swap_in_frame_num_cpu.push_back(new int[queue_num]);
//             m_swap_out_frame_num_cpu.push_back(new int[queue_num]);
//             m_swap_in_frame_num_gpu.push_back(cpu_params.swap_in_frame_num);
//             m_swap_out_frame_num_gpu.push_back(cpu_params.swap_out_frame_num);

//             // initialize params
//             int offset = std::accumulate(m_frame_queue_num_per_node.begin(), m_frame_queue_num_per_node.begin() + m_batch_start_index[i], 0);
//             cudaMemcpyAsync(cpu_params.egress_queues, m_egress_queues.data() + offset, sizeof(GPUQueue<Frame *> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemcpyAsync(cpu_params.egress_status, m_egress_status.data() + offset, sizeof(FrameQueueStatus *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             // cudaMemcpyAsync(cpu_params.egress_tss, m_egress_tss.data() + offset, sizeof(GPUQueue<int64_t> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
//             cudaMemsetAsync(cpu_params.swap_in_frame_num, 0, sizeof(int) * queue_num, m_streams[i]);
//             cudaMemsetAsync(cpu_params.swap_out_frame_num, 0, sizeof(int) * queue_num, m_streams[i]);

//             // initialize swap in/out frame egress
//             /**
//              * TODO: INITIALIZE THE SWAP IN/OUT FRAME EGRESS
//              */
//             m_swap_in_frame_egress_cpu.push_back(new Frame *[queue_num * MAX_TRANSMITTED_PACKET_NUM]);
//             m_swap_in_frame_egress_gpu.push_back(cpu_params.swap_in_frame_egress);
//             m_swap_in_frame_egress_gpu_backup.push_back(new Frame *[queue_num * MAX_TRANSMITTED_PACKET_NUM]);
//             m_swap_out_frame_egress_cpu.push_back(new Frame *[queue_num * MAX_SWAP_FRAME_NUM]);
//             m_swap_out_frame_egress_gpu.push_back(cpu_params.swap_out_frame_egress);
//             m_swap_out_frame_egress_gpu_backup.push_back(new Frame *[queue_num * MAX_SWAP_FRAME_NUM]);
//             InitializeSwapFramePtr(i);

//             // initialize swap in/out cache space
//             /**
//              * TODO: INITIALIZE THE SWAP IN/OUT CACHE SPACE
//              */
//             m_swap_in_cache_space_cpu.push_back(new VDES::Frame[queue_num * MAX_TRANSMITTED_PACKET_NUM]);
//             m_swap_out_cache_space_cpu.push_back(new VDES::Frame[queue_num * MAX_SWAP_FRAME_NUM]);
//             m_swap_in_cache_space_gpu.push_back(cpu_params.swap_in_cache_space);
//             m_swap_out_cache_space_gpu.push_back(cpu_params.swap_out_cache_space);

//             cudaMemsetAsync(cpu_params.swap_in_cache_space, 0, sizeof(VDES::Frame) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
//             cudaMemsetAsync(cpu_params.swap_out_cache_space, 0, sizeof(VDES::Frame) * queue_num * MAX_SWAP_FRAME_NUM, m_streams[i]);

//             cpu_params.timeslot_start = m_timeslot_start;
//             cpu_params.timeslot_end = m_timeslot_end;
//             cpu_params.lookahead_timeslot_num = m_lookahead_timeslot_num;
//             cpu_params.queue_num = queue_num;

//             CacheParam *gpu_params;
//             cudaMallocAsync(&gpu_params, sizeof(CacheParam), m_streams[i]);
//             cudaMemcpyAsync(gpu_params, &cpu_params, sizeof(CacheParam), cudaMemcpyHostToDevice, m_streams[i]);

//             m_kernel_params.push_back(gpu_params);
//             cudaStreamSynchronize(m_streams[i]);
//         }
//     }

//     void CacheController::InitializeSwapFramePtr(int batch_id)
//     {
//         int queue_num = m_frame_queue_num_per_batch[batch_id];
//         /**
//          * TODO: check the calculation of the total swap in/out frame num
//          */
//         int total_swap_in_num = queue_num * MAX_TRANSMITTED_PACKET_NUM;
//         int total_swap_out_num = queue_num * MAX_SWAP_FRAME_NUM;

//         // initialize swap in frame egress, gpu ptr
//         /**
//          * TODO: MISMATCH BETWEEN CPU AND GPU POINTERS
//          */
//         auto gpu_frames = frame_pool->allocate(total_swap_in_num);
//         cudaMemcpyAsync(m_swap_in_frame_egress_gpu[batch_id], gpu_frames.data(), sizeof(Frame *) * total_swap_in_num, cudaMemcpyHostToDevice, m_streams[batch_id]);

//         // initialize swap out frame egress, cpu ptr
//         auto cpu_frames = frame_pool_cpu->allocate(total_swap_out_num);
//         cudaMemcpyAsync(m_swap_out_frame_egress_gpu[batch_id], cpu_frames.data(), sizeof(Frame *) * total_swap_out_num, cudaMemcpyHostToDevice, m_streams[batch_id]);

//         // backup frame ptrs
//         memcpy(m_swap_in_frame_egress_cpu[batch_id], gpu_frames.data(), sizeof(Frame *) * total_swap_in_num);
//         memcpy(m_swap_in_frame_egress_gpu_backup[batch_id], gpu_frames.data(), sizeof(Frame *) * total_swap_in_num);
//         memcpy(m_swap_out_frame_egress_cpu[batch_id], cpu_frames.data(), sizeof(Frame *) * total_swap_out_num);
//         memcpy(m_swap_out_frame_egress_gpu_backup[batch_id], cpu_frames.data(), sizeof(Frame *) * total_swap_out_num);
//     }

//     void CacheController::SetBatches(int *start_index, int *end_index, int batch_num)
//     {
//         m_batch_start_index.insert(m_batch_start_index.end(), start_index, start_index + batch_num);
//         m_batch_end_index.insert(m_batch_end_index.end(), end_index, end_index + batch_num);
//     }

//     void CacheController::SetEgressProperties(GPUQueue<Frame *> **egress_queues, FrameQueueStatus **egress_status, int *frame_queue_num_per_node, int node_num)
//     {
//         int queue_num = std::accumulate(frame_queue_num_per_node, frame_queue_num_per_node + node_num, 0);
//         m_egress_queues.insert(m_egress_queues.end(), egress_queues, egress_queues + queue_num);
//         // m_egress_tss.insert(m_egress_tss.end(), egress_tss, egress_tss + queue_num);
//         m_egress_status.insert(m_egress_status.end(), egress_status, egress_status + queue_num);
//         m_frame_queue_num_per_node.insert(m_frame_queue_num_per_node.end(), frame_queue_num_per_node, frame_queue_num_per_node + node_num);
//     }

//     void CacheController::SetLookaheadTimeslotNum(int lookahead_timeslot_num)
//     {
//         m_lookahead_timeslot_num = lookahead_timeslot_num;
//     }

//     void CacheController::SetTimeSlot(int64_t *timeslot_start, int64_t *timeslot_end)
//     {
//         m_timeslot_start = timeslot_start;
//         m_timeslot_end = timeslot_end;
//     }

//     void CacheController::CacheInFrames(int batch_id)
//     {
//         int batch_size = m_frame_queue_num_per_batch[batch_id];
//         int batch_num = m_batch_start_index.size();
//         int queue_num = std::accumulate(m_frame_queue_num_per_node.begin() + m_batch_start_index[batch_id], m_frame_queue_num_per_node.begin() + m_batch_end_index[batch_id], 0);

//         cudaMemcpyAsync(m_swap_in_frame_num_cpu[batch_id], m_swap_in_frame_num_gpu[batch_id], sizeof(int) * queue_num, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_swap_in_frame_egress_cpu[batch_id], m_swap_in_frame_egress_gpu[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyDeviceToHost, m_streams[batch_id]);

//         cudaStreamSynchronize(m_streams[batch_id]);

//         cudaMemcpyAsync(m_swap_in_frame_egress_gpu[batch_id], m_swap_in_frame_egress_gpu_backup[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyHostToDevice, m_streams[batch_id]);

//         // copy frame to cache space
//         int *swap_in_num = m_swap_in_frame_num_cpu[batch_id];
//         int total_used_frame_num = std::accumulate(swap_in_num, swap_in_num + batch_size, 0);
//         auto alloc_frames = frame_pool->allocate(total_used_frame_num);
//         std::vector<VDES::Frame *> recycle_frames;
//         int frame_index = 0;
//         /**
//          * TODO: Use queue_num instead of batch_size
//          */
//         for (int i = 0; i < queue_num; i++)
//         {
//             int offset = i * MAX_TRANSMITTED_PACKET_NUM;
//             VDES::Frame *dst = m_swap_in_cache_space_cpu[batch_id] + offset;
//             VDES::Frame **src = m_swap_in_frame_egress_cpu[batch_id] + offset;

//             for (int j = 0; j < swap_in_num[i]; j++)
//             {
//                 // copy frame to cache space
//                 VDES::Frame tmp_frame;
//                 memcpy(dst + j, src[j], sizeof(VDES::Frame));
//                 // recycle cpu frames
//                 recycle_frames.push_back(src[j]);
//                 // reset comsumed frames using gpu ptr
//                 src[j] = alloc_frames[frame_index++];
//             }
//         }

//         cudaStreamSynchronize(m_streams[batch_id]);
//         cudaMemcpyAsync(m_swap_in_cache_space_gpu[batch_id], m_swap_in_cache_space_cpu[batch_id], sizeof(VDES::Frame) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyHostToDevice, m_streams[batch_id]);
//         LaunchCopyFrameCacheSpaceKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);

//         cudaMemcpyAsync(m_swap_in_frame_egress_gpu[batch_id], m_swap_in_frame_egress_cpu[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyHostToDevice, m_streams[batch_id]);
//         // todo：backup frame ptrs
//         memcpy(m_swap_in_frame_egress_gpu_backup[batch_id], m_swap_in_frame_egress_cpu[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM);

//         frame_pool_cpu->deallocate(recycle_frames.data(), total_used_frame_num);
//     }

//     void CacheController::CacheOutFrames(int batch_id)
//     {
//         int batch_size = m_frame_queue_num_per_batch[batch_id];

//         cudaMemcpyAsync(m_swap_out_cache_space_cpu[batch_id], m_swap_out_cache_space_gpu[batch_id], sizeof(VDES::Frame) * MAX_SWAP_FRAME_NUM * batch_size, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_swap_out_frame_egress_cpu[batch_id], m_swap_out_frame_egress_gpu[batch_id], sizeof(Frame *) * MAX_SWAP_FRAME_NUM * batch_size, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaMemcpyAsync(m_swap_out_frame_num_cpu[batch_id], m_swap_out_frame_num_gpu[batch_id], sizeof(int) * batch_size, cudaMemcpyDeviceToHost, m_streams[batch_id]);
//         cudaStreamSynchronize(m_streams[batch_id]);

//         // copy cache space to frame
//         int *swap_out_num = m_swap_out_frame_num_cpu[batch_id];
//         int total_used_frame_num = std::accumulate(swap_out_num, swap_out_num + batch_size, 0);
//         auto alloc_frames = frame_pool_cpu->allocate(total_used_frame_num);
//         std::vector<VDES::Frame *> recycle_frames;
//         int frame_index = 0;

//         for (int i = 0; i < batch_size; i++)
//         {
//             int offset = i * MAX_SWAP_FRAME_NUM;
//             VDES::Frame *src = m_swap_out_cache_space_cpu[batch_id] + offset;
//             VDES::Frame **dst = m_swap_out_frame_egress_gpu_backup[batch_id] + offset;
//             VDES::Frame **gpu_frames = m_swap_out_frame_egress_cpu[batch_id] + offset;

//             for (int j = 0; j < swap_out_num[i]; j++)
//             {
//                 // copy cache space to frame
//                 memcpy(dst[j], src + j, sizeof(VDES::Frame));
//                 // recycle gpu frames
//                 recycle_frames.push_back(gpu_frames[j]);
//                 // reset comsumed frames using cpu ptr
//                 gpu_frames[j] = alloc_frames[frame_index++];
//             }
//         }
//         // reset used frame ptr
//         cudaMemcpyAsync(m_swap_out_frame_egress_gpu[batch_id], m_swap_out_frame_egress_cpu[batch_id], sizeof(Frame *) * MAX_SWAP_FRAME_NUM * batch_size, cudaMemcpyHostToDevice, m_streams[batch_id]);
//         memcpy(m_swap_out_frame_egress_gpu_backup[batch_id], m_swap_out_frame_egress_cpu[batch_id], sizeof(Frame *) * MAX_SWAP_FRAME_NUM * batch_size);

//         frame_pool->deallocate(recycle_frames.data(), total_used_frame_num);
//     }

//     void CacheController::Run(int batch_id)
//     {
//         /**
//          * TODO: Launch another kernel
//          */
//         BackupFrameQueueInfo(batch_id);
//         CacheFrame(batch_id);
//         cudaStreamSynchronize(m_streams[batch_id]);
//         CacheInFrames(batch_id);
//         CacheOutFrames(batch_id);
//     }

//     void CacheController::Run()
//     {
//     }

//     void CacheController::BackupFrameQueueInfo(int batch_id)
//     {
//         LaunchBackupFrameQueueInfoKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
//     }

//     void CacheController::CacheFrame(int batch_id)
//     {
//         LaunchCacheFrameKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
//     }

//     void CacheController::CopyFrameCacheSpace(int batch_id)
//     {
//         LaunchCopyFrameCacheSpaceKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
//     }

//     void CacheController::SetStreams(cudaStream_t *streams, int num)
//     {
//         m_streams.insert(m_streams.end(), streams, streams + num);
//     }

//     void CacheController::BuildGraph(int batch_id)
//     {
//         cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
//         LaunchCacheFrameKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_cache_graphs[batch_id]);
//         cudaGraphInstantiate(&m_cache_graph_execs[batch_id], m_cache_graphs[batch_id], NULL, NULL, 0);

//         cudaStreamBeginCapture(m_streams[batch_id], cudaStreamCaptureModeGlobal);
//         LaunchBackupFrameQueueInfoKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
//         cudaStreamEndCapture(m_streams[batch_id], &m_backup_graphs[batch_id]);
//         cudaGraphInstantiate(&m_backup_graph_execs[batch_id], m_backup_graphs[batch_id], NULL, NULL, 0);
//     }

//     void CacheController::BuildGraph()
//     {
//         int batch_num = m_batch_start_index.size();
//         for (int i = 0; i < batch_num; i++)
//         {
//             BuildGraph(i);
//         }
//     }

//     cudaGraph_t CacheController::GetCacheGraph(int batch_id)
//     {
//         return m_cache_graphs[batch_id];
//     }

//     cudaGraph_t CacheController::GetBackupGraph(int batch_id)
//     {
//         return m_backup_graphs[batch_id];
//     }
// }

// // #include "packet_cache.h"
// // #include "thread_pool.h"
// // #include "conf.h"
// // #include <algorithm>
// // #include <numeric>

// // namespace VDES
// // {
// //     CacheController::CacheController()
// //     {
// //     }

// //     CacheController::~CacheController()
// //     {
// //     }

// //     void CacheController::InitializeKernelParams()
// //     {
// //         int batch_num = m_batch_start_index.size();
// //         for (int i = 0; i < batch_num; i++)
// //         {
// //             cudaStream_t stream;
// //             cudaStreamCreate(&stream);
// //             m_streams.push_back(stream);

// //             int queue_num = std::accumulate(m_frame_queue_num_per_node.begin() + m_batch_start_index[i], m_frame_queue_num_per_node.begin() + m_batch_end_index[i], 0);
// //             m_frame_queue_num_per_batch.push_back(queue_num);

// //             dim3 block_dim(KERNEL_BLOCK_WIDTH);
// //             dim3 grid_dim((queue_num + block_dim.x - 1) / block_dim.x);
// //             m_grid_dims.push_back(grid_dim);
// //             m_block_dims.push_back(block_dim);

// //             // allocate gpu memory
// //             CacheParam cpu_params;
// //             cudaMallocAsync(&cpu_params.egress_queues, sizeof(GPUQueue<Frame *>) * queue_num, m_streams[i]);
// //             cudaMallocAsync(&cpu_params.egress_status, sizeof(FrameQueueStatus *) * queue_num, m_streams[i]);
// //             cudaMallocAsync(&cpu_params.egress_tss, sizeof(GPUQueue<int64_t> *) * queue_num, m_streams[i]);
// //             cudaMallocAsync(&cpu_params.swap_in_frame_num, sizeof(int) * queue_num, m_streams[i]);
// //             cudaMallocAsync(&cpu_params.swap_out_frame_num, sizeof(int) * queue_num, m_streams[i]);
// //             cudaMallocAsync(&cpu_params.swap_in_frame_egress, sizeof(Frame **) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
// //             cudaMallocAsync(&cpu_params.swap_out_frame_egress, sizeof(Frame **) * queue_num * MAX_SWAP_FRAME_NUM, m_streams[i]);
// //             cudaMallocAsync(&cpu_params.swap_in_cache_space, sizeof(VDES::Frame) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
// //             cudaMallocAsync(&cpu_params.swap_out_cache_space, sizeof(VDES::Frame) * queue_num * MAX_SWAP_FRAME_NUM, m_streams[i]);

// //             // initialize params
// //             int offset = std::accumulate(m_frame_queue_num_per_node.begin(), m_frame_queue_num_per_node.begin() + m_batch_start_index[i], 0);
// //             cudaMemcpyAsync(cpu_params.egress_queues, m_egress_queues.data() + offset, sizeof(GPUQueue<Frame *>) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
// //             cudaMemcpyAsync(cpu_params.egress_status, m_egress_status.data() + offset, sizeof(FrameQueueStatus *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
// //             cudaMemcpyAsync(cpu_params.egress_tss, m_egress_tss.data() + offset, sizeof(GPUQueue<int64_t> *) * queue_num, cudaMemcpyHostToDevice, m_streams[i]);
// //             cudaMemsetAsync(cpu_params.swap_in_frame_num, 0, sizeof(int) * queue_num, m_streams[i]);
// //             cudaMemsetAsync(cpu_params.swap_out_frame_num, 0, sizeof(int) * queue_num, m_streams[i]);

// //             // initialize swap in/out frame egress
// //             m_swap_in_frame_egress_cpu.push_back(new Frame *[queue_num * MAX_TRANSMITTED_PACKET_NUM]);
// //             m_swap_in_frame_egress_gpu_backup.push_back(new Frame *[queue_num * MAX_TRANSMITTED_PACKET_NUM]);
// //             m_swap_out_frame_egress_cpu.push_back(new Frame *[queue_num * MAX_SWAP_FRAME_NUM]);
// //             m_swap_out_frame_egress_gpu_backup.push_back(new Frame *[queue_num * MAX_SWAP_FRAME_NUM]);
// //             InitializeSwapFramePtr(i);

// //             cudaMemsetAsync(cpu_params.swap_in_cache_space, 0, sizeof(VDES::Frame) * queue_num * MAX_TRANSMITTED_PACKET_NUM, m_streams[i]);
// //             cudaMemsetAsync(cpu_params.swap_out_cache_space, 0, sizeof(VDES::Frame) * queue_num * MAX_SWAP_FRAME_NUM, m_streams[i]);

// //             cpu_params.timeslot_start = m_timeslot_start;
// //             cpu_params.timeslot_end = m_timeslot_end;
// //             cpu_params.lookahead_timeslot_num = m_lookahead_timeslot_num;
// //             cpu_params.queue_num = queue_num;

// //             CacheParam *gpu_params;
// //             cudaMallocAsync(&gpu_params, sizeof(CacheParam), m_streams[i]);
// //             cudaMemcpyAsync(gpu_params, &cpu_params, sizeof(CacheParam), cudaMemcpyHostToDevice, m_streams[i]);

// //             m_kernel_params.push_back(gpu_params);
// //             cudaStreamSynchronize(m_streams[i]);
// //         }
// //     }

// //     void CacheController::InitializeSwapFramePtr(int batch_id)
// //     {
// //         int queue_num = m_frame_queue_num_per_batch[batch_id];
// //         int total_swap_in_num = total_swap_in_num * MAX_TRANSMITTED_PACKET_NUM;
// //         int total_swap_out_num = total_swap_out_num * MAX_SWAP_FRAME_NUM;

// //         // initialize swap in frame egress, gpu ptr
// //         auto gpu_frames = frame_pool->allocate(total_swap_in_num);
// //         cudaMemcpyAsync(m_swap_in_frame_egress_gpu[batch_id], gpu_frames.data(), sizeof(Frame *) * total_swap_in_num, cudaMemcpyHostToDevice, m_streams[batch_id]);

// //         // initialize swap out frame egress, cpu ptr
// //         auto cpu_frames = frame_pool_cpu->allocate(total_swap_out_num);
// //         cudaMemcpyAsync(m_swap_out_frame_egress_cpu[batch_id], cpu_frames.data(), sizeof(Frame *) * total_swap_out_num, cudaMemcpyHostToDevice, m_streams[batch_id]);

// //         // backup frame ptrs
// //         memcpy(m_swap_in_frame_egress_cpu[batch_id], gpu_frames.data(), sizeof(Frame *) * total_swap_in_num);
// //         memcpy(m_swap_in_frame_egress_gpu_backup[batch_id], gpu_frames.data(), sizeof(Frame *) * total_swap_in_num);
// //         memcpy(m_swap_out_frame_egress_cpu[batch_id], cpu_frames.data(), sizeof(Frame *) * total_swap_out_num);
// //         memcpy(m_swap_out_frame_egress_gpu_backup[batch_id], cpu_frames.data(), sizeof(Frame *) * total_swap_out_num);
// //     }

// //     void CacheController::SetBatches(int *start_index, int *end_index, int batch_num)
// //     {
// //         m_batch_start_index.insert(m_batch_start_index.end(), start_index, start_index + batch_num);
// //         m_batch_end_index.insert(m_batch_end_index.end(), end_index, end_index + batch_num);
// //     }

// //     void CacheController::SetEgressProperties(GPUQueue<Frame *> **egress_queues, GPUQueue<int64_t> **egress_tss, FrameQueueStatus *egress_status, int *frame_queue_num_per_node, int node_num)
// //     {
// //         int queue_num = std::accumulate(frame_queue_num_per_node, frame_queue_num_per_node + node_num, 0);
// //         m_egress_queues.insert(m_egress_queues.end(), egress_queues, egress_queues + queue_num);
// //         m_egress_tss.insert(m_egress_tss.end(), egress_tss, egress_tss + queue_num);
// //         m_egress_status.insert(m_egress_status.end(), egress_status, egress_status + queue_num);
// //         m_frame_queue_num_per_node.insert(m_frame_queue_num_per_node.end(), frame_queue_num_per_node, frame_queue_num_per_node + node_num);
// //     }

// //     void CacheController::SetLookaheadTimeslotNum(int lookahead_timeslot_num)
// //     {
// //         m_lookahead_timeslot_num = lookahead_timeslot_num;
// //     }

// //     void CacheController::SetTimeSlot(int64_t *timeslot_start, int64_t *timeslot_end)
// //     {
// //         m_timeslot_start = timeslot_start;
// //         m_timeslot_end = timeslot_end;
// //     }

// //     void CacheController::CacheInFrames(int batch_id)
// //     {
// //         int batch_size = m_frame_queue_num_per_batch[batch_id];
// //         int batch_num = m_batch_start_index.size();
// //         cudaMemcpyAsync(m_swap_in_frame_num_cpu[batch_id], m_swap_in_frame_num_gpu[batch_id], sizeof(int) * batch_size, cudaMemcpyDeviceToHost, m_streams[batch_id]);
// //         cudaMemcpyAsync(m_swap_in_frame_egress_cpu[batch_id], m_swap_in_frame_egress_gpu[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyDeviceToHost, m_streams[batch_id]);

// //         cudaStreamSynchronize(m_streams[batch_id]);

// //         cudaMemcpyAsync(m_swap_in_frame_egress_gpu[batch_id], m_swap_in_frame_egress_gpu_backup[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyHostToDevice, m_streams[batch_id]);

// //         // copy frame to cache space
// //         int *swap_in_num = m_swap_in_frame_num_cpu[batch_id];
// //         int total_used_frame_num = std::accumulate(swap_in_num, swap_in_num + batch_size, 0);
// //         auto alloc_frames = frame_pool->allocate(total_used_frame_num);
// //         std::vector<VDES::Frame *> recycle_frames;
// //         int frame_index = 0;
// //         for (int i = 0; i < batch_size; i++)
// //         {
// //             int offset = i * MAX_TRANSMITTED_PACKET_NUM;
// //             VDES::Frame *dst = m_swap_in_cache_space_cpu[batch_id] + offset;
// //             VDES::Frame **src = m_swap_in_frame_egress_cpu[batch_id] + offset;

// //             for (int j = 0; j < swap_in_num[i]; j++)
// //             {
// //                 // copy frame to cache space
// //                 memcpy(dst + j, src[j], sizeof(VDES::Frame));
// //                 // recycle cpu frames
// //                 recycle_frames.push_back(src[j]);
// //                 // reset comsumed frames using gpu ptr
// //                 src[j] = alloc_frames[frame_index++];
// //             }
// //         }

// //         cudaStreamSynchronize(m_streams[batch_id]);
// //         cudaMemcpyAsync(m_swap_in_cache_space_gpu[batch_id], m_swap_in_cache_space_cpu[batch_id], sizeof(VDES::Frame) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyHostToDevice, m_streams[batch_id]);
// //         LaunchCopyFrameCacheSpaceKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
// //         cudaMemcpyAsync(m_swap_in_frame_egress_gpu[batch_id], m_swap_in_frame_egress_cpu[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM, cudaMemcpyHostToDevice, m_streams[batch_id]);
// //         // todo：backup frame ptrs
// //         memcpy(m_swap_in_frame_egress_gpu_backup[batch_id], m_swap_in_frame_egress_cpu[batch_id], sizeof(Frame *) * batch_size * MAX_TRANSMITTED_PACKET_NUM);

// //         frame_pool_cpu->deallocate(recycle_frames.data(), total_used_frame_num);
// //     }

// //     void CacheController::CacheOutFrames(int batch_id)
// //     {
// //         int batch_size = m_frame_queue_num_per_batch[batch_id];

// //         cudaMemcpyAsync(m_swap_out_cache_space_cpu[batch_id], m_swap_out_cache_space_gpu[batch_id], sizeof(VDES::Frame) * MAX_SWAP_FRAME_NUM * batch_size, cudaMemcpyDeviceToHost, m_streams[batch_id]);
// //         cudaMemcpyAsync(m_swap_out_frame_egress_cpu[batch_id], m_swap_out_frame_egress_gpu[batch_id], sizeof(Frame *) * MAX_SWAP_FRAME_NUM * batch_size, cudaMemcpyDeviceToHost, m_streams[batch_id]);
// //         cudaMemcpyAsync(m_swap_out_frame_num_cpu[batch_id], m_swap_out_frame_num_gpu[batch_id], sizeof(int) * batch_size, cudaMemcpyDeviceToHost, m_streams[batch_id]);
// //         cudaStreamSynchronize(m_streams[batch_id]);

// //         // copy cache space to frame
// //         int *swap_out_num = m_swap_out_frame_num_cpu[batch_id];
// //         int total_used_frame_num = std::accumulate(swap_out_num, swap_out_num + batch_size, 0);
// //         auto alloc_frames = frame_pool_cpu->allocate(total_used_frame_num);
// //         std::vector<VDES::Frame *> recycle_frames;
// //         int frame_index = 0;
// //         for (int i = 0; i < batch_size; i++)
// //         {
// //             int offset = i * MAX_SWAP_FRAME_NUM;
// //             VDES::Frame *src = m_swap_out_cache_space_cpu[batch_id] + offset;
// //             VDES::Frame **dst = m_swap_out_frame_egress_gpu_backup[batch_id] + offset;
// //             VDES::Frame **gpu_frames = m_swap_out_frame_egress_cpu[batch_id] + offset;

// //             for (int j = 0; j < swap_out_num[i]; j++)
// //             {
// //                 // copy cache space to frame
// //                 memcpy(dst[j], src + j, sizeof(VDES::Frame));
// //                 // recycle gpu frames
// //                 recycle_frames.push_back(gpu_frames[j]);
// //                 // reset comsumed frames using cpu ptr
// //                 gpu_frames[j] = alloc_frames[frame_index++];
// //             }
// //         }

// //         // reset used frame ptr
// //         cudaMemcpyAsync(m_swap_out_frame_egress_gpu[batch_id], m_swap_out_frame_egress_cpu[batch_id], sizeof(Frame *) * MAX_SWAP_FRAME_NUM * batch_size, cudaMemcpyHostToDevice, m_streams[batch_id]);
// //         memcpy(m_swap_out_frame_egress_gpu_backup[batch_id], m_swap_out_frame_egress_cpu[batch_id], sizeof(Frame *) * MAX_SWAP_FRAME_NUM * batch_size);
// //     }

// //     void CacheController::Run(int batch_id)
// //     {
// //         LaunchCopyFrameCacheSpaceKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
// //         cudaStreamSynchronize(m_streams[batch_id]);
// //         CacheInFrames(batch_id);
// //         LaunchCopyFrameCacheSpaceKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
// //         CacheOutFrames(batch_id);
// //     }

// //     void CacheController::Run()
// //     {
// //     }

// //     void CacheController::BackupFrameQueueInfo(int batch_id)
// //     {
// //         LaunchBackupFrameQueueInfoKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
// //     }

// //     void CacheController::CacheFrame(int batch_id)
// //     {
// //         LaunchCacheFrameKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
// //     }

// //     void CacheController::CopyFrameCacheSpace(int batch_id)
// //     {
// //         LaunchCopyFrameCacheSpaceKernel(m_grid_dims[batch_id], m_block_dims[batch_id], m_kernel_params[batch_id], m_streams[batch_id]);
// //     }

// // }