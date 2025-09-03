#ifndef PACHET_CACHE_SIZE_H
#define PACHET_CACHE_SIZE_H

#include <vector>
#include "gpu_queue.cuh"
#include "packet_definition.h"
#include <map>
#include <cstring>

namespace VDES
{

    typedef struct
    {
        // cache window, [0, start] cache in, [start, end] do nothing, [end, size] cache out
        uint32_t cache_win_start;
        // uint16_t cache_middle;
        uint32_t cache_win_end;
        // the packet status in cache window, 1 means in gpu, 0 means in cpu
        uint8_t *packet_status_in_cache_win;

        // lastly coming packet offset
        int last_cache_out_offset;

        // transmitted frame in last timeslot
        uint32_t transmitted_frame_num;

        // ingress info
        int head;
        int size;

    } FrameQueueStatus;

    typedef struct
    {
        GPUQueue<Frame *> **egress_queues;
        FrameQueueStatus **egress_status;

        // cache properties
        int *swap_in_frame_num;
        int *swap_out_frame_num;

        // frame ptr
        // gpu ptr
        Frame **swap_in_frame_egress;
        // cpu ptr
        Frame **swap_out_frame_egress;

        // cache space, frame_num* MAX_Transmitted_frame_num
        Frame *swap_in_cache_space;
        Frame *swap_out_cache_space;

        int64_t *timeslot_start;
        int64_t *timeslot_end;

        int64_t lookahead_timeslot_num;

        int queue_num;

    } CacheParam;

    typedef struct
    {
        int *swap_in_frame_num;
    } G2GCopyParam;

    // only cache frames
    class CacheController
    {
    private:
        // kernel parameters
        std::vector<CacheParam *> m_kernel_params;
        std::vector<cudaStream_t> m_streams;
        std::vector<cudaGraph_t> m_graphs;
        std::vector<cudaGraphExec_t> m_graph_execs;

        // std::vector<cudaStream_t>
        std::vector<dim3> m_grid_dims;
        std::vector<dim3> m_block_dims;

        // ingress and egress are considered from the perspective of nodes
        std::vector<GPUQueue<Frame *> *> m_egress_queues;
        std::vector<FrameQueueStatus *> m_egress_status;
        // std::vector<GPUQueue<int64_t> *> m_egress_tss;

        // cache properties
        std::vector<int *> m_swap_in_frame_num_cpu;
        std::vector<int *> m_swap_out_frame_num_cpu;
        std::vector<int *> m_swap_in_frame_num_gpu;
        std::vector<int *> m_swap_out_frame_num_gpu;

        std::vector<Frame **> m_swap_in_frame_egress_cpu;
        std::vector<Frame **> m_swap_out_frame_egress_cpu;
        std::vector<Frame **> m_swap_in_frame_egress_gpu;
        std::vector<Frame **> m_swap_out_frame_egress_gpu;

        // gpu ptr in cpu memory
        std::vector<Frame **> m_swap_in_frame_egress_gpu_backup;
        // cpu ptr in cpu memory
        std::vector<Frame **> m_swap_out_frame_egress_gpu_backup;

        // cache space,
        std::vector<Frame *> m_swap_in_cache_space_cpu;
        std::vector<Frame *> m_swap_out_cache_space_cpu;
        std::vector<Frame *> m_swap_in_cache_space_gpu;
        std::vector<Frame *> m_swap_out_cache_space_gpu;

        // the lookahead time window for cache
        int m_lookahead_timeslot_num;

        // batch properties, node-level dividing
        std::vector<int> m_batch_start_index;
        std::vector<int> m_batch_end_index;
        std::vector<int> m_frame_queue_num_per_batch;
        std::vector<int> m_frame_queue_num_per_node;

        // time slot, storing on GPU
        int64_t *m_timeslot_start;
        int64_t *m_timeslot_end;

        // cache batch size, the number of frame queues
        int m_cache_batch_size;

    public:
        CacheController();
        ~CacheController();

        // initialize kernel parameters
        void InitializeKernelParams();
        void InitializeSwapFramePtr(int batch_id);

        void SetBatches(int *start_index, int *end_index, int batch_num);

        // initialize node properties
        void SetEgressProperties(GPUQueue<Frame *> **egress_queues, FrameQueueStatus **egress_status, int *frame_queue_num_per_node, int node_num);
        void SetStreams(cudaStream_t *streams, int stream_num);

        // cache frames
        void CacheInFrames(int batch_id);
        void CacheOutFrames(int batch_id);

        // Run Cache Strageties
        void Run(int batch_id);
        void Run();

        void SetLookaheadTimeslotNum(int lookahead_timeslot_num);
        void SetTimeSlot(int64_t *timeslot_start, int64_t *timeslot_end);

        // kernel launchers
        void BackupFrameQueueInfo(int batch_id);
        void CacheFrame(int batch_id);
        void CopyFrameCacheSpace(int batch_id);
        void UpdateFrameQueue(int batch_id);

        void BuildGraph(int batch_id);
        void BuildGraph();
    };

    void LaunchCacheFrameKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream);
    void LaunchCopyFrameCacheSpaceKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream);
    void LaunchBackupFrameQueueInfoKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream);

} // namespace VDES

#endif

// #ifndef PACHET_CACHE_SIZE_H
// #define PACHET_CACHE_SIZE_H

// #include <vector>
// #include "gpu_queue.cuh"
// #include "packet_definition.h"
// #include <map>
// #include <cstring>

// namespace VDES
// {

//     typedef struct
//     {
//         // uint8_t device_type;
//         // uint8_t next_hop_device_type;

//         // cache window, [0, start] cache in, [start, end] do nothing, [end, size] cache out
//         uint16_t cache_win_start;
//         // uint16_t cache_middle;
//         uint16_t cache_win_end;
//         // the packet status in cache window, 0 means not in gpu, 1 means in cpu
//         uint8_t *packet_status_in_cache_win;

//         // lastly coming packet offset
//         int last_cache_out_offset;

//         // transmitted frame in last timeslot
//         uint16_t transmitted_frame_num;

//         // ingress info
//         int head;
//         int size;

//     } FrameQueueStatus;

//     typedef struct
//     {
//         GPUQueue<Frame *> **egress_queues;
//         FrameQueueStatus **egress_status;
//         GPUQueue<int64_t> **egress_tss;

//         // cache properties
//         int *swap_in_frame_num;
//         int *swap_out_frame_num;

//         // frame ptr
//         // gpu ptr
//         Frame **swap_in_frame_egress;
//         // cpu ptr
//         Frame **swap_out_frame_egress;

//         // cache space, frame_num* MAX_Transmitted_frame_num
//         Frame *swap_in_cache_space;
//         Frame *swap_out_cache_space;

//         int64_t *timeslot_start;
//         int64_t *timeslot_end;

//         int64_t lookahead_timeslot_num;

//         int queue_num;

//     } CacheParam;

//     typedef struct
//     {
//         int *swap_in_frame_num;
//     } G2GCopyParam;

//     // only cache frames
//     class CacheController
//     {
//     private:
//         // kernel parameters
//         std::vector<CacheParam *> m_kernel_params;
//         std::vector<cudaStream_t> m_streams;

//         // std::vector<cudaStream_t>
//         std::vector<dim3> m_grid_dims;
//         std::vector<dim3> m_block_dims;

//         // ingress and egress are considered from the perspective of nodes
//         std::vector<GPUQueue<Frame *> *> m_egress_queues;
//         std::vector<FrameQueueStatus> m_egress_status;
//         std::vector<GPUQueue<int64_t> *> m_egress_tss;

//         // cache properties
//         std::vector<int *> m_swap_in_frame_num_cpu;
//         std::vector<int *> m_swap_out_frame_num_cpu;
//         std::vector<int *> m_swap_in_frame_num_gpu;
//         std::vector<int *> m_swap_out_frame_num_gpu;

//         std::vector<Frame **> m_swap_in_frame_egress_cpu;
//         std::vector<Frame **> m_swap_out_frame_egress_cpu;
//         std::vector<Frame **> m_swap_in_frame_egress_gpu;
//         std::vector<Frame **> m_swap_out_frame_egress_gpu;

//         // gpu ptr in cpu memory
//         std::vector<Frame **> m_swap_in_frame_egress_gpu_backup;
//         // cpu ptr in cpu memory
//         std::vector<Frame **> m_swap_out_frame_egress_gpu_backup;

//         // cache space,
//         std::vector<Frame *> m_swap_in_cache_space_cpu;
//         std::vector<Frame *> m_swap_out_cache_space_cpu;
//         std::vector<Frame *> m_swap_in_cache_space_gpu;
//         std::vector<Frame *> m_swap_out_cache_space_gpu;

//         // the lookahead time window for cache
//         int m_lookahead_timeslot_num;

//         // batch properties, node-level dividing
//         std::vector<int> m_batch_start_index;
//         std::vector<int> m_batch_end_index;
//         std::vector<int> m_frame_queue_num_per_batch;
//         std::vector<int> m_frame_queue_num_per_node;

//         // time slot, storing on GPU
//         int64_t *m_timeslot_start;
//         int64_t *m_timeslot_end;

//         // cache batch size, the number of frame queues
//         int m_cache_batch_size;

//     public:
//         CacheController();
//         ~CacheController();

//         // initialize kernel parameters
//         void InitializeKernelParams();
//         void InitializeSwapFramePtr(int batch_id);

//         void SetBatches(int *start_index, int *end_index, int batch_num);

//         // initialize node properties
//         void SetEgressProperties(GPUQueue<Frame *> **egress_queues, GPUQueue<int64_t> **egress_tss, FrameQueueStatus *egress_status, int *frame_queue_num_per_node, int node_num);

//         // cache frames
//         void CacheInFrames(int batch_id);
//         void CacheOutFrames(int batch_id);

//         // Run Cache Strageties
//         void Run(int batch_id);
//         void Run();

//         void SetLookaheadTimeslotNum(int lookahead_timeslot_num);
//         void SetTimeSlot(int64_t *timeslot_start, int64_t *timeslot_end);

//         // kernel launchers
//         void BackupFrameQueueInfo(int batch_id);
//         void CacheFrame(int batch_id);
//         void CopyFrameCacheSpace(int batch_id);
//     };

//     void LaunchCacheFrameKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream);
//     void LaunchCopyFrameCacheSpaceKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream);
//     void LaunchBackupFrameQueueInfoKernel(dim3 grid_dim, dim3 block_dim, CacheParam *kernel_param, cudaStream_t stream);

// } // namespace VDES

// #endif