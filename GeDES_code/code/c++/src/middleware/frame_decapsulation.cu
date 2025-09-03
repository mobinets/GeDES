#include "frame_decapsulation.h"

#define ECN_OFFSET 1

__global__ void decapsulate_frame(VDES::FrameDecapsulationParams *param)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < param->node_num)
    {
        // classification
        int frame_queue_num = param->frame_queue_num_per_node[tid];
        int frame_queue_offset = param->frame_queue_offset_per_node[tid];
        GPUQueue<VDES::Frame *> **frame_queues = param->frame_ingresses + frame_queue_offset;
        int *recycle_frame_num = param->recycle_num_per_frame_queue + frame_queue_offset;
        GPUQueue<uint8_t *> **next_hop = param->packet_ingresses;
        int *l3_timestamp_offset = param->l3_timestamp_offset;

        memset(recycle_frame_num, 0, frame_queue_num * sizeof(int));
        memset(param->recycle_frames + tid * MAX_TRANSMITTED_PACKET_NUM, 0, MAX_TRANSMITTED_PACKET_NUM);

        // decapsulation
        for (int i = 0; i < frame_queue_num; i++)
        {
            int size = frame_queues[i]->size;
            int queue_id = frame_queue_offset + i;
            VDES::Frame **recycle_frames = param->recycle_frames + queue_id * MAX_TRANSMITTED_PACKET_NUM;

            // printf("tid:%d, recived frame num: %d\n", tid, size);

#if ENABLE_CACHE

            for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
            {
                // reset the number of swapped packets
                int index = j * param->queue_num + queue_id;

                param->swap_packet_num[index] = 0;
            }
#endif

            for (int j = 0; j < size; j++)
            {
                VDES::Frame *frame = frame_queues[i]->get_element(j);
                // uint16_t protocol_type;
                uint8_t protocol_type = frame->type[1];
                int next_hop_index = protocol_type * param->node_num + tid;
                uint8_t *payload;
                memcpy(&payload, frame->data, sizeof(uint8_t *));

#if ENABLE_CACHE
                if (frame->device == uint8_t(VDES::Device::CPU))
                {
                    // swap packets for cache
                    int *swap_id = param->swap_packet_num + protocol_type * param->queue_num + queue_id;
                    uint8_t *temp = payload;
                    payload = param->swap_in_packet_buffer[(protocol_type * param->queue_num + queue_id) * MAX_TRANSMITTED_PACKET_NUM + *swap_id];
                    param->swap_in_packet_buffer[(protocol_type * param->queue_num + queue_id) * MAX_TRANSMITTED_PACKET_NUM + *swap_id] = temp;
                    (*swap_id)++;
                }
#endif

                memcpy(payload + l3_timestamp_offset[protocol_type], frame->timestamp, 8);
                // int64_t timestamp;
                // memcpy(&timestamp, frame->timestamp, 8);
                // printf("%ld\n", timestamp);
#if ENABLE_DCTCP
                memcpy(payload + ECN_OFFSET, &frame->fcs[0], 1);
#endif

                // submit payload to up layer
                next_hop[next_hop_index]->append_element(payload);

                // recycle frame
                recycle_frames[j] = frame;
            }
            recycle_frame_num[i] = size;
            // Remeber to uncomment this code.
            frame_queues[i]->clear();
        }
    }
}

__global__ void sort_packets_by_timestamp(VDES::FrameDecapsulationParams *param)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < param->node_num)
    {
        for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
        {
            for (int j = 0; j < param->frame_queue_num_per_node[tid]; j++)
            {
                int queue_id = i * param->node_num + j;
                GPUQueue<uint8_t *> *packet_queue = param->packet_ingresses[queue_id];
                int timestamp_offset = param->l3_timestamp_offset[i];
                int size = packet_queue->size;

                // sort packets, swap if necessary
                uint64_t time_m, time_n;
                for (int m = 0; m < size - 1; m++)
                {
                    for (int n = m + 1; n < size; n++)
                    {
                        memcpy(&time_m, packet_queue->get_element(m) + timestamp_offset, 8);
                        memcpy(&time_n, packet_queue->get_element(n) + timestamp_offset, 8);

                        if (time_m > time_n)
                        {
                            // exchange packets
                            uint8_t *temp_packet = packet_queue->get_element(m);
                            packet_queue->set_element(m, packet_queue->get_element(n));
                            packet_queue->set_element(n, temp_packet);
                        }
                    }
                }
            }
        }
    }
}

namespace VDES
{
    void LaunchFrameDecapsulationKernel(dim3 grid_dim, dim3 block_dim, FrameDecapsulationParams *kernel_params, cudaStream_t stream)
    {
        decapsulate_frame<<<grid_dim, block_dim, 0, stream>>>(kernel_params);
    }

    void LaunchSortPacketKernel(dim3 grid_dim, dim3 block_dim, FrameDecapsulationParams *kernel_params, cudaStream_t stream)
    {
        sort_packets_by_timestamp<<<grid_dim, block_dim, 0, stream>>>(kernel_params);
    }

}

// __global__ void decapsulate_frame(VDES::FrameDecapsulationParams *param)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     if (tid < param->node_num)
//     {
//         // classification
//         int frame_queue_num = param->frame_queue_num_per_node[tid];
//         int frame_queue_offset = param->frame_queue_offset_per_node[tid];
//         GPUQueue<VDES::Frame *> **frame_queues = param->frame_ingresses + frame_queue_offset;
//         int *recycle_frame_num = param->recycle_num_per_frame_queue + frame_queue_offset;
//         GPUQueue<uint8_t *> **next_hop = param->packet_ingresses;
//         int *l3_timestamp_offset = param->l3_timestamp_offset;

//         memset(recycle_frame_num, 0, frame_queue_num * sizeof(int));
//         memset(param->recycle_frames + tid * MAX_TRANSMITTED_PACKET_NUM, 0, MAX_TRANSMITTED_PACKET_NUM);

//         // decapsulation
//         for (int i = 0; i < frame_queue_num; i++)
//         {
//             int size = frame_queues[i]->size;
//             int queue_id = frame_queue_offset + i;
//             VDES::Frame **recycle_frames = param->recycle_frames + queue_id * MAX_TRANSMITTED_PACKET_NUM;

// #if ENABLE_CACHE

//             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 // reset the number of swapped packets
//                 int index = j * param->queue_num + queue_id;

//                 param->swap_packet_num[index] = 0;
//             }
// #endif

//             for (int j = 0; j < size; j++)
//             {
//                 VDES::Frame *frame = frame_queues[i]->get_element(j);
//                 // uint16_t protocol_type;
//                 uint8_t protocol_type = frame->type[1];
//                 int next_hop_index = protocol_type * param->node_num + tid;
//                 uint8_t *payload;
//                 memcpy(&payload, frame->data, sizeof(uint8_t *));

// #if ENABLE_CACHE
//                 if (frame->device == uint8_t(VDES::Device::CPU))
//                 {
//                     // swap packets for cache
//                     int *swap_id = param->swap_packet_num + protocol_type * param->queue_num + queue_id;
//                     uint8_t *temp = payload;
//                     payload = param->swap_in_packet_buffer[(protocol_type * param->queue_num + queue_id) * MAX_TRANSMITTED_PACKET_NUM + *swap_id];
//                     param->swap_in_packet_buffer[(protocol_type * param->queue_num + queue_id) * MAX_TRANSMITTED_PACKET_NUM + *swap_id] = temp;
//                     (*swap_id)++;
//                 }
// #endif

//                 memcpy(payload + l3_timestamp_offset[protocol_type], frame->timestamp, 8);

//                 // submit payload to up layer
//                 next_hop[next_hop_index]->append_element(payload);

//                 // recycle frame
//                 recycle_frames[j] = frame;
//             }
//             recycle_frame_num[i] = size;
//             // Remeber to uncomment this code.
//             frame_queues[i]->clear();
//         }
//     }
// }

// __global__ void sort_packets_by_timestamp(VDES::FrameDecapsulationParams *param)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < param->node_num)
//     {
//         for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
//         {
//             for (int j = 0; j < param->frame_queue_num_per_node[tid]; j++)
//             {
//                 int queue_id = i * param->node_num + j;
//                 GPUQueue<uint8_t *> *packet_queue = param->packet_ingresses[queue_id];
//                 int timestamp_offset = param->l3_timestamp_offset[i];
//                 int size = packet_queue->size;

//                 // sort packets, swap if necessary
//                 uint64_t time_m, time_n;
//                 for (int m = 0; m < size - 1; m++)
//                 {
//                     for (int n = m + 1; n < size; n++)
//                     {
//                         memcpy(&time_m, packet_queue->get_element(m) + timestamp_offset, 8);
//                         memcpy(&time_n, packet_queue->get_element(n) + timestamp_offset, 8);

//                         if (time_m > time_n)
//                         {
//                             // exchange packets
//                             uint8_t *temp_packet = packet_queue->get_element(m);
//                             packet_queue->set_element(m, packet_queue->get_element(n));
//                             packet_queue->set_element(n, temp_packet);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// namespace VDES
// {
//     void LaunchFrameDecapsulationKernel(dim3 grid_dim, dim3 block_dim, FrameDecapsulationParams *kernel_params, cudaStream_t stream)
//     {
//         decapsulate_frame<<<grid_dim, block_dim, 0, stream>>>(kernel_params);
//     }

//     void LaunchSortPacketKernel(dim3 grid_dim, dim3 block_dim, FrameDecapsulationParams *kernel_params, cudaStream_t stream)
//     {
//         sort_packets_by_timestamp<<<grid_dim, block_dim, 0, stream>>>(kernel_params);
//     }

// }

// #include "frame_decapsulation.h"

// __global__ void decapsulate_frame(VDES::FrameDecapsulationParams *param)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     if (tid < param->node_num)
//     {
//         // classification
//         int frame_queue_num = param->frame_queue_num_per_node[tid];
//         int frame_queue_offset = param->frame_queue_offset_per_node[tid];
//         // TODO(fix): offset of the ingress queue should start from
//         GPUQueue<VDES::Frame *> **frame_queues = param->frame_ingresses + frame_queue_offset;
//         int *recycle_frame_num = param->recycle_num_per_frame_queue + frame_queue_offset;
//         GPUQueue<uint8_t *> **next_hop = param->packet_ingresses;
//         int *l3_timestamp_offset = param->l3_timestamp_offset;

//         // decapsulation
//         for (int i = 0; i < frame_queue_num; i++)
//         {
//             int size = frame_queues[i]->size;
//             int queue_id = frame_queue_offset + i;
//             VDES::Frame **recycle_frames = param->recycle_frames + queue_id * MAX_TRANSMITTED_PACKET_NUM;

//             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
//             {
//                 // reset the number of swapped packets
//                 int index = j * param->queue_num + queue_id;

//                 param->swap_packet_num[index] = 0;
//             }

//             for (int j = 0; j < size; j++)
//             {
//                 VDES::Frame *frame = frame_queues[i]->get_element(j);
//                 uint8_t protocol_type = frame->type[1];
//                 int next_hop_index = protocol_type * param->node_num + tid;
//                 uint8_t *payload = *((uint8_t **)frame->data);
//                 VDES::Ipv4Packet *packet = (VDES::Ipv4Packet *)payload;

//                 if (frame->device == uint8_t(VDES::Device::CPU))
//                 {
//                     // swap packets for cache
//                     int *swap_id = param->swap_packet_num + protocol_type * param->queue_num + queue_id;
//                     uint8_t *temp = payload;
//                     payload = param->swap_in_packet_buffer[(protocol_type * param->queue_num + queue_id) * MAX_TRANSMITTED_PACKET_NUM + *swap_id];
//                     param->swap_in_packet_buffer[(protocol_type * param->queue_num + queue_id) * MAX_TRANSMITTED_PACKET_NUM + *swap_id] = temp;
//                     (*swap_id)++;
//                 }

//                 memcpy(payload + l3_timestamp_offset[protocol_type], frame->timestamp, 8);

//                 // submit payload to up layer
//                 next_hop[next_hop_index]->append_element(payload);
//                 // next_hop_ts[next_hop_index]->append_element(frame_queues_ts[i]->get_element(j));

//                 // recycle frame
//                 recycle_frames[j] = frame;
//             }
//             recycle_frame_num[i] = size;
//             frame_queues[i]->clear();
//             // frame_queues_ts[i]->clear();
//         }
//     }
// }

// __global__ void sort_packets_by_timestamp(VDES::FrameDecapsulationParams *param)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < param->node_num)
//     {
//         for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
//         {
//             for (int j = 0; j < param->node_num; j++)
//             {
//                 int queue_id = i * param->node_num + j;
//                 GPUQueue<uint8_t *> *packet_queue = param->packet_ingresses[queue_id];
//                 int timestamp_offset = param->l3_timestamp_offset[i];
//                 // GPUQueue<int64_t> *packet_queue_ts = param->packet_ingresses_ts[queue_id];
//                 int size = packet_queue->size;

//                 // sort packets, swap if necessary
//                 int time_m, time_n;
//                 for (int m = 0; m < size - 1; m++)
//                 {
//                     for (int n = m + 1; n < size; n++)
//                     {
//                         memcpy(&time_m, packet_queue->get_element(m) + timestamp_offset, 8);
//                         memcpy(&time_n, packet_queue->get_element(n) + timestamp_offset, 8);

//                         if (time_m > time_n)
//                         {
//                             // exchange packets
//                             uint8_t *temp_packet = packet_queue->get_element(m);
//                             packet_queue->set_element(m, packet_queue->get_element(n));
//                             packet_queue->set_element(n, temp_packet);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// namespace VDES
// {
//     void LaunchFrameDecapsulationKernel(dim3 grid_dim, dim3 block_dim, FrameDecapsulationParams *kernel_params, cudaStream_t stream)
//     {
//         decapsulate_frame<<<grid_dim, block_dim, 0, stream>>>(kernel_params);
//     }

//     void LaunchSortPacketKernel(dim3 grid_dim, dim3 block_dim, FrameDecapsulationParams *kernel_params, cudaStream_t stream)
//     {
//         sort_packets_by_timestamp<<<grid_dim, block_dim, 0, stream>>>(kernel_params);
//     }

// }

// // #include "frame_decapsulation.h"

// // __global__ void decapsulate_frame(VDES::FrameDecapsulationParams *param)
// // {
// //     int tid = blockIdx.x * blockDim.x + threadIdx.x;

// //     if (tid < param->node_num)
// //     {
// //         // classification
// //         int frame_queue_num = param->frame_queue_num_per_node[tid];
// //         int frame_queue_offset = param->frame_queue_offset_per_node[tid];
// //         GPUQueue<VDES::Frame *> **frame_queues = param->frame_ingresses + frame_queue_num;
// //         GPUQueue<int64_t> **frame_queues_ts = param->frame_ingresses_ts + frame_queue_num;
// //         int *recycle_frame_num = param->recycle_num_per_frame_queue + frame_queue_offset;
// //         GPUQueue<void *> **next_hop = param->packet_ingresses;
// //         GPUQueue<int64_t> **next_hop_ts = param->packet_ingresses_ts;

// //         // decapsulation
// //         for (int i = 0; i < frame_queue_num; i++)
// //         {
// //             int size = frame_queues[i]->size;
// //             int queue_id = frame_queue_offset + i;
// //             VDES::Frame **recycle_frames = param->recycle_frames + queue_id * MAX_TRANSMITTED_PACKET_NUM;

// //             for (int j = 0; j < NetworkProtocolType::COUNT_NetworkProtocolType; j++)
// //             {
// //                 // reset the number of swapped packets
// //                 param->swap_packet_num[j * param->node_num + queue_id] = 0;
// //             }

// //             for (int j = 0; j < size; j++)
// //             {
// //                 VDES::Frame *frame = frame_queues[i]->get_element(j);
// //                 uint16_t protocol_type = *((uint16_t *)frame->type);
// //                 int next_hop_index = protocol_type * param->node_num + tid;
// //                 void *payload = *((void **)frame->data);

// //                 if (frame->device[1] == uint8_t(VDES::Device::CPU))
// //                 {
// //                     // swap packets for cache
// //                     int *swap_id = param->swap_packet_num + protocol_type * param->node_num + queue_id;
// //                     void *temp = payload;
// //                     payload = param->swap_in_packet_buffer[(protocol_type * param->node_num + queue_id) * MAX_TRANSMITTED_PACKET_NUM + *swap_id];
// //                     param->swap_in_packet_buffer[(protocol_type * param->node_num + queue_id) * MAX_TRANSMITTED_PACKET_NUM + *swap_id] = temp;
// //                     (*swap_id)++;
// //                 }

// //                 // submit payload to up layer
// //                 next_hop[next_hop_index]->append_element(payload);
// //                 next_hop_ts[next_hop_index]->append_element(frame_queues_ts[i]->get_element(j));

// //                 // recycle frame
// //                 recycle_frames[j] = frame;
// //             }
// //             recycle_frame_num[i] = size;
// //             frame_queues[i]->clear();
// //             frame_queues_ts[i]->clear();
// //         }
// //     }
// // }

// // __global__ void sort_packets_by_timestamp(VDES::FrameDecapsulationParams *param)
// // {
// //     int tid = blockIdx.x * blockDim.x + threadIdx.x;
// //     if (tid < param->node_num)
// //     {
// //         for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
// //         {
// //             for (int j = 0; j < param->node_num; j++)
// //             {
// //                 int queue_id = i * param->node_num + j;
// //                 GPUQueue<void *> *packet_queue = param->packet_ingresses[queue_id];
// //                 GPUQueue<int64_t> *packet_queue_ts = param->packet_ingresses_ts[queue_id];
// //                 int size = packet_queue->size;
// //                 // sort packets, swap if necessary
// //                 for (int m = 0; m < size - 1; m++)
// //                 {
// //                     for (int n = m + 1; n < size; n++)
// //                     {
// //                         if (packet_queue_ts->get_element(m) > packet_queue_ts->get_element(n))
// //                         {
// //                             // exchange packets
// //                             void *temp_packet = packet_queue->get_element(m);
// //                             packet_queue->set_element(m, packet_queue->get_element(n));
// //                             packet_queue->set_element(n, temp_packet);
// //                             int64_t temp_ts = packet_queue_ts->get_element(m);
// //                             packet_queue_ts->set_element(m, packet_queue_ts->get_element(n));
// //                             packet_queue_ts->set_element(n, temp_ts);
// //                         }
// //                     }
// //                 }
// //             }
// //         }
// //     }
// // }

// // namespace VDES
// // {
// //     void LaunchFrameDecapsulationKernel(dim3 grid_dim, dim3 block_dim, FrameDecapsulationParams *kernel_params, cudaStream_t stream)
// //     {
// //         decapsulate_frame<<<grid_dim, block_dim, 0, stream>>>(kernel_params);
// //     }

// //     void LaunchSortPacketKernel(dim3 grid_dim, dim3 block_dim, FrameDecapsulationParams *kernel_params, cudaStream_t stream)
// //     {
// //         sort_packets_by_timestamp<<<grid_dim, block_dim, 0, stream>>>(kernel_params);
// //     }

// // }