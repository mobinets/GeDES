#include "p2p_channel.h"

__global__ void p2p_transmit_frames(VDES::P2PParams *param)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < param->queue_num)
    {
        GPUQueue<VDES::Frame *> *ingress = param->ingresses[tid];
        GPUQueue<VDES::Frame *> *egress = param->egresses[tid];

        int64_t current_time = max(*(param->timeslot_start_time), param->last_tx_end_time[tid]);
        int64_t timeslot_end = *(param->timeslot_end_time);
        int tx_rate = param->link_transmission_rate[tid];
        int propogation_delay = param->propogation_delay_per_ch[tid];
        uint16_t remainder_packet_len = param->remainder_of_packet_len[tid];

        int transmit_num = 0;

        int size = ingress->size;
        for (int i = 0; i < size; i++)
        {
            VDES::Frame *frame = ingress->get_element(i);
            uint64_t arrival_time;
            memcpy(&arrival_time, frame->timestamp, sizeof(uint64_t));
            arrival_time = max(current_time, arrival_time);

            if (arrival_time >= timeslot_end || transmit_num >= MAX_TRANSMITTED_PACKET_NUM)
            {
                break;
            }

            uint16_t frame_len;
            memcpy(&frame_len, frame->frame_len, sizeof(uint16_t));
            current_time = arrival_time + (frame_len * 8 + remainder_packet_len) / tx_rate;
            remainder_packet_len = (frame_len * 8 + remainder_packet_len) % tx_rate;
            arrival_time = current_time + propogation_delay;
            memcpy(frame->timestamp, &arrival_time, sizeof(uint64_t));
            transmit_num++;
        }
        // printf("tid: %d, transmit: %d, cache queue size: %d\n", tid, transmit_num, size - transmit_num);
        // printf("tid: %d, transmit_num: %d, total queue size: %d\n", tid, transmit_num, size);
        // printf("tid: %d, size: %d\n", tid, size);

        egress->append_elements(ingress, transmit_num);
        ingress->remove_elements(transmit_num);
        param->last_tx_end_time[tid] = current_time;
        param->remainder_of_packet_len[tid] = remainder_packet_len;

        // examine whether frames are completed
        __syncthreads();
        if (threadIdx.x == 0)
        {
            bool is_completed = true;
            for (size_t i = 0; i < blockDim.x && tid + i < param->queue_num; i++)
            {
                if (param->egresses[tid + i]->size != 0 || param->ingresses[tid + i]->size != 0)
                {
                    is_completed = false;
                    break;
                }
            }
            param->is_completed[blockIdx.x] = is_completed;
        }
    }
}

namespace VDES
{
    void LuanchP2PTransmiFramesKernel(dim3 grid, dim3 block, VDES::P2PParams *p2p_params, cudaStream_t stream)
    {
        p2p_transmit_frames<<<grid, block, 0, stream>>>(p2p_params);
    }

}

// __global__ void p2p_transmit_frames(VDES::P2PParams *param)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < param->queue_num)
//     {
//         GPUQueue<VDES::Frame *> *ingress = param->ingresses[tid];
//         GPUQueue<VDES::Frame *> *egress = param->egresses[tid];

//         int64_t current_time = max(*(param->timeslot_start_time), param->last_tx_end_time[tid]);
//         int64_t timeslot_end = *(param->timeslot_end_time);
//         int tx_rate = param->link_transmission_rate[tid];
//         int propogation_delay = param->propogation_delay_per_ch[tid];
//         uint16_t remainder_packet_len = param->remainder_of_packet_len[tid];

//         int transmit_num = 0;
//         /**
//          * TODO: size is equal to the ingress queue size.
//          */
//         int size = ingress->size;
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Frame *frame = ingress->get_element(i);
//             uint64_t arrival_time;
//             /**
//              * TODO: Use timestamp instead of data.
//              */
//             memcpy(&arrival_time, frame->timestamp, sizeof(uint64_t));
//             arrival_time = max(current_time, arrival_time);

//             if (arrival_time >= timeslot_end || transmit_num >= MAX_TRANSMITTED_PACKET_NUM)
//             {
//                 break;
//             }

//             uint16_t frame_len;
//             memcpy(&frame_len, frame->frame_len, sizeof(uint16_t));
//             current_time = arrival_time + (frame_len * 8 + remainder_packet_len) / tx_rate;
//             remainder_packet_len = (frame_len * 8 + remainder_packet_len) % tx_rate;
//             arrival_time = current_time + propogation_delay;
//             /**
//              * TODO: Use timestamp instead of data.
//              */
//             memcpy(frame->timestamp, &arrival_time, sizeof(uint64_t));
//             transmit_num++;
//         }

//         egress->append_elements(ingress, transmit_num);
//         ingress->remove_elements(transmit_num);
//         param->last_tx_end_time[tid] = current_time;
//         param->remainder_of_packet_len[tid] = remainder_packet_len;
//     }
// }

// namespace VDES
// {
//     void LuanchP2PTransmiFramesKernel(dim3 grid, dim3 block, VDES::P2PParams *p2p_params, cudaStream_t stream)
//     {
//         p2p_transmit_frames<<<grid, block, 0, stream>>>(p2p_params);
//     }

// }

// #include "p2p_channel.h"

// __global__ void p2p_transmit_frames(VDES::P2PParams *param)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < param->queue_num)
//     {
//         GPUQueue<VDES::Frame *> *ingress = param->ingresses[tid];
//         GPUQueue<VDES::Frame *> *egress = param->egresses[tid];

//         int64_t current_time = max(*(param->timeslot_start_time), param->last_tx_end_time[tid]);
//         int64_t timeslot_end = *(param->timeslot_end_time);
//         int tx_rate = param->link_transmission_rate[tid];
//         int propogation_delay = param->propogation_delay_per_ch[tid];

//         int transmit_num = 0;
//         int size = 0;
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Frame *frame = ingress->get_element(i);
//             int64_t arrival_time;
//             memcpy(&arrival_time, frame->data, sizeof(int64_t));
//             arrival_time = max(current_time, arrival_time);

//             if (arrival_time >= timeslot_end || transmit_num >= MAX_TRANSMITTED_PACKET_NUM)
//             {
//                 break;
//             }

//             uint16_t frame_len;
//             memcpy(&frame_len, frame->frame_len, sizeof(uint16_t));
//             current_time = arrival_time + frame_len / tx_rate;
//             arrival_time += propogation_delay;
//             memcpy(frame->data, &arrival_time, sizeof(int64_t));
//             transmit_num++;
//         }

//         egress->append_elements(ingress, transmit_num);
//         ingress->remove_elements(transmit_num);
//         param->last_tx_end_time[tid] = current_time;
//     }
// }

// namespace VDES
// {
//     void LuanchP2PTransmiFramesKernel(dim3 grid, dim3 block, VDES::P2PParams *p2p_params, cudaStream_t stream)
//     {
//         p2p_transmit_frames<<<grid, block, 0, stream>>>(p2p_params);
//     }

// }