#include "frame_encapsulation.h"

__device__ uint8_t *lookup_mac_address(GPUQueue<VDES::ARPRule *> *arp, uint32_t ip, uint8_t *null_mac)
{
    int end = arp->size;
    int start = 0;

    // binary search for the mac address
    while (start < end)
    {
        int mid = (start + end) / 2;
        VDES::ARPRule *rule = arp->queue[mid];
        if (rule->ip == ip)
        {
            return rule->mac;
        }
        else if (rule->ip < ip)
        {
            start = mid + 1;
        }
        else
        {
            end = mid;
        }
    }
    return null_mac;
}

__device__ uint8_t *fattree_arp_lookup(uint16_t k, uint32_t ip_group_size, uint32_t ip_assign_base, uint32_t ip, uint8_t *mac)
{
    int group_id = (ip - ip_assign_base) / ip_group_size;
    int inner_id = (ip - ip_assign_base) % ip_group_size;
    int node_id = k / 2 * group_id + inner_id;
    memcpy(mac, &node_id, sizeof(node_id));
    return mac;
}

__device__ void sort_frames_encapsulation(GPUQueue<VDES::Frame *> *frame_queue, int origin_size)
{
    int size = frame_queue->size;
    int64_t time_i = 0;
    int64_t time_j = 0;
    for (int i = origin_size; i < size; i++)
    {
        memcpy(&time_i, frame_queue->get_element(i)->timestamp, 8);
        int index = i;
        for (int j = i + 1; j < size; j++)
        {

            memcpy(&time_j, frame_queue->get_element(j)->timestamp, 8);

            if (time_i > time_j)
            {
                time_i = time_j;
                index = j;
            }
        }

        if (index != i)
        {
            auto tmp = frame_queue->queue[i];
            frame_queue->queue[i] = frame_queue->queue[index];
            frame_queue->queue[index] = tmp;
        }
    }
}

__global__ void encapsulate_frames(VDES::FrameEncapsulationParams *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < params->queue_num)
    {
        int node_id = params->node_id_per_frame_queue[tid];
        auto packet_queues = params->packets_egresses;
        auto frame_queue = params->frame_queues[tid];

        params->alloc_num_per_frame_queue[tid] = 0;
        for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
        {
            params->alloc_num_per_frame_queue[tid] += packet_queues[tid + i * params->queue_num]->size;
        }
        params->alloc_num_per_frame_queue[tid] = min(params->alloc_num_per_frame_queue[tid], frame_queue->get_remaining_capacity());

        __syncthreads();

        int inner_id = tid - params->frame_queue_offset_per_node[node_id];
        auto alloc_frames = params->alloc_frames + params->alloc_offset_per_node[node_id];

        int alloc_offset = 0;
        for (int i = 0; i < inner_id; i++)
        {
            alloc_offset += params->alloc_num_per_frame_queue[inner_id - i - 1];
        }
        alloc_frames = alloc_frames + alloc_offset;

        // encapsulate packets into frames
        int frame_offset = 0;
        uint8_t *src_mac = params->frame_queue_macs[tid];
        uint8_t *dst_mac = NULL;

#if ENABLE_FATTREE_MODE
        dst_mac = params->mac_addr_ft[tid];

#else
        dst_mac = params->null_mac;
#endif

        for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
        {
            auto packet_q = packet_queues[tid + i * params->queue_num];
            int packet_m = min(packet_q->size, frame_queue->get_remaining_capacity());

#if ENABLE_CACHE
            uint8_t **swap_out_l3_packets = params->swap_out_l3_packets + params->alloc_offset_per_node[node_id] + alloc_offset + i * params->max_packet_num;
            uint8_t **l3_cache_ptr = params->l3_cache_ptr + params->alloc_offset_per_node[node_id] + alloc_offset + i * params->max_packet_num;
            /**
             * @TODO: UPDATE THE I * NODE_NUM HERE
             */
            int *swap_out_l3_packets_num = params->swap_out_l3_packets_num + i * params->node_num;
            swap_out_l3_packets_num[tid] = 0;
#endif

            int len_offset = params->l3_packet_len_offset[i];
            int timestamp_offset = params->l3_packet_timestamp_offset[i];
            int ip_offset = params->l3_dst_ip_offset[i];

            for (int j = 0; j < packet_m; j++)
            {
                auto packet = packet_q->get_element(j);
                auto frame = alloc_frames[frame_offset + j];

                uint32_t dst_ip;
                memcpy(&dst_ip, packet + ip_offset, 4);

#if ENABLE_FATTREE_MODE
                dst_mac = fattree_arp_lookup(params->k, params->ip_group_size, params->base_ip, dst_ip, dst_mac);

#else
                dst_mac = lookup_mac_address(params->arp_tables[i], dst_ip, params->null_mac);
#endif

                // fill in the frame
                memcpy(frame->dst_mac, dst_mac, 6);

                memcpy(frame->src_mac, src_mac, 6);

                uint16_t frame_len;
                memcpy(&frame_len, packet + len_offset, 2);
                frame_len += 40; // 40 bytes for the header
                memcpy(frame->frame_len, &frame_len, 2);

                frame->type[1] = i;

#if ENABLE_DCTCP
                frame->fcs[0] = 0;
#endif

                memcpy(frame->timestamp, packet + timestamp_offset, 8);
                // int64_t timestamp;
                // memcpy(&timestamp, frame->timestamp, 8);
                // printf("%ld\n", timestamp);

#if ENABLE_CACHE
                int packet_size = params->l3_packet_size[i];
                // cache out
                memcpy(l3_cache_ptr[j], packet, packet_size);

                memcpy(&swap_out_l3_packets[j], frame->data, 8);
                frame->device = (uint8_t)VDES::Device::CPU;
                packet = swap_out_l3_packets[j];
                swap_out_l3_packets_num[tid]++;
#endif
                memcpy(frame->data, &packet, 8);
            }

            frame_offset += packet_m;

            // packet_q->clear();
            packet_q->remove_elements(packet_m);
        }

        frame_queue->append_elements(alloc_frames, frame_offset);
    }
}

namespace VDES
{
    void LaunchEncapsulateFrameKernel(dim3 grid_dim, dim3 block_dim, VDES::FrameEncapsulationParams *params, cudaStream_t stream)
    {
        encapsulate_frames<<<grid_dim, block_dim, 0, stream>>>(params);
    }
} // namespace VDES

// #include "frame_encapsulation.h"

// __device__ uint8_t *lookup_mac_address(GPUQueue<VDES::ARPRule *> *arp, uint32_t ip, uint8_t *null_mac)
// {
//     int end = arp->size;
//     int start = 0;

//     // binary search for the mac address
//     while (start < end)
//     {
//         int mid = (start + end) / 2;
//         VDES::ARPRule *rule = arp->queue[mid];
//         if (rule->ip == ip)
//         {
//             return rule->mac;
//         }
//         else if (rule->ip < ip)
//         {
//             start = mid + 1;
//         }
//         else
//         {
//             end = mid;
//         }
//     }
//     return null_mac;
// }

// __device__ uint8_t *fattree_arp_lookup(uint16_t k, uint32_t ip_group_size, uint32_t ip_assign_base, uint32_t ip, uint8_t *mac)
// {
//     int group_id = (ip - ip_assign_base) / ip_group_size;
//     int inner_id = (ip - ip_assign_base) % ip_group_size;
//     int node_id = k / 2 * group_id + inner_id;
//     memcpy(mac, &node_id, sizeof(node_id));
//     return mac;
// }

// __device__ void sort_frames(GPUQueue<VDES::Frame *> *frame_queue, int origin_size)
// {
//     int size = frame_queue->size;
//     int64_t time_i = 0;
//     int64_t time_j = 0;
//     for (int i = origin_size; i < size; i++)
//     {
//         for (int j = i + 1; j < size; j++)
//         {
//             memcpy(&time_i, frame_queue->get_element(i)->timestamp, 8);
//             memcpy(&time_j, frame_queue->get_element(j)->timestamp, 8);

//             if (time_i < time_j)
//             {
//                 VDES::Frame *temp = frame_queue->get_element(i);
//                 frame_queue->set_element(i, frame_queue->get_element(j));
//                 frame_queue->set_element(j, temp);
//             }
//         }
//     }
// }

// __global__ void encapsulate_frames(VDES::FrameEncapsulationParams *params)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < params->queue_num)
//     {
//         int node_id = params->node_id_per_frame_queue[tid];
//         auto packet_queues = params->packets_egresses;
//         auto frame_queue = params->frame_queues[tid];
//         // auto frame_queue_ts = params->frame_queue_tss[tid];
//         int origin_size = frame_queue->size;

//         params->alloc_num_per_frame_queue[tid] = 0;
//         for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
//         {
//             // auto queue = packet_queues[tid + i * params->queue_num];
//             params->alloc_num_per_frame_queue[tid] += packet_queues[tid + i * params->queue_num]->size;
//         }
//         params->alloc_num_per_frame_queue[tid] = min(params->alloc_num_per_frame_queue[tid], frame_queue->get_remaining_capacity());

//         __threadfence();

//         int inner_id = tid - params->frame_queue_offset_per_node[node_id];
//         auto alloc_frames = params->alloc_frames + params->alloc_offset_per_node[node_id];

//         int alloc_offset = 0;
//         for (int i = 0; i < inner_id; i++)
//         {
//             alloc_offset += params->alloc_num_per_frame_queue[inner_id - i - 1];
//         }
//         alloc_frames = alloc_frames + alloc_offset;

//         // encapsulate packets into frames
//         int frame_offset = 0;
//         uint8_t *src_mac = params->frame_queue_macs[tid];
//         uint8_t *dst_mac = NULL;

//         if (ENABLE_FATTREE_MODE)
//         {
//             /**
//              * TODO: Maybe we should use frame_queue_id
//              */
//             dst_mac = params->mac_addr_ft[tid];
//         }
//         else
//         {
//             dst_mac = params->null_mac;
//         }

//         for (int i = 0; i < NetworkProtocolType::COUNT_NetworkProtocolType; i++)
//         {
//             auto packet_q = packet_queues[tid + i * params->queue_num];
//             int packet_m = min(packet_q->size, frame_queue->get_remaining_capacity());

//             uint8_t **swap_out_l3_packets = params->swap_out_l3_packets + params->alloc_offset_per_node[node_id] + alloc_offset + i * params->max_packet_num;
//             uint8_t **l3_cache_ptr = params->l3_cache_ptr + params->alloc_offset_per_node[node_id] + alloc_offset + i * params->max_packet_num;
//             int *swap_out_l3_packets_num = params->swap_out_l3_packets_num + i * params->max_packet_num;
//             swap_out_l3_packets_num[tid] = 0;

//             int packet_size = params->l3_packet_size[i];
//             int len_offset = params->l3_packet_len_offset[i];
//             int timestamp_offset = params->l3_packet_timestamp_offset[i];
//             int ip_offset = params->l3_dst_ip_offset[i];

//             for (int j = 0; j < packet_m; j++)
//             {
//                 auto packet = packet_q->get_element(j);
//                 auto frame = alloc_frames[frame_offset + j];
//                 uint32_t dst_ip;
//                 memcpy(&dst_ip, packet + ip_offset, 4);

//                 if (ENABLE_FATTREE_MODE)
//                 {
//                     dst_mac = fattree_arp_lookup(params->k, params->ip_group_size, params->base_ip, dst_ip, dst_mac);
//                 }
//                 else
//                 {
//                     dst_mac = lookup_mac_address(params->arp_tables[i], dst_ip, params->null_mac);
//                 }
//                 // fill in the frame
//                 memcpy(frame->dst_mac, dst_mac, 6);

//                 memcpy(frame->src_mac, src_mac, 6);

//                 uint16_t frame_len;
//                 memcpy(&frame_len, packet + len_offset, 2);
//                 frame_len += 40; // 40 bytes for the header
//                 memcpy(frame->frame_len, &frame_len, 2);

//                 frame->type[1] = i;

//                 memcpy(frame->timestamp, packet + timestamp_offset, 8);

//                 // cache out
//                 memcpy(l3_cache_ptr[j], packet, packet_size);

//                 memcpy(frame->data, &swap_out_l3_packets[j], sizeof(uint8_t *));
//                 frame->device = (uint8_t)VDES::Device::CPU;
//                 swap_out_l3_packets_num[tid]++;
//             }

//             frame_offset += packet_m;

//             packet_q->clear();
//         }

//         frame_queue->append_elements(alloc_frames, frame_offset);

//         // sort packets
//         sort_frames(frame_queue, origin_size);
//     }
// }

// namespace VDES
// {
//     void LaunchEncapsulateFrameKernel(dim3 grid_dim, dim3 block_dim, VDES::FrameEncapsulationParams *params, cudaStream_t stream)
//     {
//         encapsulate_frames<<<grid_dim, block_dim, 0, stream>>>(params);
//     }
// } // namespace VDES
