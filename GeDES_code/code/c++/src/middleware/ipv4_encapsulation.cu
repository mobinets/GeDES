#include "ipv4_encapsulation.h"

__global__ void ipv4_encapsulation(VDES::IPv4EncapsulationParams *param)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < param->node_num)
    {
        GPUQueue<uint8_t *> **l4_packet_queues = param->l4_packet_queues + tid;
        GPUQueue<VDES::Ipv4Packet *> *ipv4_queue = param->ipv4_packet_queues[tid];

#if ENABLE_CACHE
        uint8_t **l4_swap_out_packets = nullptr;
        uint8_t **l4_cache_space = nullptr;
        int *l4_swap_out_packet_num = nullptr;
        int *l4_swap_out_offset = nullptr;
        l4_swap_out_packets = param->l4_swap_out_packets + param->l4_swap_out_offset[tid];
        l4_cache_space = param->l4_cache_space + param->l4_swap_out_offset[tid];
        l4_swap_out_packet_num = param->l4_swap_out_packet_num;
        l4_swap_out_offset = param->l4_swap_out_offset;
#endif

        VDES::Ipv4Packet **alloc_ipv4_packets = param->alloc_ipv4_packets + param->packet_offset_per_node[tid];

        int node_num = param->node_num;

        int ipv4_offset = 0;
        for (int i = 0; i < TransportProtocolType::COUNT_TransportProtocolType; i++)
        {
            GPUQueue<uint8_t *> *l4_queue = l4_packet_queues[i * node_num];
            int l4_queue_size = l4_queue->size;

            int src_ip_offset = param->l4_src_ip_offset[i];
            int dst_ip_offset = param->l4_dst_ip_offset[i];
            int timestamp_offset = param->l4_timestamp_offset[i];
            int len_offset = param->l4_len_offset[i];

#if ENABLE_CACHE
            int packet_size = param->l4_packet_size[i];
            l4_swap_out_packets = l4_swap_out_packets + i * param->max_packet_num;
            l4_cache_space = l4_cache_space + i * param->max_packet_num;
#endif

            for (int j = 0; j < l4_queue_size; j++)
            {
                // fill ipv4 packet
                VDES::Ipv4Packet *ipv4_packet = alloc_ipv4_packets[ipv4_offset];
                ipv4_offset++;

                uint8_t *l4_packet = l4_queue->get_element(j);
#if DEBUG_MODE
                uint64_t payload_in_tcp;
                memcpy(&payload_in_tcp, l4_packet + 20, sizeof(uint64_t));
                uint32_t sequence_in_tcp;
                memcpy(&sequence_in_tcp, l4_packet + 4, sizeof(uint32_t));
                VDES::TCPPacket tmp_packet;
                memcpy(&tmp_packet, l4_packet, sizeof(VDES::TCPPacket));
#endif

                // fill ips
                memcpy(ipv4_packet->src_ip, l4_packet + src_ip_offset, 4);
                memcpy(ipv4_packet->dst_ip, l4_packet + dst_ip_offset, 4);

                // fill timestamp
                memcpy(ipv4_packet->timestamp, l4_packet + timestamp_offset, 8);

                // int64_t timestamp;
                // memcpy(&timestamp, ipv4_packet->timestamp, 8);
                // printf("%ld\n", timestamp);

                // fill protocol
                ipv4_packet->protocol = i;
                ipv4_packet->time_to_live = IPV4_DEFAULT_TTL;

                // fill packet len
                uint16_t packet_len;
                memcpy(&packet_len, l4_packet + len_offset, 2);
                if (i == TransportProtocolType::TCP)
                {
                    // tcp header and ipv4 header
                    packet_len += 40;
                }
                memcpy(ipv4_packet->total_len, &packet_len, 2);

// fill payload, swap out l4 packet1
#if ENSABLE_CACHE
                memcpy(l4_cache_space[j], l4_packet, packet_size);
                l4_packet = l4_swap_out_packets[j];
#endif
                memcpy(ipv4_packet->payload, &l4_packet, 8);
            }
#if ENABLE_CACHE
            l4_swap_out_packet_num[i * node_num + tid] = l4_queue_size;
#endif

            // l4_queue->clear();
            l4_queue->remove_elements(l4_queue_size);
        }

        param->used_packet_num_per_node[tid] = ipv4_offset;
        ipv4_queue->append_elements(alloc_ipv4_packets, ipv4_offset);

        // printf("tid: %d, send packet num: %d\n", tid, ipv4_offset);
    }
}

namespace VDES
{
    void LaunchIPv4EncapsulationKernel(dim3 grid, dim3 block, IPv4EncapsulationParams *params, cudaStream_t stream)
    {
        ipv4_encapsulation<<<grid, block, 0, stream>>>(params);
    }

}

// __global__ void ipv4_encapsulation(VDES::IPv4EncapsulationParams *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->node_num)
//     {
//         GPUQueue<uint8_t *> **l4_packet_queues = param->l4_packet_queues + tid;
//         GPUQueue<VDES::Ipv4Packet *> *ipv4_queue = param->ipv4_packet_queues[tid];

//         uint8_t **l4_swap_out_packets = NULL;
//         uint8_t **l4_cache_space = NULL;
//         int *l4_swap_out_packet_num = NULL;
//         int *l4_swap_out_offset = NULL;

// #if ENABLE_CACHE

//         l4_swap_out_packets = param->l4_swap_out_packets + param->l4_swap_out_offset[tid];
//         l4_cache_space = param->l4_cache_space + param->l4_swap_out_offset[tid];
//         l4_swap_out_packet_num = param->l4_swap_out_packet_num;
//         l4_swap_out_offset = param->l4_swap_out_offset;
// #endif

//         VDES::Ipv4Packet **alloc_ipv4_packets = param->alloc_ipv4_packets + param->packet_offset_per_node[tid];

//         int node_num = param->node_num;

//         int ipv4_offset = 0;
//         for (int i = 0; i < TransportProtocolType::COUNT_TransportProtocolType; i++)
//         {
//             GPUQueue<uint8_t *> *l4_queue = l4_packet_queues[i * node_num];
//             int l4_queue_size = l4_queue->size;

//             int src_ip_offset = param->l4_src_ip_offset[i];
//             int dst_ip_offset = param->l4_dst_ip_offset[i];
//             int timestamp_offset = param->l4_timestamp_offset[i];
//             int len_offset = param->l4_len_offset[i];
//             int packet_size = param->l4_packet_size[i];

// #if ENABLE_CACHE

//             l4_swap_out_packets = l4_swap_out_packets + i * param->max_packet_num;
//             l4_cache_space = l4_cache_space + i * param->max_packet_num;
// #endif

//             for (int j = 0; j < l4_queue_size; j++)
//             {
//                 // fill ipv4 packet
//                 VDES::Ipv4Packet *ipv4_packet = alloc_ipv4_packets[ipv4_offset];
//                 ipv4_offset++;

//                 uint8_t *l4_packet = l4_queue->get_element(j);

//                 // fill ips
//                 memcpy(ipv4_packet->src_ip, l4_packet + src_ip_offset, 4);
//                 memcpy(ipv4_packet->dst_ip, l4_packet + dst_ip_offset, 4);

//                 // fill timestamp
//                 memcpy(ipv4_packet->timestamp, l4_packet + timestamp_offset, 8);

//                 // fill packet len
//                 uint16_t packet_len;
//                 memcpy(&packet_len, l4_packet + len_offset, 2);
//                 if (i == TransportProtocolType::TCP)
//                 {
//                     // tcp header and ipv4 header
//                     packet_len += 40;
//                 }
//                 memcpy(ipv4_packet->total_len, &packet_len, 2);

//                 // fill protocol
//                 ipv4_packet->protocol = i;
//                 ipv4_packet->time_to_live = IPV4_DEFAULT_TTL;

// // fill payload, swap out l4 packet1
// #if ENABLE_CACHE

//                 // memcpy(ipv4_packet->payload, l4_swap_out_packets + j, 8);
//                 memcpy(l4_cache_space[j], l4_packet, packet_size);
//                 l4_packet = l4_swap_out_packets[j];
// #endif

//                 memcpy(ipv4_packet->payload, &l4_packet, 8);
//             }

// #if ENABLE_CACHE
//             l4_swap_out_packet_num[i * node_num + tid] = l4_queue_size;
// #endif

//             l4_queue->clear();
//         }

//         param->used_packet_num_per_node[tid] = ipv4_offset;
//         ipv4_queue->append_elements(alloc_ipv4_packets, ipv4_offset);
//     }
// }

// namespace VDES
// {
//     void LaunchIPv4EncapsulationKernel(dim3 grid, dim3 block, IPv4EncapsulationParams *params, cudaStream_t stream)
//     {
//         ipv4_encapsulation<<<grid, block, 0, stream>>>(params);
//     }

// }

// #include "ipv4_encapsulation.h"

// __global__ void ipv4_encapsulation(VDES::IPv4EncapsulationParams *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->node_num)
//     {
//         GPUQueue<uint8_t *> **l4_packet_queues = param->l4_packet_queues + tid;
//         GPUQueue<VDES::Ipv4Packet *> *ipv4_queue = param->ipv4_packet_queues[tid];
//         uint8_t **l4_swap_out_packets = param->l4_swap_out_packets + param->l4_swap_out_offset[tid];
//         uint8_t **l4_cache_space = param->l4_cache_space + param->l4_swap_out_offset[tid];
//         int *l4_swap_out_packet_num = param->l4_swap_out_packet_num;
//         int *l4_swap_out_offset = param->l4_swap_out_offset;

//         VDES::Ipv4Packet **alloc_ipv4_packets = param->alloc_ipv4_packets + param->packet_offset_per_node[tid];

//         int node_num = param->node_num;

//         int ipv4_offset = 0;
//         for (int i = 0; i < TransportProtocolType::COUNT_TransportProtocolType; i++)
//         {
//             GPUQueue<uint8_t *> *l4_queue = l4_packet_queues[i * node_num];
//             int l4_queue_size = l4_queue->size;

//             int src_ip_offset = param->l4_src_ip_offset[i];
//             int dst_ip_offset = param->l4_dst_ip_offset[i];
//             int timestamp_offset = param->l4_timestamp_offset[i];
//             int packet_size = param->l4_packet_size[i];

//             l4_swap_out_packets = l4_swap_out_packets + i * node_num;
//             l4_cache_space = l4_cache_space + i * node_num;

//             for (int j = 0; j < l4_queue_size; j++)
//             {
//                 // fill ipv4 packet
//                 VDES::Ipv4Packet *ipv4_packet = alloc_ipv4_packets[ipv4_offset];
//                 ipv4_offset++;

//                 uint8_t *l4_packet = l4_queue->get_element(j);

//                 // fill ips
//                 memcpy(ipv4_packet->src_ip, l4_packet + src_ip_offset, 4);
//                 memcpy(ipv4_packet->dst_ip, l4_packet + dst_ip_offset, 4);

//                 // fill timestamp
//                 memcpy(ipv4_packet->timestamp, l4_packet + timestamp_offset, 8);

//                 // fill protocol
//                 ipv4_packet->protocol = i;

//                 // fill payload, swap out l4 packet1
//                 memcpy(ipv4_packet->payload, l4_swap_out_packets + j, 8);
//                 memcpy(l4_cache_space + j, l4_packet, packet_size);
//             }

//             l4_swap_out_packet_num[i * node_num + tid] = l4_queue_size;

//             l4_queue->clear();
//         }

//         param->used_packet_num_per_node[tid] = ipv4_offset;
//         ipv4_queue->append_elements(alloc_ipv4_packets, ipv4_offset);
//     }
// }

// namespace VDES
// {
//     void LaunchIPv4EncapsulationKernel(dim3 grid, dim3 block, IPv4EncapsulationParams *params, cudaStream_t stream)
//     {
//         ipv4_encapsulation<<<grid, block, 0, stream>>>(params);
//     }

// }
