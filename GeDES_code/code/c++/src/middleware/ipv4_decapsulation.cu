#include "ipv4_decapsulation.h"
#define ECN_OFFSET_TCP 13

__global__ void ipv4_decapsulation(VDES::IPv4DecapsulationParam *param)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < param->node_num)
    {
        int node_num = param->node_num;
        GPUQueue<VDES::Ipv4Packet *> *ipv4_queue = param->ipv4_queues[tid];
        GPUQueue<uint8_t *> **l4_queues = param->l4_queues + tid;
        VDES::Ipv4Packet **recycle_ipv4_packets = param->recycle_ipv4_packets + param->recycle_offset_per_node[tid];

#if ENABLE_CACHE
        int *l4_swap_in_packet_num = nullptr;
        uint8_t **l4_swap_in_packets = nullptr;
        int cache_packet_num = 0;
        l4_swap_in_packet_num = param->l4_swap_in_packets_num;
        l4_swap_in_packets = param->l4_swap_in_packets + param->l4_swap_in_offset_per_node[tid];
        cache_packet_num = param->cache_packet_num;

        for (int i = 0; i < TransportProtocolType::COUNT_TransportProtocolType; i++)
        {
            l4_swap_in_packet_num[tid + i * node_num] = 0;
        }
#endif

        int *l4_src_ip_offset = param->l4_src_ip_offset;
        int *l4_dst_ip_offset = param->l4_dst_ip_offset;
        int *l4_timestamp_offset = param->l4_timestamp_offset;

        int size = ipv4_queue->size;

        // classification
        for (int i = 0; i < size; i++)
        {
            VDES::Ipv4Packet *ipv4_packet = ipv4_queue->get_element(i);
            uint8_t protocol = ipv4_packet->protocol;
            uint8_t *l4_packet;
            memcpy(&l4_packet, ipv4_packet->payload, sizeof(void *));
#if ENABLE_CACHE
            if (ipv4_packet->device == (uint8_t)VDES::Device::CPU)
            {
                // cache in
                uint8_t *temp = l4_packet;
                int index = l4_swap_in_packet_num[tid + protocol * node_num] + protocol * cache_packet_num;
                l4_packet = l4_swap_in_packets[index];
                l4_swap_in_packets[index] = temp;
                l4_swap_in_packet_num[tid + protocol * node_num]++;
            }
#endif

            memcpy(l4_packet + l4_src_ip_offset[protocol], ipv4_packet->src_ip, 4);
            memcpy(l4_packet + l4_dst_ip_offset[protocol], ipv4_packet->dst_ip, 4);
            memcpy(l4_packet + l4_timestamp_offset[protocol], ipv4_packet->timestamp, 8);

#if ENABLE_DCTCP
            if (protocol == TransportProtocolType::TCP)
            {
                memcpy(l4_packet + ECN_OFFSET_TCP, &ipv4_packet->dscp_ecn, 1);
            }
#endif

            l4_queues[protocol * node_num]->append_element(l4_packet);
#if !ENABLE_CACHE
            recycle_ipv4_packets[i] = ipv4_packet;
#endif
        }
        ipv4_queue->clear();

#if !ENABLE_CACHE
        param->recycle_ipv4_packets_num[tid] = size;
#endif
    }
}

namespace VDES
{
    void LaunchIPv4DecapsulationKernel(dim3 grid, dim3 block, IPv4DecapsulationParam *param, cudaStream_t stream)
    {
        ipv4_decapsulation<<<grid, block, 0, stream>>>(param);
    }

}

// __global__ void ipv4_decapsulation(VDES::IPv4DecapsulationParam *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->node_num)
//     {
//         int node_num = param->node_num;
//         GPUQueue<VDES::Ipv4Packet *> *ipv4_queue = param->ipv4_queues[tid];
//         GPUQueue<uint8_t *> **l4_queues = param->l4_queues + tid;
//         VDES::Ipv4Packet **recycle_ipv4_packets = param->recycle_ipv4_packets + param->recycle_offset_per_node[tid];

//         int *l4_swap_in_packet_num = NULL;
//         uint8_t **l4_swap_in_packets = NULL;
//         int cache_packet_num = 0;

// #if ENABLE_CACHE

//         l4_swap_in_packet_num = param->l4_swap_in_packets_num;
//         l4_swap_in_packets = param->l4_swap_in_packets + param->l4_swap_in_offset_per_node[tid];
//         cache_packet_num = param->cache_packet_num;

//         for (int i = 0; i < TransportProtocolType::COUNT_TransportProtocolType; i++)
//         {
//             l4_swap_in_packet_num[tid + i * node_num] = 0;
//         }

// #endif

//         int *l4_src_ip_offset = param->l4_src_ip_offset;
//         int *l4_dst_ip_offset = param->l4_dst_ip_offset;
//         int *l4_timestamp_offset = param->l4_timestamp_offset;

//         int size = ipv4_queue->size;

//         // classification
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Ipv4Packet *ipv4_packet = ipv4_queue->get_element(i);
//             uint8_t protocol = ipv4_packet->protocol;
//             uint8_t *l4_packet;
//             memcpy(&l4_packet, ipv4_packet->payload, sizeof(void *));
// #if ENABLE_CACHE
//             if (ipv4_packet->device == (uint8_t)VDES::Device::CPU)
//             {
//                 // cache in
//                 uint8_t *temp = l4_packet;
//                 int index = l4_swap_in_packet_num[tid + protocol * node_num] + protocol * cache_packet_num;
//                 l4_packet = l4_swap_in_packets[index];
//                 l4_swap_in_packets[index] = temp;
//                 l4_swap_in_packet_num[tid + protocol * node_num]++;
//             }
// #endif

//             memcpy(l4_packet + l4_src_ip_offset[protocol], ipv4_packet->src_ip, 4);
//             memcpy(l4_packet + l4_dst_ip_offset[protocol], ipv4_packet->dst_ip, 4);
//             memcpy(l4_packet + l4_timestamp_offset[protocol], ipv4_packet->timestamp, 8);

//             l4_queues[protocol * node_num]->append_element(l4_packet);
// #if !ENABLE_CACHE
//             recycle_ipv4_packets[i] = ipv4_packet;
// #endif
//         }
//         ipv4_queue->clear();
// #if !ENABLE_CACHE
//         param->recycle_ipv4_packets_num[tid] = size;
// #endif
//     }
// }

// namespace VDES
// {
//     void LaunchIPv4DecapsulationKernel(dim3 grid, dim3 block, IPv4DecapsulationParam *param, cudaStream_t stream)
//     {
//         ipv4_decapsulation<<<grid, block, 0, stream>>>(param);
//     }

// }

// #include "ipv4_decapsulation.h"

// __global__ void ipv4_decapsulation(VDES::IPv4DecapsulationParam *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->node_num)
//     {
//         int node_num = param->node_num;
//         GPUQueue<VDES::Ipv4Packet *> *ipv4_queue = param->ipv4_queues[tid];
//         GPUQueue<uint8_t *> **l4_queues = param->l4_queues + tid;
//         VDES::Ipv4Packet **recycle_ipv4_packets = param->recycle_ipv4_packets + param->recycle_offset_per_node[tid];
//         int *l4_swap_in_packet_num = param->l4_swap_in_packets_num;
//         uint8_t **l4_swap_in_packets = param->l4_swap_in_packets + param->l4_swap_in_offset_per_node[tid];
//         int cache_packet_num = param->cache_packet_num;
//         int *l4_src_ip_offset = param->l4_src_ip_offset;
//         int *l4_dst_ip_offset = param->l4_dst_ip_offset;
//         int *l4_timestamp_offset = param->l4_timestamp_offset;

//         int size = ipv4_queue->size;

//         // classification
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Ipv4Packet *ipv4_packet = ipv4_queue->get_element(i);
//             uint8_t protocol = ipv4_packet->protocol;
//             uint8_t *l4_packet;
//             memcpy(&l4_packet, ipv4_packet->payload, sizeof(void *));
//             if (ipv4_packet->device == (uint8_t)VDES::Device::CPU)
//             {
//                 // cache in
//                 uint8_t *temp = l4_packet;
//                 int index = l4_swap_in_packet_num[tid + protocol * node_num] + protocol * cache_packet_num;
//                 l4_packet = l4_swap_in_packets[index];
//                 l4_swap_in_packets[index] = temp;
//                 l4_swap_in_packet_num[tid + protocol * node_num]++;
//             }

//             memcpy(l4_packet + l4_src_ip_offset[protocol], ipv4_packet->src_ip, 4);
//             memcpy(l4_packet + l4_dst_ip_offset[protocol], ipv4_packet->dst_ip, 4);
//             memcpy(l4_packet + l4_timestamp_offset[protocol], &ipv4_packet->timestamp, 8);

//             l4_queues[protocol * node_num]->append_element(l4_packet);
//             recycle_ipv4_packets[i] = ipv4_packet;
//         }
//         ipv4_queue->clear();

//         param->recycle_ipv4_packets_num[tid] = size;
//     }
// }

// namespace VDES
// {
//     void LaunchIPv4DecapsulationKernel(dim3 grid, dim3 block, IPv4DecapsulationParam *param, cudaStream_t stream)
//     {
//         ipv4_decapsulation<<<grid, block, 0, stream>>>(param);
//     }

// }