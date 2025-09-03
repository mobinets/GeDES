#include "ipv4_controller.h"

__device__ void lookup_routing_table(VDES::Ipv4Packet *packet, GPUQueue<VDES::IPv4RoutingRule *> *routing_table, int table_size)
{
    uint32_t dst_ip;
    memcpy(&dst_ip, packet->dst_ip, sizeof(uint32_t));

    uint8_t err_code = (uint8_t)VDES::RoutingError::DESTINATION_UNRECHEABLE;
    for (int i = 0; i < table_size; i++)
    {
        VDES::IPv4RoutingRule *rule = routing_table->get_element(i);
        if ((rule->dst & rule->mask) == (dst_ip & rule->mask))
        {
#if ENABLE_FATTREE_MODE
            // simplify routing proceeding for experiments
            if (rule->gw == 0)
            {
                memcpy(packet->next_hop, &rule->gw, sizeof(uint32_t));
            }
            else
            {
                memcpy(packet->next_hop, packet->dst_ip, sizeof(uint32_t));
            }
#else
            memcpy(packet->next_hop, &rule->gw, sizeof(uint32_t));
#endif

            packet->next_if = rule->if_id;
            err_code = (uint8_t)VDES::RoutingError::NO_ERROR;
            break;
        }
    }

    packet->err_code = err_code;
}

__global__ void routing_ipv4_packets(VDES::IPv4Params *param)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int node_id = tid % param->node_num;
    int inner_id = tid / param->node_num;

    int th_num = blockDim.x * gridDim.x;
    int collaborate_th_num = (th_num + param->node_num - node_id - 1) / param->node_num;

    GPUQueue<VDES::Ipv4Packet *> *ingress = param->ingresses[node_id];
    GPUQueue<VDES::IPv4RoutingRule *> *routing_table = param->routing_tables[node_id];
    int routing_table_size = routing_table->size;

    int size = ingress->size;

    for (int i = inner_id; i < size; i += collaborate_th_num)
    {
        VDES::Ipv4Packet *packet = ingress->get_element(i);
        // lookup routing table
        lookup_routing_table(packet, routing_table, routing_table_size);
        uint32_t next_hop;
        memcpy(&next_hop, packet->next_hop, sizeof(uint32_t));
        if (packet->err_code == (uint8_t)VDES::RoutingError::NO_ERROR && next_hop != 0 && packet->time_to_live == 0)
        {
            packet->err_code = (uint8_t)VDES::RoutingError::TTL_EXPIRED;
        }
    }
}

__device__ void handle_routing_errors(GPUQueue<VDES::Ipv4Packet *> *error_queue, VDES::IPv4Params *param)
{
    // do nothing but only remove error packets from error queue
    error_queue->clear();
}

__global__ void forward_ipv4_packets(VDES::IPv4Params *param)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < param->node_num)
    {
        // ingress queue
        GPUQueue<VDES::Ipv4Packet *> *ingress = param->ingresses[tid];

        // egress queue
        int offset = param->egress_offset_per_node[tid];
        GPUQueue<VDES::Ipv4Packet *> **egress = param->egresses + offset;

        // error queue
        GPUQueue<VDES::Ipv4Packet *> *error_queue = param->error_queues[tid];

        // local delivery queue
        GPUQueue<VDES::Ipv4Packet *> *local_delivery = param->local_egresses[tid];

        // forwarding
        int size = ingress->size;
        // int swap_num = 0;
        for (int i = 0; i < size; i++)
        {
            VDES::Ipv4Packet *packet = ingress->get_element(i);
            GPUQueue<VDES::Ipv4Packet *> *dst;

            if (packet->err_code == (uint8_t)VDES::RoutingError::NO_ERROR)
            {
                // forward to egress
                uint32_t next_hop;
                memcpy(&next_hop, packet->next_hop, sizeof(uint32_t));
                if (next_hop == 0)
                {
                    // local delivery
                    dst = local_delivery;
                }
                else
                {
                    // forward to next hop
                    dst = egress[packet->next_if];
                    packet->time_to_live--;
                }
            }
            else
            {
                // insert to error queue
                dst = error_queue;
            }
            // append to dst queue
            dst->append_element(packet);
        }

        // clear ingress queues
        ingress->clear();

        // update egress remaining capacity, for tcp layer
        int egress_num = param->egress_num_per_node[tid];
        int remaining_capacity = 0;
        for (int i = 0; i < egress_num; i++)
        {
            remaining_capacity += egress[i]->get_remaining_capacity();
        }
        param->egress_remaining_capacity[tid] = remaining_capacity;

        // handle error packets
        handle_routing_errors(error_queue, param);
    }
}

namespace VDES
{
    void LaunchRoutingIPv4PacketsKernel(dim3 grid_dim, dim3 block_dim, IPv4Params *params, cudaStream_t stream)
    {
        routing_ipv4_packets<<<grid_dim, block_dim, 0, stream>>>(params);
    }

    void LaunchForwardIPv4PacketsKernel(dim3 grid_dim, dim3 block_dim, IPv4Params *params, cudaStream_t stream)
    {
        forward_ipv4_packets<<<grid_dim, block_dim, 0, stream>>>(params);
    }

}

// __device__ void lookup_routing_table(VDES::Ipv4Packet *packet, GPUQueue<VDES::IPv4RoutingRule *> *routing_table, int table_size)
// {
//     uint32_t dst_ip;
//     memcpy(&dst_ip, packet->dst_ip, sizeof(uint32_t));

//     uint8_t err_code = (uint8_t)VDES::RoutingError::DESTINATION_UNRECHEABLE;
//     for (int i = 0; i < table_size; i++)
//     {
//         VDES::IPv4RoutingRule *rule = routing_table->get_element(i);
//         if ((rule->dst & rule->mask) == (dst_ip & rule->mask))
//         {
//             if (ENABLE_FATTREE_MODE)
//             {
//                 // simplify routing proceeding for experiments
//                 if (rule->gw == 0)
//                 {
//                     memcpy(packet->next_hop, &rule->gw, sizeof(uint32_t));
//                 }
//                 else
//                 {
//                     memcpy(packet->next_hop, packet->dst_ip, sizeof(uint32_t));
//                 }
//             }
//             else
//             {
//                 memcpy(packet->next_hop, &rule->gw, sizeof(uint32_t));
//             }

//             packet->next_if = rule->if_id;
//             err_code = (uint8_t)VDES::RoutingError::NO_ERROR;
//             break;
//         }
//     }

//     packet->err_code = err_code;
// }

// __global__ void routing_ipv4_packets(VDES::IPv4Params *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int node_id = tid % param->node_num;
//     int inner_id = tid / param->node_num;

//     int th_num = blockDim.x * gridDim.x;
//     int collaborate_th_num = (th_num + param->node_num - node_id - 1) / param->node_num;

//     GPUQueue<VDES::Ipv4Packet *> *ingress = param->ingresses[node_id];
//     GPUQueue<VDES::IPv4RoutingRule *> *routing_table = param->routing_tables[node_id];
//     int routing_table_size = routing_table->size;

//     int size = ingress->size;

//     for (int i = inner_id; i < size; i += collaborate_th_num)
//     {
//         VDES::Ipv4Packet *packet = ingress->get_element(i);
//         // lookup routing table

//         lookup_routing_table(packet, routing_table, routing_table_size);
//         uint32_t next_hop;
//         memcpy(&next_hop, packet->next_hop, sizeof(uint32_t));
//         if (packet->err_code == (uint8_t)VDES::RoutingError::NO_ERROR && next_hop != 0 && packet->time_to_live == 0)
//         {
//             packet->err_code = (uint8_t)VDES::RoutingError::TTL_EXPIRED;
//         }
//     }
// }

// __device__ void handle_routing_errors(GPUQueue<VDES::Ipv4Packet *> *error_queue, VDES::IPv4Params *param)
// {
//     // do nothing but only remove error packets from error queue
//     error_queue->clear();
// }

// __global__ void forward_ipv4_packets(VDES::IPv4Params *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->node_num)
//     {
//         // ingress queue
//         GPUQueue<VDES::Ipv4Packet *> *ingress = param->ingresses[tid];

//         // egress queue
//         int offset = param->egress_offset_per_node[tid];
//         GPUQueue<VDES::Ipv4Packet *> **egress = param->egresses + offset;

//         // error queue
//         GPUQueue<VDES::Ipv4Packet *> *error_queue = param->error_queues[tid];

//         // local delivery queue
//         GPUQueue<VDES::Ipv4Packet *> *local_delivery = param->local_egresses[tid];

//         // swap out cache space
//         // VDES::Ipv4Packet **swap_out_packets = param->swap_out_packets + param->swap_offset_per_node[tid];
//         // VDES::Ipv4Packet *swap_out_cache = param->swap_out_cache_space + param->swap_offset_per_node[tid];

//         // forwarding
//         int size = ingress->size;
//         // int swap_num = 0;
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Ipv4Packet *packet = ingress->get_element(i);
//             // int64_t ts = ingress_ts->get_element(i);
//             GPUQueue<VDES::Ipv4Packet *> *dst;
//             GPUQueue<int64_t> *dst_ts;

//             if (packet->err_code == (uint8_t)VDES::RoutingError::NO_ERROR)
//             {
//                 // forward to egress
//                 uint32_t next_hop;
//                 memcpy(&next_hop, packet->next_hop, sizeof(uint32_t));
//                 if (next_hop == 0)
//                 {
//                     // local delivery
//                     dst = local_delivery;
//                 }
//                 else
//                 {
//                     // forward to next hop
//                     dst = egress[packet->next_if];
//                     packet->time_to_live--;

//                     // cache ipv4 packets
//                     // memcpy(swap_out_cache + swap_num, packet, sizeof(VDES::Ipv4Packet));
//                     // packet = swap_out_packets[swap_num];
//                     // swap_num++;
//                 }
//             }
//             else
//             {
//                 // insert to error queue
//                 dst = error_queue;
//             }
//             // append to dst queue
//             dst->append_element(packet);
//         }

//         // param->swap_out_num[tid] = swap_num;

//         // clear ingress queues
//         ingress->clear();

//         // update egress remaining capacity, for tcp layer
//         int egress_num = param->egress_num_per_node[tid];
//         int remaining_capacity = 0;
//         for (int i = 0; i < egress_num; i++)
//         {
//             remaining_capacity += egress[i]->get_remaining_capacity();
//         }
//         param->egress_remaining_capacity[tid] = remaining_capacity;

//         // handle error packets
//         handle_routing_errors(error_queue, param);
//     }
// }

// namespace VDES
// {
//     void LaunchRoutingIPv4PacketsKernel(dim3 grid_dim, dim3 block_dim, IPv4Params *params, cudaStream_t stream)
//     {
//         routing_ipv4_packets<<<grid_dim, block_dim, 0, stream>>>(params);
//     }

//     void LaunchForwardIPv4PacketsKernel(dim3 grid_dim, dim3 block_dim, IPv4Params *params, cudaStream_t stream)
//     {
//         forward_ipv4_packets<<<grid_dim, block_dim, 0, stream>>>(params);
//     }

// }

// #include "ipv4_controller.h"

// __device__ void lookup_routing_table(VDES::Ipv4Packet *packet, GPUQueue<VDES::IPv4RoutingRule *> *routing_table, int table_size)
// {
//     uint32_t dst_ip;
//     memcpy(&dst_ip, packet->dst_ip, sizeof(uint32_t));

//     uint8_t err_code = (uint8_t)VDES::RoutingError::DESTINATION_UNRECHEABLE;
//     for (int i = 0; i < table_size; i++)
//     {
//         VDES::IPv4RoutingRule *rule = routing_table->get_element(i);
//         if ((rule->dst & rule->mask) == (dst_ip & rule->mask))
//         {
//             memcpy(packet->next_hop, &rule->gw, sizeof(uint32_t));
//             packet->next_if = rule->if_id;
//             err_code = (uint8_t)VDES::RoutingError::NO_ERROR;
//             break;
//         }
//     }

//     packet->err_code = err_code;
// }

// __global__ void routing_ipv4_packets(VDES::IPv4Params *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int node_id = tid % param->node_num;
//     int inner_id = tid / param->node_num;

//     GPUQueue<VDES::Ipv4Packet *> *ingress = param->ingresses[node_id];
//     GPUQueue<int64_t> *ingress_ts = param->ingress_tss[node_id];
//     GPUQueue<VDES::IPv4RoutingRule *> *routing_table = param->routing_tables[node_id];
//     int routing_table_size = routing_table->size;

//     int size = ingress->size;

//     for (int i = inner_id; i < size; i += param->node_num)
//     {
//         VDES::Ipv4Packet *packet = ingress->get_element(i);
//         // lookup routing table
//         lookup_routing_table(packet, routing_table, routing_table_size);
//         uint32_t next_hop;
//         memcpy(&next_hop, packet->next_hop, sizeof(uint32_t));
//         if (packet->err_code != (uint8_t)VDES::RoutingError::NO_ERROR || next_hop != 0)
//         {
//             packet->err_code = (uint8_t)VDES::RoutingError::TTL_EXPIRED;
//         }
//     }
// }

// __device__ void handle_routing_errors(GPUQueue<VDES::Ipv4Packet *> *error_queue, GPUQueue<int64_t> *error_ts, VDES::IPv4Params *param)
// {
//     // do nothing but only remove error packets from error queue
//     error_queue->clear();
//     error_ts->clear();
// }

// __global__ void forward_ipv4_packets(VDES::IPv4Params *param)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < param->node_num)
//     {
//         // ingress queue
//         GPUQueue<VDES::Ipv4Packet *> *ingress = param->ingresses[tid];
//         GPUQueue<int64_t> *ingress_ts = param->ingress_tss[tid];

//         // egress queue
//         int offset = param->egress_offset_per_node[tid];
//         GPUQueue<VDES::Ipv4Packet *> **egress = param->egresses + offset;
//         GPUQueue<int64_t> **egress_ts = param->egress_tss + offset;
//         GPUQueue<uint32_t> **egress_ip = param->egress_ip + offset;

//         // error queue
//         GPUQueue<VDES::Ipv4Packet *> *error_queue = param->error_queues[tid];
//         GPUQueue<int64_t> *error_ts = param->error_queue_tss[tid];

//         // local delivery queue
//         GPUQueue<VDES::Ipv4Packet *> *local_delivery = param->local_egresses[tid];
//         GPUQueue<int64_t> *local_delivery_ts = param->local_egress_tss[tid];

//         // swap out cache space
//         VDES::Ipv4Packet **swap_out_packets = param->swap_out_packets + param->swap_offset_per_node[tid];
//         VDES::Ipv4Packet *swap_out_cache = param->swap_out_cache_space + param->swap_offset_per_node[tid];

//         // forwarding
//         int size = ingress->size;
//         int swap_num = 0;
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Ipv4Packet *packet = ingress->get_element(i);
//             int64_t ts = ingress_ts->get_element(i);
//             GPUQueue<VDES::Ipv4Packet *> *dst;
//             GPUQueue<int64_t> *dst_ts;

//             if (packet->err_code == (uint8_t)VDES::RoutingError::NO_ERROR)
//             {
//                 // forward to egress
//                 uint32_t next_hop;
//                 memcpy(&next_hop, packet->next_hop, sizeof(uint32_t));
//                 if (next_hop == 0)
//                 {
//                     // local delivery
//                     dst_ts = local_delivery_ts;
//                     dst = local_delivery;
//                 }
//                 else
//                 {
//                     // forward to next hop
//                     dst_ts = egress_ts[packet->next_if];
//                     dst = egress[packet->next_if];
//                     egress_ip[packet->next_if]->append_element(next_hop);
//                     packet->time_to_live--;

//                     // cache ipv4 packets
//                     memcpy(swap_out_cache + swap_num, packet, sizeof(VDES::Ipv4Packet));
//                     // VDES::Ipv4Packet *tmep_packet = packet;
//                     packet = swap_out_packets[swap_num];
//                     // swap_out_packets[swap_num] = tmep_packet;
//                     swap_num++;
//                 }
//             }
//             else
//             {
//                 // insert to error queue
//                 dst_ts = error_ts;
//                 dst = error_queue;
//             }
//             // append to dst queue
//             dst->append_element(packet);
//             dst_ts->append_element(ts);
//         }

//         param->swap_out_num[tid] = swap_num;

//         // clear ingress queues
//         ingress->clear();
//         ingress_ts->clear();

//         // handle error packets
//         handle_routing_errors(error_queue, error_ts, param);
//     }
// }

// namespace VDES
// {
//     void LaunchRoutingIPv4PacketsKernel(dim3 grid_dim, dim3 block_dim, IPv4Params *params, cudaStream_t stream)
//     {
//         routing_ipv4_packets<<<grid_dim, block_dim, 0, stream>>>(params);
//     }

//     void LaunchForwardIPv4PacketsKernel(dim3 grid_dim, dim3 block_dim, IPv4Params *params, cudaStream_t stream)
//     {
//         forward_ipv4_packets<<<grid_dim, block_dim, 0, stream>>>(params);
//     }

// }