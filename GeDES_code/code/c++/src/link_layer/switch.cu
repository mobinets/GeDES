#include "switch.h"

__device__ bool is_same_mac(uint8_t *mac1, uint8_t *mac2)
{
    for (int i = 0; i < 6; i++)
    {
        if (mac1[i] != mac2[i])
        {
            return false;
        }
    }
    return true;
}

__device__ void lookup_mac_table(GPUQueue<VDES::Frame *> *ingress, GPUQueue<VDES::MacForwardingRule *> *table, int queue_num, int *receive_packet_num)
{
    int tab_size = table->size;
    int size = ingress->size;

    for (int i = 0; i < size; i++)
    {
        VDES::Frame *frame = ingress->get_element(i);

        // if don't have mac table, forward to error queue
        frame->port = queue_num;
        for (int j = 0; j < tab_size; j++)
        {
            if (is_same_mac(frame->dst_mac, table->get_element(j)->mac))
            {
                frame->port = table->get_element(j)->port;
                receive_packet_num[frame->port]++;
                break;
            }
        }
    }
}

__device__ void ft_forward_frames(VDES::SwitchParams *params, GPUQueue<VDES::Frame *> *ingress, int node_id, int *receive_packet_num)
{
    int half_ft_k = params->ft_k / 2;
    int ft_k = params->ft_k;
    int ft_k_sq_quarter = params->ft_k_sq_quarter;
    int sw_id = params->sw_id_per_node[node_id];
    int sw_pod = sw_id / ft_k;
    int sw_inner_id = sw_id % params->ft_k;
    uint8_t current_port_up_forward = params->ft_current_port_up_forward[node_id];

    int size = ingress->size;

    if (sw_pod >= ft_k)
    {
        for (int i = 0; i < size; i++)
        {
            VDES::Frame *frame = ingress->get_element(i);
            uint32_t dst_node_id;
            memcpy(&dst_node_id, frame->dst_mac, sizeof(uint32_t));

            frame->port = dst_node_id / ft_k_sq_quarter;
            receive_packet_num[frame->port]++;
        }
    }
    else if (sw_inner_id >= half_ft_k)
    {
        for (int i = 0; i < size; i++)
        {
            VDES::Frame *frame = ingress->get_element(i);
            uint32_t dst_node_id;
            memcpy(&dst_node_id, frame->dst_mac, sizeof(uint32_t));

            int dst_pod_id = dst_node_id / ft_k_sq_quarter;
            uint port_id = 0;
            if (dst_pod_id == sw_pod)
            {
                port_id = (dst_node_id % ft_k_sq_quarter) / half_ft_k;
            }
            else
            {
                port_id = (current_port_up_forward + dst_node_id) % half_ft_k + half_ft_k;
                // current_port_up_forward = (current_port_up_forward + 1) % half_ft_k;
            }
            frame->port = port_id;
            receive_packet_num[port_id]++;
        }
        params->ft_current_port_up_forward[node_id] = (current_port_up_forward + 1) % half_ft_k;
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            VDES::Frame *frame = ingress->get_element(i);
            uint32_t dst_node_id;
            memcpy(&dst_node_id, frame->dst_mac, sizeof(uint32_t));

            int dst_pod_id = dst_node_id / ft_k_sq_quarter;
            int dst_sw_inner_id = (dst_node_id % ft_k_sq_quarter) / half_ft_k;

            uint8_t port_id = 0;
            if (dst_pod_id == sw_pod && dst_sw_inner_id == sw_inner_id)
            {
                /* code */
                port_id = dst_node_id % half_ft_k;
            }
            else
            {
                port_id = (current_port_up_forward + dst_node_id) % half_ft_k + half_ft_k;
                // current_port_up_forward = (current_port_up_forward + 1) % half_ft_k;
            }
            frame->port = port_id;
            receive_packet_num[port_id]++;
        }
        params->ft_current_port_up_forward[node_id] = (current_port_up_forward + 1) % half_ft_k;
    }
}

__device__ void calculate_frame_offset_in_egress(int *receive_packet_num, int queue_num, int inner_queue_id)
{

    int last_packet_num = 0;
    int offset = 0;
    for (int j = 0; j < queue_num; j++)
    {
        int index = j * (queue_num + 1) + inner_queue_id;
        last_packet_num = receive_packet_num[index];
        receive_packet_num[index] = offset;
        offset += last_packet_num;
    }
    receive_packet_num[inner_queue_id * (queue_num + 1) + queue_num] = offset;
}

__device__ void sort_frames_switch(GPUQueue<VDES::Frame *> *egress, int origin_size)
{
    int size = egress->size;
    int64_t timestamp1;
    int64_t timestamp2;
    for (int i = origin_size; i < size; i++)
    {
        int min_index = i;
        VDES::Frame *tmp_frame = egress->get_element(i);
        memcpy(&timestamp1, egress->get_element(i)->timestamp, 8);
        for (int j = i + 1; j < size; j++)
        {
            memcpy(&timestamp2, egress->get_element(j)->timestamp, 8);
            if (timestamp1 > timestamp2)
            {
                timestamp1 = timestamp2;
                min_index = j;
            }
        }
        if (min_index != i)
        {
            VDES::Frame *temp = egress->get_element(i);
            egress->set_element(i, egress->get_element(min_index));
            egress->set_element(min_index, temp);
        }
    }
}

__global__ void forward_frames(VDES::SwitchParams *params)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < params->queue_num)
    {
        int node_id = params->node_offset_per_queue[tid];
        int inner_id = tid - params->queue_offset_per_node[node_id];
        int current_queue_num = params->queue_num_per_node[node_id];
        int *receive_packet_num = params->received_packets_per_queue[node_id] + inner_id * (current_queue_num + 1);

        memset(receive_packet_num, 0, sizeof(int) * (current_queue_num + 1));
        GPUQueue<VDES::Frame *> *ingress = params->ingresses[tid];

#if ENABLE_FATTREE_MODE
        ft_forward_frames(params, ingress, node_id, receive_packet_num);
#else
        int sw_id = params->sw_id_per_node[node_id];
        lookup_mac_table(ingress, params->mac_forwarding_table[sw_id], current_queue_num, receive_packet_num);
#endif

        // computing the offset to insert packets into the egress queue
        GPUQueue<VDES::Frame *> *egress = params->egresses[tid];
        int orgin_size = egress->size;
        __syncthreads();
        calculate_frame_offset_in_egress(params->received_packets_per_queue[node_id], current_queue_num, inner_id);
        __syncthreads();

        // insert pacekts into egress queue
        GPUQueue<VDES::Frame *> **egresses = params->egresses + params->queue_offset_per_node[node_id];

        int drop_frame_num = 0;
        VDES::Frame **drop_frames = params->drop_frames + tid * MAX_TRANSMITTED_PACKET_NUM;
        VDES::RecycleFramePayload *drop_cache = params->drop_cache + tid * MAX_TRANSMITTED_PACKET_NUM;
        int size = ingress->size;

#if ENABLE_DCTCP
        // int dctcp_k = DCTCP_K + MAX_TRANSMITTED_PACKET_NUM;
        int dctcp_k = DCTCP_K;

#endif

        for (int i = 0; i < size; i++)
        {
            VDES::Frame *frame = ingress->get_element(i);
            uint8_t port = frame->port;
            if (port < current_queue_num && egresses[port]->get_remaining_capacity() > receive_packet_num[port])
            {
                // forward egress to the corresponding egress queue
                egresses[port]->append_element(receive_packet_num[port], frame);
#if ENABLE_DCTCP
                if (egresses[port]->size > dctcp_k)
                {
                    frame->fcs[0] = 1;
                }
#endif
                receive_packet_num[port]++;
            }
            else
            {
                // drop the frame
                drop_frames[drop_frame_num] = frame;
                drop_cache[drop_frame_num].protocol = frame->type[1];
                memcpy(&drop_cache[drop_frame_num].payload, frame->data, 8);
                drop_frame_num++;
            }
        }
        params->drop_frame_num[tid] = drop_frame_num;
        // update egress queue size
        __syncthreads();

        egress->size = min(egress->size + receive_packet_num[current_queue_num], egress->queue_capacity);
        ingress->remove_elements(ingress->size);
        sort_frames_switch(egress, orgin_size);
    }
}

namespace VDES
{

    void LaunchForwardFramesKernel(dim3 grid, dim3 block, SwitchParams *params, cudaStream_t stream)
    {
        forward_frames<<<grid, block, 0, stream>>>(params);
    }
}

// __device__ bool is_same_mac(uint8_t *mac1, uint8_t *mac2)
// {
//     for (int i = 0; i < 6; i++)
//     {
//         if (mac1[i] != mac2[i])
//         {
//             return false;
//         }
//     }
//     return true;
// }

// __device__ void lookup_mac_table(GPUQueue<VDES::Frame *> *ingress, GPUQueue<VDES::MacForwardingRule *> *table, int queue_num, int *receive_packet_num)
// {
//     int tab_size = table->size;
//     int size = ingress->size;

//     for (int i = 0; i < size; i++)
//     {
//         VDES::Frame *frame = ingress->get_element(i);

//         // if don't have mac table, forward to error queue
//         frame->port = queue_num;
//         for (int j = 0; j < tab_size; j++)
//         {
//             if (is_same_mac(frame->dst_mac, table->get_element(j)->mac))
//             {
//                 frame->port = table->get_element(j)->port;
//                 receive_packet_num[frame->port]++;
//                 break;
//             }
//         }
//     }
// }

// __device__ void ft_forward_frames(VDES::SwitchParams *params, GPUQueue<VDES::Frame *> *ingress, int node_id, int *receive_packet_num)
// {
//     int half_ft_k = params->ft_k / 2;
//     int ft_k = params->ft_k;
//     int ft_k_sq_quarter = params->ft_k_sq_quarter;
//     int sw_id = params->sw_id_per_node[node_id];
//     int sw_pod = sw_id / ft_k;
//     int sw_inner_id = sw_id % params->ft_k;
//     uint8_t current_port_up_forward = params->ft_current_port_up_forward[node_id];

//     int size = ingress->size;

//     if (sw_pod >= ft_k)
//     {
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Frame *frame = ingress->get_element(i);
//             uint32_t dst_node_id;
//             memcpy(&dst_node_id, frame->dst_mac, sizeof(uint32_t));

//             frame->port = dst_node_id / ft_k_sq_quarter;
//             receive_packet_num[frame->port]++;
//         }
//     }
//     else if (sw_inner_id >= half_ft_k)
//     {
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Frame *frame = ingress->get_element(i);
//             uint32_t dst_node_id;
//             memcpy(&dst_node_id, frame->dst_mac, sizeof(uint32_t));

//             int dst_pod_id = dst_node_id / ft_k_sq_quarter;
//             uint port_id = 0;
//             if (dst_pod_id == sw_pod)
//             {
//                 port_id = (dst_node_id % ft_k_sq_quarter) / half_ft_k;
//             }
//             else
//             {
//                 port_id = current_port_up_forward + half_ft_k;
//                 current_port_up_forward = (current_port_up_forward + 1) % half_ft_k;
//             }
//             frame->port = port_id;
//             receive_packet_num[port_id]++;
//         }
//         params->ft_current_port_up_forward[node_id] = current_port_up_forward;
//     }
//     else
//     {
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Frame *frame = ingress->get_element(i);
//             uint32_t dst_node_id;
//             memcpy(&dst_node_id, frame->dst_mac, sizeof(uint32_t));

//             int dst_pod_id = dst_node_id / ft_k_sq_quarter;
//             int dst_sw_inner_id = (dst_node_id % ft_k_sq_quarter) / half_ft_k;

//             uint8_t port_id = 0;
//             if (dst_pod_id == sw_pod && dst_sw_inner_id == sw_inner_id)
//             {
//                 /* code */
//                 port_id = dst_node_id % half_ft_k;
//             }
//             else
//             {
//                 port_id = current_port_up_forward + half_ft_k;
//                 current_port_up_forward = (current_port_up_forward + 1) % half_ft_k;
//             }
//             frame->port = port_id;
//             receive_packet_num[port_id]++;
//         }
//         params->ft_current_port_up_forward[node_id] = current_port_up_forward;
//     }
// }

// __device__ void calculate_frame_offset_in_egress(int *receive_packet_num, int queue_num, int inner_queue_id)
// {

//     int last_packet_num = 0;
//     int offset = 0;
//     for (int j = 0; j < queue_num; j++)
//     {
//         int index = j * (queue_num + 1) + inner_queue_id;
//         last_packet_num = receive_packet_num[index];
//         receive_packet_num[index] = offset;
//         offset += last_packet_num;
//     }
//     receive_packet_num[inner_queue_id * (queue_num + 1) + queue_num] = offset;
// }

// __device__ void sort_frames_switch(GPUQueue<VDES::Frame *> *egress, int origin_size)
// {
//     int size = egress->size;
//     int64_t timestamp1;
//     int64_t timestamp2;
//     for (int i = origin_size; i < size; i++)
//     {
//         int min_index = i;
//         VDES::Frame *tmp_frame = egress->get_element(i);
//         memcpy(&timestamp1, egress->get_element(i)->timestamp, 8);
//         for (int j = i + 1; j < size; j++)
//         {
//             memcpy(&timestamp2, egress->get_element(j)->timestamp, 8);
//             if (timestamp1 > timestamp2)
//             {
//                 timestamp1 = timestamp2;
//                 min_index = j;
//             }
//         }
//         if (min_index != i)
//         {
//             VDES::Frame *temp = egress->get_element(i);
//             egress->set_element(i, egress->get_element(min_index));
//             egress->set_element(min_index, temp);
//         }
//     }
// }

// __global__ void forward_frames(VDES::SwitchParams *params)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < params->queue_num)
//     {
//         int node_id = params->node_offset_per_queue[tid];
//         int inner_id = tid - params->queue_offset_per_node[node_id];
//         int current_queue_num = params->queue_num_per_node[node_id];
//         int *receive_packet_num = params->received_packets_per_queue[node_id] + inner_id * (current_queue_num + 1);
//         int sw_id = params->sw_id_per_node[node_id];

//         memset(receive_packet_num, 0, sizeof(int) * (current_queue_num + 1));
//         GPUQueue<VDES::Frame *> *ingress = params->ingresses[tid];

// #if ENABLE_FATTREE_MODE
//         ft_forward_frames(params, ingress, node_id, receive_packet_num);
// #else
//         lookup_mac_table(ingress, params->mac_forwarding_table[sw_id], current_queue_num, receive_packet_num);
// #endif

//         // computing the offset to insert packets into the egress queue
//         GPUQueue<VDES::Frame *> *egress = params->egresses[tid];
//         int orgin_size = egress->size;
//         __syncthreads();
//         calculate_frame_offset_in_egress(params->received_packets_per_queue[node_id], current_queue_num, inner_id);
//         __syncthreads();

//         // insert pacekts into egress queue
//         GPUQueue<VDES::Frame *> **egresses = params->egresses + params->queue_offset_per_node[node_id];

//         int drop_frame_num = 0;
//         /**
//          * TODO: USE MAX_TRANSMITTED_PACKET_NUM instead
//          */
//         VDES::Frame **drop_frames = params->drop_frames + tid * MAX_TRANSMITTED_PACKET_NUM;
//         VDES::RecycleFramePayload *drop_cache = params->drop_cache + tid * MAX_TRANSMITTED_PACKET_NUM;

//         int size = ingress->size;

//         for (int i = 0; i < size; i++)
//         {
//             VDES::Frame *frame = ingress->get_element(i);
//             uint8_t port = frame->port;
//             if (port < current_queue_num && egresses[port]->get_remaining_capacity() > receive_packet_num[port])
//             {
//                 // forward egress to the corresponding egress queue
//                 egresses[port]->append_element(receive_packet_num[port], frame);
//                 receive_packet_num[port]++;
//             }
//             else
//             {
//                 // drop the frame
//                 drop_frames[drop_frame_num] = frame;
//                 drop_cache[drop_frame_num].protocol = frame->type[1];
//                 memcpy(&drop_cache[drop_frame_num].payload, frame->data, 8);
//                 drop_frame_num++;
//             }
//         }
//         params->drop_frame_num[tid] = drop_frame_num;
//         // update egress queue size
//         __syncthreads();
//         /**
//          * @warning: Clear the ingress.
//          */
//         egress->size = min(egress->size + receive_packet_num[current_queue_num], egress->queue_capacity);
//         ingress->remove_elements(ingress->size);
//         sort_frames_switch(egress, orgin_size);
//     }
// }

// namespace VDES
// {

//     void LaunchForwardFramesKernel(dim3 grid, dim3 block, SwitchParams *params, cudaStream_t stream)
//     {
//         forward_frames<<<grid, block, 0, stream>>>(params);
//     }
// }

// #include "switch.h"

// typedef bool (*UpForwardPtr)(int, int, int, int);

// __device__ bool is_same_mac(uint8_t *mac1, uint8_t *mac2)
// {
//     for (int i = 0; i < 6; i++)
//     {
//         if (mac1[i] != mac2[i])
//         {
//             return false;
//         }
//     }
//     return true;
// }

// __device__ void lookup_mac_table(GPUQueue<VDES::Frame *> *ingress, GPUQueue<VDES::MacForwardingRule *> *table, int queue_num)
// {
//     int tab_size = table->size;
//     int size = ingress->size;

//     for (int i = 0; i < size; i++)
//     {
//         VDES::Frame *frame = ingress->get_element(i);

//         // if don't have mac table, forward to error queue
//         frame->port = queue_num;
//         for (int j = 0; j < size; j++)
//         {
//             if (is_same_mac(frame->dst_mac, table->get_element(j)->mac))
//             {
//                 frame->port = table->get_element(j)->port;
//                 break;
//             }
//         }
//     }
// }

// __device__ void ft_forward_frames(VDES::SwitchParams *params, GPUQueue<VDES::Frame *> *ingress, int node_id, int *receive_packet_num)
// {
//     int half_ft_k = params->ft_k / 2;
//     int ft_k = params->ft_k;
//     int ft_k_sq_quarter = params->ft_k_sq_quarter;
//     int sw_id = params->sw_id_per_node[node_id];
//     int sw_pod = sw_id / ft_k;
//     int sw_inner_id = sw_id % params->ft_k;
//     uint8_t current_port_up_forward = params->ft_current_port_up_forward[node_id];

//     int size = ingress->size;

//     if (sw_pod > ft_k)
//     {
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Frame *frame = ingress->get_element(i);
//             uint32_t dst_node_id;
//             memcpy(&dst_node_id, frame->dst_mac, sizeof(uint32_t));

//             frame->port = dst_node_id / ft_k_sq_quarter;
//             receive_packet_num[frame->port]++;
//         }
//     }
//     else if (sw_inner_id >= half_ft_k)
//     {
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Frame *frame = ingress->get_element(i);
//             uint32_t dst_node_id;
//             memcpy(&dst_node_id, frame->dst_mac, sizeof(uint32_t));

//             int dst_pod_id = dst_node_id / ft_k_sq_quarter;
//             uint port_id = 0;
//             if (dst_pod_id == sw_pod)
//             {
//                 port_id = (dst_node_id % ft_k_sq_quarter) / half_ft_k;
//             }
//             else
//             {
//                 port_id = current_port_up_forward + half_ft_k;
//                 current_port_up_forward = (current_port_up_forward + 1) % half_ft_k;
//             }
//             frame->port = port_id;
//             receive_packet_num[port_id]++;
//         }
//         params->ft_current_port_up_forward[node_id] = current_port_up_forward;
//     }
//     else
//     {
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Frame *frame = ingress->get_element(i);
//             uint32_t dst_node_id;
//             memcpy(&dst_node_id, frame->dst_mac, sizeof(uint32_t));

//             int dst_pod_id = dst_node_id / ft_k_sq_quarter;
//             int dst_sw_inner_id = (dst_node_id % ft_k_sq_quarter) / half_ft_k;

//             uint8_t port_id = 0;
//             if (dst_node_id == sw_id && dst_sw_inner_id == sw_inner_id)
//             {
//                 /* code */
//                 port_id = dst_node_id % half_ft_k;
//             }
//             else
//             {
//                 port_id = current_port_up_forward + half_ft_k;
//                 current_port_up_forward = (current_port_up_forward + 1) % half_ft_k;
//             }
//             frame->port = port_id;
//             receive_packet_num[port_id]++;
//         }
//         params->ft_current_port_up_forward[node_id] = current_port_up_forward;
//     }
// }

// __device__ void calculate_frame_offset_in_egress(int *receive_packet_num, int queue_num, int inner_queue_id, int remaining_capacity)
// {

//     int last_packet_num = 0;
//     int offset = 0;
//     for (int j = 0; j < queue_num; j++)
//     {
//         int index = j * (queue_num + 1) + inner_queue_id;
//         last_packet_num = receive_packet_num[index];
//         receive_packet_num[index] = offset;
//         offset += last_packet_num;
//     }
//     receive_packet_num[inner_queue_id * (queue_num + 1) + queue_num] = offset;
// }

// __device__ void sort_frames(GPUQueue<VDES::Frame *> *egress, int origin_size)
// {
//     int size = egress->size;
//     int64_t timestamp1;
//     int64_t timestamp2;
//     for (int i = origin_size; i < size; i++)
//     {
//         int min_index = i;
//         memcpy(&timestamp1, egress->get_element(i)->timestamp, 8);
//         for (int j = i + 1; j < size; j++)
//         {
//             memcpy(&timestamp2, egress->get_element(j)->timestamp, 8);
//             if (timestamp1 > timestamp2)
//             {
//                 timestamp1 = timestamp2;
//                 min_index = j;
//             }
//         }
//         if (min_index != i)
//         {
//             VDES::Frame *temp = egress->get_element(i);
//             egress->set_element(i, egress->get_element(min_index));
//             egress->set_element(min_index, temp);
//         }
//     }
// }

// __global__ void forward_frames(VDES::SwitchParams *params)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < params->queue_num)
//     {
//         int node_id = params->node_offset_per_queue[tid];
//         int inner_id = tid - params->queue_offset_per_node[node_id];
//         int current_queue_num = params->queue_num_per_node[node_id];
//         int *receive_packet_num = params->received_packets_per_queue[node_id] + inner_id * (current_queue_num + 1);
//         int sw_id = params->sw_id_per_node[node_id];

//         memset(receive_packet_num, 0, sizeof(int) * (current_queue_num + 1));
//         GPUQueue<VDES::Frame *> *ingress = params->ingresses[tid];

//         if (ENABLE_FATTREE_MODE == 1)
//         {
//             ft_forward_frames(params, ingress, node_id, receive_packet_num);
//         }
//         else
//         {
//             lookup_mac_table(ingress, params->mac_forwarding_table[sw_id], current_queue_num);
//         }

//         // computing the offset to insert packets into the egress queue
//         GPUQueue<VDES::Frame *> *egress = params->egresses[tid];
//         __threadfence();
//         calculate_frame_offset_in_egress(params->received_packets_per_queue[node_id], current_queue_num, inner_id, egress->get_remaining_capacity());
//         __threadfence();

//         // insert pacekts into egress queue
//         GPUQueue<VDES::Frame *> **egresses = params->egresses + params->queue_offset_per_node[node_id];
//         int drop_frame_num = 0;
//         VDES::Frame **drop_frames = params->drop_frames + tid * MAX_GENERATED_PACKET_NUM;

//         int size = ingress->size;
//         for (int i = 0; i < size; i++)
//         {
//             VDES::Frame *frame = ingress->get_element(i);
//             uint8_t port = frame->port;
//             if (port < current_queue_num && egresses[port]->get_remaining_capacity() > receive_packet_num[port])
//             {
//                 // forward egress to the corresponding egress queue
//                 egresses[port]->set_element(receive_packet_num[port], frame);
//                 receive_packet_num[port]++;
//             }
//             else
//             {
//                 // drop the frame
//                 drop_frames[drop_frame_num] = frame;
//                 drop_frame_num++;
//             }
//         }
//         // update egress queue size
//         egress->size = min(egress->size + receive_packet_num[current_queue_num], egress->queue_capacity);

//         params->drop_frame_num[tid] = drop_frame_num;
//     }
// }

// namespace VDES
// {

//     void LaunchForwardFramesKernel(dim3 grid, dim3 block, SwitchParams *params, cudaStream_t stream)
//     {
//         forward_frames<<<grid, block, 0, stream>>>(params);
//     }
// }