#include "tcp_controller.h"

__device__ bool is_ack(VDES::TCPPacket *p)
{
    return p->data_offset_reserved_flags[0] & 0x1 == 1;
}

__device__ int classify_tcp_packets(GPUQueue<VDES::TCPPacket *> *recv_queue)
{
    int size = recv_queue->size;
    int offset = size;

    // scan the first non-ack packet
    for (int i = 0; i < size; i++)
    {
        if (!is_ack(recv_queue->get_element(i)))
        {
            offset = i;
            break;
        }
    }

    // swap packets to adhere to the architecture |ack packets| non-ack packets|
    for (int end = offset + 1; end < size; end++)
    {
        if (is_ack(recv_queue->get_element(end)))
        {
            // swap
            VDES::TCPPacket *temp = recv_queue->get_element(offset);
            recv_queue->set_element(offset, recv_queue->get_element(end));
            recv_queue->set_element(end, temp);
            offset++;
        }
    }

    // sort non-ack packets by sequence number
    for (int i = offset; i < size; i++)
    {
        for (int j = i + 1; j < size; j++)
        {
            // try reinforced cast first, if failed, use memcpy or memcmp
            uint32_t seq_pre;
            uint32_t seq_post;
            memcpy(&seq_pre, recv_queue->get_element(i)->sequence_number, sizeof(uint32_t));
            memcpy(&seq_post, recv_queue->get_element(j)->sequence_number, sizeof(uint32_t));

            if (seq_pre > seq_post)
            {
                // swap
                VDES::TCPPacket *temp = recv_queue->get_element(i);
                recv_queue->set_element(i, recv_queue->get_element(j));
                recv_queue->set_element(j, temp);
            }
        }
    }

    return offset;
}

__device__ int get_tcp_connection_id(VDES::TCPPacket *p, VDES::TCPConnection **con, int con_num)
{
    for (int i = 0; i < con_num; i++)
    {
        // try reinforced cast first, if failed, use memcpy or memcmp
        uint32_t src_ip;
        uint32_t dst_ip;
        memcpy(&src_ip, p->src_ip, sizeof(uint32_t));
        memcpy(&dst_ip, p->dst_ip, sizeof(uint32_t));
        uint16_t src_port;
        uint16_t dst_port;
        memcpy(&src_port, p->src_port, sizeof(uint16_t));
        memcpy(&dst_port, p->dst_port, sizeof(uint16_t));
        if (dst_ip == con[i]->src_ip && src_ip == con[i]->dst_ip && src_port == con[i]->src_port && dst_port == con[i]->dst_port)
        {
            return i;
        }
    }

    // non-existing connection
    return -1;
}

__device__ void update_cwnd(VDES::TCPConnection *con, int64_t timeslot_end)
{
    // bool ack_repeat_over_3_times = con->repeat_ack_count >= 3 && con->repeat_ack_count > con->former_repeat_ack_count;
    bool ack_repeat_over_3_times = con->former_repeat_ack_count < 3 && con->repeat_ack_count >= 3;
    bool rto_timeout = (con->retrans_timer + con->rto) < timeslot_end;

    if ((ack_repeat_over_3_times || rto_timeout) && con->planned_bytes > con->acked_bytes)
    {
        // congestion avoidance
        con->ssthresh = con->cwnd >> 1;
        con->cwnd = con->ssthresh + 3 * con->mss;
        con->una = con->snd;
    }
    else
    {
#if !ENABLE_DCTCP
        for (int i = 0; i < con->ack_count; i++)
        {
            if (con->cwnd >= con->ssthresh)
            {
                // linear increase
                con->cwnd = con->cwnd + (con->ack_count - i) * con->mss;
                break;
            }
            else
            {
                // quick recovery
                con->cwnd = min(con->cwnd * 2, con->ssthresh);
            }
        }
#else
        if (con->snd > con->dctcp_window_end && con->dctcp_acked_bytes > 0)
        {
            float alpha = (float)con->dctcp_ecn_masked_bytes / con->dctcp_acked_bytes;
            con->dctcp_alpha = con->dctcp_alpha * (1 - DCTCP_G) + alpha * DCTCP_G;
            con->dctcp_window_end = con->una;
            con->dctcp_acked_bytes = 0;
            con->dctcp_ecn_masked_bytes = 0;
            if (alpha > 0)
            {
                con->cwnd = (uint32_t)(con->cwnd * (1 - con->dctcp_alpha / 2));
            }
        }
        if (con->dctcp_ecn_masked_bytes == 0)
        {
            con->cwnd += (con->ack_count * con->mss);
        }
#endif
        con->ack_count = 0;
    }
    // update restransmission timer
    if (rto_timeout)
    {
        con->retrans_timer = timeslot_end;
    }
    con->former_repeat_ack_count = con->repeat_ack_count;
}

__device__ void fill_ack_packet(VDES::TCPConnection *con, VDES::TCPPacket *ack_packet, int64_t *timestamp)
{
    memcpy(ack_packet->src_port, &con->src_port, 2);
    memcpy(ack_packet->dst_port, &con->dst_port, 2);
    memcpy(ack_packet->src_ip, &con->src_ip, 4);
    memcpy(ack_packet->dst_ip, &con->dst_ip, 4);
    memcpy(ack_packet->acknowledgement_number, &con->expected_seq, 4);

    memcpy(ack_packet->send_timestamp, timestamp, 8);
    ack_packet->data_offset_reserved_flags[0] = ack_packet->data_offset_reserved_flags[0] | 0x1;
    memset(ack_packet->payload_len, 0, 2);
}

__device__ void handle_ack_packets(GPUQueue<VDES::TCPPacket *> *recv_queue, int ack_num, VDES::TCPConnection **cons, int con_num)
{
    for (int i = 0; i < ack_num; i++)
    {
        VDES::TCPPacket *p = recv_queue->get_element(i);
        int tcp_id = get_tcp_connection_id(p, cons, con_num);
        if (tcp_id >= 0)
        {
            uint32_t ack;
            memcpy(&ack, p->acknowledgement_number, 4);
            VDES::TCPConnection *con = cons[tcp_id];
            uint32_t snd = con->snd;
            // if (con->una - snd >= ack - snd && ack != snd)
            uint32_t acked_bytes = ack - snd;
            if (acked_bytes > 0 && acked_bytes <= con->swnd)
            // if (con->una - snd >= ack - snd && ack != snd)
            {
                con->repeat_ack_count = 0;
                con->former_repeat_ack_count = 0;
                // reset retransmission timer
                memcpy(&con->retrans_timer, p->recv_timestamp, 8);
                // printf("%ld\n", con->retrans_timer);

                // update send window
                con->acked_bytes += acked_bytes;
                con->snd = ack;
                con->una = max(con->una, ack);

                if (con->acked_bytes >= con->flows->queue[con->acked_flows].byte_offset)
                {

                    int64_t timestamp_end;
                    memcpy(&timestamp_end, p->recv_timestamp, sizeof(int64_t));
                    con->flows->queue[con->acked_flows].tiimestamp_end = timestamp_end;
                    // printf("%d, %ld\n", con->acked_flows, con->flows->queue[con->acked_flows].tiimestamp_end);
                    con->acked_flows++;
                }

                con->ack_count++;

#if ENABLE_DCTCP
                con->dctcp_acked_bytes += acked_bytes;
                if (p->ECE == 1)
                {
                    con->dctcp_ecn_masked_bytes += acked_bytes;
                }
#endif
            }
            else if (ack == snd)
            {
                con->repeat_ack_count++;
                con->ack_count++;
            }
            else
            {
                // discard ack packets with ack number less than snd
                con->ack_count++;
            }
        }
    }
}

__device__ int handle_non_ack_packets(GPUQueue<VDES::TCPPacket *> *recv_queue, int ack_num, VDES::TCPConnection **cons, int con_num)
{
    int packet_num = recv_queue->size;

    VDES::TCPConnection *con = cons[0];
    VDES::RecvPacketRecord *record = con->records;
    int expired_packet_num = 0;

    VDES::RecvPacketRecord *last = NULL;
    for (int i = ack_num; i < packet_num; i++)
    {
        uint32_t seq;
        VDES::TCPPacket *p = recv_queue->get_element(i);

        memcpy(&seq, p->sequence_number, 4);
        uint32_t seq_end = seq + 1460;

        if (seq < con->expected_seq)
        {
            expired_packet_num++;
            continue;
        }

#if ENABLE_DCTCP
        if (p->data_offset_reserved_flags[1] == 1)
        {
            con->ece = 1;
        }
#endif

        if (record != NULL || last != NULL)
        {

            while (record != NULL)
            {
                if (seq_end < record->start_id)
                {
                    if (con->records_pool->size <= 0)
                    {
                        break;
                    }

                    VDES::RecvPacketRecord *cur = con->records_pool->get_head();
                    cur->start_id = seq;
                    cur->end_id = seq_end;

                    if (last == NULL)
                    {
                        cur->next = con->records;
                        con->records = cur;
                    }
                    else
                    {
                        cur->next = last->next;
                        last->next = cur;
                    }
                    con->record_num++;
                    record = cur;
                    break;
                }
                else if (seq_end == record->start_id)
                {
                    record->start_id = seq;
                    break;
                }
                else if (seq_end <= record->end_id)
                {
                    break;
                }
                else if (seq == record->end_id)
                {
                    record->end_id = seq_end;
                    break;
                }
                last = record;
                record = record->next;
            }

            if (record == NULL && con->records_pool->size > 0)
            {
                VDES::RecvPacketRecord *cur = con->records_pool->get_head();
                cur->start_id = seq;
                cur->end_id = seq_end;
                cur->next = NULL;

                last->next = cur;
                record = cur;
                con->record_num++;
            }
        }
        else
        {
            VDES::RecvPacketRecord *cur = con->records_pool->get_head();
            cur->start_id = seq;
            cur->end_id = seq_end;
            cur->next = NULL;

            con->records = cur;
            record = cur;
            con->record_num++;
        }
    }
    return expired_packet_num;

    // enable for multiple TCP connections
    // for (int i = ack_num; i < packet_num; i++)
    // {
    //     int tcp_id = get_tcp_connection_id(recv_queue->get_element(i), cons, con_num);
    //     if (tcp_id >= 0)
    //     {
    //         VDES::TCPConnection *con = cons[tcp_id];
    //         uint32_t expected_seq = con->expected_seq;
    //         uint32_t seq;
    //         memcpy(&seq, recv_queue->get_element(i)->sequence_number, 4);
    //         seq += 1460;

    //         VDES::RecvPacketRecord *record = con->records;

    //         bool find = false;
    //         VDES::RecvPacketRecord *last = NULL;
    //         while (record != NULL)
    //         {
    //             if (seq < record->start_id)
    //             {
    //                 VDES::RecvPacketRecord *cur = con->records_pool->get_element(0);
    //                 con->records_pool->remove_elements(1);
    //                 cur->start_id = seq - 1460;
    //                 cur->end_id = seq;
    //                 if (last != NULL)
    //                 {
    //                     cur->next = last->next;
    //                     last->next = cur;
    //                 }
    //                 else
    //                 {
    //                     cur->next = con->records;
    //                     con->records = cur;
    //                 }

    //                 find = true;
    //                 con->record_num++;
    //                 break;
    //             }
    //             else if (seq == record->start_id)
    //             {
    //                 record->start_id = seq - 1460;
    //                 find = true;
    //                 break;
    //             }
    //         }

    //         if (!find)
    //         {
    //             VDES::RecvPacketRecord *cur = con->records_pool->get_element(0);
    //             con->records_pool->remove_elements(1);
    //             cur->start_id = seq - 1460;
    //             cur->end_id = seq;
    //             cur->next = NULL;
    //             con->record_num++;

    //             if (last == NULL)
    //             {
    //                 con->records = cur;
    //             }
    //             else
    //             {
    //                 last->next = cur;
    //             }
    //         }
    //         record = record->next;
    //     }
    // }
}

__device__ int send_ack_packets(VDES::TCPParams *params, int non_ack_num, VDES::TCPConnection **cons, int con_num, VDES::TCPPacket **alloc_packets)
{
    int ack_response = 0;
    for (int i = 0; i < con_num; i++)
    {
        VDES::TCPConnection *con = cons[i];
        VDES::RecvPacketRecord *record = con->records;
        int64_t expected_num = con->expected_seq;

        while (record != NULL)
        {
            if (expected_num >= record->start_id)
            {
                expected_num = max(expected_num, record->end_id);

                // recycle records
                con->records_pool->append_element(record);
                record = record->next;
                con->record_num--;
            }
            else
            {
                break;
            }
        }

        con->records = record;

        int acked_bytes = expected_num - con->expected_seq;

        if (acked_bytes > 0)
        {
            // con->acked_bytes += acked_bytes;
            // int former_unack_packet_num = con->unacked_packets;
            con->unacked_packets += (acked_bytes / 1460);
            // int ack_packet_num = con->unacked_packets / con->packets_num_per_ack;
            int ack_packet_num = con->unacked_packets;
            // con->unacked_packets %= con->packets_num_per_ack;

            ack_packet_num = min(ack_packet_num, 5);
            int seq_step = (acked_bytes / 1460) / ack_packet_num;
            for (int j = 0; j < ack_packet_num; j++)
            {
                con->expected_seq = expected_num - (ack_packet_num - j - 1) * seq_step * 1460;
                VDES::TCPPacket *ack_packet = alloc_packets[j];
                fill_ack_packet(con, ack_packet, params->timeslot_end_time);
#if ENABLE_DCTCP
                ack_packet->data_offset_reserved_flags[1] = 0;
                ack_packet->ECE = con->ece;
#endif
                // ack_response++;
                // con->unacked_packets = 0;
                // con->repeat_non_ack_num = 0;
            }
            ack_response += ack_packet_num;
            if (ack_packet_num > 0)
            {
                con->unacked_packets = 0;
                con->repeat_non_ack_num = 0;
            }

            //             con->expected_seq = expected_num;
            //             if (ack_packet_num > 0)
            //             {
            //                 VDES::TCPPacket *ack_packet = alloc_packets[0];
            //                 fill_ack_packet(con, ack_packet, params->timeslot_end_time);
            // #if ENABLE_DCTCP
            //                 ack_packet->data_offset_reserved_flags[1] = 0;
            //                 ack_packet->ECE = con->ece;
            //                 con->ece = 0;
            // #endif
            //                 ack_response++;
            //                 con->unacked_packets = 0;
            //                 con->repeat_non_ack_num = 0;
            //             }

            // for (int j = 0; j < ack_packet_num; j++)
            // {
            //     con->expected_seq += ((con->packets_num_per_ack - former_unack_packet_num) * 1460);
            //     VDES::TCPPacket *ack_packet = alloc_packets[j];
            //     fill_ack_packet(con, ack_packet, params->timeslot_end_time);
            //     former_unack_packet_num = 0;
            // }
            // con->expected_seq = expected_num;
            // ack_response += ack_packet_num;
        }
        else if (non_ack_num > 0)
        {
            con->repeat_non_ack_num++;
            if (con->repeat_non_ack_num > con->packets_num_per_ack)
            {
                VDES::TCPPacket *ack_packet = alloc_packets[0];
                fill_ack_packet(con, ack_packet, params->timeslot_end_time);
#if ENABLE_DCTCP
                ack_packet->data_offset_reserved_flags[1] = 0;
                ack_packet->ECE = con->ece;
#endif
                con->unacked_packets = 0;
                ack_response++;
                con->repeat_non_ack_num = 0;
            }
        }
#if ENABLE_DCTCP
        con->ece = 0;
#endif
    }

    return ack_response;
}

__global__ void receive_tcp_packets(VDES::TCPParams *params)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < params->node_num)
    {
        GPUQueue<VDES::TCPPacket *> *recv_queue = params->recv_queues[tid];

        // resort packets as |ack packets| non-ack packets
        int ack_num = classify_tcp_packets(recv_queue);
        params->used_packet_num_per_node[tid] = 0;

        VDES::TCPConnection **cons = params->tcp_cons + tid * MAX_TCP_CONNECTION_NUM;
        int con_num = params->tcp_cons_num_per_node[tid];

        // handle ack packets
        handle_ack_packets(recv_queue, ack_num, cons, con_num);

        // handle non-ack packets
        VDES::TCPPacket **alloc_packets = params->alloc_packets + params->packet_offset_per_node[tid];
        // int packet_num = recv_queue->size;
        // int ack_response_num = 0;
        // for (int i = ack_num; i < packet_num; i++)
        // {
        //     int tcp_id = get_tcp_connection_id(recv_queue->get_element(i), cons, con_num);
        //     if (tcp_id >= 0)
        //     {
        //         VDES::TCPConnection *con = cons[tcp_id];

        //         uint32_t expected_seq = con->expected_seq;
        //         uint32_t seq;
        //         memcpy(&seq, recv_queue->get_element(i)->sequence_number, 4);
        //         if (seq == expected_seq)
        //         {
        //             // receive packets, and update expected_seq
        //             uint16_t payload_len = *(uint16_t *)recv_queue->get_element(i)->payload_len;
        //             con->expected_seq += payload_len;
        //             con->rx_num += payload_len;
        //             con->unacked_packets++;

        //             if (con->unacked_packets < con->packets_num_per_ack)
        //             {
        //                 continue;
        //             }
        //         }

        //         // ack packets immediately
        //         VDES::TCPPacket *ack_packet = alloc_packets[ack_response_num];
        //         ack_response_num++;
        //         fill_ack_packet(con, ack_packet, params->timeslot_end_time);
        //         con->unacked_packets = 0;
        //     }
        // }
        int expired_packet_num = handle_non_ack_packets(recv_queue, ack_num, cons, con_num);
        int non_ack_num = recv_queue->size - ack_num - expired_packet_num;
        int ack_response_num = send_ack_packets(params, non_ack_num, cons, con_num, alloc_packets);
        params->used_packet_num_per_node[tid] = ack_response_num;

        for (int i = 0; i < con_num; i++)
        {
            // update cwnd
            int64_t timestamp;
            memcpy(&timestamp, params->timeslot_end_time, 8);
            update_cwnd(cons[i], timestamp);
        }
#if !ENBALE_CACHE
        VDES::TCPPacket **recycle_packets = params->recycle_packets + params->packet_offset_per_node[tid];
        memcpy(recycle_packets, recv_queue->queue, 8 * recv_queue->size);
        params->recycle_tcp_packet_num[tid] = recv_queue->size;
#endif
        recv_queue->clear();
    }
}

__device__ int get_queueing_packet_num(VDES::TCPConnection *con)
{
    uint64_t snd_bytes = con->una - con->snd;
    int64_t queuing_bytes = min((uint64_t)con->cwnd, con->planned_bytes - con->acked_bytes) - snd_bytes;
    queuing_bytes = max(min(queuing_bytes, ((int64_t)con->swnd) - (int64_t)snd_bytes), (int64_t)0);
    int num = (queuing_bytes + con->mss - 1) / con->mss;
    return num;

    // return (queuing_bytes + con->mss - 1) / con->mss;
}

__device__ void fill_tcp_packets(VDES::TCPConnection *con, VDES::TCPPacket *packet, uint16_t payload_len, VDES::Payload *payload, int64_t *timestamp)
{
    memcpy(packet->src_port, &con->dst_port, 2);
    memcpy(packet->dst_port, &con->src_port, 2);
    memcpy(packet->src_ip, &con->src_ip, 4);
    memcpy(packet->dst_ip, &con->dst_ip, 4);
    packet->data_offset_reserved_flags[0] = packet->data_offset_reserved_flags[0] & 0xfe;

    // fill payload
    memcpy(packet->payload, &payload, sizeof(VDES::Payload *));
    memcpy(packet->payload_len, &payload_len, 2);
    memcpy(packet->send_timestamp, timestamp, 8);

    // fill sequence number
    memcpy(packet->sequence_number, &con->una, 4);
    con->una += payload_len;
    con->tx_num += payload_len;
}

__global__ void send_tcp_packets(VDES::TCPParams *params)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < params->node_num)
    {
        GPUQueue<VDES::TCPPacket *> *send_queue = params->send_queues[tid];

        // computing remaining space for sending packets,
        // please update this part if your traffic model is different
        // our model is designed that a node can only run a transport protocol
        int remaining_capacity = min(params->remaining_nic_cache_space_per_node[tid], send_queue->get_remaining_capacity());

        // send packets
        int con_num = params->tcp_cons_num_per_node[tid];
        VDES::TCPConnection **cons = params->tcp_cons + tid * MAX_TCP_CONNECTION_NUM;
        VDES::TCPPacket **alloc_packets = params->alloc_packets + params->packet_offset_per_node[tid];
        // consumed by ack packets
        int used_packet_num = params->used_packet_num_per_node[tid];
        remaining_capacity -= used_packet_num;

        for (int i = 0; i < con_num; i++)
        {
            VDES::TCPConnection *con = cons[i];

            // update planed bytes
            GPUQueue<VDES::Flow> *flows = con->flows;
            int64_t timestamp_end;
            memcpy(&timestamp_end, params->timeslot_end_time, sizeof(int64_t));
            if (flows->size > 0 && flows->get_element(0).timestamp <= timestamp_end)
            {
                con->planned_bytes += flows->get_element(0).flow_size;
                flows->queue[flows->head].byte_offset = con->planned_bytes;
                flows->remove_elements(1);
            }

            int send_packet_num = min(get_queueing_packet_num(con), remaining_capacity);
            send_packet_num = min(send_packet_num, MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM - used_packet_num);
            if (send_packet_num <= 0)
            {
                break;
            }

            // send packets
            for (int j = 0; j < send_packet_num; j++)
            {
                VDES::TCPPacket *packet = alloc_packets[used_packet_num];
                used_packet_num++;
                fill_tcp_packets(con, packet, con->mss, NULL, params->timeslot_end_time);
#if ENABLE_DCTCP
                packet->data_offset_reserved_flags[1] = 0;
#endif
            }
            remaining_capacity -= send_packet_num;
        }

        if (used_packet_num > 0)
        {
            send_queue->append_elements(alloc_packets, used_packet_num);
        }
        params->used_packet_num_per_node[tid] = used_packet_num;

        // examine whether all connections are completed
        bool is_completed = true;
        bool is_temporary_completed = true;
        for (int i = 0; i < con_num; i++)
        {
            if (cons[i]->acked_bytes < cons[i]->planned_bytes)
            {
                is_completed = false;
                is_temporary_completed = false;
                break;
            }
            else
            {
                if (cons[i]->flows->size > 0)
                {
                    is_completed = false;
                }
            }
        }

        params->is_completed_temporary_traffic_plan[tid] = is_temporary_completed;
        params->is_completed_traffic_plan[tid] = is_completed;
    }
}

namespace VDES
{
    void LaunchReceiveTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream)
    {
        receive_tcp_packets<<<grid_dim, block_dim, 0, stream>>>(tcp_params);
    }

    void LaunchSendTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream)
    {
        send_tcp_packets<<<grid_dim, block_dim, 0, stream>>>(tcp_params);
    }
}

// __device__ bool is_ack(VDES::TCPPacket *p)
// {
//     return p->data_offset_reserved_flags[0] & 0x1 == 1;
// }

// __device__ int classify_tcp_packets(GPUQueue<VDES::TCPPacket *> *recv_queue)
// {
//     int size = recv_queue->size;
//     int offset = size;

//     // scan the first non-ack packet
//     for (int i = 0; i < size; i++)
//     {
//         if (!is_ack(recv_queue->get_element(i)))
//         {
//             offset = i;
//             break;
//         }
//     }

//     // swap packets to adhere to the architecture |ack packets| non-ack packets|
//     for (int end = offset + 1; end < size; end++)
//     {
//         if (is_ack(recv_queue->get_element(end)))
//         {
//             // swap
//             VDES::TCPPacket *temp = recv_queue->get_element(offset);
//             recv_queue->set_element(offset, recv_queue->get_element(end));
//             recv_queue->set_element(end, temp);
//             offset++;
//         }
//     }

//     // sort non-ack packets by sequence number
//     for (int i = offset; i < size; i++)
//     {
//         for (int j = i + 1; j < size; j++)
//         {
//             // try reinforced cast first, if failed, use memcpy or memcmp
//             uint32_t seq_pre;
//             uint32_t seq_post;
//             memcpy(&seq_pre, recv_queue->get_element(i)->sequence_number, sizeof(uint32_t));
//             memcpy(&seq_post, recv_queue->get_element(j)->sequence_number, sizeof(uint32_t));

//             if (seq_pre > seq_post)
//             {
//                 // swap
//                 VDES::TCPPacket *temp = recv_queue->get_element(i);
//                 recv_queue->set_element(i, recv_queue->get_element(j));
//                 recv_queue->set_element(j, temp);
//             }
//         }
//     }

//     return offset;
// }

// __device__ int get_tcp_connection_id(VDES::TCPPacket *p, VDES::TCPConnection **con, int con_num)
// {
//     for (int i = 0; i < con_num; i++)
//     {
//         // try reinforced cast first, if failed, use memcpy or memcmp
//         uint32_t src_ip;
//         uint32_t dst_ip;
//         memcpy(&src_ip, p->src_ip, sizeof(uint32_t));
//         memcpy(&dst_ip, p->dst_ip, sizeof(uint32_t));
//         uint16_t src_port;
//         uint16_t dst_port;
//         memcpy(&src_port, p->src_port, sizeof(uint16_t));
//         memcpy(&dst_port, p->dst_port, sizeof(uint16_t));
//         if (dst_ip == con[i]->src_ip && src_ip == con[i]->dst_ip && src_port == con[i]->src_port && dst_port == con[i]->dst_port)
//         {
//             return i;
//         }
//     }

//     // non-existing connection
//     return -1;
// }

// __device__ void update_cwnd(VDES::TCPConnection *con, int64_t timeslot_end)
// {
//     bool ack_repeat_over_3_times = con->former_repeat_ack_count < 3 && con->repeat_ack_count >= 3;
//     bool rto_timeout = (con->retrans_timer + con->rto) < timeslot_end;

//     if (ack_repeat_over_3_times || rto_timeout)
//     {
//         // congestion avoidance
//         con->ssthresh = con->cwnd >> 1;
//         con->cwnd = con->ssthresh;
//         con->una = con->snd;
//     }
//     else
//     {
//         for (int i = 0; i < con->ack_count; i++)
//         {
//             if (con->cwnd >= con->ssthresh)
//             {
//                 // linear increase
//                 con->cwnd = con->cwnd + (con->ack_count - i) * con->mss * con->packets_num_per_ack;
//                 break;
//             }
//             else
//             {
//                 // quick recovery
//                 con->cwnd = min(con->cwnd * 2, con->ssthresh);
//             }
//         }
//         con->ack_count = 0;
//     }
//     // update restransmission timer
//     if (rto_timeout)
//     {
//         con->retrans_timer = timeslot_end;
//     }
//     con->former_repeat_ack_count = con->repeat_ack_count;
// }

// __device__ void fill_ack_packet(VDES::TCPConnection *con, VDES::TCPPacket *ack_packet, int64_t *timestamp)
// {

//     memcpy(ack_packet->src_port, &con->src_port, 2);
//     memcpy(ack_packet->dst_port, &con->dst_port, 2);
//     memcpy(ack_packet->src_ip, &con->src_ip, 4);
//     memcpy(ack_packet->dst_ip, &con->dst_ip, 4);
//     memcpy(ack_packet->acknowledgement_number, &con->expected_seq, 4);
//     memcpy(ack_packet->send_timestamp, timestamp, 8);
//     ack_packet->data_offset_reserved_flags[0] = ack_packet->data_offset_reserved_flags[0] | 0x1;
//     memset(ack_packet->payload_len, 0, 2);
// }

// __global__ void receive_tcp_packets(VDES::TCPParams *params)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < params->node_num)
//     {
//         GPUQueue<VDES::TCPPacket *> *recv_queue = params->recv_queues[tid];
//         // GPUQueue<VDES::TCPPacket *> *send_queue = params->send_queues[tid];
//         // resort packets as |ack packets| non-ack packets
//         int ack_num = classify_tcp_packets(recv_queue);
//         params->used_packet_num_per_node[tid] = 0;

//         VDES::TCPConnection **cons = params->tcp_cons + tid * MAX_TCP_CONNECTION_NUM;
//         int con_num = params->tcp_cons_num_per_node[tid];

//         // handle ack packets
//         for (int i = 0; i < ack_num; i++)
//         {
//             VDES::TCPPacket *p = recv_queue->get_element(i);
//             int tcp_id = get_tcp_connection_id(p, cons, con_num);
//             if (tcp_id >= 0)
//             {
//                 uint32_t ack;
//                 memcpy(&ack, p->acknowledgement_number, 4);
//                 VDES::TCPConnection *con = cons[tcp_id];
//                 uint32_t snd = con->snd;
//                 if (con->una - snd >= ack - snd && ack != snd)
//                 {
//                     con->repeat_ack_count = 0;
//                     // reset retransmission timer
//                     memcpy(&con->retrans_timer, p->recv_timestamp, 8);

//                     // update send window
//                     con->acked_bytes += (ack - snd);
//                     con->snd = ack;
//                 }
//                 else if (ack == snd)
//                 {
//                     con->repeat_ack_count++;
//                 }
//                 // discard ack packets with ack number less than snd

//                 con->ack_count++;
//             }
//         }

//         // handle non-ack packets
//         VDES::TCPPacket **alloc_packets = params->alloc_packets + params->packet_offset_per_node[tid];
//         int packet_num = recv_queue->size;
//         int ack_response_num = 0;
//         for (int i = ack_num; i < packet_num; i++)
//         {
//             int tcp_id = get_tcp_connection_id(recv_queue->get_element(i), cons, con_num);
//             if (tcp_id >= 0)
//             {
//                 VDES::TCPConnection *con = cons[tcp_id];
//                 /**
//                  * TODO: Use uint instead of int.
//                  */
//                 uint32_t expected_seq = con->expected_seq;
//                 uint32_t seq;
//                 memcpy(&seq, recv_queue->get_element(i)->sequence_number, 4);
//                 if (seq == expected_seq)
//                 {
//                     // receive packets, and update expected_seq
//                     uint16_t payload_len = *(uint16_t *)recv_queue->get_element(i)->payload_len;
//                     con->expected_seq += payload_len;
//                     con->rx_num += payload_len;
//                     con->unacked_packets++;

//                     // push payoads to recv cache, enable this line if needed
//                     // con->recv_cache->append_element(recv_queue->get_element(i)->payload);

//                     if (con->unacked_packets < con->packets_num_per_ack)
//                     {
//                         continue;
//                     }
//                 }

//                 // ack packets immediately
//                 VDES::TCPPacket *ack_packet = alloc_packets[ack_response_num];
//                 ack_response_num++;
//                 fill_ack_packet(con, ack_packet, params->timeslot_end_time);
//                 // send_queue->append_element(ack_packet);
//                 con->unacked_packets = 0;
//             }
//         }
//         params->used_packet_num_per_node[tid] = ack_response_num;

//         for (int i = 0; i < con_num; i++)
//         {
//             // update cwnd
//             int64_t timestamp;
//             memcpy(&timestamp, params->timeslot_end_time, 8);
//             update_cwnd(cons[i], timestamp);
//         }

// #if !ENBALE_CACHE
//         VDES::TCPPacket **recycle_packets = params->recycle_packets + params->packet_offset_per_node[tid];
//         memcpy(recycle_packets, recv_queue->queue, 8 * recv_queue->size);
//         params->recycle_tcp_packet_num[tid] = recv_queue->size;
// #endif

//         recv_queue->clear();
//     }
// }

// __device__ int get_queueing_packet_num(VDES::TCPConnection *con)
// {
//     //     uint64_t queuing_bytes = min((uint64_t)con->cwnd, con->planned_bytes - con->acked_bytes) - (con->una - con->snd);
//     //     return (queuing_bytes + con->mss - 1) / con->mss;
//     uint64_t snd_bytes = con->una - con->snd;
//     int64_t queuing_bytes = min((uint64_t)con->cwnd, con->planned_bytes - con->acked_bytes) - snd_bytes;
//     queuing_bytes = max(min(queuing_bytes, con->swnd - snd_bytes), (int64_t)0);
//     int num = (queuing_bytes + con->mss - 1) / con->mss;
//     return num;
// }

// __device__ void fill_tcp_packets(VDES::TCPConnection *con, VDES::TCPPacket *packet, uint16_t payload_len, VDES::Payload *payload, int64_t *timestamp)
// {
//     memcpy(packet->src_port, &con->dst_port, 2);
//     memcpy(packet->dst_port, &con->src_port, 2);
//     memcpy(packet->src_ip, &con->dst_ip, 4);
//     memcpy(packet->dst_ip, &con->src_ip, 4);
//     packet->data_offset_reserved_flags[0] = packet->data_offset_reserved_flags[0] & 0xfe;

//     // fill payload
//     memcpy(packet->payload, &payload, sizeof(VDES::Payload *));
//     memcpy(packet->payload_len, &payload_len, 2);
//     memcpy(packet->send_timestamp, timestamp, 8);

//     // fill sequence number
//     memcpy(packet->sequence_number, &con->una, 4);
//     con->una += payload_len;
//     con->tx_num += payload_len;
// }

// __global__ void send_tcp_packets(VDES::TCPParams *params)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < params->node_num)
//     {
//         GPUQueue<VDES::TCPPacket *> *send_queue = params->send_queues[tid];

//         // computing remaining space for sending packets,
//         // please update this part if your traffic model is different
//         // our model is designed that a node can only run a transport protocol
//         int remaining_capacity = params->remaining_nic_cache_space_per_node[tid];

//         // send packets
//         int con_num = params->tcp_cons_num_per_node[tid];
//         VDES::TCPConnection **cons = params->tcp_cons + tid * MAX_TCP_CONNECTION_NUM;
//         VDES::TCPPacket **alloc_packets = params->alloc_packets + params->packet_offset_per_node[tid];
//         // consumed by ack packets
//         int used_packet_num = params->used_packet_num_per_node[tid];
//         remaining_capacity -= used_packet_num;

//         // int all_send_packet_num = ;
//         for (int i = 0; i < con_num; i++)
//         {
//             VDES::TCPConnection *con = cons[i];
//             int send_packet_num = min(get_queueing_packet_num(con), remaining_capacity);
//             send_packet_num = min(send_packet_num, MAX_TRANSMITTED_PACKET_NUM + MAX_GENERATED_PACKET_NUM - used_packet_num);
//             if (send_packet_num <= 0)
//             {
//                 break;
//             }

//             /**
//              * TODO: Use j instead of i to avoid reading confusion.
//              */
//             // send packets
//             for (int j = 0; j < send_packet_num; j++)
//             {
//                 VDES::TCPPacket *packet = alloc_packets[used_packet_num];
//                 used_packet_num++;
//                 fill_tcp_packets(con, packet, con->mss, NULL, params->timeslot_end_time);
//                 // send_queue->append_element(packet);
//             }
//             remaining_capacity -= send_packet_num;
//         }

//         if (used_packet_num > 0)
//         {
//             send_queue->append_elements(alloc_packets, used_packet_num);
//         }
//         params->used_packet_num_per_node[tid] = used_packet_num;
//         // removed updated used packet num
//     }
// }

// namespace VDES
// {
//     void LaunchReceiveTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream)
//     {
//         receive_tcp_packets<<<grid_dim, block_dim, 0, stream>>>(tcp_params);
//     }

//     void LaunchSendTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream)
//     {
//         send_tcp_packets<<<grid_dim, block_dim, 0, stream>>>(tcp_params);
//     }
// }
// #include "tcp_controller.h"

// __device__ bool is_ack(VDES::TCPPacket *p)
// {
//     return p->data_offset_reserved_flags[0] & 0x1 == 1;
// }

// __device__ int classify_tcp_packets(GPUQueue<VDES::TCPPacket *> *recv_queue)
// {
//     int offset = 0;
//     int size = recv_queue->size;

//     // scan the first non-ack packet
//     for (int i = 0; i < size; i++)
//     {
//         if (!is_ack(recv_queue->get_element(i)))
//         {
//             offset = i;
//             break;
//         }
//     }

//     // swap packets to adhere to the architecture |ack packets| non-ack packets|
//     for (int end = offset + 1; end < size; end++)
//     {
//         if (is_ack(recv_queue->get_element(end)))
//         {
//             // swap
//             VDES::TCPPacket *temp = recv_queue->get_element(offset);
//             recv_queue->set_element(offset, recv_queue->get_element(end));
//             recv_queue->set_element(end, temp);
//             offset++;
//         }
//     }

//     // sort non-ack packets by sequence number
//     for (int i = offset + 1; i < size; i++)
//     {
//         for (int j = i + 1; j < size; j++)
//         {
//             // try reinforced cast first, if failed, use memcpy or memcmp
//             if (*(uint32_t *)(recv_queue->get_element(i)->sequence_number) > *(uint32_t *)(recv_queue->get_element(j)->sequence_number))
//             {
//                 // swap
//                 VDES::TCPPacket *temp = recv_queue->get_element(i);
//                 recv_queue->set_element(i, recv_queue->get_element(j));
//                 recv_queue->set_element(j, temp);
//             }
//         }
//     }

//     return offset;
// }

// __device__ int get_tcp_connection_id(VDES::TCPPacket *p, VDES::TCPConnection **con, int con_num)
// {
//     for (int i = 0; i < con_num; i++)
//     {
//         // try reinforced cast first, if failed, use memcpy or memcmp
//         if (*(uint16_t *)(p->dst_port) == con[i]->src_port && *(uint16_t *)(p->src_port) == con[i]->dst_port)
//         {
//             return i;
//         }
//     }

//     // non-existing connection
//     return -1;
// }

// __device__ void update_cwnd(VDES::TCPConnection *con, int64_t timeslot_end)
// {
//     bool ack_repeat_over_3_times = con->former_repeat_ack_count < 3 && con->repeat_ack_count >= 3;
//     bool rto_timeout = (con->retrans_timer + con->rto) > timeslot_end;

//     if (ack_repeat_over_3_times || rto_timeout)
//     {
//         // congestion avoidance
//         con->ssthresh = con->cwnd >> 2;
//         con->cwnd = con->ssthresh + 3 * con->mss;
//     }
//     else
//     {
//         for (int i = 0; i < con->ack_count; i++)
//         {
//             if (con->cwnd >= con->ssthresh)
//             {
//                 // linear increase
//                 con->cwnd = con->cwnd + (con->ack_count - i) * con->mss * con->packets_num_per_ack;
//                 break;
//             }
//             else
//             {
//                 // quick recovery
//                 con->cwnd = min(con->cwnd * 2, con->ssthresh);
//             }
//         }
//     }
//     // update restransmission timer
//     con->retrans_timer = min(timeslot_end, con->retrans_timer + con->rto);
//     con->former_repeat_ack_count = con->repeat_ack_count;
// }

// __device__ void fill_ack_packet(VDES::TCPConnection *con, VDES::TCPPacket *ack_packet, int timestamp)
// {

//     memcpy(ack_packet->src_port, &con->dst_port, 2);
//     memcpy(ack_packet->dst_port, &con->src_port, 2);
//     memcpy(ack_packet->src_ip, &con->dst_ip, 4);
//     memcpy(ack_packet->dst_ip, &con->src_ip, 4);
//     memcpy(ack_packet->acknowledgement_number, &con->expected_seq, 4);
//     memcpy(ack_packet->send_timestamp, &timestamp, 8);
//     ack_packet->data_offset_reserved_flags[0] = ack_packet->data_offset_reserved_flags[0] | 0x1;
// }

// __global__ void receive_tcp_packets(VDES::TCPParams *params)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < params->node_num)
//     {
//         GPUQueue<VDES::TCPPacket *> *recv_queue = params->recv_queues[tid];
//         // GPUQueue<VDES::TCPPacket *> *send_queue = params->send_queues[tid];
//         // resort packets as |ack packets| non-ack packets
//         int ack_num = classify_tcp_packets(recv_queue);

//         VDES::TCPConnection **cons = params->tcp_cons + tid * MAX_TCP_CONNECTION_NUM;
//         int con_num = params->tcp_cons_num_per_node[tid];

//         // handle ack packets
//         for (int i = 0; i < ack_num; i++)
//         {
//             VDES::TCPPacket *p = recv_queue->get_element(i);
//             int tcp_id = get_tcp_connection_id(p, cons, con_num);
//             if (tcp_id >= 0)
//             {
//                 uint32_t ack;
//                 memcpy(&ack, p->acknowledgement_number, 4);
//                 VDES::TCPConnection *con = cons[tcp_id];
//                 int snd = con->snd;
//                 if (ack > snd)
//                 {
//                     con->repeat_ack_count = 0;
//                     // reset retransmission timer
//                     memcpy(&con->retrans_timer, p->recv_timestamp, 8);

//                     // update send window
//                     con->acked_bytes += (ack - snd);
//                     con->snd = ack;
//                 }
//                 else if (ack == snd)
//                 {
//                     con->repeat_ack_count++;
//                 }
//                 // discard ack packets with ack number less than snd

//                 con->ack_count++;
//             }
//         }

//         // handle non-ack packets
//         VDES::TCPPacket **alloc_packets = params->alloc_packets + params->packet_offset_per_node[tid];
//         int packet_num = recv_queue->size;
//         int ack_response_num = 0;
//         for (int i = ack_num; i < packet_num; i++)
//         {
//             int tcp_id = get_tcp_connection_id(recv_queue->get_element(i), cons, con_num);
//             if (tcp_id >= 0)
//             {
//                 VDES::TCPConnection *con = cons[tcp_id];
//                 int expected_seq = con->expected_seq;
//                 int seq;
//                 memcpy(&seq, recv_queue->get_element(i)->sequence_number, 4);
//                 if (seq == expected_seq)
//                 {
//                     // receive packets, and update expected_seq
//                     uint16_t payload_len = *(uint16_t *)recv_queue->get_element(i)->payload_len;
//                     con->expected_seq += payload_len;
//                     con->rx_num += payload_len;
//                     con->unacked_packets++;

//                     // push payoads to recv cache, enable this line if needed
//                     // con->recv_cache->append_element(recv_queue->get_element(i)->payload);

//                     if (con->unacked_packets < con->packets_num_per_ack)
//                     {
//                         continue;
//                     }
//                 }

//                 // ack packets immediately
//                 VDES::TCPPacket *ack_packet = alloc_packets[ack_response_num];
//                 ack_response_num++;
//                 fill_ack_packet(con, ack_packet, params->timeslot_end_time[0]);
//                 // send_queue->append_element(ack_packet);
//                 con->unacked_packets = 0;
//             }
//         }

//         for (int i = 0; i < con_num; i++)
//         {
//             // update cwnd
//             update_cwnd(cons[i], params->timeslot_end_time[0]);
//         }
//         recv_queue->clear();
//     }
// }

// __device__ int get_queueing_packet_num(VDES::TCPConnection *con)
// {
//     uint64_t queuing_bytes = min((uint64_t)con->cwnd, con->planned_bytes - con->acked_bytes);
//     return (queuing_bytes + con->mss - 1) / con->mss;
// }

// __device__ void fill_tcp_packets(VDES::TCPConnection *con, VDES::TCPPacket *packet, uint16_t payload_len, VDES::Payload *payload, int timestamp)
// {
//     memcpy(packet->src_port, &con->dst_port, 2);
//     memcpy(packet->dst_port, &con->src_port, 2);
//     memcpy(packet->src_ip, &con->dst_ip, 4);
//     memcpy(packet->dst_ip, &con->src_ip, 4);
//     packet->data_offset_reserved_flags[0] = packet->data_offset_reserved_flags[0] | 0xfe;

//     // fill payload
//     memcpy(packet->payload, &payload, sizeof(VDES::Payload *));
//     memcpy(packet->payload_len, &payload_len, 2);
//     memcpy(packet->send_timestamp, &timestamp, 8);

//     // fill sequence number
//     memcpy(packet->sequence_number, &con->una, 4);
//     con->una += payload_len;
//     con->tx_num += payload_len;
// }

// __global__ void send_tcp_packets(VDES::TCPParams *params)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < params->node_num)
//     {
//         GPUQueue<VDES::TCPPacket *> *send_queue = params->send_queues[tid];

//         // computing remaining space for sending packets,
//         // please update this part if your traffic model is different
//         // our model is designed that a node can only run a transport protocol
//         int remaining_capacity = params->remaining_nic_cache_space_per_node[tid];

//         // send packets
//         int con_num = params->tcp_cons_num_per_node[tid];
//         VDES::TCPConnection **cons = params->tcp_cons + tid * MAX_TCP_CONNECTION_NUM;
//         VDES::TCPPacket **alloc_packets = params->alloc_packets + params->packet_offset_per_node[tid];
//         // consumed by ack packets
//         int used_packet_num = params->used_packet_num_per_node[tid];
//         remaining_capacity -= used_packet_num;

//         // int all_send_packet_num = ;
//         for (int i = 0; i < con_num; i++)
//         {
//             VDES::TCPConnection *con = cons[i];
//             int send_packet_num = min(get_queueing_packet_num(con), remaining_capacity);
//             if (send_packet_num < 0)
//             {
//                 break;
//             }

//             // send packets
//             for (int i = 0; i < send_packet_num; i++)
//             {
//                 VDES::TCPPacket *packet = alloc_packets[used_packet_num];
//                 used_packet_num++;
//                 fill_tcp_packets(con, packet, con->mss, NULL, params->timeslot_end_time[0]);
//                 // send_queue->append_element(packet);
//             }
//             remaining_capacity -= send_packet_num;
//         }

//         if (used_packet_num > 0)
//         {
//             send_queue->append_elements(alloc_packets, used_packet_num);
//         }
//         params->used_packet_num_per_node[tid] = 0;
//     }
// }

// namespace VDES
// {
//     void LaunchReceiveTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream)
//     {
//         receive_tcp_packets<<<grid_dim, block_dim, 0, stream>>>(tcp_params);
//     }

//     void LaunchSendTCPPacketKernel(dim3 grid_dim, dim3 block_dim, TCPParams *tcp_params, cudaStream_t stream)
//     {
//         send_tcp_packets<<<grid_dim, block_dim, 0, stream>>>(tcp_params);
//     }
// }