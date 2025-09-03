#include "component.h"
#include <iostream>
#include <cstring>
#include "protocol_type.h"

namespace VDES
{

    NIC *CreateNIC(int ingress_capacity, int egress_capacity)
    {
        NIC *nic = new NIC;
        nic->ingress = create_gpu_queue<Frame *>(ingress_capacity);
        nic->egress = create_gpu_queue<Frame *>(egress_capacity);
        nic->is_connected = false;
        return nic;
    }

    void SetNICMacAddr(NIC *nic, uint8_t *mac_addr)
    {
        if (nic == NULL)
        {
            std::cout << "Error: NIC is NULL" << std::endl;
            return;
        }
        memcpy(nic->mac_addr, mac_addr, 6);
    }

    P2PChanenl *CreateP2PChannel(int tx_rate, int popogation_delay)
    {
        P2PChanenl *p2p_channel = new P2PChanenl;
        p2p_channel->tx_rate = tx_rate;
        p2p_channel->popogation_delay = popogation_delay;
        return p2p_channel;
    }

    bool ConnectNICByP2P(NIC *nic1, NIC *nic2, P2PChanenl *ch)
    {
        if (nic1 == NULL || nic2 == NULL || nic1->is_connected || nic2->is_connected)
        {
            return false;
        }

        ch->nic1 = nic1;
        ch->nic2 = nic2;

        nic1->is_connected = true;
        nic2->is_connected = true;
        return true;
    }

    Node *CreateNode(int nic_num, int ingress_capacity, int egress_capacity)

    {
        Node *node = new Node;
        for (int i = 0; i < nic_num; i++)
        {
            node->nics.push_back(CreateNIC(ingress_capacity, egress_capacity));
        }

        return node;
    }

    Switch *CreateSwitch(int port_num, int ingress_capacity, int egress_capacity)
    {
        Switch *sw = new Switch;
        sw->port_num = port_num;
        sw->nics = new NIC *[port_num];
        for (int i = 0; i < port_num; i++)
        {
            sw->nics[i] = CreateNIC(ingress_capacity, egress_capacity);
        }
        return sw;
    }

    bool ConnectDevices(Node *node1, int nic_id, Switch *sw1, int port_id, P2PChanenl *ch)
    {
        return ConnectNICByP2P(node1->nics[nic_id], sw1->nics[port_id], ch);
    }

    bool ConnectDevices(Switch *sw1, int port_id1, Switch *sw2, int port_id2, P2PChanenl *ch)
    {
        return ConnectNICByP2P(sw1->nics[port_id1], sw2->nics[port_id2], ch);
    }

    IPv4Protocol *CreateIPv4Protocol(int nic_num)
    {
        IPv4Protocol *ipv4 = new IPv4Protocol;
        ipv4->protocol_header.protocol_type = NetworkProtocolType::IPv4;
        ipv4->protocol_header.network_layer = 3;
        ipv4->ingress = create_gpu_queue<Ipv4Packet *>(nic_num * NODE_DEFAULT_INGRESS_QUEUE_SIZE);
        ipv4->nic_num = nic_num;
        ipv4->egresses = new GPUQueue<Ipv4Packet *> *[nic_num];
        for (int i = 0; i < nic_num; i++)
        {
            ipv4->egresses[i] = create_gpu_queue<Ipv4Packet *>(NODE_DEFAULT_INGRESS_QUEUE_SIZE * nic_num);
        }
        ipv4->local_delivery = create_gpu_queue<Ipv4Packet *>(NODE_DEFAULT_INGRESS_QUEUE_SIZE * nic_num);
        ipv4->routing_table = create_gpu_queue<IPv4RoutingRule *>(MAX_ROUTING_TABLE_SIZE);
        ipv4->error_queue = create_gpu_queue<Ipv4Packet *>(NODE_DEFAULT_INGRESS_QUEUE_SIZE * nic_num);
        ipv4->ipv4_interfaces = new IPv4Interface[nic_num];
        return ipv4;
    }

    TCPProtocol *CreateTCPProtocol(int nic_num)
    {
        TCPProtocol *tcp = new TCPProtocol;
        tcp->protocol_header.protocol_type = TransportProtocolType::TCP;
        tcp->protocol_header.network_layer = 4;
        tcp->tcp_cons_num = 0;
        tcp->tcp_cons = new TCPConnection *[MAX_TCP_CONNECTION_NUM];
        tcp->ingress = create_gpu_queue<TCPPacket *>(NODE_DEFAULT_INGRESS_QUEUE_SIZE * nic_num);
        tcp->egress = create_gpu_queue<TCPPacket *>(NODE_DEFAULT_INGRESS_QUEUE_SIZE * nic_num);
        tcp->tcp_port_offset = 1000;
        return tcp;
    }

    IPv4Protocol *GetIPv4Protocol(Node *node)
    {
        for (int i = 0; i < node->l3_protocols.size(); i++)
        {
            if (node->l3_protocols[i]->protocol_type == NetworkProtocolType::IPv4)
            {
                return (IPv4Protocol *)node->l3_protocols[i];
            }
        }
        return NULL;
    }

    TCPProtocol *GetTCPProtocol(Node *node)
    {
        for (int i = 0; i < node->l4_protocols.size(); i++)
        {
            if (node->l4_protocols[i]->protocol_type == TransportProtocolType::TCP)
            {
                return (TCPProtocol *)node->l4_protocols[i];
            }
        }
        return NULL;
    }

    bool InstallIPv4Protocol(Node *node)
    {
        for (int i = 0; i < node->l3_protocols.size(); i++)
        {
            if (node->l3_protocols[i]->protocol_type == NetworkProtocolType::IPv4)
            {
                return false;
            }
        }

        int nic_num = node->nics.size();
        IPv4Protocol *ipv4 = CreateIPv4Protocol(nic_num);
        node->l3_protocols.push_back((ProtocolHeader *)ipv4);
        return true;
    }

    bool InstallTCPProtocol(Node *node)
    {
        for (int i = 0; i < node->l4_protocols.size(); i++)
        {
            if (node->l3_protocols[i]->protocol_type == TransportProtocolType::TCP)
            {
                return false;
            }
        }

        int nic_num = node->nics.size();
        TCPProtocol *tcp = CreateTCPProtocol(nic_num);
        node->l4_protocols.push_back((ProtocolHeader *)tcp);
        return true;
    }

    TCPConnection *CreateTCPConnection(int64_t planned_bytes)
    {
        TCPConnection *tcp_con = new TCPConnection;
        tcp_con->acked_flows = 0;
        tcp_con->src_ip = 0;
        tcp_con->dst_ip = 0;
        tcp_con->src_port = 0;
        tcp_con->dst_port = 0;
        tcp_con->mss = 1460;
        tcp_con->ssthresh = 10 * tcp_con->mss;
        tcp_con->cwnd = 10 * tcp_con->mss;
        tcp_con->rto = 1000000; // 1000us
        tcp_con->retrans_timer = tcp_con->rto;
        tcp_con->snd = 0;
        tcp_con->una = 0;
        tcp_con->unacked_packets = 0;
        // ack once for every 5 packets
        tcp_con->packets_num_per_ack = 5;
        tcp_con->ack_count = 0;
        tcp_con->expected_seq = 0;
        tcp_con->former_repeat_ack_count = 0;
        tcp_con->repeat_ack_count = 0;
        tcp_con->tx_num = 0;
        tcp_con->rx_num = 0;
        tcp_con->planned_bytes = planned_bytes;
        tcp_con->acked_bytes = 0;
        // tcp_con->swnd = (NODE_DEFAULT_EGRESS_QUEUE_SIZE / SWND_RATE) * tcp_con->mss;
        tcp_con->swnd = 10000 * tcp_con->mss;
        tcp_con->record_num = 0;
        tcp_con->flows = create_gpu_queue<Flow>(100);
        tcp_con->records = NULL;
        tcp_con->records_pool = create_gpu_queue<RecvPacketRecord *>(200);
        RecvPacketRecord *records;
        cudaMalloc(&records, sizeof(RecvPacketRecord) * 200);
        std::vector<RecvPacketRecord *> records_ptr;
        for (int i = 0; i < 200; i++)
        {
            records_ptr.push_back(records + i);
        }
        GPUQueue<RecvPacketRecord *> cpu_pool;
        cudaMemcpy(&cpu_pool, tcp_con->records_pool, sizeof(GPUQueue<RecvPacketRecord *>), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_pool.queue, records_ptr.data(), sizeof(RecvPacketRecord *) * 200, cudaMemcpyHostToDevice);
        cpu_pool.size = 200;
        cudaMemcpy(tcp_con->records_pool, &cpu_pool, sizeof(GPUQueue<RecvPacketRecord *>), cudaMemcpyHostToDevice);

#if ENABLE_DCTCP
        tcp_con->dctcp_acked_bytes = 0;
        tcp_con->dctcp_ecn_masked_bytes = 0;
        tcp_con->dctcp_window_end = tcp_con->cwnd;
        tcp_con->dctcp_alpha = 0;
        tcp_con->ece = 0;
#endif

        return tcp_con;
    }

    bool ConnectTCPConnection(Node *node1, int64_t planed_bytes1, int nic_id1, Node *node2, int64_t planed_bytes2, int nic_id2)
    {

        IPv4Protocol *ipv41 = GetIPv4Protocol(node1);
        IPv4Protocol *ipv42 = GetIPv4Protocol(node2);
        TCPProtocol *tcp1 = GetTCPProtocol(node1);
        TCPProtocol *tcp2 = GetTCPProtocol(node2);

        if (tcp1->tcp_cons_num >= MAX_TCP_CONNECTION_NUM || tcp2->tcp_cons_num >= MAX_TCP_CONNECTION_NUM)
        {
            return false;
        }

        TCPConnection *tcp_con1 = CreateTCPConnection(planed_bytes1);
        TCPConnection *tcp_con2 = CreateTCPConnection(planed_bytes2);

        tcp_con1->src_ip = ipv41->ipv4_interfaces[nic_id1].ip;
        tcp_con1->dst_ip = ipv42->ipv4_interfaces[nic_id2].ip;
        tcp_con1->src_port = tcp1->tcp_port_offset;
        tcp_con1->dst_port = tcp2->tcp_port_offset;
        tcp_con2->src_ip = ipv42->ipv4_interfaces[nic_id2].ip;
        tcp_con2->dst_ip = ipv41->ipv4_interfaces[nic_id1].ip;
        tcp_con2->src_port = tcp2->tcp_port_offset;
        tcp_con2->dst_port = tcp1->tcp_port_offset;
        (tcp1->tcp_port_offset)++;
        (tcp2->tcp_port_offset)++;

        tcp1->tcp_cons[tcp1->tcp_cons_num] = tcp_con1;
        tcp2->tcp_cons[tcp2->tcp_cons_num] = tcp_con2;
        tcp1->tcp_cons_num++;
        tcp2->tcp_cons_num++;

        return false;
    }

    bool AllocateIPAddr(Node **node, int node_num, uint32_t ip_base, uint32_t mask)
    {
        int addr_num = 0;
        uint32_t remaing_addr_num = -1;
        remaing_addr_num = remaing_addr_num & (~mask);

        for (int i = 0; i < node_num; i++)
        {
            auto ipv4 = GetIPv4Protocol(node[i]);
            for (int j = 0; j < ipv4->nic_num; j++)
            {
                if (remaing_addr_num <= addr_num)
                {
                    return false;
                }
                ipv4->ipv4_interfaces[j].ip = ip_base + addr_num;
                ipv4->ipv4_interfaces[j].mask = mask;
                addr_num++;
            }
        }

        return true;
    }

} // namespace VDES

// namespace VDES
// {

//     NIC *CreateNIC(int ingress_capacity, int egress_capacity)
//     {
//         NIC *nic = new NIC;
//         nic->ingress = create_gpu_queue<Frame *>(ingress_capacity);
//         nic->egress = create_gpu_queue<Frame *>(egress_capacity);
//         nic->is_connected = false;
//         return nic;
//     }

//     void SetNICMacAddr(NIC *nic, uint8_t *mac_addr)
//     {
//         if (nic == NULL)
//         {
//             std::cout << "Error: NIC is NULL" << std::endl;
//             return;
//         }
//         memcpy(nic->mac_addr, mac_addr, 6);
//     }

//     P2PChanenl *CreateP2PChannel(int tx_rate, int popogation_delay)
//     {
//         P2PChanenl *p2p_channel = new P2PChanenl;
//         p2p_channel->tx_rate = tx_rate;
//         p2p_channel->popogation_delay = popogation_delay;
//         return p2p_channel;
//     }

//     bool ConnectNICByP2P(NIC *nic1, NIC *nic2, P2PChanenl *ch)
//     {
//         if (nic1 == NULL || nic2 == NULL || nic1->is_connected || nic2->is_connected)
//         {
//             std::cout << "Error: NIC is NULL or already connected" << std::endl;
//             return false;
//         }

//         ch->nic1 = nic1;
//         ch->nic2 = nic2;

//         nic1->is_connected = true;
//         nic2->is_connected = true;
//         return true;
//     }

//     Node *CreateNode(int nic_num, int ingress_capacity, int egress_capacity)

//     {
//         Node *node = new Node;
//         for (int i = 0; i < nic_num; i++)
//         {
//             node->nics.push_back(CreateNIC(ingress_capacity, egress_capacity));
//         }

//         return node;
//     }

//     Switch *CreateSwitch(int port_num, int ingress_capacity, int egress_capacity)
//     {
//         Switch *sw = new Switch;
//         sw->port_num = port_num;
//         sw->nics = new NIC *[port_num];
//         for (int i = 0; i < port_num; i++)
//         {
//             sw->nics[i] = CreateNIC(ingress_capacity, egress_capacity);
//         }
//         return sw;
//     }

//     bool ConnectDevices(Node *node1, int nic_id, Switch *sw1, int port_id, P2PChanenl *ch)
//     {
//         return ConnectNICByP2P(node1->nics[nic_id], sw1->nics[port_id], ch);
//     }

//     bool ConnectDevices(Switch *sw1, int port_id1, Switch *sw2, int port_id2, P2PChanenl *ch)
//     {
//         return ConnectNICByP2P(sw1->nics[port_id1], sw2->nics[port_id2], ch);
//     }

//     IPv4Protocol *CreateIPv4Protocol(int nic_num)
//     {
//         IPv4Protocol *ipv4 = new IPv4Protocol;
//         ipv4->protocol_header.protocol_type = NetworkProtocolType::IPv4;
//         ipv4->protocol_header.network_layer = 3;
//         ipv4->ingress = create_gpu_queue<Ipv4Packet *>(nic_num * NODE_DEFAULT_INGRESS_QUEUE_SIZE);
//         ipv4->nic_num = nic_num;
//         ipv4->egresses = new GPUQueue<Ipv4Packet *> *[nic_num];
//         for (int i = 0; i < nic_num; i++)
//         {
//             ipv4->egresses[i] = create_gpu_queue<Ipv4Packet *>(NODE_DEFAULT_INGRESS_QUEUE_SIZE * nic_num);
//         }
//         ipv4->local_delivery = create_gpu_queue<Ipv4Packet *>(NODE_DEFAULT_INGRESS_QUEUE_SIZE * nic_num);
//         ipv4->routing_table = create_gpu_queue<IPv4RoutingRule *>(MAX_ROUTING_TABLE_SIZE);
//         ipv4->error_queue = create_gpu_queue<Ipv4Packet *>(NODE_DEFAULT_INGRESS_QUEUE_SIZE * nic_num);
//         ipv4->ipv4_interfaces = new IPv4Interface[nic_num];
//         return ipv4;
//     }

//     TCPProtocol *CreateTCPProtocol(int nic_num)
//     {
//         TCPProtocol *tcp = new TCPProtocol;
//         tcp->protocol_header.protocol_type = TransportProtocolType::TCP;
//         tcp->protocol_header.network_layer = 4;
//         tcp->tcp_cons_num = 0;
//         tcp->tcp_cons = new TCPConnection *[MAX_TCP_CONNECTION_NUM];
//         tcp->ingress = create_gpu_queue<TCPPacket *>(NODE_DEFAULT_INGRESS_QUEUE_SIZE * nic_num);
//         tcp->egress = create_gpu_queue<TCPPacket *>(NODE_DEFAULT_INGRESS_QUEUE_SIZE * nic_num);
//         tcp->tcp_port_offset = 1000;
//         return tcp;
//     }

//     IPv4Protocol *GetIPv4Protocol(Node *node)
//     {
//         for (int i = 0; i < node->l3_protocols.size(); i++)
//         {
//             if (node->l3_protocols[i]->protocol_type == NetworkProtocolType::IPv4)
//             {
//                 return (IPv4Protocol *)node->l3_protocols[i];
//             }
//         }
//     }

//     TCPProtocol *GetTCPProtocol(Node *node)
//     {
//         for (int i = 0; i < node->l4_protocols.size(); i++)
//         {
//             if (node->l4_protocols[i]->protocol_type == TransportProtocolType::TCP)
//             {
//                 return (TCPProtocol *)node->l4_protocols[i];
//             }
//         }
//     }

//     bool InstallIPv4Protocol(Node *node)
//     {
//         for (int i = 0; i < node->l3_protocols.size(); i++)
//         {
//             if (node->l3_protocols[i]->protocol_type == NetworkProtocolType::IPv4)
//             {
//                 return false;
//             }
//         }

//         int nic_num = node->nics.size();
//         IPv4Protocol *ipv4 = CreateIPv4Protocol(nic_num);
//         node->l3_protocols.push_back((ProtocolHeader *)ipv4);
//         return true;
//     }

//     bool InstallTCPProtocol(Node *node)
//     {
//         for (int i = 0; i < node->l4_protocols.size(); i++)
//         {
//             if (node->l3_protocols[i]->protocol_type == TransportProtocolType::TCP)
//             {
//                 return false;
//             }
//         }

//         int nic_num = node->nics.size();
//         TCPProtocol *tcp = CreateTCPProtocol(nic_num);
//         node->l4_protocols.push_back((ProtocolHeader *)tcp);
//         return true;
//     }

//     TCPConnection *CreateTCPConnection(int64_t planned_bytes)
//     {
//         TCPConnection *tcp_con = new TCPConnection;
//         tcp_con->src_ip = 0;
//         tcp_con->dst_ip = 0;
//         tcp_con->src_port = 0;
//         tcp_con->dst_port = 0;
//         tcp_con->mss = 1460;
//         tcp_con->ssthresh = 65535;
//         tcp_con->cwnd = 10 * tcp_con->mss;
//         tcp_con->rto = 1000000; // 1000us
//         tcp_con->retrans_timer = tcp_con->rto;
//         tcp_con->snd = 0;
//         tcp_con->una = 0;
//         tcp_con->unacked_packets = 0;
//         // ack once for every 5 packets
//         tcp_con->packets_num_per_ack = 5;
//         tcp_con->ack_count = 0;
//         tcp_con->expected_seq = 0;
//         tcp_con->former_repeat_ack_count = 0;
//         tcp_con->repeat_ack_count = 0;
//         tcp_con->tx_num = 0;
//         tcp_con->rx_num = 0;
//         tcp_con->planned_bytes = planned_bytes;
//         tcp_con->acked_bytes = 0;
//         tcp_con->swnd = (NODE_DEFAULT_EGRESS_QUEUE_SIZE / SWND_RATE) * tcp_con->mss;

//         return tcp_con;
//     }

//     bool ConnectTCPConnection(Node *node1, int64_t planed_bytes1, int nic_id1, Node *node2, int64_t planed_bytes2, int nic_id2)
//     {

//         IPv4Protocol *ipv41 = GetIPv4Protocol(node1);
//         IPv4Protocol *ipv42 = GetIPv4Protocol(node2);
//         TCPProtocol *tcp1 = GetTCPProtocol(node1);
//         TCPProtocol *tcp2 = GetTCPProtocol(node2);

//         if (tcp1->tcp_cons_num >= MAX_TCP_CONNECTION_NUM || tcp2->tcp_cons_num >= MAX_TCP_CONNECTION_NUM)
//         {
//             return false;
//         }

//         TCPConnection *tcp_con1 = CreateTCPConnection(planed_bytes1);
//         TCPConnection *tcp_con2 = CreateTCPConnection(planed_bytes2);

//         tcp_con1->src_ip = ipv41->ipv4_interfaces[nic_id1].ip;
//         tcp_con1->dst_ip = ipv42->ipv4_interfaces[nic_id2].ip;
//         tcp_con1->src_port = tcp1->tcp_port_offset;
//         tcp_con1->dst_port = tcp2->tcp_port_offset;
//         tcp_con2->src_ip = ipv42->ipv4_interfaces[nic_id2].ip;
//         tcp_con2->dst_ip = ipv41->ipv4_interfaces[nic_id1].ip;
//         tcp_con2->src_port = tcp2->tcp_port_offset;
//         tcp_con2->dst_port = tcp1->tcp_port_offset;
//         (tcp1->tcp_port_offset)++;
//         (tcp2->tcp_port_offset)++;

//         tcp1->tcp_cons[tcp1->tcp_cons_num] = tcp_con1;
//         tcp2->tcp_cons[tcp2->tcp_cons_num] = tcp_con2;
//         tcp1->tcp_cons_num++;
//         tcp2->tcp_cons_num++;

//         return false;
//     }

//     bool AllocateIPAddr(Node **node, int node_num, uint32_t ip_base, uint32_t mask)
//     {
//         int addr_num = 0;
//         uint32_t remaing_addr_num = -1;
//         remaing_addr_num = remaing_addr_num & (~mask);

//         for (int i = 0; i < node_num; i++)
//         {
//             auto ipv4 = GetIPv4Protocol(node[i]);
//             for (int j = 0; j < ipv4->nic_num; j++)
//             {
//                 if (remaing_addr_num <= addr_num)
//                 {
//                     return false;
//                 }
//                 ipv4->ipv4_interfaces[j].ip = ip_base + addr_num;
//                 ipv4->ipv4_interfaces[j].mask = mask;
//                 addr_num++;
//             }
//         }

//         return true;
//     }

// } // namespace VDES
