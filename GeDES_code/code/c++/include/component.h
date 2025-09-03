#ifndef COMPONENT_VDES_H
#define COMPONENT_VDES_H

#include "gpu_queue.cuh"
#include "packet_definition.h"
#include "conf.h"
#include "tcp_controller.h"
#include <vector>

namespace VDES
{

    typedef struct
    {
        GPUQueue<Frame *> *ingress;
        GPUQueue<Frame *> *egress;
        uint8_t mac_addr[6];
        bool is_connected;
    } NIC;

    typedef struct
    {
        uint8_t network_layer;
        uint8_t protocol_type;
    } ProtocolHeader;

    typedef struct
    {
        ProtocolHeader protocol_header;
        GPUQueue<Ipv4Packet *> *ingress;
        GPUQueue<Ipv4Packet *> **egresses;
        IPv4Interface *ipv4_interfaces;
        int nic_num;

        GPUQueue<Ipv4Packet *> *local_delivery;

        // routing table
        GPUQueue<IPv4RoutingRule *> *routing_table;
        GPUQueue<Ipv4Packet *> *error_queue;

    } IPv4Protocol;

    typedef struct
    {
        ProtocolHeader protocol_header;
        TCPConnection **tcp_cons;
        int tcp_cons_num;

        GPUQueue<TCPPacket *> *ingress;
        GPUQueue<TCPPacket *> *egress;

        int tcp_port_offset;

    } TCPProtocol;

    typedef struct
    {
        NIC *nic1;
        NIC *nic2;
        int tx_rate;
        int popogation_delay;
    } P2PChanenl;

    typedef struct
    {
        int node_id;
        std::vector<NIC *> nics;
        std::vector<ProtocolHeader *> l3_protocols;
        std::vector<ProtocolHeader *> l4_protocols;
    } Node;

    typedef struct
    {
        // unified node id, used to network devices
        int node_id;

        // switch id
        int sw_id;
        int port_num;
        NIC **nics;
    } Switch;

    NIC *CreateNIC(int ingress_capacity, int egress_capacity);
    void SetNICMacAddr(NIC *nic, uint8_t *mac_addr);
    P2PChanenl *CreateP2PChannel(int tx_rate, int popogation_delay);
    bool ConnectNICByP2P(NIC *nic1, NIC *nic2, P2PChanenl *ch);
    Node *CreateNode(int nic_num, int ingress_capacity, int egress_capacity);
    Switch *CreateSwitch(int port_num, int ingress_capacity, int egress_capacity);
    bool ConnectDevices(Node *node1, int nic_id, Switch *sw1, int port_id, P2PChanenl *ch);
    bool ConnectDevices(Switch *sw1, int port_id1, Switch *sw2, int port_id2, P2PChanenl *ch);
    IPv4Protocol *CreateIPv4Protocol(int nic_num);
    TCPProtocol *CreateTCPProtocol(int nic_num);
    bool InstallIPv4Protocol(Node *node);
    bool InstallTCPProtocol(Node *node);
    TCPConnection *CreateTCPConnection(int64_t planned_bytes);
    bool ConnectTCPConnection(Node *node1, int64_t planed_bytes1, int nic_id1, Node *node2, int64_t planed_bytes2, int nic_id2);
    IPv4Protocol *GetIPv4Protocol(Node *node);
    TCPProtocol *GetTCPProtocol(Node *node);
    bool AllocateIPAddr(Node **node, int node_num, uint32_t ip_base, uint32_t mask);
    void HostNodeCallback(void *user_data);

} // namespace VDES

#endif