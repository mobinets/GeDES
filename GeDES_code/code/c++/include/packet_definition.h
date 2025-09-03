#ifndef PACKET_DEFINITION_H
#define PACKET_DEFINITION_H

#include <cstdint>
#include <cuda_runtime.h>
#define ENABLE_DCTCP_PACKET 1

namespace VDES
{
    enum class Device
    {
        CPU = 0,
        GPU
    };

    typedef struct Frame
    {
        uint8_t dst_mac[6];
        uint8_t src_mac[6];
        uint8_t data[8];
        // used the first byte as the ecn idetifier
        uint8_t fcs[4];
        // transmission time of the frame,
        uint8_t timestamp[8];

        // Frame length in bytes, including FCS
        uint8_t frame_len[2];

        uint8_t type[2];
        uint8_t device;
        uint8_t port;

    } Frame;

    typedef struct Ipv4Packet
    {
        uint8_t version_header_len;
        uint8_t dscp_ecn;
        uint8_t total_len[2];
        uint8_t identification[2];
        uint8_t flags_fragment_offset[2];
        uint8_t time_to_live;
        uint8_t protocol;
        uint8_t header_checksum[2];
        uint8_t src_ip[4];
        uint8_t dst_ip[4];

        // TODO: MOVE params for routing
        uint8_t next_hop[4];

        // the length of options is scalable
        // uint8_t options[4];

        // data
        uint8_t payload[8];
        // TODO: THE LOCATION OF THE NEXT PACKET
        uint8_t device;

        // egress interface id
        uint8_t next_if;
        // identify the routing error type
        uint8_t err_code;

        uint8_t timestamp[8];
    } Ipv4Packet;

    // TODO: CHANGE THE STRUCTURE OF TCPPACKET AND PAYLOAD
    typedef struct TCPPacket
    {
        uint8_t src_port[2];
        uint8_t dst_port[2];
        uint8_t sequence_number[4];
        uint8_t acknowledgement_number[4];
        // used the second byte to identify ecn
        uint8_t data_offset_reserved_flags[2];
        uint8_t window_size[2];
        uint8_t checksum[2];
        uint8_t urgent_pointer[2];

        // scalable option field
        // uint8_t options[4];

        // params for directing to correct tcp connection
        // uint8_t connection_id;
        // TODO: FILL THE REST OF THE FIELDS
        // uint8_t padding1[3];

        // data
        uint8_t payload[8];
        uint8_t payload_len[2];
        uint8_t send_timestamp[8];

        // extra fields for packet tracing
        uint8_t src_ip[4];
        uint8_t dst_ip[4];
        uint8_t recv_timestamp[8];

#if ENABLE_DCTCP_PACKET
        // Used for DCTCP
        uint8_t ECE;
#endif
    } TCPPacket;

    typedef struct Payload
    {
        // the pointer to the start of the payload
        uint8_t payload[8];
        // the length of the payload in bytes
        uint8_t payload_len[2];
        // TODO: FILL THE REST OF THE FIELDS
        // uint8_t padding[2];
        // sequence number of the payload, used for reordering
        uint8_t sequence_number[4];
        // send and receive timestamp of the payload
        uint8_t send_timestamp[8];
        uint8_t recv_timestamp[8];
    } Payload;

    typedef struct UDPPacket
    {
        uint8_t src_port[2];
        uint8_t dst_port[2];
        uint8_t length[2];
        uint8_t checksum[2];

        // TODO: FILL DATA TO ALIGN THE STRUCTURE
        // data
        uint8_t payload[8];

        // param for directing to correct udp connection
        uint8_t connection_id;
        uint8_t padding[7];
    } UDPPacket;

    typedef struct
    {
        // ip
        uint32_t dst;
        uint32_t mask;

        // the ip of next hop, gw=0 means local delivery
        uint32_t gw;

        // if id
        uint8_t if_id;

        // priority score
        int priority;

    } IPv4RoutingRule;

    typedef struct
    {
        // ip and mask
        uint32_t ip;
        uint32_t mask;
        // mac
        uint8_t mac[6];

    } IPv4Interface;

    enum class RoutingError
    {
        NO_ERROR,
        DESTINATION_UNRECHEABLE,
        TTL_EXPIRED,
    };

    typedef struct
    {
        // ip
        uint32_t ip;
        // mac
        uint8_t mac[6];

    } ARPRule;

    typedef struct
    {
        // mac
        uint8_t mac[6];

        // next hop
        uint8_t port;
    } MacForwardingRule;

    typedef struct
    {
        int64_t timestamp;
        int64_t flow_size;
        int64_t tiimestamp_end;
        int64_t byte_offset;
    } Flow;

}

#endif