#ifndef PROTOCOL_TYPE_H
#define PROTOCOL_TYPE_H

// TODO: REPLACE COUNT
enum NetworkProtocolType
{
    // IPv4 = 0x0800,
    // IPv6 = 0x86DD,
    // ARP = 0x0806,
    // RARP = 0x8035
    IPv4,
    // disable 
    // IPv6,
    // ARP,
    // RARP,

    // the number of protocols
    COUNT_NetworkProtocolType
};

// enum TransportProtocolType
// {
//     TCP = 0x06,
//     UDP = 0x11
// };

enum TransportProtocolType
{
    TCP,
    // UDP,
    // ICMP,
    COUNT_TransportProtocolType
};

#endif // PROTOCOL_TYPE_H