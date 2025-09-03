#ifndef ARP_PROTOCOL_H
#define ARP_PROTOCOL_H

#include <cuda_runtime.h>
#include <memory> 

namespace VDES {
    struct ArpCache
    {
        int32_t ip;
        uint8_t mac[6];
    };
}




#endif // ARP_PROTOCOL_H