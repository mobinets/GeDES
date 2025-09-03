// configuration variables
#ifndef CONFIG_VDES_H
#define CONFIG_VDES_H
#include <stdint.h>
#include <cuda_runtime.h>
#include "memory_pool.h"
#include "packet_definition.h"
#include <memory>

#define TCP_PACKET_DEFAULT_LEN 20
#define UDP_PACKET_DEFAULT_LEN 8
#define IPV4_PACKET_DEFAULT_LEN 20
#define IPV4_VERSION_AND_IHL 0x45
#define IPV4_DEFAULT_TTL 255
#define FRAME_DEFAULT_HEADER_LEN 18

#define NODE_DEFAULT_INGRESS_QUEUE_SIZE 100
#define NODE_DEFAULT_EGRESS_QUEUE_SIZE 200
#define Switch_DEFAULT_INGRESS_QUEUE_SIZE 100
#define Switch_DEFAULT_EGRESS_QUEUE_SIZE 200
#define MAX_ROUTING_TABLE_SIZE 30

#define MAX_TRANSMITTED_PACKET_NUM 18 // 18 packets per timeslot
#define MAX_GENERATED_PACKET_NUM 20   // 20 packets per timeslot
#define MAX_SWAP_FRAME_NUM 15         // 15 frames per timeslot
#define TIMESLOT_LENGTH 1000          // ns
#define MAX_TCP_CONNECTION_NUM 10     // 10 TCP connections per node
#define DCTCP_K 30                    // the threshold in DCTCP to trigger ECN
#define DCTCP_G 0.0625

// 0 denotes close, 1 denotes open
#define SWITCH_MAC_ROUTING 1
#define ENABLE_ARP_CACHE 1
#define ENABLE_FATTREE_MODE 1
#define SEND_NULL_PAYLOAD 1
#define ENABLE_CACHE 0
#define ENABLE_HUGE_GRAPH 1
#define SWND_RATE 10
#define ENABLE_GPU_MEM_POOL 1
#define ENABLE_DCTCP 1

#define MAX_PACKET_WARP 32

inline __device__ int GLOBAL_CACHE_STRATEGY_GPU = 0;

namespace VDES
{

    // typedef struct
    // {
    //     uint8_t arr[ARRAY_UNIT_SIZE];
    // } ARRAY_UNIT;

    extern int THREAD_POOL_SIZE;
    extern int CPU_PARALLELISM_BATCH_UNIT;
    extern int KERNEL_BLOCK_WIDTH;
    extern int KERNEL_BLOCK_HEIGHT;

    extern int INITIALIZED_MEMORY_POOL_SIZE;
    extern int MEMORY_POOL_EXPAND_SIZE;
    extern int MAC_ENTITY_NUM;
    extern int DEFAULT_QUEUE_SIZE;

    typedef enum
    {
        UNKNOWN,
        L2,
        L3,
        L4
    } DeviceType;

    /**
     * TODO: GLOBAL CACHE STRATEGY OPTION, DEFAULT FALSE.
     */
    extern int GLOBAL_CACHE_STRATEGY;

    extern MemoryPoolGPU<VDES::Frame> *frame_pool;
    extern MemoryPoolGPU<VDES::Ipv4Packet> *ipv4_packet_pool;
    extern MemoryPoolGPU<VDES::TCPPacket> *tcp_packet_pool;
    extern MemoryPoolGPU<VDES::UDPPacket> *udp_packet_pool;
    // TEMP: PAYLOAD POOL
    extern MemoryPoolGPU<VDES::Payload> *payload_pool;
    // extern MemoryPoolGPU<VDES::ArpCache> *arp_cache_pool;
    // extern MemoryPoolGPU<ARRAY_UNIT> *array_pool;

    // TODO: CPU memory pool
    extern MemoryPoolCPU<VDES::Frame> *frame_pool_cpu;
    extern MemoryPoolCPU<VDES::Ipv4Packet> *ipv4_packet_pool_cpu;
    extern MemoryPoolCPU<VDES::TCPPacket> *tcp_packet_pool_cpu;
    extern MemoryPoolCPU<VDES::UDPPacket> *udp_packet_pool_cpu;

    void SetThreadPoolSize(int size);
    void SetCpuParallelismBatchUnit(int value);
    void SetKernelBlockWidth(int value);
    void SetKernelBlockHeight(int value);
    void SetMacEntityNum(int value);
    void SetInitialMemoryPoolSize(int size);


    void InitializeMemoryPools();

}

#endif // CONFIG_H