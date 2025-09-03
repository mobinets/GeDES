#include "conf.h"
// TODO: INCLUDE ADDITIONAL HEADERS HERE
#include "memory_pool.h"
#include "packet_definition.h"
#include "arp_protocol.h"

namespace VDES
{
    int THREAD_POOL_SIZE = 16;
    int CPU_PARALLELISM_BATCH_UNIT = 100;
    int KERNEL_BLOCK_WIDTH = 32;
    int KERNEL_BLOCK_HEIGHT = 32;
    int MAC_ENTITY_NUM = 20;
    int DEFAULT_QUEUE_SIZE = 50;

    /** true if we enable the global cache strategy(moving unused data from GPU to CPU)
     *  false if we disable the global cache strategy(keeping all data in GPU memory)
     */
    int GLOBAL_CACHE_STRATEGY = 0;

    // MemoryPoolGPU<VDES::BasicPacket> *basic_packet_pool;

    MemoryPoolGPU<VDES::Frame> *frame_pool = NULL;
    MemoryPoolGPU<VDES::Ipv4Packet> *ipv4_packet_pool = NULL;
    MemoryPoolGPU<VDES::TCPPacket> *tcp_packet_pool;
    MemoryPoolGPU<VDES::UDPPacket> *udp_packet_pool;
    MemoryPoolGPU<VDES::Payload> *payload_pool;
    MemoryPoolGPU<VDES::ArpCache> *arp_cache_pool;

    MemoryPoolCPU<VDES::Frame> *frame_pool_cpu;
    MemoryPoolCPU<VDES::Ipv4Packet> *ipv4_packet_pool_cpu;
    MemoryPoolCPU<VDES::TCPPacket> *tcp_packet_pool_cpu;
    MemoryPoolCPU<VDES::UDPPacket> *udp_packet_pool_cpu;
    // MemoryPoolCPU<VDES::BasicPacket> *basic_packet_pool_cpu;
    // MemoryPoolGPU<ARRAY_UNIT> *array_pool;

    int INITIALIZED_MEMORY_POOL_SIZE = 60000000;
    int MEMORY_POOL_EXPAND_SIZE = 1000000;

    void SetThreadPoolSize(int size)
    {
        THREAD_POOL_SIZE = size;
    }

    int GetThreadPoolSize()
    {
        return THREAD_POOL_SIZE;
    }

    void SetCpuParallelismBatchUnit(int value)
    {
        CPU_PARALLELISM_BATCH_UNIT = value;
    }

    void SetKernelBlockWidth(int value)
    {
        KERNEL_BLOCK_WIDTH = value;
    }

    void SetKernelBlockHeight(int value)
    {
        KERNEL_BLOCK_HEIGHT = value;
    }

    void SetMacEntityNum(int value)
    {
        MAC_ENTITY_NUM = value;
    }

    void InitializeMemoryPools()
    {
        // TODO: ADD ADDITIONAL MEMORY POOLS HERE.
        // basic_packet_pool = new MemoryPoolGPU<VDES::BasicPacket>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);
        // basic_packet_pool_cpu = new MemoryPoolCPU<VDES::BasicPacket>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);

        frame_pool = new MemoryPoolGPU<VDES::Frame>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);
        ipv4_packet_pool = new MemoryPoolGPU<VDES::Ipv4Packet>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);
        tcp_packet_pool = new MemoryPoolGPU<VDES::TCPPacket>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);
        // udp_packet_pool = new MemoryPoolGPU<VDES::UDPPacket>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);
        // payload_pool = new MemoryPoolGPU<VDES::Payload>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);
        // arp_cache_pool = new MemoryPoolGPU<VDES::ArpCache>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);
        // array_pool = new MemoryPoolGPU<ARRAY_UNIT>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);

        // frame_pool_cpu = new MemoryPoolCPU<VDES::Frame>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);
        // ipv4_packet_pool_cpu = new MemoryPoolCPU<VDES::Ipv4Packet>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);
        // tcp_packet_pool_cpu = new MemoryPoolCPU<VDES::TCPPacket>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);
        // udp_packet_pool_cpu = new MemoryPoolCPU<VDES::UDPPacket>(INITIALIZED_MEMORY_POOL_SIZE, MEMORY_POOL_EXPAND_SIZE);
    }

    void SetInitialMemoryPoolSize(int size)
    {
        INITIALIZED_MEMORY_POOL_SIZE = size;
    }
}