#include "gpu_queue.cuh"
#include "packet_definition.h"
#include "switch.h"
#include "arp_protocol.h"
#include "component.h"
#include "tcp_controller.h"

template <typename T>
GPUQueue<T> *create_gpu_queue(int capacity)
{
    GPUQueue<T> queue_host;
    GPUQueue<T> *queue_dev;

    queue_host.init(capacity);

    cudaMalloc((void **)&queue_dev, sizeof(GPUQueue<T>));
    cudaMemcpy(queue_dev, &queue_host, sizeof(GPUQueue<T>), cudaMemcpyHostToDevice);

    return queue_dev;
}

template <typename R>
GPUQueue<R> *convert_gpu_queue_type(void *gpu_queue)
{
    GPUQueue<R> *queue_dev = (GPUQueue<R> *)gpu_queue;
    return queue_dev;
}

template GPUQueue<int8_t> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<int16_t> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<int32_t> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<int64_t> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<VDES::Frame *> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<void *> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<VDES::Ipv4Packet *> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<unsigned char> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<VDES::IPv4RoutingRule *> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<VDES::TCPPacket *> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<VDES::Payload *> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<uint16_t> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<uint32_t> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<uint64_t> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<VDES::UDPPacket *> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<VDES::ArpCache *> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<VDES::MacForwardingRule *> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<VDES::Flow> *convert_gpu_queue_type(void *gpu_queue);
template GPUQueue<VDES::RecvPacketRecord *> *convert_gpu_queue_type(void *gpu_queue);

// TODO: ADD DATATPYE VOID*
// instantiate the template for the required data type
template GPUQueue<int8_t> *create_gpu_queue(int capacity);
template GPUQueue<int16_t> *create_gpu_queue(int capacity);
template GPUQueue<int32_t> *create_gpu_queue(int capacity);
template GPUQueue<int64_t> *create_gpu_queue(int capacity);
template GPUQueue<VDES::Frame *> *create_gpu_queue(int capacity);
template GPUQueue<void *> *create_gpu_queue(int capacity);
template GPUQueue<VDES::Ipv4Packet *> *create_gpu_queue(int capacity);
template GPUQueue<unsigned char> *create_gpu_queue(int capacity);
template GPUQueue<VDES::IPv4RoutingRule *> *create_gpu_queue(int capacity);
template GPUQueue<VDES::TCPPacket *> *create_gpu_queue(int capacity);
template GPUQueue<VDES::Payload *> *create_gpu_queue(int capacity);
template GPUQueue<uint16_t> *create_gpu_queue(int capacity);
template GPUQueue<uint32_t> *create_gpu_queue(int capacity);
template GPUQueue<uint64_t> *create_gpu_queue(int capacity);
template GPUQueue<VDES::UDPPacket *> *create_gpu_queue(int capacity);
template GPUQueue<VDES::ArpCache *> *create_gpu_queue(int capacity);
template GPUQueue<VDES::MacForwardingRule *> *create_gpu_queue(int capacity);
template GPUQueue<VDES::Flow> *create_gpu_queue(int capacity);
template GPUQueue<VDES::RecvPacketRecord *> *create_gpu_queue(int capacity);
