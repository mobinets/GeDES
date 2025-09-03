#ifndef IPV4_ENCAPSULATION_H
#define IPV4_ENCAPSULATION_H
#include "conf.h"
#include "gpu_queue.cuh"
#include "packet_definition.h"
#include "protocol_type.h"
#include <cstring>
#include <numeric>
#include <vector>

namespace VDES
{
  typedef struct
  {
    GPUQueue<uint8_t *> **l4_packet_queues;
    // structured by |TCP|UDP|
    GPUQueue<Ipv4Packet *> **ipv4_packet_queues;

    int node_num;

#if ENABLE_CACHE
    // swap out packets
    uint8_t **l4_cache_space;
    uint8_t **l4_swap_out_packets;
    int *l4_swap_out_packet_num;
    int *l4_swap_out_offset;
#endif

    int *l4_packet_size;
    int *l4_src_ip_offset;
    int *l4_dst_ip_offset;
    int *l4_timestamp_offset;
    int *l4_len_offset;

    int max_packet_num;

    // alloc ipv4 packets
    Ipv4Packet **alloc_ipv4_packets;
    int *packet_offset_per_node;
    int *used_packet_num_per_node;

  } IPv4EncapsulationParams;

  class IPv4EncapsulationController
  {

  private:
    std::vector<cudaStream_t> m_streams;
    std::vector<IPv4EncapsulationParams *> m_kernel_params;
    std::vector<cudaGraph_t> m_graphs;
    std::vector<cudaGraphExec_t> m_graph_execs;

    // kernel params
    std::vector<GPUQueue<uint8_t *> *> m_l4_packet_queues;
    std::vector<GPUQueue<Ipv4Packet *> *> m_ipv4_packet_queues;

#if ENBALE_CACHE
    // cache space
    // gpu ptr on gpu, the ptr of cache space
    std::vector<uint8_t **> m_l4_cache_space_ptr_gpu;
    std::vector<uint8_t **> m_l4_cache_space_ptr_cpu;

    // cpu ptr on gpu
    std::vector<uint8_t **> m_l4_swap_out_packets_ptr_gpu;
    // cpu ptr on cpu
    std::vector<uint8_t **> m_l4_swap_out_packets_ptr_cpu;
    // std::vector<uint8_t **> m_l4_swap_out_packets_ptr_cpu_backup;
    std::vector<uint8_t *> m_l4_cache_space_gpu;
    std::vector<uint8_t *> m_l4_cache_space_cpu;
    std::vector<int *> m_l4_swap_out_packet_num_gpu;
    std::vector<int *> m_l4_swap_out_packet_num_cpu;
#endif

    std::vector<int> m_cache_sizes;
    std::vector<int> m_max_packet_num;

    // alloc ipv4 packets
    std::vector<Ipv4Packet **> m_alloc_ipv4_packets_gpu;
    std::vector<Ipv4Packet **> m_alloc_ipv4_packets_cpu;
    std::vector<int *> m_used_packet_num_per_node_gpu;
    std::vector<int *> m_used_packet_num_per_node_cpu;

    // packet offset
    std::vector<int *> m_packet_offset_per_node;

    // batch properties
    std::vector<int> m_batch_start_index;
    std::vector<int> m_batch_end_index;

    std::vector<int> m_nic_num_per_node;

    std::vector<int> m_packet_sizes;

#if ENABLE_HUGE_GRAPH

    std::vector<cudaMemcpy3DParms> m_memcpy_param;
    std::vector<cudaHostNodeParams> m_host_param;

#endif

  public:
    IPv4EncapsulationController();
    ~IPv4EncapsulationController();

    void InitalizeKernelParams();
    void SetIPv4PacketQueue(GPUQueue<Ipv4Packet *> **queues, int node_num);
    void SetL4PacketQueue(GPUQueue<uint8_t *> **queues, int node_num);
    void SetBatchProperties(int *batch_start_index, int *batch_end_index,
                            int node_num);
    void SetNICNumPerNode(int *nic_num_per_node, int node_num);
    void SetStreams(cudaStream_t *streams, int node_num);

    void CacheOutL4Packets(int batch_id);
    void UpdateUsedIPv4Packets(int batch_id);
    void BuildGraph(int batch_id);
    void BuildGraph();
    void LaunchInstance(int batch_id);

    cudaGraph_t GetGraph(int batch_id);

#if ENABLE_CACHE
    void CacheOutL4Packets(int batch_id);
#endif

    void Run(int batch_id);
    void Run();

#if ENABLE_HUGE_GRAPH
    std::vector<cudaMemcpy3DParms> &GetMemcpyParams();
    std::vector<cudaHostNodeParams> &GetHostParams();
#endif

    std::vector<void *> GetAllocateInfo();
  };

  void LaunchIPv4EncapsulationKernel(dim3 grid, dim3 block,
                                     IPv4EncapsulationParams *params,
                                     cudaStream_t stream);
  std::vector<void *> GetAllocateInfo();
} // namespace VDES

#endif