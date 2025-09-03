#include "vdes_timer.h"
#include <vector>
#include <memory>

namespace VDES
{
    VDESTimer::VDESTimer()
    {
        // 构造函数实现
        m_kernel_params = new TimerParam();
        // 初始化其他成员变量
        m_temporary_completed_node_num_gpu = nullptr;
        m_completed_node_num_gpu = nullptr;
        m_transmission_completed_sw_num_gpu = nullptr;
        m_is_node_temporary_completed_gpu = nullptr;
        m_is_node_payload_completed_gpu = nullptr;
        m_is_transmission_completed_gpu = nullptr;
        m_is_ch_completed_gpu = nullptr;
    }

    VDESTimer::~VDESTimer()
    {
        // 析构函数实现
        if (m_kernel_params != nullptr)
            cudaFree(m_kernel_params);
        // 释放其他 GPU 内存
        if (m_temporary_completed_node_num_gpu != nullptr)
            cudaFree(m_temporary_completed_node_num_gpu);
        if (m_completed_node_num_gpu != nullptr)
            cudaFree(m_completed_node_num_gpu);
        if (m_transmission_completed_sw_num_gpu != nullptr)
            cudaFree(m_transmission_completed_sw_num_gpu);
        if (m_is_node_temporary_completed_gpu != nullptr)
            cudaFree(m_is_node_temporary_completed_gpu);
        if (m_is_node_payload_completed_gpu != nullptr)
            cudaFree(m_is_node_payload_completed_gpu);
        if (m_is_transmission_completed_gpu != nullptr)
            cudaFree(m_is_transmission_completed_gpu);

        if (m_is_ch_completed_gpu != nullptr)
            cudaFree(m_is_ch_completed_gpu);
    }

    void VDESTimer::InitKernelParams()
    {
        // 初始化内核参数
        TimerParam cpu_param;
        cpu_param.time_start = m_timestamp_start_gpu;
        cpu_param.time_end = m_timestamp_end_gpu;

        // cpu_param.completed_node_num = m_completed_node_num_gpu;
        // cpu_param.temporary_completed_node_num = m_temporary_completed_node_num_gpu;
        // cpu_param.transmission_completed_sw_num = m_transmission_completed_sw_num_gpu;

        cudaMalloc(&cpu_param.completed_node_num, sizeof(int64_t));
        cudaMalloc(&cpu_param.temporary_completed_node_num, sizeof(int64_t));
        cudaMalloc(&cpu_param.transmission_completed_sw_num, sizeof(int64_t));
        m_completed_node_num_gpu = cpu_param.completed_node_num;
        m_temporary_completed_node_num_gpu = cpu_param.temporary_completed_node_num;
        m_transmission_completed_sw_num_gpu = cpu_param.transmission_completed_sw_num;

        cudaMalloc(&cpu_param.ch_completed_ch_num, sizeof(int64_t));
        m_ch_completed_ch_num_gpu = cpu_param.ch_completed_ch_num;

        m_is_finished = false;
        cudaMalloc(&m_is_finished_gpu, sizeof(bool));
        cudaMemcpy(m_is_finished_gpu, &m_is_finished, sizeof(bool), cudaMemcpyHostToDevice);

        cpu_param.is_node_payload_completed = m_is_node_payload_completed_gpu;
        cpu_param.is_node_temporary_completed = m_is_node_temporary_completed_gpu;
        cpu_param.is_transmission_completed = m_is_transmission_completed_gpu;

        cpu_param.is_ch_completed = m_is_ch_completed_gpu;

        cpu_param.is_finished = m_is_finished_gpu;
        cpu_param.node_num = m_node_num_gpu;
        cpu_param.sw_node = m_sw_num_gpu;

        cpu_param.ch_block_num = m_ch_block_num_gpu;

        cudaMalloc(&m_kernel_params, sizeof(TimerParam));
        cudaMemcpy(m_kernel_params, &cpu_param, sizeof(TimerParam), cudaMemcpyHostToDevice);
    }

    void VDESTimer::SetTemporaryCompletedNodeNum(uint8_t *temporary_state)
    {
        // 设置临时完成节点数量
        m_is_node_temporary_completed_gpu = temporary_state;
    }

    void VDESTimer::SetCompletedNodeNum(uint8_t *completed_state)
    {
        // 设置完成节点数量
        m_is_node_payload_completed_gpu = completed_state;
    }

    void VDESTimer::SetTransmissionCompleted(uint8_t *transmission_completed)
    {
        // 设置传输完成状态
        // m_is_transmission_completed_gpu = transmission_completed;
        m_is_ch_completed_gpu = transmission_completed;
    }

    void VDESTimer::SetNodeNum(int node_num)
    {
        // 设置节点数量
        m_node_num_gpu = node_num;
    }

    void VDESTimer::SetSwitchNode(int sw_node)
    {
        // 设置交换节点数量
        m_sw_num_gpu = sw_node;
    }

    void VDESTimer::SetChannelBlockNum(int ch_block_num)
    {
        m_ch_block_num_gpu = ch_block_num;
    }

    void VDESTimer::SetTimestamp(int64_t *time_start, int64_t *time_end)
    {
        // 检查指针是否为空
        if (time_start == nullptr || time_end == nullptr)
        {
            // 处理错误，例如抛出异常或记录日志
            return;
        }

        m_timestamp_start_gpu = time_start;
        m_timestamp_end_gpu = time_end;
    }

    void VDESTimer::SetFlowStartInstants(GPUQueue<int64_t> *flow_start_instants)
    {
        // 检查指针是否为空
        if (flow_start_instants == nullptr)
        {
            // 处理错误，例如抛出异常或记录日志
            return;
        }

        // 将流开始时间实例赋值给成员变量
        m_flow_start_instants_gpu = flow_start_instants;
    }

    void VDESTimer::SetIsFinished(bool *is_finished)
    {
        // 检查指针是否为空
        if (is_finished == nullptr)
        {
            // 处理错误，例如抛出异常或记录日志
            return;
        }

        // 将任务完成状态赋值给成员变量
        m_is_finished_gpu = is_finished;
    }

    bool VDESTimer::IsFinished()
    {
        // 检查成员变量是否为空
        if (m_is_finished_gpu == nullptr)
        {
            // 处理错误，例如抛出异常或记录日志
            return false;
        }

        // 返回任务完成状态
        return m_is_finished;
    }

    void VDESTimer::BuildGraphs()
    {
        cudaGraph_t main_graph;
        cudaGraphCreate(&main_graph, 0);
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        void *d_temp_storage = nullptr;
        size_t temp_storage_size = 0;
        Sum(d_temp_storage, temp_storage_size, m_is_node_payload_completed_gpu, m_completed_node_num_gpu, m_node_num_gpu, stream);

        cudaGraph_t sub_graph_1;
        cudaMalloc(&d_temp_storage, temp_storage_size);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        Sum(d_temp_storage, temp_storage_size, m_is_node_payload_completed_gpu, m_completed_node_num_gpu, m_node_num_gpu, stream);
        cudaStreamEndCapture(stream, &sub_graph_1);

        cudaGraph_t sub_graph_2;
        cudaMalloc(&d_temp_storage, temp_storage_size);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        Sum(d_temp_storage, temp_storage_size, m_is_node_temporary_completed_gpu, m_temporary_completed_node_num_gpu, m_node_num_gpu, stream);
        cudaStreamEndCapture(stream, &sub_graph_2);

        // d_temp_storage = nullptr;
        // temp_storage_size = 0;
        // Sum(d_temp_storage, temp_storage_size, m_is_transmission_completed_gpu, m_transmission_completed_sw_num_gpu, m_sw_num_gpu, stream);

        // cudaGraph_t sub_graph_3;
        // cudaMalloc(&d_temp_storage, temp_storage_size);
        // cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        // Sum(d_temp_storage, temp_storage_size, m_is_transmission_completed_gpu, m_transmission_completed_sw_num_gpu, m_sw_num_gpu, stream);
        // cudaStreamEndCapture(stream, &sub_graph_3);

        d_temp_storage = nullptr;
        temp_storage_size = 0;
        Sum(d_temp_storage, temp_storage_size, m_is_ch_completed_gpu, m_ch_completed_ch_num_gpu, m_ch_block_num_gpu, stream);

        cudaGraph_t sub_graph_3;
        cudaMalloc(&d_temp_storage, temp_storage_size);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        Sum(d_temp_storage, temp_storage_size, m_is_ch_completed_gpu, m_ch_completed_ch_num_gpu, m_ch_block_num_gpu, stream);
        cudaStreamEndCapture(stream, &sub_graph_3);

        cudaGraph_t sub_graph_4;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        UpdateTimer(m_kernel_params, stream);
        cudaMemcpyAsync(&m_is_finished, m_is_finished_gpu, sizeof(bool), cudaMemcpyDeviceToHost, stream);
        cudaStreamEndCapture(stream, &sub_graph_4);

        cudaGraphNode_t node1, node2, node3, node4;
        cudaGraphAddChildGraphNode(&node1, main_graph, NULL, 0, sub_graph_1);
        cudaGraphAddChildGraphNode(&node2, main_graph, NULL, 0, sub_graph_2);
        cudaGraphAddChildGraphNode(&node3, main_graph, NULL, 0, sub_graph_3);
        std::vector<cudaGraphNode_t> dependencies;
        dependencies.push_back(node1);
        dependencies.push_back(node2);
        dependencies.push_back(node3);

        cudaGraphAddChildGraphNode(&node4, main_graph, dependencies.data(), dependencies.size(), sub_graph_4);
        m_graphs.push_back(main_graph);

        // cudaGraphExec_t exec;
        // cudaGraphInstantiate(&exec, main_graph, NULL, NULL, 0);
        // m_graph_execs.push_back(exec);
    }

    cudaGraph_t VDESTimer::GetGraph()
    {
        // 返回第一个子图
        return m_graphs[0];
    }

} // namespace VDES
