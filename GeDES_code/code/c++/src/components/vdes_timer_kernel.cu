#include "vdes_timer.h"
#include "packet_definition.h"
#include <cub/device/device_reduce.cuh>

__global__ void increase_timer(int64_t *time_start, int64_t *time_end)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0)
    {
        *time_end += 1000;
        *time_start += 1000;
    }
}

__global__ void update_timer(VDES::TimerParam *param)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0)
    {
        // if (*param->ch_completed_ch_num == param->ch_block_num)
        // {
        //     if (*param->completed_node_num == param->node_num)
        //     {
        //         /* code */
        //         *(param->is_finished) = true;
        //         return;
        //     }
        //     else if (*param->temporary_completed_node_num == param->node_num)
        //     {
        //         /* code */
        //         int size = param->flow_start_instants->size;
        //         for (int i = 0; i < size; i++)
        //         {
        //             if (param->flow_start_instants->get_element(i) >= *(param->time_end))
        //             {
        //                 *(param->time_start) = param->flow_start_instants->get_element(i);
        //                 *(param->time_end) = *(param->time_start) + 1000;
        //                 param->flow_start_instants->remove_elements(i);
        //                 break;
        //             }
        //         }
        //     }
        // }
        if (*param->ch_completed_ch_num == param->ch_block_num && *param->completed_node_num == param->node_num)
        {
            *(param->is_finished) = true;
            return;
        }
        else
        {
            *(param->time_start) += 1000;
            *(param->time_end) += 1000;

            // printf("%ld\n", *(param->time_start));
        }
    }
}

void Sum(void *d_temp_storage, size_t &temp_storage_bytes, uint8_t *d_in, int64_t *d_out, int64_t num, cudaStream_t stream)
{
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num, stream);
}

void UpdateTimer(VDES::TimerParam *param, cudaStream_t stream)
{
    update_timer<<<1, 1, 0, stream>>>(param);
}

cudaGraph_t create_timer_graph(int64_t *time_start, int64_t *time_end)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    LaunchTimer(time_start, time_end, stream);
    cudaStreamEndCapture(stream, &graph);

    return graph;
}

void LaunchTimer(int64_t *time_start, int64_t *time_end, cudaStream_t stream)
{
    increase_timer<<<1, 1, 0, stream>>>(time_start, time_end);
}