// queue operation on GPU

#ifndef GPU_QUEUE_CUH
#define GPU_QUEUE_CUH

// TODO: INCLUDE THE REQUIREMENT LIBRARIES
// #include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
// #include <cstdlib>
// #include <device_functions.h>

template <typename T>
struct GPUQueue
{
    T *queue;
    // the total capacity of the queues
    int queue_capacity;
    // the number of elements
    int size;
    // the header of round-robin queue
    int head;

    int removed_record;

    __host__ void init(int capacity)
    {
        cudaError_t err = cudaMalloc(&queue, sizeof(T) * capacity);
        if (err != cudaSuccess)
        {
            printf("Error: cudaMalloc failed in GPUQueue::init\n");
            return;
        }
        // TODO: HEAD
        head = 0;
        // TODO: INITIALIZE SIZE AND CAPACITY
        size = 0;
        queue_capacity = capacity;
    }

    __host__ void destroy()
    {
        cudaFree(queue);
    }

    __device__ __host__ T get_element(int index)
    {
        int index_in_queue = (index + head) % queue_capacity;
        // TODO: QUEUE
        return queue[index_in_queue];
    }

    __device__ __host__ void set_element(int index, T value)
    {
        int index_in_queue = (index + head) % queue_capacity;
        queue[index_in_queue] = value;
    }

    __device__ __host__ void append_element(T value)
    {
        int index_in_queue = (head + size) % queue_capacity;
        queue[index_in_queue] = value;
        size++;
    }

    __device__ bool append_elements(T *values, int num)
    {
        if (num <= 0)
        {
            return true;
        }

        int remaining_size = queue_capacity - size;
        if (remaining_size < num)
        {
            return false;
        }
        else
        {
            int tail = (head + size) % queue_capacity;
            int tail_size = std::min(queue_capacity - tail, remaining_size);
            int copy_size = std::min(tail_size, num);
            /**
             * TODO: The start copying position of the queue is queue + tail
             */

            memcpy(queue + tail, values, copy_size * sizeof(T));
            if (copy_size < num)
            {
                memcpy(queue, values + copy_size, (num - copy_size) * sizeof(T));
            }
            size += num;
            return true;
        }
    }

    __device__ bool append_elements(GPUQueue<T> *gpu_queue, int num)
    {
        int num1 = std::min(gpu_queue->queue_capacity - gpu_queue->head, num);
        int num2 = std::max(num - num1, 0);

        append_elements(gpu_queue->queue + gpu_queue->head, num1);
        append_elements(gpu_queue->queue, num2);

        return true;
    }

    __device__ __host__ void remove_elements(int remove_size)
    {
        head = (head + remove_size) % queue_capacity;
        size -= remove_size;
        removed_record = remove_size;
    }

    __device__ __host__ void clear()
    {
        head = 0;
        size = 0;
    }

    __device__ void memcpy_gpu(T *dst, T *src, int size)
    {
        for (int i = 0; i < size; i++)
        {
            dst[i] = src[i];
        }
    }

    __device__ T get_element(int head1, int index1)
    {
        return queue[(head1 + index1) % queue_capacity];
    }

    __device__ T set_element(int head1, int index1, T value)
    {
        return queue[(head1 + index1) % queue_capacity] = value;
    }

    __device__ int get_remaining_capacity()
    {
        return queue_capacity - size;
    }

    __device__ T append_element(int index1, T value)
    {
        return queue[(head + size + index1) % queue_capacity] = value;
    }

    __device__ T get_head()
    {
        int old_head = head;
        head = (head + 1) % queue_capacity;
        size--;
        return queue[old_head];
    }
};

template <typename T>
GPUQueue<T> *create_gpu_queue(int capacity);

template <typename R>
GPUQueue<R> *convert_gpu_queue_type(void *gpu_queue);

#endif // GPU_QUEUE_CUH