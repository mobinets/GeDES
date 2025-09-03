// This is a simple memory pool implementation based on c++17 standard.
// It is designed as a template class, and can be used to allocate memory of any type.
// It allows parallel allocation and deallocation.
// It enables allocation and deallocation of batches of memory chunks.
// It is implemented based on a round-robin queue of free memory chunks.
// It don't need to manage allocated memory chunks, as it only needs to keep track of free chunks.

#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <queue>
#include <mutex>
#include <optional>
// TODO: ADD ADITIONAL HEADERS HERE
#include <algorithm>
#include <cstdio>

namespace VDES
{

    template <typename T>
    class MemoryPoolGPU
    {
    public:
        // TODO: USE LARGER SPACE FOR INITIALIZATION
        MemoryPoolGPU(size_t initialSize = 100, int expand_size = 100)
        {
            expandPool(initialSize);
            this->expand_size = expand_size;
        }

        ~MemoryPoolGPU()
        {
            for (auto ptr : allocatedChunks)
            {
                cudaFree(ptr);
            }
        }

        T *allocate()
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (freeChunks.empty())
            {
                // TODO: EXPAND TWO TIMES SIZE
                expandPool(expand_size);
            }
            if (!freeChunks.empty())
            {
                T *chunk = freeChunks.back();
                freeChunks.erase(freeChunks.end() - 1);
                return chunk;
            }
            return NULL;
        }

        // allocate a batch of memory chunks
        std::vector<T *> allocate(int size)
        {
            std::vector<T *> chunks;
            chunks.reserve(size);

            mutex.lock();
            if (size > freeChunks.size())
            {
                // TODO: EXPAND TWO TIMES SIZE
                expandPool(std::max(size, expand_size) << 1);
            }

            // TODO: CHECK FREE CHUNKS SIZE
            if (freeChunks.empty())
            {
                printf("No enough free chunks\n");
                return chunks;
            }

            chunks.insert(chunks.end(), freeChunks.end() - size, freeChunks.end());
            freeChunks.erase(freeChunks.end() - size, freeChunks.end());
            mutex.unlock();
            // TODO: RETURN THE CHUNKS
            return chunks;
        }

        void deallocate(T *chunk)
        {
            std::lock_guard<std::mutex> lock(mutex);
            freeChunks.push_back(chunk);
        }

        void deallocate(T **chunks, int num)
        {
            mutex.lock();
            if (num > freeChunks.capacity() - freeChunks.size())
            {
                freeChunks.reserve(freeChunks.size() + num << 2);
            }
            freeChunks.insert(freeChunks.end(), chunks, chunks + num);
            mutex.unlock();
        }

        void set_expand_size(int size)
        {
            expand_size = size;
        }

        void resize(int size)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (freeChunks.size() > size)
            {
                int free_chunk_num = freeChunks.size() - size;
                for (int i = 0; i < free_chunk_num; i++)
                {
                    T *chunk = freeChunks.back();
                    freeChunks.erase(freeChunks.end() - 1);
                    cudaFree(chunk);
                    // delete chunk from alloatedChunks vector

                    auto it = std::find(allocatedChunks.begin(), allocatedChunks.end(), chunk);
                    if (it != allocatedChunks.end())
                    {
                        allocatedChunks.erase(it);
                    }
                }
            }
        }

    private:
        void expandPool(int size)
        {
            T *newChunks;
            cudaError_t err = cudaMalloc(&newChunks, sizeof(T) * size);
            if (err != cudaSuccess)
            {
                printf("error malloc: %s, size: %d,", cudaGetErrorString(err), size);
                return;
            }
            for (int i = 0; i < size; i++)
            {
                freeChunks.push_back(newChunks + i);
            }
            allocatedChunks.push_back(newChunks);
        }

        std::vector<T *> freeChunks;
        std::vector<T *> allocatedChunks;
        std::mutex mutex;
        int expand_size = 100;
    };

    template <typename T>
    class MemoryPoolCPU
    {
    public:
        // TODO: USE LARGER SPACE FOR INITIALIZATION
        MemoryPoolCPU(int initialSize = 100, int expand_size = 100)
        {
            expandPool(initialSize);
            this->expand_size = expand_size;
        }

        ~MemoryPoolCPU()
        {
            for (auto ptr : allocatedChunks)
            {
                cudaFree(ptr);
            }
        }

        T *allocate()
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (freeChunks.empty())
            {
                // TODO: EXPAND TWO TIMES SIZE
                expandPool(expand_size);
            }
            if (!freeChunks.empty())
            {
                T *chunk = freeChunks.back();
                freeChunks.erase(freeChunks.end() - 1);
                return chunk;
            }
            return NULL;
        }

        // allocate a batch of memory chunks
        std::vector<T *> allocate(int size)
        {
            std::vector<T *> chunks;
            chunks.reserve(size);

            std::lock_guard<std::mutex> lock(mutex);
            if (size > freeChunks.size())
            {
                // TODO: EXPAND TWO TIMES SIZE
                expandPool(std::max(size, expand_size) << 1);
            }

            if (freeChunks.empty())
            {
                printf("No enough free chunks\n");
                return chunks;
            }

            chunks.insert(chunks.end(), freeChunks.end() - size, freeChunks.end());
            freeChunks.erase(freeChunks.end() - size, freeChunks.end());

            // TODO: RETURN THE CHUNKS
            return chunks;
        }

        void deallocate(T *chunk)
        {
            std::lock_guard<std::mutex> lock(mutex);
            freeChunks.push_back(chunk);
        }

        void deallocate(T **chunks, int num)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (num > freeChunks.capacity() - freeChunks.size())
            {
                freeChunks.reserve(freeChunks.size() + num << 2);
            }
            freeChunks.insert(freeChunks.end(), chunks, chunks + num);
        }

        void set_expand_size(int size)
        {
            expand_size = size;
        }

        void resize(int size)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (freeChunks.size() > size)
            {
                int free_chunk_num = freeChunks.size() - size;
                for (int i = 0; i < free_chunk_num; i++)
                {
                    T *chunk = freeChunks.front();
                    freeChunks.erase(freeChunks.end() - 1);
                    cudaFree(chunk);
                    // delete chunk from alloatedChunks vector

                    auto it = std::find(allocatedChunks.begin(), allocatedChunks.end(), chunk);
                    if (it != allocatedChunks.end())
                    {
                        allocatedChunks.erase(it);
                    }
                }
            }
        }

    private:
        void expandPool(int size)
        {
            T *newChunks;
            newChunks = new T[size];
            for (size_t i = 0; i < size; i++)
            {
                freeChunks.push_back(newChunks + i);
            }
            allocatedChunks.push_back(newChunks);
        }

        std::vector<T *> freeChunks;
        std::vector<T *> allocatedChunks;
        std::mutex mutex;
        int expand_size = 100;
    };

} // namespace VDES
#endif // MEMORY_POOL_H
