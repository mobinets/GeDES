// the implementation of thread pool

#include "thread_pool.h"

namespace VDES
{
    ThreadPool *global_pool = nullptr;

    ThreadPool::ThreadPool(size_t num_threads) : m_stop(false)
    {
        for (size_t i = 0; i < num_threads; ++i)
        {
            m_workers.emplace_back([this]
                                   {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(m_queue_mutex);
                        m_condition.wait(lock, [this] { return m_stop || !m_tasks.empty(); });
                        if (m_stop && m_tasks.empty()) {
                            return;
                        }
                        task = std::move(m_tasks.front());
                        m_tasks.pop_front();
                    }

                    task();
                } });
        }
    }

    ThreadPool::~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_stop = true;
        }
        m_condition.notify_all();
        for (std::thread &worker : m_workers)
        {
            worker.join();
        }
    }

    void ThreadPool::stop()
    {
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_stop = true;
        }
        m_condition.notify_all();
    }

    void create_thread_pool(int num_threads)
    {
        if (global_pool == nullptr)
        {
            global_pool = new ThreadPool(num_threads);
        }
    }

}