//  implements an efficient thread pool to execute tasks concurrently

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <future>
#include <thread>
#include <mutex>
#include <functional>

namespace VDES
{

    class ThreadPool
    {
    public:
        ThreadPool(size_t num_threads);
        ~ThreadPool();

        template <class F, class... Args>
        auto enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>;
        template <class F, class... Args>
        auto enqueue_head(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>;

        void stop();

    private:
        std::vector<std::thread> m_workers;
        std::deque<std::function<void()>> m_tasks;

        std::mutex m_queue_mutex;
        std::condition_variable m_condition;
        bool m_stop;
    };

    template <class F, class... Args>
    auto ThreadPool::enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            if (m_stop)
            {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            m_tasks.emplace([task]()
                            { (*task)(); });
        }
        m_condition.notify_one();
        return res;
    }

    // insert to queue haed
    template <class F, class... Args>
    auto ThreadPool::enqueue_head(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);

            if (m_stop)
            {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            m_tasks.emplace_front([task]()
                                  { (*task)(); });
        }
        m_condition.notify_one();
        return res;
    }

    extern ThreadPool *global_pool;

    void create_thread_pool(int num_threads);

}
#endif // THREAD_POOL_H