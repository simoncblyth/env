// gcc std_thread_split_workers_0.cpp -std=c++11 -lstdc++ -o /tmp/std_thread_split_workers_0 && /tmp/std_thread_split_workers_0

// https://eli.thegreenplace.net/2016/the-promises-and-challenges-of-stdasync-task-based-parallelism-in-c11/

#include <vector>
#include <thread>
#include <numeric>
#include <iostream>

// Demonstrates how to launch two threads and return two results to the caller
// that will have to wait on those threads. Gives half the input vector to
// one thread, and the other half to another.


void accumulate_block_worker(int* data, size_t count, int* result) {
  *result = std::accumulate(data, data + count, 0);
}

std::vector<std::thread> launch_split_workers_with_std_thread(std::vector<int>& v, std::vector<int>* results) 
{
    std::vector<std::thread> threads;
    threads.emplace_back(accumulate_block_worker, v.data()               , v.size() / 2, &((*results)[0]));
    threads.emplace_back(accumulate_block_worker, v.data() + v.size() / 2, v.size() / 2, &((*results)[1]));
    return threads;
}



void launch_split_workers_with_std_thread()
{
    // Usage
    std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> results(2, 0);
    
    std::vector<std::thread> threads = launch_split_workers_with_std_thread(v, &results);
    for (auto& t : threads) t.join();

    std::cout << "results from launch_split_workers_with_std_thread: " << results[0] << " and " << results[1] << "\n";
}




/////////////////////////////////////
#include <future>


int accumulate_block_worker_ret(int* data, size_t count) {
  return std::accumulate(data, data + count, 0);
}

using int_futures = std::vector<std::future<int>>;

int_futures launch_split_workers_with_std_async(std::vector<int>& v) 
{
    int_futures futures;
    futures.push_back(std::async(std::launch::async, accumulate_block_worker_ret, v.data()               , v.size() / 2));
    futures.push_back(std::async(std::launch::async, accumulate_block_worker_ret, v.data() + v.size() / 2, v.size() / 2));
    return futures;
}

void launch_split_workers_with_std_async()
{
    std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8};
    int_futures futures = launch_split_workers_with_std_async(v);
    std::cout << "results from launch_split_workers_with_std_async: "
               << futures[0].get() << " and " << futures[1].get() << "\n";
}

/**
https://eli.thegreenplace.net/2016/the-promises-and-challenges-of-stdasync-task-based-parallelism-in-c11/

I like how the future decouples the task from the result. In more complex code,
you can pass the future somewhere else, and it encapsulates both the thread to
wait on and the result you'll end up with. The alternative of using std::thread
directly is more cumbersome, because there are two things to pass around.

**/

int main(int argc, char** argv)
{
    launch_split_workers_with_std_thread() ; 
    launch_split_workers_with_std_async() ;

    return 0 ; 
}










