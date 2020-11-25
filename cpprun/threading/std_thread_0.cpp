// gcc std_thread_0.cpp -std=c++11 -lstdc++ -o /tmp/std_thread_0 && /tmp/std_thread_0
#include <string>
#include <iostream>
#include <thread>

void task1(std::string msg)
{
    std::cout << "task1 says: " << msg << std::endl ; 
}

int main()
{
    std::cout << "before thread creation" << std::endl ; 

    std::thread t1(task1, "Hello");

    std::cout << "after thread creation" << std::endl ; 

    t1.join();

    std::cout << "after join" << std::endl ; 

    return 0 ; 
}
