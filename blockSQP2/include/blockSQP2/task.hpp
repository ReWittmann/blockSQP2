#ifndef BLOCKSQP2_TASK_HPP
#define BLOCKSQP2_TASK_HPP

#include <functional>
#include <chrono>
#include <thread>
#include <future>
#include <memory>
#include <stdexcept>


namespace blockSQP2{

//Compound exception for exceptions thrown in threads
class MultiException : public std::exception{
    const std::unique_ptr<std::exception_ptr[]> exceptions;
    const int N_except;
    const std::string msg;
    const std::string header;
    
    public:
    MultiException(std::string_view arg_header, std::unique_ptr<std::exception_ptr[]> arg_exceptions, int arg_N_except);
    std::string create_msg(std::string_view arg_what, std::string_view offset = "") const;
    const char* what() const noexcept;
};


class BasicThreadHandler{
    public:
    virtual ~BasicThreadHandler();
    virtual int get_N_threads() const = 0;
    virtual void place(std::move_only_function<void()> arg_task, int ind) = 0;
    virtual void wait(int ind) = 0;
    virtual std::future_status wait_for(std::chrono::nanoseconds duration, int ind) = 0;
    virtual void assert_noexcept(int ind) = 0;
    
    virtual void assert_noexcept();
    virtual void combine_except(std::exception_ptr arg_except); //Combine given exception with all exceptions thrown from tasks into a MultiException.
};

// Add functionality to passed tasks
inline std::move_only_function<void()> observe_end(std::move_only_function<void()> arg_task, std::future<void> *ret_end_future);
inline std::move_only_function<void()> set_affinity(std::move_only_function<void()> arg_task, int ind);
inline std::move_only_function<void()> catch_exceptions(std::move_only_function<void()> arg_task, std::future<void> *ret_exception_future);


class ThreadHandler : public BasicThreadHandler{
    public:
    std::unique_ptr<std::future<void>[]> exception_futures;
    std::unique_ptr<std::future<void>[]> end_futures;
    std::unique_ptr<std::jthread[]> threads;
    int N_threads;
    
    ThreadHandler(int arg_N_threads);
    ~ThreadHandler();
    int get_N_threads() const {return N_threads;}
    void place(std::move_only_function<void()> arg_task, int ind);
    void wait(int ind);
    std::future_status wait_for(std::chrono::nanoseconds arg_duration, int ind);
    // void assert_noexcept();
    void assert_noexcept(int ind);
    // void combine_except(std::exception_ptr arg_except);
};


class ThreadSubHandler : public BasicThreadHandler{
    public:
    BasicThreadHandler *parent;
    std::unique_ptr<int[]> thread_ind;
    int N_threads;
    
    ThreadSubHandler(BasicThreadHandler *arg_parent, std::unique_ptr<int[]> arg_thread_ind, int arg_N_threads):
        parent(arg_parent), thread_ind(std::move(arg_thread_ind)), N_threads(arg_N_threads){}
    ThreadSubHandler(): parent(nullptr), thread_ind(nullptr), N_threads(0){}
    ThreadSubHandler(const ThreadSubHandler &other):
            parent(other.parent), thread_ind(std::make_unique<int[]>(other.N_threads)), N_threads(other.N_threads){
        thread_ind = std::make_unique<int[]>(N_threads);
        std::copy(other.thread_ind.get(), other.thread_ind.get() + N_threads, thread_ind.get());
    }
    ThreadSubHandler &operator=(const ThreadSubHandler &other){
        if (this != &other){
            parent = other.parent; 
            N_threads = other.N_threads;
            thread_ind = std::make_unique<int[]>(N_threads);
            std::copy(other.thread_ind.get(), other.thread_ind.get() + N_threads, thread_ind.get());
        }
        return *this;
    }
    
    int get_N_threads() const {return N_threads;}
    void place(std::move_only_function<void()> arg_task, int ind){parent->place(std::move(arg_task), thread_ind[ind]);}
    void wait(int ind){parent->wait(thread_ind[ind]);}
    std::future_status wait_for(std::chrono::nanoseconds arg_duration, int ind){return parent->wait_for(arg_duration, thread_ind[ind]);}
    void assert_noexcept(int ind);
};

// Fake ThreadHandler that simply executes tasks in the calling thread.
class ThisThreadHandle : public BasicThreadHandler{
    public:
    ThisThreadHandle(){};
    int get_N_threads() const {return 1;}
    void place(std::move_only_function<void()> arg_task, int ind){arg_task();}
    void wait(int ind){};
    std::future_status wait_for(std::chrono::nanoseconds duration, int ind){return std::future_status::ready;}
    void assert_noexcept(){}
    void assert_noexcept(int ind){}
    void combine_except(std::exception_ptr arg_except){std::rethrow_exception(arg_except);}
};



#include <blockSQP2/inline/task.ipp>
}
#endif