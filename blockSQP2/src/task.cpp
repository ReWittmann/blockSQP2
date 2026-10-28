#include <blockSQP2/task.hpp>
#include <string>

namespace blockSQP2{
    
    BasicThreadHandler::~BasicThreadHandler(){}
    
    void BasicThreadHandler::assert_noexcept(){
        int N_threads = get_N_threads();
        std::unique_ptr<std::exception_ptr[]> exceptions = std::make_unique<std::exception_ptr[]>(N_threads);
        int N_except = 0;
        for (int i = 0; i < N_threads; i++){
            try{
                assert_noexcept(i);
                exceptions[i] = nullptr;
            }
            catch (const std::exception &){
                exceptions[i] = std::current_exception();
                N_except++;
            }
        }
        if (N_except > 0) throw MultiException("BasicThreadHandler: Exception(s) in thread(s):", std::move(exceptions), N_threads);
    }
    
    void BasicThreadHandler::combine_except(std::exception_ptr arg_except){
        int N_threads = get_N_threads();
        std::unique_ptr<std::exception_ptr[]> exceptions = std::make_unique<std::exception_ptr[]>(N_threads + 1);
        for (int i = 0; i < N_threads; i++){
            try{
                assert_noexcept(i);
                exceptions[i+1] = nullptr;
            }
            catch (const std::exception &){
                exceptions[i+1] = std::current_exception();
            }
        }
        exceptions[0] = arg_except;
        throw MultiException("BasicThreadHandler: Exception(s) in thread(s):", std::move(exceptions), N_threads + 1);
    }
    
    MultiException::MultiException(std::string_view arg_header, std::unique_ptr<std::exception_ptr[]> arg_exceptions, int arg_N_except):
            exceptions(std::move(arg_exceptions)), N_except(arg_N_except), msg(create_msg(arg_header)), header(arg_header){}
    
    const char* MultiException::what() const noexcept {return msg.c_str();}
    
    std::string MultiException::create_msg(std::string_view arg_what, std::string_view offset) const {
        std::string ret_msg = std::string("MultiException: ");
        ret_msg.append(arg_what).append("\n");// Exception(s) thrown in threads:\n");
        for (int i = 0; i < N_except; i++){
            if (exceptions[i]) try {std::rethrow_exception(exceptions[i]);}
            catch (const MultiException &ME){
                ret_msg.append(" --> " + std::to_string(i) + ": std::exception(\"");
                ret_msg.append(ME.create_msg(ME.header, "   "));
                ret_msg.append(offset).append("  \")\n");
            }
            catch (const std::exception &e){
                ret_msg.append(offset).append(" --> " + std::to_string(i) + ": std::exception(\"");
                ret_msg.append(e.what()).append("\")\n");
            }
            catch (...){
                ret_msg.append(" --> thread " + std::to_string(i) + ": could not handle exception (not a subclass of std::exception)");
            }
        }
        return ret_msg;
    }
    
    
    ThreadHandler::ThreadHandler(int arg_N_threads):
            exception_futures(std::make_unique<std::future<void>[]>(arg_N_threads)),
            end_futures(std::make_unique<std::future<void>[]>(arg_N_threads)),
            threads(std::make_unique<std::jthread[]>(arg_N_threads)),
            N_threads(arg_N_threads){}
    
    ThreadHandler::~ThreadHandler(){}
    
    void ThreadHandler::place(task_T arg_task, int ind){
        if (ind >= N_threads) throw std::runtime_error("Cannot assign task to thread " + std::to_string(ind) + ", only " + std::to_string(get_N_threads()) + " are available");
        threads[ind] = std::jthread(set_affinity(
                                        catch_exceptions(
                                            observe_end(
                                                std::move(arg_task),
                                                &end_futures[ind]),
                                            &exception_futures[ind]),
                                        ind)
                                    );
    }
    
    void ThreadHandler::wait(int ind){
        end_futures[ind].wait();
    }
    
    std::future_status ThreadHandler::wait_for(std::chrono::nanoseconds arg_duration, int ind){
        return end_futures[ind].wait_for(arg_duration);
    }
    
    
    void ThreadHandler::assert_noexcept(int ind){
        if (exception_futures[ind].valid()){
            exception_futures[ind].wait();
            exception_futures[ind].get();
        }
    }
    
    void ThreadSubHandler::assert_noexcept(int ind){parent->assert_noexcept(thread_ind[ind]);}
}
