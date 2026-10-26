inline std::move_only_function<void()> observe_end(std::move_only_function<void()> arg_task, std::future<void> *ret_end_future){
    std::promise<void> end_promise;
    *ret_end_future = end_promise.get_future();
    return std::move_only_function<void()>([cpt_task = std::move(arg_task), cpt_promise = std::move(end_promise)]() mutable {
        cpt_task();
        cpt_promise.set_value();
    });
}

#ifdef LINUX
    inline std::move_only_function<void()> set_affinity(std::move_only_function<void()> arg_task, int ind){
        return arg_task;
    }
#else
    inline std::move_only_function<void()> set_affinity(std::move_only_function<void()> arg_task, int ind){
        return arg_task;
    }
#endif

inline std::move_only_function<void()> catch_exceptions(std::move_only_function<void()> arg_task, std::future<void> *ret_exception_future){
    std::promise<void> exception_promise;
    *ret_exception_future = exception_promise.get_future();
    return std::move_only_function<void()>([cpt_task = std::move(arg_task), cpt_promise = std::move(exception_promise)]() mutable {
        try {cpt_task(); cpt_promise.set_value();}
        catch (...) {cpt_promise.set_exception(std::current_exception());}
    });
}