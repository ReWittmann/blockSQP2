template <task_C task_T> inline task_T observe_end(task_T arg_task, std::future<void> *ret_end_future){
    std::promise<void> end_promise;
    *ret_end_future = end_promise.get_future();
    return task_T([cpt_task = std::move(arg_task), cpt_promise = std::move(end_promise)]() mutable {
        cpt_task();
        cpt_promise.set_value();
    });
}

#ifdef LINUX
    template<task_C task_T> inline task_T set_affinity(task_T arg_task, int ind){
        return arg_task;
    }
#else
    template<task_C task_T> inline task_T set_affinity(task_T arg_task, int ind){
        return arg_task;
    }
#endif

template<task_C task_T> inline task_T catch_exceptions(task_T arg_task, std::future<void> *ret_exception_future){
    std::promise<void> exception_promise;
    *ret_exception_future = exception_promise.get_future();
    return task_T([cpt_task = std::move(arg_task), cpt_promise = std::move(exception_promise)]() mutable {
        try {cpt_task(); cpt_promise.set_value();}
        catch (...) {cpt_promise.set_exception(std::current_exception());}
    });
}