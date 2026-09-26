#pragma once

#include <pthread.h>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

class EngineThread {
    pthread_t handle{};
    bool active = false;

    struct StartData {
        std::function<void()> function;
    };

    static void* trampoline(void* raw){
        std::unique_ptr<StartData> data(static_cast<StartData*>(raw));
        data->function();
        return nullptr;
    }

public:
    EngineThread() = default;

    EngineThread(const EngineThread&) = delete;
    EngineThread& operator=(const EngineThread&) = delete;

    EngineThread(EngineThread&& other) noexcept
        : handle(other.handle), active(other.active) {
        other.active = false;
    }

    EngineThread& operator=(EngineThread&& other) noexcept {
        if (this != &other){
            if (active) pthread_join(handle, nullptr);
            handle = other.handle;
            active = other.active;
            other.active = false;
        }
        return *this;
    }

    ~EngineThread(){
        if (active) pthread_join(handle, nullptr);
    }

    template<class Function, class... Args>
    void start(Function&& function, Args&&... args){
        if (active) pthread_join(handle, nullptr);

        auto data = std::make_unique<StartData>();
        data->function = std::bind(std::forward<Function>(function),
                                   std::forward<Args>(args)...);

        pthread_attr_t attr;
        if (pthread_attr_init(&attr) != 0)
            throw std::runtime_error("pthread_attr_init failed");

        // Android normally gives new pthreads about 1 MiB. Search recursion plus
        // move-generation locals can be deeper than a desktop build expects.
        // 8 MiB matches the traditional desktop-sized safety margin without
        // changing the engine's search code or strength.
        constexpr size_t ENGINE_STACK_SIZE = 8ULL * 1024ULL * 1024ULL;
        if (pthread_attr_setstacksize(&attr, ENGINE_STACK_SIZE) != 0){
            pthread_attr_destroy(&attr);
            throw std::runtime_error("pthread_attr_setstacksize failed");
        }

        int result = pthread_create(&handle, &attr, trampoline, data.get());
        pthread_attr_destroy(&attr);

        if (result != 0)
            throw std::runtime_error("pthread_create failed");

        data.release();
        active = true;
    }

    bool joinable() const noexcept {
        return active;
    }

    void join(){
        if (active){
            pthread_join(handle, nullptr);
            active = false;
        }
    }
};
