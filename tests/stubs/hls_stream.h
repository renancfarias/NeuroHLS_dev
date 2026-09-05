#pragma once

#include <queue>

namespace hls {
template<typename T>
class stream {
public:
    bool empty() const { return values_.empty(); }
    void write(const T& value) { values_.push(value); }
    T read() {
        T value = values_.front();
        values_.pop();
        return value;
    }

private:
    std::queue<T> values_;
};
}
