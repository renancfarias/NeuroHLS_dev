#pragma once

template<int Width>
class ap_uint {
public:
    ap_uint(unsigned int value = 0) : value_(value) {}
    template<int OtherWidth>
    ap_uint(const ap_uint<OtherWidth>& value) : value_((unsigned int)value) {}
    ap_uint& operator=(unsigned int value) { value_ = value; return *this; }
    operator unsigned int() const { return value_; }

private:
    unsigned int value_;
};
