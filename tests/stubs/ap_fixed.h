#pragma once

#include <cmath>
#include "ap_int.h"

template<int Width, int IntegerBits>
class ap_fixed {
public:
    ap_fixed(double value = 0.0) : value_(value) {}

    template<int OtherWidth, int OtherIntegerBits>
    ap_fixed(const ap_fixed<OtherWidth, OtherIntegerBits>& value)
        : value_((double)value) {}

    template<typename T>
    ap_fixed& operator=(const T& value) { value_ = (double)value; return *this; }

    operator double() const { return value_; }

    template<typename T> ap_fixed& operator+=(const T& value) { value_ += (double)value; return *this; }
    template<typename T> ap_fixed& operator-=(const T& value) { value_ -= (double)value; return *this; }

    ap_fixed operator>>(int shift) const {
        return ap_fixed(std::ldexp(value_, -shift));
    }

    ap_fixed operator<<(int shift) const {
        return ap_fixed(std::ldexp(value_, shift));
    }

    ap_fixed& operator>>=(int shift) {
        value_ = std::ldexp(value_, -shift);
        return *this;
    }

    ap_fixed& operator<<=(int shift) {
        value_ = std::ldexp(value_, shift);
        return *this;
    }

private:
    double value_;
};

template<int Width, int IntegerBits>
class ap_ufixed : public ap_fixed<Width, IntegerBits> {
public:
    using ap_fixed<Width, IntegerBits>::ap_fixed;

    class range_proxy {
    public:
        explicit range_proxy(ap_ufixed& owner) : owner_(owner) {}
        template<int OtherWidth>
        range_proxy& operator=(const ap_uint<OtherWidth>& value) {
            owner_ = (unsigned int)value;
            return *this;
        }
    private:
        ap_ufixed& owner_;
    };

    range_proxy range(int, int) { return range_proxy(*this); }
};
