#ifndef __AP_FIXED_H__
#define __AP_FIXED_H__

#include <cmath>
#include <iostream>

// Implementação simplificada de ap_fixed para testes
template<int W, int I>
class ap_fixed {
private:
    double value;
    static constexpr int F = W - I;  // Bits fracionários
    static constexpr double scale = (1 << F);  // 2^F
    static constexpr double max_val = (1 << (W-1)) / scale - 1.0/scale;
    static constexpr double min_val = -(1 << (W-1)) / scale;
    
    void saturate() {
        if (value > max_val) value = max_val;
        if (value < min_val) value = min_val;
    }
    
public:
    // Construtores
    ap_fixed() : value(0.0) {}
    ap_fixed(double v) : value(v) { saturate(); }
    ap_fixed(float v) : value((double)v) { saturate(); }
    ap_fixed(int v) : value((double)v) { saturate(); }
    
    // Conversão para float
    float to_float() const { return (float)value; }
    double to_double() const { return value; }
    
    // Operador de atribuição
    ap_fixed& operator=(double v) {
        value = v;
        saturate();
        return *this;
    }
    
    ap_fixed& operator=(float v) {
        value = (double)v;
        saturate();
        return *this;
    }
    
    // Operadores aritméticos
    ap_fixed operator+(const ap_fixed& other) const {
        return ap_fixed(value + other.value);
    }
    
    ap_fixed operator-(const ap_fixed& other) const {
        return ap_fixed(value - other.value);
    }
    
    ap_fixed operator*(const ap_fixed& other) const {
        return ap_fixed(value * other.value);
    }
    
    ap_fixed operator/(const ap_fixed& other) const {
        return ap_fixed(value / other.value);
    }
    
    // Operadores com double
    friend ap_fixed operator+(double a, const ap_fixed& b) {
        return ap_fixed(a + b.value);
    }
    
    friend ap_fixed operator-(double a, const ap_fixed& b) {
        return ap_fixed(a - b.value);
    }
    
    friend ap_fixed operator*(double a, const ap_fixed& b) {
        return ap_fixed(a * b.value);
    }
    
    friend ap_fixed operator/(double a, const ap_fixed& b) {
        return ap_fixed(a / b.value);
    }
    
    ap_fixed operator+(double other) const {
        return ap_fixed(value + other);
    }
    
    ap_fixed operator-(double other) const {
        return ap_fixed(value - other);
    }
    
    ap_fixed operator*(double other) const {
        return ap_fixed(value * other);
    }
    
    ap_fixed operator/(double other) const {
        return ap_fixed(value / other);
    }
    
    // Operadores de comparação
    bool operator>=(const ap_fixed& other) const {
        return value >= other.value;
    }
    
    bool operator<=(const ap_fixed& other) const {
        return value <= other.value;
    }
    
    bool operator>(const ap_fixed& other) const {
        return value > other.value;
    }
    
    bool operator<(const ap_fixed& other) const {
        return value < other.value;
    }
    
    bool operator==(const ap_fixed& other) const {
        return value == other.value;
    }
    
    // Operadores compostos
    ap_fixed& operator+=(const ap_fixed& other) {
        value += other.value;
        saturate();
        return *this;
    }
    
    ap_fixed& operator-=(const ap_fixed& other) {
        value -= other.value;
        saturate();
        return *this;
    }
    
    ap_fixed& operator*=(const ap_fixed& other) {
        value *= other.value;
        saturate();
        return *this;
    }
    
    ap_fixed& operator/=(const ap_fixed& other) {
        value /= other.value;
        saturate();
        return *this;
    }
    
    // Operador de negação
    ap_fixed operator-() const {
        return ap_fixed(-value);
    }
    
    // Operador de stream
    friend std::ostream& operator<<(std::ostream& os, const ap_fixed& f) {
        os << f.value;
        return os;
    }
};

#endif // __AP_FIXED_H__
