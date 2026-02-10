def get_closest_divisor(number: int, candidite_divisor: int):
    
    smaller_divisor = candidite_divisor

    while smaller_divisor > 0 and number % smaller_divisor != 0:
        smaller_divisor -= 1

    greater_divisor = candidite_divisor

    while number % greater_divisor != 0:
        greater_divisor += 1

    if greater_divisor - candidite_divisor < candidite_divisor - smaller_divisor:
        return greater_divisor
    
    return smaller_divisor