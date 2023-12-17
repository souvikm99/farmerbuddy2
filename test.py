def generate_numbers():
    yield 1
    yield 2
    yield 3

# Iterating over the generator and unpacking the yielded values
num1, num2, num3 = generate_numbers()

print(num1)   # Output: 1
print(num2)   # Output: 2
print(num3)   # Output: 3
