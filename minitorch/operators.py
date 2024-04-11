
##task 0.1

import math

def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y

def id(x: float) -> float:
    """$f(x) = x$"""
    return x

def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y

def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -x

def lt(x: float, y: float) -> float:
    """$f(x) = 1.0$ if x is less than y else 0.0"""
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    """$f(x) = 1.0$ if x is equal to y else 0.0"""
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    """$f(x) = x$ if x is greater than y else y"""
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    """$f(x) = 1$ if $|x - y| < 1e-2$, else 0"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0

def sigmoid(x: float) -> float:
    """
    $f(x) = \frac{1.0}{1.0 + e^{-x}}$ if x >= 0, else $\frac{e^x}{1.0 + e^x}$ (for stability)
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    """$f(x) = x$ if x is greater than 0, else 0"""
    return max(0.0, x)



EPS = 1e-6

def mul(x: float, y: float) -> float:
    """$f(x, y) = x \cdot y$"""
    return x * y

def id(x: float) -> float:
    """$f(x) = x$"""
    return x

def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y

def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -x

def lt(x: float, y: float) -> float:
    """$f(x) = 1.0$ if $x$ is less than $y$, else $0.0$"""
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    """$f(x) = 1.0$ if $x$ is equal to $y$, else $0.0$"""
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    """$f(x) = x$ if $x$ is greater than $y$, else $y$"""
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    """$f(x) = 1$ if $|x - y| < 1e-2$, else $0$"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0

def sigmoid(x: float) -> float:
    """
    $f(x) = \frac{1.0}{1.0 + e^{-x}}$ if $x \geq 0$, else $\frac{e^x}{1.0 + e^x}$ (for stability)
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    """$f(x) = x$ if $x$ is greater than 0, else 0"""
    return max(0.0, x)

def log_back(x: float, d: float) -> float:
    """If $f = \log$, compute $d \cdot f'(x)$"""
    return d / (x + EPS)

def inv(x: float) -> float:
    """$f(x) = 1/x$"""
    return 1.0 / x

def inv_back(x: float, d: float) -> float:
    """If $f(x) = 1/x$, compute $d \cdot f'(x)$"""
    return -d / (x ** 2)

def relu_back(x: float, d: float) -> float:
    """If $f = \text{ReLU}$, compute $d \cdot f'(x)$"""
    return d if x > 0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.


# minitorch/operators.py

from typing import Callable, Iterable

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    Parameters:
        fn (Callable[[float], float]): Function from one value to one value.

    Returns:
        Callable[[Iterable[float]], Iterable[float]]: A function that takes a list, applies fn to each element, and returns a new list.
    """
    def apply_map(ls):
        return [fn(x) for x in ls]
    return apply_map

def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Negate each element in ls using map.

    Parameters:
        ls (Iterable[float]): List of elements.

    Returns:
        Iterable[float]: List with negated elements.
    """
    return map(lambda x: -x)(ls)

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipWith (or map2).

    Parameters:
        fn (Callable[[float, float], float]): Combine two values.

    Returns:
        Callable[[Iterable[float], Iterable[float]], Iterable[float]]: Function that takes two equally sized lists ls1 and ls2, and produces a new list by applying fn(x, y) on each pair of elements.
    """
    def apply_zip_with(ls1, ls2):
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return apply_zip_with

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Add the elements of ls1 and ls2 using zipWith and add.

    Parameters:
        ls1 (Iterable[float]): First list of elements.
        ls2 (Iterable[float]): Second list of elements.

    Returns:
        Iterable[float]: List with added elements.
    """
    return zipWith(lambda x, y: x + y)(ls1, ls2)

def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    """
    Higher-order reduce.

    Parameters:
        fn (Callable[[float, float], float]): Combine two values.
        start (float): Start value.

    Returns:
        Callable[[Iterable[float]], float]: Function that takes a list ls of elements and reduces them using fn.
    """
    def apply_reduce(ls):
        result = start
        for x in ls:
            result = fn(result, x)
        return result
    return apply_reduce
