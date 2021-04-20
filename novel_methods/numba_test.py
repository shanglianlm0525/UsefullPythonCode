# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/20 9:24
# @Author : liumin
# @File : numba_test.py

import math
import time
from numba import njit, prange

"""
    total prime num is 664579
    cost 53.910285234451294s
    
    total prime num is 664579
    cost 3.5231616497039795s
    
    total prime num is 664579
    cost 1.087864875793457s
"""

# @njit 相当于 @jit(nopython=True)
@njit
def is_prime(num):
    if num == 2:
        return True
    if num <= 1 or not num % 2:
        return False
    for div in range(3, int(math.sqrt(num) + 1), 2):
        if not num % div:
            return False
    return True

# @njit 相当于 @jit(nopython=True)
@njit(parallel = True)
def run_program(N):
    total = 0
    for i in prange(N):
        if is_prime(i):
            total += 1
    return total


if __name__ == "__main__":
    N = 10000000
    start = time.time()
    total = run_program(N)
    end = time.time()
    print(f"total prime num is {total}")
    print(f"cost {end - start}s")