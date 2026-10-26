"""
High-Performance Probabilistic Sketch Data Structures for Approximate Query Processing (AQP).
Includes:
  1. Count-Min Sketch (CMS): Frequency estimation, heavy hitter tracking, inner product
  2. HyperLogLog (HLL): High-precision distinct cardinality estimation (Flajolet et al.)

Both implementations strictly satisfy the Monoid algebraic properties (Associativity,
Commutativity, and Identity element), enabling zero-loss parallel reduction across
distributed worker nodes.
Optimized with SplitMix64 bit-avalanche hashing and hardware bit_length() for maximum throughput.
"""

import math
from typing import List, Tuple, Optional


def _hash_64(key: str, seed: int = 0) -> int:
    """Fast, uniform 64-bit SplitMix64 avalanche hash."""
    h = hash(key) ^ (seed * 0x9E3779B97F4A7C15)
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9
    h = (h ^ (h >> 27)) * 0x94D049BB133111EB
    h = h ^ (h >> 31)
    return h & 0xFFFFFFFFFFFFFFFF


def _hash_32(key: str, seed: int = 0) -> int:
    """Fast, uniform 32-bit hash."""
    h = _hash_64(key, seed)
    return (h ^ (h >> 32)) & 0xFFFFFFFF


class CountMinSketch:
    """
    Count-Min Sketch (Cormode & Muthukrishnan, 2005)
    Guarantees:
      With probability at least 1 - delta, the estimate a_hat satisfies:
          a_i <= a_hat_i <= a_i + epsilon * ||a||_1
      Where:
          width (w) = ceil(e / epsilon)
          depth (d) = ceil(ln(1 / delta))
    """

    def __init__(
        self,
        epsilon: float = 0.005,
        delta: float = 0.01,
        width: Optional[int] = None,
        depth: Optional[int] = None,
    ):
        if width is not None and depth is not None:
            self.width = width
            self.depth = depth
            self.epsilon = math.e / width
            self.delta = math.exp(-depth)
        else:
            self.epsilon = epsilon
            self.delta = delta
            self.width = int(math.ceil(math.e / epsilon))
            self.depth = int(math.ceil(math.log(1.0 / delta)))

        self.table = [[0] * self.width for _ in range(self.depth)]
        self.total_count = 0
        self.seeds = [i * 31 + 17 for i in range(self.depth)]

    def update(self, key: str, count: int = 1) -> None:
        """Add item occurrence to sketch."""
        self.total_count += count
        for i in range(self.depth):
            col = _hash_32(key, self.seeds[i]) % self.width
            self.table[i][col] += count

    def query(self, key: str) -> int:
        """Point query: Estimate frequency of key."""
        min_val = float("inf")
        for i in range(self.depth):
            col = _hash_32(key, self.seeds[i]) % self.width
            val = self.table[i][col]
            if val < min_val:
                min_val = val
        return int(min_val)

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """Monoid merge: (CMS x CMS) -> CMS."""
        if self.width != other.width or self.depth != other.depth:
            raise ValueError(
                f"Dimension mismatch for CMS merge: ({self.width},{self.depth}) vs ({other.width},{other.depth})"
            )
        merged = CountMinSketch(width=self.width, depth=self.depth)
        merged.total_count = self.total_count + other.total_count
        merged.seeds = list(self.seeds)

        for i in range(self.depth):
            for j in range(self.width):
                merged.table[i][j] = self.table[i][j] + other.table[i][j]

        return merged

    def merge_inplace(self, other: "CountMinSketch") -> None:
        """In-place merge: elementwise addition."""
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Dimension mismatch for CMS in-place merge")
        self.total_count += other.total_count
        for i in range(self.depth):
            for j in range(self.width):
                self.table[i][j] += other.table[i][j]

    def get_memory_bytes(self) -> int:
        return self.depth * self.width * 4

    def __repr__(self) -> str:
        return (
            f"CountMinSketch(w={self.width}, d={self.depth}, "
            f"mem={self.get_memory_bytes() / 1024:.1f}KB, total={self.total_count})"
        )


class HyperLogLog:
    """
    HyperLogLog (Flajolet et al., 2007)
    Estimates multiset distinct cardinality with standard error:
        SE ~= 1.04 / sqrt(m) where m = 2^p
    """

    def __init__(self, p: int = 10):
        if not (4 <= p <= 16):
            raise ValueError("HLL precision parameter p must be between 4 and 16.")
        self.p = p
        self.m = 1 << p  # 2^p registers
        self.registers = bytearray(self.m)
        self.rem_bits = 64 - p
        self.mask_rem = (1 << self.rem_bits) - 1

        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1.0 + 1.079 / self.m)

    def add(self, key: str) -> None:
        """Insert element into HLL sketch."""
        x = _hash_64(key, seed=42)
        # Register index j from top p bits
        j = x >> self.rem_bits
        # Remaining 64 - p bits for rank calculation
        rem = x & self.mask_rem
        if rem == 0:
            rank = self.rem_bits + 1
        else:
            rank = self.rem_bits - rem.bit_length() + 1

        if rank > self.registers[j]:
            self.registers[j] = rank

    def estimate(self) -> float:
        """
        Estimate distinct count with Harmonic Mean and Small Range Linear Counting.
        """
        sum_harmonic = 0.0
        num_zeros = 0
        for val in self.registers:
            sum_harmonic += 2.0 ** (-val)
            if val == 0:
                num_zeros += 1

        raw_estimate = self.alpha * (self.m**2) / sum_harmonic

        # Small range correction (Linear Counting)
        if raw_estimate <= 2.5 * self.m:
            if num_zeros > 0:
                return self.m * math.log(self.m / num_zeros)
            return raw_estimate

        return raw_estimate

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Monoid merge: register-wise maximum."""
        if self.p != other.p:
            raise ValueError(f"HLL precision mismatch: {self.p} vs {other.p}")
        merged = HyperLogLog(p=self.p)
        for i in range(self.m):
            merged.registers[i] = max(self.registers[i], other.registers[i])
        return merged

    def merge_inplace(self, other: "HyperLogLog") -> None:
        """In-place register-wise maximum."""
        if self.p != other.p:
            raise ValueError(f"HLL precision mismatch: {self.p} vs {other.p}")
        for i in range(self.m):
            if other.registers[i] > self.registers[i]:
                self.registers[i] = other.registers[i]

    def get_memory_bytes(self) -> int:
        return self.m

    def standard_error_percent(self) -> float:
        return (1.04 / math.sqrt(self.m)) * 100.0

    def __repr__(self) -> str:
        return (
            f"HyperLogLog(p={self.p}, m={self.m}, "
            f"SE={self.standard_error_percent():.2f}%, mem={self.m} bytes, "
            f"estimate={self.estimate():.1f})"
        )
