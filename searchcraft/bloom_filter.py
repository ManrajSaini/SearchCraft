import math
import hashlib


class BloomFilter:
    """
    A space-efficient probabilistic data structure that answers:
      "Is this element in the set?"

    Answers are either:
      DEFINITELY NOT  — 100% accurate, no false negatives
      PROBABLY YES    — may be wrong (false positive rate ≈ p, see below)

    How it works:
      - An array of `size` bits, all initialised to 0
      - `num_hashes` independent hash functions
      - add(x)          → set bits at each of the k hash positions to 1
      - might_contain(x)→ if ANY of those k bits is 0 → definitely absent
                          if ALL k bits are 1          → probably present

    Sizing formulas (stored as class methods for transparency):
      optimal size     m = -(n × ln p) / (ln 2)²
      optimal hashes   k =  (m / n) × ln 2

    Where n = expected number of items, p = desired false-positive rate.

    With size=50_000 and num_hashes=5 for ~375 tokens:
      actual false-positive rate ≈ < 0.0001%  (effectively zero at this scale)
    """

    def __init__(self, size: int = 50_000, num_hashes: int = 5):
        self.size       = size
        self.num_hashes = num_hashes
        self._bits      = bytearray(math.ceil(size / 8))  # packed bit array
        self._count     = 0   # number of items added

    # ── bit operations ────────────────────────────────────────────────────────

    def _set_bit(self, pos: int) -> None:
        self._bits[pos // 8] |= (1 << (pos % 8))

    def _get_bit(self, pos: int) -> bool:
        return bool(self._bits[pos // 8] & (1 << (pos % 8)))

    # ── hash positions ────────────────────────────────────────────────────────

    def _hash_positions(self, item: str) -> list[int]:
        """
        Generate `num_hashes` independent bit positions for `item`.

        Uses double-hashing: h_i(x) = (h1(x) + i * h2(x)) mod size
        This avoids needing k truly independent hash functions — one
        SHA-256 and one MD5 are enough to derive all k positions.
        """
        h1 = int(hashlib.sha256(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.md5(item.encode()).hexdigest(),    16)
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, item: str) -> None:
        """Add an item to the filter."""
        for pos in self._hash_positions(item):
            self._set_bit(pos)
        self._count += 1

    def might_contain(self, item: str) -> bool:
        """
        Return True  if the item PROBABLY exists (all k bits are set).
        Return False if the item DEFINITELY does not exist (any bit is 0).
        """
        return all(self._get_bit(pos) for pos in self._hash_positions(item))

    @property
    def count(self) -> int:
        """Number of items added."""
        return self._count

    def false_positive_rate(self) -> float:
        """
        Estimated false-positive probability given the current fill level.
        p = (1 - e^(-k*n/m))^k
        """
        if self._count == 0:
            return 0.0
        exponent = -self.num_hashes * self._count / self.size
        return (1 - math.exp(exponent)) ** self.num_hashes

    # ── sizing helpers (useful at design time) ────────────────────────────────

    @staticmethod
    def optimal_size(n: int, p: float = 0.01) -> int:
        """Return the optimal bit-array size for n items at false-positive rate p."""
        return math.ceil(-(n * math.log(p)) / (math.log(2) ** 2))

    @staticmethod
    def optimal_hashes(m: int, n: int) -> int:
        """Return the optimal number of hash functions given bit size m and n items."""
        return max(1, round((m / n) * math.log(2)))
