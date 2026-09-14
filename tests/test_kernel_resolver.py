import unittest
from pathlib import Path
from src.preprocessing.kernel_resolver import KernelResolver


class TestKernelResolver(unittest.TestCase):
    def test_init(self):
        resolver = KernelResolver("/mock/isisdata")
        self.assertEqual(resolver.isis_data_dir, "/mock/isisdata")
        self.assertEqual(resolver.priorities["reconstructed"], 3)
        self.assertEqual(resolver.priorities["predicted"], 2)


if __name__ == "__main__":
    unittest.main()
