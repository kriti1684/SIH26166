import unittest


class TestModuleImports(unittest.TestCase):
    def test_model_import(self):
        from src.models.matching import Matching
        self.assertTrue(callable(Matching))

    def test_preprocessing_imports(self):
        from src.preprocessing.normalizer import normalize_tiff
        from src.preprocessing.band_selector import select_best_band_for_wac
        from src.preprocessing.kernel_resolver import KernelResolver
        from src.preprocessing.spice_georeference import compute_gcps
        self.assertTrue(callable(normalize_tiff))
        self.assertTrue(callable(select_best_band_for_wac))
        self.assertTrue(callable(compute_gcps))

    def test_registration_imports(self):
        from src.registration.coarse_alignment import run_coarse_alignment
        from src.registration.tiled_matching import run_tiled_matching
        from src.registration.drift_profiler import run_drift_profile
        from src.registration.hybrid_transform import fit_hybrid_transform
        from src.registration.warp import run_warp
        from src.registration.verifier import run_verification
        self.assertTrue(callable(run_coarse_alignment))
        self.assertTrue(callable(run_tiled_matching))
        self.assertTrue(callable(run_drift_profile))
        self.assertTrue(callable(fit_hybrid_transform))
        self.assertTrue(callable(run_warp))
        self.assertTrue(callable(run_verification))


if __name__ == "__main__":
    unittest.main()
