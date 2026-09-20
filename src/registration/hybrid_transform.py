"""
src/registration/hybrid_transform.py
=========================================
Physics-Grounded 3-Layer Hybrid Transformation (Task 3.3).

Generates a sub-pixel accurate, multi-layer deformation model from tie points:
  - Layer 1: RANSAC Affine (Rigid baseline)
  - Layer 2: 1D Scanline Polynomial (absorbs pushbroom pitch/rate jitter along Y-axis)
  - Layer 3: Regularized Thin Plate Spline (TPS) (compensates micro-topographic relief)
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

# ─── 3-Layer Transformation Engine ───────────────────────────────────────────

class HybridTransform:
    def __init__(self):
        self.affine_matrix = None
        self.poly_coeffs_x = None  # 1D drift in X along Y
        self.poly_coeffs_y = None  # 1D drift in Y along Y
        self.tps_rbf_x = None
        self.tps_rbf_y = None
        self.layer_params = {}
        
    def fit(
        self, 
        src_pts: np.ndarray, 
        ref_pts: np.ndarray, 
        image_shape: Tuple[int, int] = None,
        poly_degree: int = 2,
        tps_smoothing: float = 0.05
    ) -> Dict[str, Any]:
        """
        Fit the 3-layer deformation model.
        Returns fit statistics and residuals at each layer.
        """
        src_pts = np.ascontiguousarray(src_pts, dtype=np.float32)
        ref_pts = np.ascontiguousarray(ref_pts, dtype=np.float32)
        assert len(src_pts) == len(ref_pts), "Must have matching number of points"
        assert len(src_pts) >= 6, "Need at least 6 points for full hybrid model"
        
        N = len(src_pts)
        
        # ── Layer 1: Affine Baseline (RANSAC) ─────────────────────────────────
        # Use a high RANSAC threshold here to capture all valid inliers, 
        # since actual relief displacement can cause points to deviate from pure affine.
        A, inliers = cv2.estimateAffine2D(src_pts, ref_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        
        if A is None:
            raise ValueError("Failed to fit Affine transformation (Layer 1).")
            
        self.affine_matrix = A
        inlier_mask = inliers.ravel() == 1
        
        # We build the subsequent layers ONLY on the inliers to avoid fitting to blunders
        src_in = src_pts[inlier_mask]
        ref_in = ref_pts[inlier_mask]
        
        # Apply Layer 1
        src_homog = np.hstack([src_in, np.ones((len(src_in), 1))])
        pred_L1 = (A @ src_homog.T).T
        
        res_L1 = ref_in - pred_L1
        rmse_L1 = np.sqrt(np.mean(np.sum(res_L1**2, axis=1)))
        
        # ── Layer 2: 1D Scanline Drift Polynomial ─────────────────────────────
        # For pushbroom sensors, unmodeled spacecraft jitter causes cross-track (X) 
        # and along-track (Y) drift as a function of the scanline (Y).
        # We fit a polynomial: dX = f(Y), dY = g(Y)
        
        y_coords = src_in[:, 1]
        dx = res_L1[:, 0]
        dy = res_L1[:, 1]
        
        # Fit polynomial to residuals
        self.poly_coeffs_x = np.polyfit(y_coords, dx, poly_degree)
        self.poly_coeffs_y = np.polyfit(y_coords, dy, poly_degree)
        
        pred_dx = np.polyval(self.poly_coeffs_x, y_coords)
        pred_dy = np.polyval(self.poly_coeffs_y, y_coords)
        
        pred_L2 = pred_L1 + np.column_stack([pred_dx, pred_dy])
        
        res_L2 = ref_in - pred_L2
        rmse_L2 = np.sqrt(np.mean(np.sum(res_L2**2, axis=1)))
        
        # ── Layer 3: Regularized Thin Plate Spline (TPS) ──────────────────────
        # Fits local topographic relief displacement. We use RBFInterpolator with
        # smoothing to prevent overfitting to noise.
        # ONLY IF inliers > 50 and coverage_pct > 35.0%
        
        coverage_pct = 0.0
        if len(src_in) >= 3:
            hull = cv2.convexHull(src_in.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            if image_shape is not None:
                img_area = image_shape[0] * image_shape[1]
            else:
                img_area = (np.max(src_pts[:, 0]) - np.min(src_pts[:, 0])) * (np.max(src_pts[:, 1]) - np.min(src_pts[:, 1])) + 1e-6
            coverage_pct = (hull_area / img_area) * 100.0
            
        inlier_count = len(src_in)
        
        if inlier_count > 50 and coverage_pct > 35.0:
            self.tps_rbf_x = RBFInterpolator(
                src_in, res_L2[:, 0], kernel='thin_plate_spline', smoothing=tps_smoothing
            )
            self.tps_rbf_y = RBFInterpolator(
                src_in, res_L2[:, 1], kernel='thin_plate_spline', smoothing=tps_smoothing
            )
            
            tps_dx = self.tps_rbf_x(src_in)
            tps_dy = self.tps_rbf_y(src_in)
            
            pred_L3 = pred_L2 + np.column_stack([tps_dx, tps_dy])
            
            res_L3 = ref_in - pred_L3
            rmse_L3 = np.sqrt(np.mean(np.sum(res_L3**2, axis=1)))
            print(f"  [HYBRID-TRANSFORM] TPS Activated: {inlier_count} inliers, {coverage_pct:.1f}% coverage.")
        else:
            self.tps_rbf_x = None
            self.tps_rbf_y = None
            rmse_L3 = rmse_L2  # Fallback to Layer 2
            print(f"  [HYBRID-TRANSFORM] TPS Bypassed: {inlier_count} inliers, {coverage_pct:.1f}% coverage. Falling back to Layer 2.")
            
        # Save inliers for serialization and inverse mapping
        self.src_inliers = src_in
        self.ref_inliers = ref_in
        self.poly_degree = poly_degree
        self.tps_smoothing = tps_smoothing

        # ── Compile parameters and stats ─────────────────────────────────────
        self.layer_params = {
            "inlier_count": int(np.sum(inlier_mask)),
            "total_points": N,
            "inlier_ratio": float(np.sum(inlier_mask)) / N,
            "coverage_pct": float(coverage_pct),
            "rmse_layer1_affine": float(rmse_L1),
            "rmse_layer2_poly": float(rmse_L2),
            "rmse_layer3_tps": float(rmse_L3),
            "poly_degree": poly_degree,
            "tps_smoothing": tps_smoothing
        }
        
        print(f"  L1 Affine RMSE: {rmse_L1:.3f} px")
        print(f"  L2 Drift RMSE:  {rmse_L2:.3f} px")
        print(f"  L3 TPS RMSE:    {rmse_L3:.3f} px")
        
        return self.layer_params
        
    def predict(self, src_pts: np.ndarray, use_tps: bool = True) -> np.ndarray:
        """
        Apply the fitted 3-layer transform to source points.
        """
        if self.affine_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        src_pts = np.asarray(src_pts)
        if src_pts.ndim == 1:
            src_pts = src_pts.reshape(1, -1)
            
        # L1: Affine
        src_homog = np.hstack([src_pts, np.ones((len(src_pts), 1))])
        out = (self.affine_matrix @ src_homog.T).T
        
        # L2: Polynomial
        y_coords = src_pts[:, 1]
        dx = np.polyval(self.poly_coeffs_x, y_coords)
        dy = np.polyval(self.poly_coeffs_y, y_coords)
        out += np.column_stack([dx, dy])
        
        # L3: TPS
        if use_tps and self.tps_rbf_x is not None and self.tps_rbf_y is not None:
            tps_dx = self.tps_rbf_x(src_pts)
            tps_dy = self.tps_rbf_y(src_pts)
            out += np.column_stack([tps_dx, tps_dy])
        
        return out

    def get_inverse_transform(self) -> "HybridTransform":
        """
        Fits and returns an inverse HybridTransform mapping from ref -> src coordinates.
        """
        if not hasattr(self, "src_inliers") or self.src_inliers is None:
            raise RuntimeError("Cannot invert transform: inliers not available.")
        inv_model = HybridTransform()
        inv_model.fit(
            src_pts=self.ref_inliers,
            ref_pts=self.src_inliers,
            poly_degree=self.poly_degree,
            tps_smoothing=self.tps_smoothing
        )
        return inv_model

    def save(self, filepath: Path) -> None:
        """Serialize model parameters and inliers to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "affine_matrix": self.affine_matrix.tolist() if self.affine_matrix is not None else None,
            "poly_coeffs_x": self.poly_coeffs_x.tolist() if self.poly_coeffs_x is not None else None,
            "poly_coeffs_y": self.poly_coeffs_y.tolist() if self.poly_coeffs_y is not None else None,
            "src_inliers": self.src_inliers.tolist() if hasattr(self, "src_inliers") and self.src_inliers is not None else None,
            "ref_inliers": self.ref_inliers.tolist() if hasattr(self, "ref_inliers") and self.ref_inliers is not None else None,
            "poly_degree": getattr(self, "poly_degree", 2),
            "tps_smoothing": getattr(self, "tps_smoothing", 0.05),
            "stats": self.layer_params
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"  [HYBRID-TRANSFORM] Exported model parameters to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "HybridTransform":
        """Reconstruct HybridTransform from saved JSON."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)

        model = cls()
        model.affine_matrix = np.array(data["affine_matrix"], dtype=np.float64) if data.get("affine_matrix") is not None else None
        model.poly_coeffs_x = np.array(data["poly_coeffs_x"], dtype=np.float64) if data.get("poly_coeffs_x") is not None else None
        model.poly_coeffs_y = np.array(data["poly_coeffs_y"], dtype=np.float64) if data.get("poly_coeffs_y") is not None else None
        model.layer_params = data.get("stats", {})
        model.poly_degree = data.get("poly_degree", 2)
        model.tps_smoothing = data.get("tps_smoothing", 0.05)

        if data.get("src_inliers") is not None and data.get("ref_inliers") is not None:
            src_in = np.array(data["src_inliers"], dtype=np.float64)
            ref_in = np.array(data["ref_inliers"], dtype=np.float64)
            model.src_inliers = src_in
            model.ref_inliers = ref_in

            # Re-fit TPS layer
            src_homog = np.hstack([src_in, np.ones((len(src_in), 1))])
            pred_L1 = (model.affine_matrix @ src_homog.T).T
            y_coords = src_in[:, 1]
            pred_dx = np.polyval(model.poly_coeffs_x, y_coords)
            pred_dy = np.polyval(model.poly_coeffs_y, y_coords)
            pred_L2 = pred_L1 + np.column_stack([pred_dx, pred_dy])
            res_L2 = ref_in - pred_L2

            model.tps_rbf_x = RBFInterpolator(
                src_in, res_L2[:, 0], kernel='thin_plate_spline', smoothing=model.tps_smoothing
            )
            model.tps_rbf_y = RBFInterpolator(
                src_in, res_L2[:, 1], kernel='thin_plate_spline', smoothing=model.tps_smoothing
            )

        return model

# ─── API Function ────────────────────────────────────────────────────────────

def compute_hybrid_transform(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    output_json: Path,
    poly_degree: int = 2,
    tps_smoothing: float = 0.05
) -> HybridTransform:
    """
    Convenience function to fit the hybrid transform and save its state.
    """
    model = HybridTransform()
    model.fit(src_pts, ref_pts, poly_degree=poly_degree, tps_smoothing=tps_smoothing)
    model.save(output_json)
    return model
