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
        self.tps_center = None
        self.tps_scale = None
        self.layer_params = {}
        
    def fit(
        self, 
        src_pts: np.ndarray, 
        ref_pts: np.ndarray, 
        image_shape: Tuple[int, int] = None,
        poly_degree: int = 2,
        tps_smoothing: float = 0.05,
        ransac_threshold: float = 5.0
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
        self.ransac_threshold = ransac_threshold
        
        # ── Layer 1: Affine Baseline (RANSAC) ─────────────────────────────────
        # Adaptive sub-pixel inlier filtering: filters out blunders and coarse noise
        A, inliers = cv2.estimateAffine2D(
            src_pts, ref_pts, cv2.RANSAC, 
            ransacReprojThreshold=ransac_threshold, 
            maxIters=5000, 
            confidence=0.999
        )
        
        # If tight threshold yields fewer than 15 inliers, relax gracefully
        if A is None or inliers is None or np.sum(inliers.ravel() == 1) < min(15, len(src_pts) // 2):
            relaxed_thresh = min(3.5, ransac_threshold * 2.0)
            A, inliers = cv2.estimateAffine2D(
                src_pts, ref_pts, cv2.RANSAC, 
                ransacReprojThreshold=relaxed_thresh, 
                maxIters=5000, 
                confidence=0.999
            )
        
        if A is None or inliers is None:
            raise ValueError("Failed to fit Affine transformation (Layer 1).")
            
        # Physical scale sanity check for pre-harmonized satellite imagery:
        # Scale factors must be physically close to 1.0 (between 0.92 and 1.08)
        sx = float(np.sqrt(A[0, 0]**2 + A[1, 0]**2))
        sy = float(np.sqrt(A[0, 1]**2 + A[1, 1]**2))
        
        if not (0.92 <= sx <= 1.08 and 0.92 <= sy <= 1.08):
            print(f"  [HYBRID-SAFETY] Affine scale out of physical bounds (sx={sx:.3f}, sy={sy:.3f}). Falling back to constrained Similarity transform...")
            # Fall back to estimateAffinePartial2D (4-DOF: rotation + translation + isotropic scale)
            A_part, inliers_part = cv2.estimateAffinePartial2D(
                src_pts, ref_pts, cv2.RANSAC, 
                ransacReprojThreshold=ransac_threshold, 
                maxIters=5000, 
                confidence=0.999
            )
            if A_part is not None:
                s_part = float(np.sqrt(A_part[0, 0]**2 + A_part[1, 0]**2))
                if 0.92 <= s_part <= 1.08:
                    A, inliers = A_part, inliers_part
                else:
                    # Normalize scale strictly to 1.0 (pure rigid rotation + translation)
                    print(f"  [HYBRID-SAFETY] Clamping scale {s_part:.3f} to 1.0 (Rigid Euclidean baseline).")
                    R = A_part[:, :2] / max(1e-6, s_part)
                    t = A_part[:, 2:]
                    A = np.hstack([R, t])
                    inliers = inliers_part

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
        
        # Prevent RankWarning: Polyfit may be poorly conditioned
        # Ensure we have enough distinct row clusters
        y_clusters = len(np.unique(np.round(y_coords / 150.0)))
        eff_poly_degree = min(poly_degree, max(0, y_clusters - 1))
        
        # Check along-track vertical span of inliers relative to total scene height
        y_span = float(np.max(y_coords) - np.min(y_coords)) if len(y_coords) > 1 else 0.0
        total_h = float(image_shape[0]) if image_shape is not None else y_span
        span_ratio = y_span / max(1.0, total_h)
        
        # If inliers span less than 25% of the total swath, clamp degree to 0 (constant translation)
        # to prevent polynomial extrapolation runaway in unanchored regions.
        if span_ratio < 0.25 and eff_poly_degree > 0:
            print(f"  [HYBRID-SAFETY] Inliers span {span_ratio*100:.1f}% of swath (< 25%). Clamping drift polynomial to constant mean offset.")
            eff_poly_degree = 0

        if eff_poly_degree == 0:
            self.poly_coeffs_x = np.array([float(np.mean(dx))])
            self.poly_coeffs_y = np.array([float(np.mean(dy))])
        else:
            self.poly_coeffs_x = np.polyfit(y_coords, dx, eff_poly_degree)
            self.poly_coeffs_y = np.polyfit(y_coords, dy, eff_poly_degree)
        
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
        along_track_span = 0.0
        if len(src_in) >= 3:
            hull = cv2.convexHull(src_in.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            # Evaluate coverage relative to the active swath bounds
            swath_w = max(50.0, float(np.max(src_in[:, 0]) - np.min(src_in[:, 0])))
            swath_h = max(50.0, float(np.max(src_in[:, 1]) - np.min(src_in[:, 1])))
            swath_area = swath_w * swath_h
            coverage_pct = (hull_area / swath_area) * 100.0
            
            total_h = float(image_shape[0]) if image_shape is not None else swath_h
            along_track_span = (swath_h / max(1.0, total_h)) * 100.0
            
        inlier_count = len(src_in)
        
        # Activate TPS if dense tie points (>=25) span the active swath
        if inlier_count >= 25 and (coverage_pct >= 20.0 or along_track_span >= 40.0):
            reg_smoothing = max(0.01, tps_smoothing)
            self.tps_smoothing = reg_smoothing
            
            # Coordinate normalization for TPS:
            # Prevents ill-conditioned matrix scaling (where r^2 ln(r) reaches 10^7)
            # and restores physical elastic stiffness to the smoothing parameter.
            self.tps_center = np.mean(src_in, axis=0)
            self.tps_scale = max(1.0, float(np.max(np.std(src_in, axis=0))))
            src_in_norm = (src_in - self.tps_center) / self.tps_scale
            
            self.tps_rbf_x = RBFInterpolator(
                src_in_norm, res_L2[:, 0], kernel='thin_plate_spline', smoothing=reg_smoothing
            )
            self.tps_rbf_y = RBFInterpolator(
                src_in_norm, res_L2[:, 1], kernel='thin_plate_spline', smoothing=reg_smoothing
            )
            
            tps_dx = self.tps_rbf_x(src_in_norm)
            tps_dy = self.tps_rbf_y(src_in_norm)
            
            pred_L3 = pred_L2 + np.column_stack([tps_dx, tps_dy])
            
            res_L3 = ref_in - pred_L3
            rmse_L3 = np.sqrt(np.mean(np.sum(res_L3**2, axis=1)))
            print(f"  [HYBRID-TRANSFORM] TPS Activated: {inlier_count} inliers, {coverage_pct:.1f}% swath coverage ({along_track_span:.1f}% span).")
        else:
            self.tps_rbf_x = None
            self.tps_rbf_y = None
            self.tps_center = None
            self.tps_scale = None
            self.tps_smoothing = tps_smoothing
            rmse_L3 = rmse_L2  # Fallback to Layer 2
            print(f"  [HYBRID-TRANSFORM] TPS Bypassed: {inlier_count} inliers, {coverage_pct:.1f}% swath coverage ({along_track_span:.1f}% span). Falling back to Layer 2.")
            
        # Save inliers for serialization and inverse mapping
        self.src_inliers = src_in
        self.ref_inliers = ref_in
        self.poly_degree = poly_degree

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
            if getattr(self, "tps_center", None) is not None and getattr(self, "tps_scale", None) is not None:
                src_pts_norm = (src_pts - self.tps_center) / self.tps_scale
            else:
                src_pts_norm = src_pts
            tps_dx = self.tps_rbf_x(src_pts_norm)
            tps_dy = self.tps_rbf_y(src_pts_norm)
            out += np.column_stack([tps_dx, tps_dy])
        
        return out

    def get_inverse_transform(self) -> "HybridTransform":
        """
        Derives and returns an exact inverse HybridTransform mapping from ref -> src coordinates.
        Uses exact mathematical matrix inversion for Layer 1 Affine, eliminating RANSAC randomness
        and collinear degeneracy, and fits reverse residual corrections for Layers 2 and 3.
        """
        if self.affine_matrix is None:
            raise RuntimeError("Cannot invert transform: affine matrix not available.")
            
        inv_model = HybridTransform()
        
        # 1. Exact analytic inverse of the 2x3 affine matrix
        A = self.affine_matrix[:2, :2]
        t = self.affine_matrix[:2, 2]
        det = float(np.linalg.det(A))
        if abs(det) < 1e-9:
            raise RuntimeError(f"Singular affine matrix (det={det:.2e}) cannot be inverted.")
            
        A_inv = np.linalg.inv(A)
        t_inv = -A_inv @ t
        inv_model.affine_matrix = np.column_stack([A_inv, t_inv])
        
        # 2. Reverse inliers & residual fitting if inliers exist
        if hasattr(self, "src_inliers") and self.src_inliers is not None and hasattr(self, "ref_inliers") and self.ref_inliers is not None:
            inv_model.src_inliers = self.ref_inliers.copy()
            inv_model.ref_inliers = self.src_inliers.copy()
            inv_model.poly_degree = getattr(self, "poly_degree", 2)
            inv_model.tps_smoothing = getattr(self, "tps_smoothing", 0.05)
            
            # Layer 1 reverse prediction
            ref_homog = np.hstack([inv_model.src_inliers, np.ones((len(inv_model.src_inliers), 1))])
            pred_L1 = (inv_model.affine_matrix @ ref_homog.T).T
            res_L1 = inv_model.ref_inliers - pred_L1
            
            # Layer 2: Reverse polynomial drift along ref Y coordinates
            y_coords = inv_model.src_inliers[:, 1]
            y_clusters = len(np.unique(np.round(y_coords / 150.0))) if len(y_coords) > 0 else 0
            eff_poly_degree = min(inv_model.poly_degree, max(0, y_clusters - 1))
            
            if eff_poly_degree == 0:
                inv_model.poly_coeffs_x = np.array([float(np.mean(res_L1[:, 0]))]) if len(res_L1) > 0 else np.array([0.0])
                inv_model.poly_coeffs_y = np.array([float(np.mean(res_L1[:, 1]))]) if len(res_L1) > 0 else np.array([0.0])
            else:
                inv_model.poly_coeffs_x = np.polyfit(y_coords, res_L1[:, 0], eff_poly_degree)
                inv_model.poly_coeffs_y = np.polyfit(y_coords, res_L1[:, 1], eff_poly_degree)
                
            pred_dx = np.polyval(inv_model.poly_coeffs_x, y_coords)
            pred_dy = np.polyval(inv_model.poly_coeffs_y, y_coords)
            pred_L2 = pred_L1 + np.column_stack([pred_dx, pred_dy])
            res_L2 = inv_model.ref_inliers - pred_L2
            
            # Layer 3: Reverse TPS (only if forward TPS was activated)
            tps_active = getattr(self, "tps_activated", False) or (self.tps_rbf_x is not None)
            if tps_active and len(inv_model.src_inliers) >= 25:
                inv_model.tps_center = np.mean(inv_model.src_inliers, axis=0)
                inv_model.tps_scale = max(1.0, float(np.max(np.std(inv_model.src_inliers, axis=0))))
                inv_in_norm = (inv_model.src_inliers - inv_model.tps_center) / inv_model.tps_scale
                
                inv_model.tps_rbf_x = RBFInterpolator(
                    inv_in_norm, res_L2[:, 0], kernel='thin_plate_spline', smoothing=inv_model.tps_smoothing
                )
                inv_model.tps_rbf_y = RBFInterpolator(
                    inv_in_norm, res_L2[:, 1], kernel='thin_plate_spline', smoothing=inv_model.tps_smoothing
                )
            else:
                inv_model.tps_rbf_x = None
                inv_model.tps_rbf_y = None
                inv_model.tps_center = None
                inv_model.tps_scale = None
        else:
            inv_model.poly_coeffs_x = np.array([0.0])
            inv_model.poly_coeffs_y = np.array([0.0])
            inv_model.tps_rbf_x = None
            inv_model.tps_rbf_y = None
            
        return inv_model

    def scale(self, scale_factor: float) -> "HybridTransform":
        """
        Scales the transformation model by a scale_factor (e.g. GSD_harm / GSD_src_native).
        Returns a new HybridTransform instance mapping native ref coords to native src coords.
        """
        scaled = HybridTransform()
        
        if self.affine_matrix is not None:
            A = self.affine_matrix.copy()
            A[0, 2] *= scale_factor
            A[1, 2] *= scale_factor
            scaled.affine_matrix = A
            
        if self.poly_coeffs_x is not None:
            scaled.poly_coeffs_x = self.poly_coeffs_x.copy()
            scaled.poly_coeffs_y = self.poly_coeffs_y.copy()
            degree = len(self.poly_coeffs_x) - 1
            for i in range(len(self.poly_coeffs_x)):
                power = degree - i
                scaled.poly_coeffs_x[i] = self.poly_coeffs_x[i] * (scale_factor ** (1 - power))
                scaled.poly_coeffs_y[i] = self.poly_coeffs_y[i] * (scale_factor ** (1 - power))
                
        if hasattr(self, "src_inliers") and self.src_inliers is not None:
            scaled.src_inliers = self.src_inliers * scale_factor
            scaled.ref_inliers = self.ref_inliers * scale_factor
            scaled.poly_degree = getattr(self, 'poly_degree', 2)
            scaled.tps_smoothing = getattr(self, 'tps_smoothing', 0.05)
            
            src_homog = np.hstack([scaled.src_inliers, np.ones((len(scaled.src_inliers), 1))])
            pred_L1 = (scaled.affine_matrix @ src_homog.T).T
            y_coords = scaled.src_inliers[:, 1]
            pred_dx = np.polyval(scaled.poly_coeffs_x, y_coords)
            pred_dy = np.polyval(scaled.poly_coeffs_y, y_coords)
            pred_L2 = pred_L1 + np.column_stack([pred_dx, pred_dy])
            res_L2 = scaled.ref_inliers - pred_L2

            if self.tps_rbf_x is not None:
                scaled.tps_center = self.tps_center * scale_factor if getattr(self, 'tps_center', None) is not None else np.mean(scaled.src_inliers, axis=0)
                scaled.tps_scale = self.tps_scale * scale_factor if getattr(self, 'tps_scale', None) is not None else max(1.0, float(np.max(np.std(scaled.src_inliers, axis=0))))
                src_in_norm = (scaled.src_inliers - scaled.tps_center) / scaled.tps_scale
                scaled.tps_rbf_x = RBFInterpolator(
                    src_in_norm, res_L2[:, 0], kernel='thin_plate_spline', smoothing=scaled.tps_smoothing
                )
                scaled.tps_rbf_y = RBFInterpolator(
                    src_in_norm, res_L2[:, 1], kernel='thin_plate_spline', smoothing=scaled.tps_smoothing
                )
            else:
                scaled.tps_rbf_x = None
                scaled.tps_rbf_y = None
                scaled.tps_center = None
                scaled.tps_scale = None
                
        return scaled

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
            "tps_activated": (self.tps_rbf_x is not None),
            "tps_center": self.tps_center.tolist() if getattr(self, "tps_center", None) is not None else None,
            "tps_scale": float(self.tps_scale) if getattr(self, "tps_scale", None) is not None else None,
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
        model.tps_center = np.array(data["tps_center"], dtype=np.float64) if data.get("tps_center") is not None else None
        model.tps_scale = float(data["tps_scale"]) if data.get("tps_scale") is not None else None

        tps_activated = data.get("tps_activated", False)
        # Fallback for legacy JSON
        if not tps_activated and "tps_activated" not in data:
            stats = data.get("stats", {})
            tps_activated = stats.get("inlier_count", 0) > 50 and stats.get("coverage_pct", 0) > 35.0

        if data.get("src_inliers") is not None and data.get("ref_inliers") is not None:
            src_in = np.array(data["src_inliers"], dtype=np.float64)
            ref_in = np.array(data["ref_inliers"], dtype=np.float64)
            model.src_inliers = src_in
            model.ref_inliers = ref_in

            if tps_activated:
                # Re-fit TPS layer only if it was activated
                src_homog = np.hstack([src_in, np.ones((len(src_in), 1))])
                pred_L1 = (model.affine_matrix @ src_homog.T).T
                y_coords = src_in[:, 1]
                pred_dx = np.polyval(model.poly_coeffs_x, y_coords)
                pred_dy = np.polyval(model.poly_coeffs_y, y_coords)
                pred_L2 = pred_L1 + np.column_stack([pred_dx, pred_dy])
                res_L2 = ref_in - pred_L2

                if model.tps_center is None:
                    model.tps_center = np.mean(src_in, axis=0)
                    model.tps_scale = max(1.0, float(np.max(np.std(src_in, axis=0))))

                src_in_norm = (src_in - model.tps_center) / model.tps_scale
                model.tps_rbf_x = RBFInterpolator(
                    src_in_norm, res_L2[:, 0], kernel='thin_plate_spline', smoothing=model.tps_smoothing
                )
                model.tps_rbf_y = RBFInterpolator(
                    src_in_norm, res_L2[:, 1], kernel='thin_plate_spline', smoothing=model.tps_smoothing
                )
            else:
                model.tps_rbf_x = None
                model.tps_rbf_y = None

        return model

# ─── API Function ────────────────────────────────────────────────────────────

def compute_hybrid_transform(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    output_json: Path,
    poly_degree: int = 2,
    tps_smoothing: float = 0.05,
    ransac_threshold: float = 1.5
) -> HybridTransform:
    """
    Convenience function to fit the hybrid transform and save its state.
    """
    model = HybridTransform()
    model.fit(src_pts, ref_pts, poly_degree=poly_degree, tps_smoothing=tps_smoothing, ransac_threshold=ransac_threshold)
    model.save(output_json)
    return model
