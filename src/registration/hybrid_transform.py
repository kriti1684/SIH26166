import json
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator


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
        # Adaptive sub-pixel inlier filtering: filters out blunders and coarse noise.
        # Layer 1 establishes the coarse linear baseline. On tall swaths, allow up to
        # 2.5 - 3.5 px to absorb mild along-track curvature without admitting 10+ px blunders.
        scene_dim = float(max(image_shape[:2])) if image_shape is not None else (float(max(np.ptp(src_pts[:, 0]), np.ptp(src_pts[:, 1]))) if N > 1 else 0.0)
        if scene_dim > 5000:
            layer1_thresh = max(ransac_threshold, min(5.5, 0.00033 * scene_dim))
        else:
            layer1_thresh = ransac_threshold

        A, inliers = cv2.estimateAffine2D(
            src_pts, ref_pts, cv2.RANSAC, 
            ransacReprojThreshold=layer1_thresh, 
            maxIters=5000, 
            confidence=0.999
        )
        
        # If threshold yields fewer than 15 inliers or collapses along-track span, relax gracefully
        inlier_count = int(np.sum(inliers.ravel() == 1)) if inliers is not None else 0
        span_y_orig = float(np.ptp(src_pts[:, 1])) if N > 1 else 0.0
        span_y_in = float(np.ptp(src_pts[inliers.ravel() == 1, 1])) if inlier_count > 1 else 0.0

        if (A is None or inliers is None or inlier_count < min(15, N // 2) or
            (scene_dim > 5000 and span_y_orig > 0.3 * scene_dim and span_y_in < 0.50 * span_y_orig)):
            relaxed_thresh = min(5.0, max(3.5, layer1_thresh * 1.5))
            A_rel, inliers_rel = cv2.estimateAffine2D(
                src_pts, ref_pts, cv2.RANSAC, 
                ransacReprojThreshold=relaxed_thresh, 
                maxIters=5000, 
                confidence=0.999
            )
            if A_rel is not None and inliers_rel is not None:
                if np.sum(inliers_rel.ravel() == 1) > inlier_count:
                    A, inliers = A_rel, inliers_rel
        
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
                ransacReprojThreshold=layer1_thresh, 
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
                    # FIX: Recompute translation from inlier centroid after scale clamp.
                    # Keeping the old t is wrong because it was optimized for s_part != 1.0,
                    # causing a centroid shift error of ~center_px * (1 - 1/s_part).
                    src_in_clamp = src_pts[inliers_part.ravel() == 1]
                    ref_in_clamp = ref_pts[inliers_part.ravel() == 1]
                    t = np.mean(ref_in_clamp.T - R @ src_in_clamp.T, axis=1).reshape(2, 1)
                    A = np.hstack([R, t])
                    inliers = inliers_part

        self.affine_matrix = A
        inlier_mask = inliers.ravel() == 1
        
        # We build the subsequent layers ONLY on the inliers to avoid fitting to blunders
        src_in = src_pts[inlier_mask]
        ref_in = ref_pts[inlier_mask]
        
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

        # Normalize Y coordinates before polyfit to prevent ill-conditioned Vandermonde.
        # Raw Y in [0, 32000] gives condition number ~10^9 for degree-2 polynomial.
        # Normalizing to zero-mean, unit-std reduces condition number to ~9.
        self.poly_y_mean = float(np.mean(y_coords)) if len(y_coords) > 0 else 0.0
        self.poly_y_std = max(1.0, float(np.std(y_coords)))
        y_norm = (y_coords - self.poly_y_mean) / self.poly_y_std

        # Prevent RankWarning: Polyfit may be poorly conditioned
        # Ensure we have enough distinct row clusters
        y_clusters = len(np.unique(np.round(y_norm * 10.0)))  # cluster on normalized coords
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
            self.poly_coeffs_x = np.polyfit(y_norm, dx, eff_poly_degree)
            self.poly_coeffs_y = np.polyfit(y_norm, dy, eff_poly_degree)

        pred_dx = np.polyval(self.poly_coeffs_x, y_norm)
        pred_dy = np.polyval(self.poly_coeffs_y, y_norm)
        
        pred_L2 = pred_L1 + np.column_stack([pred_dx, pred_dy])
        
        res_L2 = ref_in - pred_L2
        rmse_L2 = np.sqrt(np.mean(np.sum(res_L2**2, axis=1)))

        # ── Layer 2.5: Stratified Height-Binned Physical Consensus ─────────────
        # Satellite orbital trajectory and along-track drift follow continuous physical curves.
        # Across long swaths (> 5000 px), blunder filtering must be STRATIFIED along the
        # orbital flight path so that all geographic segments (North, Middle, South) retain
        # verified physical tie-point anchors, preventing localized cluster collapse.
        if (scene_dim > 5000 or (image_shape is not None and max(image_shape) > 5000)) and len(src_in) >= 20:
            err_L2 = np.sqrt(np.sum(res_L2**2, axis=1))
            n_bins = 6
            y_min = float(np.min(y_coords))
            y_max = float(np.max(y_coords))
            y_bins = np.linspace(y_min, y_max, n_bins + 1)
            
            stratified_keep = np.zeros(len(src_in), dtype=bool)
            
            # Pass 1: Select consensus points per along-track bin
            for i in range(n_bins):
                b_mask = (y_coords >= y_bins[i]) & (y_coords <= y_bins[i + 1] if i == n_bins - 1 else y_coords < y_bins[i + 1])
                b_indices = np.where(b_mask)[0]
                if len(b_indices) == 0:
                    continue
                
                b_errs = err_L2[b_indices]
                b_order = np.argsort(b_errs)
                b_sorted_idx = b_indices[b_order]
                
                # In each along-track bin, keep consensus points matching the smooth flight path
                b_med = float(np.median(b_errs))
                b_mad = float(np.median(np.abs(b_errs - b_med)))
                b_cutoff = min(1.80, max(0.80, b_med + 1.5 * b_mad * 1.4826))
                
                good_in_bin = [idx for idx in b_sorted_idx if err_L2[idx] <= b_cutoff]
                # Ensure each populated bin retains at least 6 points if available for spatial continuity
                if len(good_in_bin) < 6:
                    good_in_bin = list(b_sorted_idx[:min(len(b_sorted_idx), max(6, len(good_in_bin)))])
                
                # Cap at 25 points per bin to prevent regional over-concentration
                stratified_keep[good_in_bin[:25]] = True

            # Pass 2: Clean remaining blunder spikes where residual from smooth surface > 1.15 px
            if np.sum(stratified_keep) >= 20:
                s_strat = src_in[stratified_keep]
                r_strat = ref_in[stratified_keep]
                y_norm_s = (s_strat[:, 1] - self.poly_y_mean) / self.poly_y_std
                p_L1_s = (A @ np.hstack([s_strat, np.ones((len(s_strat), 1))]).T).T
                r_L1_s = r_strat - p_L1_s
                if eff_poly_degree == 0:
                    c_x = np.array([float(np.mean(r_L1_s[:, 0]))])
                    c_y = np.array([float(np.mean(r_L1_s[:, 1]))])
                else:
                    c_x = np.polyfit(y_norm_s, r_L1_s[:, 0], eff_poly_degree)
                    c_y = np.polyfit(y_norm_s, r_L1_s[:, 1], eff_poly_degree)
                p_L2_s = p_L1_s + np.column_stack([np.polyval(c_x, y_norm_s), np.polyval(c_y, y_norm_s)])
                res_L2_s = r_strat - p_L2_s
                err_L2_s = np.sqrt(np.sum(res_L2_s**2, axis=1))

                pass2_mask = err_L2_s <= 1.15
                if np.sum(pass2_mask) >= 15:
                    strat_indices = np.where(stratified_keep)[0]
                    stratified_keep = np.zeros_like(stratified_keep)
                    stratified_keep[strat_indices[pass2_mask]] = True
            
            # Safety check: ensure at least 15 points survive
            if np.sum(stratified_keep) < 15:
                top_indices = np.argsort(err_L2)[:min(len(err_L2), 25)]
                stratified_keep[top_indices] = True
                
            if np.sum(stratified_keep) >= 12 and np.sum(stratified_keep) < len(src_in):
                n_pruned = len(src_in) - int(np.sum(stratified_keep))
                print(f"  [HYBRID-CONSENSUS] Stratified height-binned filtering: pruned {n_pruned} blunders; {np.sum(stratified_keep)} distributed inliers remain.")
                src_in = src_in[stratified_keep]
                ref_in = ref_in[stratified_keep]

                # Update inlier_mask for the full src_pts array
                inlier_indices = np.where(inlier_mask)[0]
                inlier_mask = np.zeros_like(inlier_mask, dtype=bool)
                inlier_mask[inlier_indices[stratified_keep]] = True

                # Re-fit Layer 2 polynomial cleanly on the stratified physical consensus inliers
                y_coords = src_in[:, 1]
                y_norm = (y_coords - self.poly_y_mean) / self.poly_y_std
                pred_L1 = (A @ np.hstack([src_in, np.ones((len(src_in), 1))]).T).T
                res_L1 = ref_in - pred_L1

                if eff_poly_degree == 0:
                    self.poly_coeffs_x = np.array([float(np.mean(res_L1[:, 0]))])
                    self.poly_coeffs_y = np.array([float(np.mean(res_L1[:, 1]))])
                else:
                    self.poly_coeffs_x = np.polyfit(y_norm, res_L1[:, 0], eff_poly_degree)
                    self.poly_coeffs_y = np.polyfit(y_norm, res_L1[:, 1], eff_poly_degree)

                pred_dx = np.polyval(self.poly_coeffs_x, y_norm)
                pred_dy = np.polyval(self.poly_coeffs_y, y_norm)
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
            # FIX: Evaluate coverage relative to the FULL IMAGE area (image_shape), not
            # just the bounding box of the inlier points. The old metric guaranteed
            # coverage_pct < 10% for any cluster narrower than 32% of the swath width,
            # causing TPS to be permanently bypassed for narrow WAC/IIRS strips.
            if image_shape is not None:
                swath_area = float(image_shape[0]) * float(image_shape[1])
            else:
                swath_w = max(50.0, float(np.max(src_in[:, 0]) - np.min(src_in[:, 0])))
                swath_h = max(50.0, float(np.max(src_in[:, 1]) - np.min(src_in[:, 1])))
                swath_area = swath_w * swath_h
            coverage_pct = (hull_area / max(1.0, swath_area)) * 100.0

            swath_h_pts = max(50.0, float(np.max(src_in[:, 1]) - np.min(src_in[:, 1])))
            total_h = float(image_shape[0]) if image_shape is not None else swath_h_pts
            along_track_span = (swath_h_pts / max(1.0, total_h)) * 100.0
            
        inlier_count = len(src_in)
        
        # FIX: Relax TPS activation to inlier_count >= 15 and coverage >= 2% (image-relative).
        # The old threshold (>= 25 inliers AND coverage >= 20%) was unreachable for narrow
        # WAC/IIRS strips where the mutual overlap is a small fraction of the scene area.
        if inlier_count >= 15 and (coverage_pct >= 2.0 or along_track_span >= 30.0):
            # FIX: N-adaptive smoothing. scipy RBFInterpolator smoothing is NOT normalized
            # by N, so a fixed 0.05 severely over-smooths with N=200+ points (CV RMSE 3x
            # training), and correctly smooths with N=40. Scale inversely with N/50.
            n_pts = float(inlier_count)
            adaptive_smoothing = min(tps_smoothing, 0.01) * (50.0 / max(30.0, n_pts))
            reg_smoothing = max(0.002, min(adaptive_smoothing, 0.05))
            self.tps_smoothing = reg_smoothing
            print(f"  [HYBRID-TRANSFORM] TPS smoothing: {tps_smoothing:.4f} (input) -> {reg_smoothing:.4f} (N-adaptive, N={inlier_count})")
            
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
            
        src_homog = np.hstack([src_pts, np.ones((len(src_pts), 1))])
        out = (self.affine_matrix @ src_homog.T).T
        
        # L2: Polynomial (uses stored Y normalization to match how coefficients were fitted)
        y_coords = src_pts[:, 1]
        y_norm = (y_coords - getattr(self, 'poly_y_mean', 0.0)) / max(1.0, getattr(self, 'poly_y_std', 1.0))
        dx = np.polyval(self.poly_coeffs_x, y_norm)
        dy = np.polyval(self.poly_coeffs_y, y_norm)
        out += np.column_stack([dx, dy])
        
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
        
        A = self.affine_matrix[:2, :2]
        t = self.affine_matrix[:2, 2]
        det = float(np.linalg.det(A))
        if abs(det) < 1e-9:
            raise RuntimeError(f"Singular affine matrix (det={det:.2e}) cannot be inverted.")
            
        A_inv = np.linalg.inv(A)
        t_inv = -A_inv @ t
        inv_model.affine_matrix = np.column_stack([A_inv, t_inv])
        
        if hasattr(self, "src_inliers") and self.src_inliers is not None and hasattr(self, "ref_inliers") and self.ref_inliers is not None:
            inv_model.src_inliers = self.ref_inliers.copy()
            inv_model.ref_inliers = self.src_inliers.copy()
            inv_model.poly_degree = getattr(self, "poly_degree", 2)
            inv_model.tps_smoothing = getattr(self, "tps_smoothing", 0.05)
            
            ref_homog = np.hstack([inv_model.src_inliers, np.ones((len(inv_model.src_inliers), 1))])
            pred_L1 = (inv_model.affine_matrix @ ref_homog.T).T
            res_L1 = inv_model.ref_inliers - pred_L1
            
            # Y normalization for inverse polynomial (inverse uses ref_inliers as src)
            y_coords = inv_model.src_inliers[:, 1]
            inv_model.poly_y_mean = float(np.mean(y_coords)) if len(y_coords) > 0 else 0.0
            inv_model.poly_y_std = max(1.0, float(np.std(y_coords)))
            y_norm = (y_coords - inv_model.poly_y_mean) / inv_model.poly_y_std
            y_clusters = len(np.unique(np.round(y_norm * 10.0)))
            eff_poly_degree = min(inv_model.poly_degree, max(0, y_clusters - 1))

            if eff_poly_degree == 0:
                inv_model.poly_coeffs_x = np.array([float(np.mean(res_L1[:, 0]))]) if len(res_L1) > 0 else np.array([0.0])
                inv_model.poly_coeffs_y = np.array([float(np.mean(res_L1[:, 1]))]) if len(res_L1) > 0 else np.array([0.0])
            else:
                inv_model.poly_coeffs_x = np.polyfit(y_norm, res_L1[:, 0], eff_poly_degree)
                inv_model.poly_coeffs_y = np.polyfit(y_norm, res_L1[:, 1], eff_poly_degree)

            pred_dx = np.polyval(inv_model.poly_coeffs_x, y_norm)
            pred_dy = np.polyval(inv_model.poly_coeffs_y, y_norm)
            pred_L2 = pred_L1 + np.column_stack([pred_dx, pred_dy])
            res_L2 = inv_model.ref_inliers - pred_L2
            
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
            inv_model.poly_y_mean = 0.0
            inv_model.poly_y_std = 1.0
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
            
            # Forward Y normalization for scaled model (Y coords scale with pixel size)
            scaled.poly_y_mean = getattr(self, 'poly_y_mean', 0.0) * scale_factor
            scaled.poly_y_std = max(1.0, getattr(self, 'poly_y_std', 1.0) * scale_factor)
            src_homog = np.hstack([scaled.src_inliers, np.ones((len(scaled.src_inliers), 1))])
            pred_L1 = (scaled.affine_matrix @ src_homog.T).T
            y_coords = scaled.src_inliers[:, 1]
            y_norm = (y_coords - scaled.poly_y_mean) / scaled.poly_y_std
            pred_dx = np.polyval(scaled.poly_coeffs_x, y_norm)
            pred_dy = np.polyval(scaled.poly_coeffs_y, y_norm)
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
            "poly_y_mean": getattr(self, 'poly_y_mean', 0.0),
            "poly_y_std": getattr(self, 'poly_y_std', 1.0),
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
        # Restore Y normalization params (default to no-op if loading legacy JSON)
        model.poly_y_mean = float(data.get("poly_y_mean", 0.0))
        model.poly_y_std = max(1.0, float(data.get("poly_y_std", 1.0)))
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
                y_norm = (y_coords - getattr(model, 'poly_y_mean', 0.0)) / max(1.0, getattr(model, 'poly_y_std', 1.0))
                pred_dx = np.polyval(model.poly_coeffs_x, y_norm)
                pred_dy = np.polyval(model.poly_coeffs_y, y_norm)
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
