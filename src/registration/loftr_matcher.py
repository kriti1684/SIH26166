import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import ssl
import torch
import numpy as np
from kornia.feature import LoFTR

# Bypass SSL verification for downloading the pretrained weights
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class LoFTRMatcher:
    def __init__(self, pretrained="outdoor", match_threshold=0.05):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.matcher = LoFTR(pretrained=pretrained).to(self.device).eval()
        self.match_threshold = match_threshold
        print(f"[LoFTRMatcher] Initialized on device: {self.device} (pretrained: {pretrained})")

    def match(self, image0: np.ndarray, image1: np.ndarray, max_dim: int = 768):
        """
        Extract dense feature matches between two images using LoFTR.
        Automatically resizes large tiles to <= max_dim (multiple of 8) to
        guarantee memory safety on GPUs/CPUs, then rescales coordinates back.
        
        Args:
            image0, image1: grayscale numpy arrays [H, W] (uint8 or float32).
            max_dim: maximum resolution dimension for transformer attention.
            
        Returns:
            src_pts, ref_pts, confidence (all np.ndarray of type float32).
        """
        import cv2

        h0, w0 = image0.shape[:2]
        h1, w1 = image1.shape[:2]

        if h0 < 16 or w0 < 16 or h1 < 16 or w1 < 16:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

        scale0 = min(1.0, max_dim / float(max(h0, w0)))
        scale1 = min(1.0, max_dim / float(max(h1, w1)))

        w0_r = max(16, int(round(w0 * scale0 / 8.0) * 8))
        h0_r = max(16, int(round(h0 * scale0 / 8.0) * 8))
        w1_r = max(16, int(round(w1 * scale1 / 8.0) * 8))
        h1_r = max(16, int(round(h1 * scale1 / 8.0) * 8))

        if (w0_r, h0_r) != (w0, h0):
            im0_proc = cv2.resize(image0, (w0_r, h0_r), interpolation=cv2.INTER_AREA)
        else:
            im0_proc = image0

        if (w1_r, h1_r) != (w1, h1):
            im1_proc = cv2.resize(image1, (w1_r, h1_r), interpolation=cv2.INTER_AREA)
        else:
            im1_proc = image1

        if im0_proc.dtype == np.uint8:
            inp0 = torch.from_numpy(im0_proc).float() / 255.0
        else:
            inp0 = torch.from_numpy(im0_proc).float()
            
        if im1_proc.dtype == np.uint8:
            inp1 = torch.from_numpy(im1_proc).float() / 255.0
        else:
            inp1 = torch.from_numpy(im1_proc).float()

        inp0 = inp0.unsqueeze(0).unsqueeze(0).to(self.device)
        inp1 = inp1.unsqueeze(0).unsqueeze(0).to(self.device)

        input_dict = {"image0": inp0, "image1": inp1}

        with torch.no_grad():
            try:
                correspondences = self.matcher(input_dict)
            except Exception as e:
                print(f"[LoFTRMatcher] Inference error: {e}")
                return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        confidence = correspondences['confidence'].cpu().numpy()

        if len(mkpts0) > 0:
            mkpts0[:, 0] *= (w0 / float(w0_r))
            mkpts0[:, 1] *= (h0 / float(h0_r))
            mkpts1[:, 0] *= (w1 / float(w1_r))
            mkpts1[:, 1] *= (h1 / float(h1_r))

        valid = confidence >= self.match_threshold
        
        return mkpts0[valid], mkpts1[valid], confidence[valid]
