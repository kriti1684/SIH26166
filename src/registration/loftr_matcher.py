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
        # Load the pre-trained LoFTR model from Kornia
        self.matcher = LoFTR(pretrained=pretrained).to(self.device).eval()
        self.match_threshold = match_threshold
        print(f"[LoFTRMatcher] Initialized on device: {self.device} (pretrained: {pretrained})")

    def match(self, image0: np.ndarray, image1: np.ndarray):
        """
        Extract dense feature matches between two images using LoFTR.
        
        Args:
            image0, image1: grayscale numpy arrays [H, W] (uint8 or float32).
            
        Returns:
            src_pts, ref_pts, confidence (all np.ndarray of type float32).
        """
        # Normalize to [0, 1] as required by Kornia's LoFTR
        if image0.dtype == np.uint8:
            inp0 = torch.from_numpy(image0).float() / 255.0
        else:
            inp0 = torch.from_numpy(image0).float()
            
        if image1.dtype == np.uint8:
            inp1 = torch.from_numpy(image1).float() / 255.0
        else:
            inp1 = torch.from_numpy(image1).float()

        # LoFTR requires inputs to be [B, 1, H, W]
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

        # Filter by threshold
        valid = confidence >= self.match_threshold
        
        return mkpts0[valid], mkpts1[valid], confidence[valid]
