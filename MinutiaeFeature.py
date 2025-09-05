import cv2
import numpy as np
import math
from skimage.morphology import skeletonize, convex_hull_image, erosion, square
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class MinutiaeFeature:
    def __init__(self, x, y, orientation, mtype):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.type = mtype  # 'Termination' or 'Bifurcation'


class MinutiaeExtractor:
    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
        skeleton = skeletonize(binary)
        return np.uint8(skeleton), binary

    def _compute_orientation(self, block, mtype):
        h, w = block.shape
        cx, cy = (h - 1) / 2, (w - 1) / 2
        angle = []

        for i in range(h):
            for j in range(w):
                if ((i == 0 or i == h - 1 or j == 0 or j == w - 1) and block[i, j]):
                    angle.append(-math.degrees(math.atan2(i - cy, j - cx)))

        if (mtype == 'Termination' and len(angle) == 1) or (mtype == 'Bifurcation' and len(angle) == 3):
            return angle
        return [float('nan')]

    def extract(self, img):
        skel, mask = self.preprocess(img)
        rows, cols = skel.shape
        term_mask = np.zeros_like(skel)
        bif_mask = np.zeros_like(skel)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if skel[i, j]:
                    block = skel[i - 1:i + 2, j - 1:j + 2]
                    val = np.sum(block)
                    if val == 2:
                        term_mask[i, j] = 1
                    elif val == 4:
                        bif_mask[i, j] = 1

        refined_mask = convex_hull_image(mask)
        refined_mask = erosion(refined_mask, square(5))
        term_mask = term_mask * refined_mask
        bif_mask = bif_mask * refined_mask

        term_labeled = label(term_mask)
        bif_labeled = label(bif_mask)

        term_features = self._extract_features(skel, term_labeled, 'Termination', 2)
        bif_features = self._extract_features(skel, bif_labeled, 'Bifurcation', 1)

        return term_features, bif_features

    def _extract_features(self, skel, label_img, mtype, window):
        features = []
        props = regionprops(label_img)
        for p in props:
            cy, cx = np.round(p.centroid).astype(int)
            if cy - window < 0 or cy + window + 1 > skel.shape[0] or cx - window < 0 or cx + window + 1 > skel.shape[1]:
                continue
            block = skel[cy - window:cy + window + 1, cx - window:cx + window + 1]
            orientation = self._compute_orientation(block, mtype)
            if not any(np.isnan(orientation)):
                features.append(MinutiaeFeature(cx, cy, orientation, mtype))
        return features


class MinutiaeMatcher:
    def __init__(self, dist_thresh=15, angle_thresh=20):
        self.dist_thresh = dist_thresh
        self.angle_thresh = angle_thresh

    def match(self, f1, f2):
        m1 = [(p.x, p.y, p.orientation[0]) for p in f1 if p.type == 'Termination']
        m2 = [(p.x, p.y, p.orientation[0]) for p in f2 if p.type == 'Termination']

        if not m1 or not m2:
            return 0.0

        matched = 0
        for x1, y1, a1 in m1:
            for x2, y2, a2 in m2:
                dist = np.hypot(x1 - x2, y1 - y2)
                angle_diff = abs(a1 - a2)
                if dist < self.dist_thresh and angle_diff < self.angle_thresh:
                    matched += 1
                    break
        return matched / max(len(m1), len(m2))
