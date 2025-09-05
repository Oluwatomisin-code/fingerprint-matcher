import cv2

class SIFTMatcher:
    def __init__(self, ratio_thresh=0.75):
        self.ratio_thresh = ratio_thresh
        self.sift = cv2.SIFT_create()

    def match(self, img1, img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0.0

        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = matcher.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < self.ratio_thresh * n.distance]
        score = len(good) / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0.0
        return score
