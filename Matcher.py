import cv2
import numpy as np
from MinutiaeFeature import MinutiaeExtractor, MinutiaeMatcher
from Extractor import SIFTMatcher

def draw_minutiae(img, features, radius=3):
    out = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    for m in features:
        color = (0, 0, 255) if m.type == 'Termination' else (255, 0, 0)
        cv2.circle(out, (m.x, m.y), radius, color, 1)
    return out


def match_fingerprints(img1, img2, visualize=False):
    extractor = MinutiaeExtractor()
    matcher = MinutiaeMatcher()
    sift_matcher = SIFTMatcher()

    f1_term, f1_bif = extractor.extract(img1)
    f2_term, f2_bif = extractor.extract(img2)

    score_minutiae = matcher.match(f1_term, f2_term)
    score_sift = sift_matcher.match(img1, img2)

    final_score = max(score_minutiae, score_sift)
    matched = final_score > 0.4

    if visualize:
        vis1 = draw_minutiae(img1, f1_term + f1_bif)
        vis2 = draw_minutiae(img2, f2_term + f2_bif)
        combined = np.hstack((vis1, vis2))
        cv2.imshow("Minutiae Comparison", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Minutiae Score: {score_minutiae:.2f}, SIFT Score: {score_sift:.2f}")
    print("Fingerprint Match:", "✅ MATCHED" if matched else "❌ NOT MATCHED")
    return matched


# Example usage
if __name__ == "__main__":
    img1 = cv2.imread("fingerprint1.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("fingerprint2.png", cv2.IMREAD_GRAYSCALE)
    match_fingerprints(img1, img2, visualize=True)
