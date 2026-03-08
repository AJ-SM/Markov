import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern, hog
import matplotlib.pyplot as plt

def get_intensity_metrics(gray_img):
    """
    Computes Exposure (E) and estimated Gamma.
    E: Average pixel intensity.
    Gamma: Log-mean heuristic to estimate the gamma curve.
    """
    E = np.mean(gray_img)
    
    # Avoid log(0) errors
    mean_val = np.clip(np.mean(gray_img) / 255.0, 1e-5, 1.0)
    gamma = np.log(0.5) / np.log(mean_val)
    
    return E, gamma

def extract_lbp_features(gray_img, radius=2):
    """Computes a histogram of Local Binary Patterns."""
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    # Normalize histogram to make it scale-invariant
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    return hist

def extract_gabor_features(gray_img):
    """Computes mean response of a Gabor kernel (texture analysis)."""
    ksize = 31 # Reduced from 1000 to reasonable processing size
    sigma = 5.0
    theta = np.pi / 4
    lambd = 10.0
    gamma = 0.5
    psi = 0
    kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_32F)
    fimg = cv.filter2D(gray_img, cv.CV_8UC3, kernel)
    return np.mean(fimg)

def extract_hog_features(gray_img):
    """Computes HOG descriptor."""
    # Resize to specific dimensions required by HOG
    resized = cv.resize(gray_img, (64, 64))
    features = hog(resized, orientations=8, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), feature_vector=True)
    return features

def get_full_feature_vector(image_path):
    """Orchestrates the feature extraction."""
    img = cv.imread(image_path)
    if img is None:
        return None
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 1. Intensity
    E, gamma = get_intensity_metrics(gray)
    
    # 2. Texture
    lbp = extract_lbp_features(gray)
    gabor = extract_gabor_features(gray)
    hog_feat = extract_hog_features(gray)
    
    # 3. Concatenate all into one vector
    # Final Vector = [E, Gamma, Gabor_Mean, LBP_Hist..., HOG_Feats...]
    full_vector = np.concatenate(([E, gamma, gabor], lbp, hog_feat))
    return full_vector

# --- Main Execution ---
path = r"D:\Storeage-1\Main\ML-Model\examps\RDJ.jpg"
features = get_full_feature_vector(path)





