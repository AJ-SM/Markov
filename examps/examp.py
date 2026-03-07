import cv2 as cv 
from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt 
from skimage.feature import hog

path = f"D:\Storeage-1\Main\ML-Model\examps\RDJ.jpg"
img = cv.imread(path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


resized = cv.resize(gray, (664, 664))



# Features 1 
def LBP():
    path = f"D:\Storeage-1\Main\ML-Model\examps\RDJ.jpg"
    img = cv.imread(path)
    radius = 2
    n_points = 6 * radius
    method = 'uniform'

    lbp_image = local_binary_pattern(resized, n_points, radius, method)

  
    cv.imshow('LBP-Image',lbp_image)
    plt.axis('off') 
    plt.show()


# Features 2
def Gabor():
    path = f"D:\Storeage-1\Main\ML-Model\examps\RDJ.jpg"
    img = cv.imread(path)
    ksize = 1000      
    sigma = 10.0     
    theta = np.pi/1 
    lambd = 100.0    
    gamma = 0.7     
    psi = 12

    kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_32F)
    # fimg = cv.filter2D(img, cv.CV_8UC3, kernel)

    cv.imshow('Actual Image',img)
    cv.imshow('Gabor Kernel',kernel)
    plt.axis('off') 
    plt.show()





# Features 3
def HOG():
    # 1. Load the image
    img_path = r'D:\Storeage-1\Main\ML-Model\examps\RDJ.jpg'
    img = cv.imread(img_path)
    
    if img is None:
        print("Error: Could not read image.")
        return


    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 


    resized = cv.resize(gray, (64, 64))


    features, hog_image = hog(resized, orientations=8, 
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), 
                               visualize=True, 
                               feature_vector=True)

    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG Feature Visualization")
    plt.show()

HOG()
Gabor()
LBP()



    

