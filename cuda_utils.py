import cv2
import numpy as np

class CUDAPipeline:
    def __init__(self):
        self.stream = cv2.cuda.Stream()
        self.gpu_img = cv2.cuda_GpuMat()
        self.gpu_resized = cv2.cuda_GpuMat()
        self.gpu_hsv = cv2.cuda_GpuMat()
        self.gpu_mask = cv2.cuda_GpuMat()
        
    def process_frame(self, img, width, height):
        # Upload image to GPU
        self.gpu_img.upload(img)
        
        # Resize on GPU
        self.gpu_resized = cv2.cuda.resize(self.gpu_img, (width, height))
        
        # Convert color space on GPU
        self.gpu_hsv = cv2.cuda.cvtColor(self.gpu_resized, cv2.COLOR_BGR2HSV)
        
        # Download result
        hsv_img = self.gpu_hsv.download()
        return hsv_img
    
    def calculate_histogram(self, img, mask, nbins=3):
        # Upload images to GPU
        self.gpu_img.upload(img)
        self.gpu_mask.upload(mask)
        
        # Calculate histogram on GPU
        hist = cv2.cuda.calcHist(self.gpu_img, [0, 1], self.gpu_mask, [nbins, 2*nbins], [0, 180, 0, 256])
        
        # Normalize on GPU
        cv2.cuda.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
        
        # Download result
        return hist.download()
    
    def compare_histograms(self, hist1, hist2):
        # Upload histograms to GPU
        gpu_hist1 = cv2.cuda_GpuMat()
        gpu_hist2 = cv2.cuda_GpuMat()
        gpu_hist1.upload(hist1)
        gpu_hist2.upload(hist2)
        
        # Compare histograms on GPU
        correlation = cv2.cuda.compareHist(gpu_hist1, gpu_hist2, cv2.HISTCMP_CORREL)
        return correlation 