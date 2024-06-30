import numpy as np
import cv2

from mean_shift import mean_shift

class MeanShiftTracker(Tracker):

    def __init__(self, params):
        self.number_bins = params.number_bins
        self.sigma = params.sigma
        self.alpha = params.alpha
        self.number_iters = params.number_iters
        self.eps = params.eps
        self.use_hsv = params.use_hsv

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
            
        self.window = max(region[2], region[3])

        left = int(max(region[0], 0))
        top = int(max(region[1], 0))
        right = int(min(region[0] + region[2], image.shape[1] - 1))
        bottom = int(min(region[1] + region[3], image.shape[0] - 1))
        width = right - left
        height = bottom - top
        
        self.kernel = create_epanechnik_kernel(width=width, height=height, sigma=self.sigma)
        self.inverse_kernel = np.ones(self.kernel.shape)

        self.position = (top + height // 2, left + width // 2)
        self.template = image[top:top+self.kernel.shape[0], left:left+self.kernel.shape[1]]
        self.size = (self.template.shape[1], self.template.shape[0])

        if self.use_hsv:
            self.q = extract_histogram_hsv(self.template, self.number_bins, weights=self.kernel)
        else:
            self.q = extract_histogram(self.template, self.number_bins, weights=self.kernel)
        self.q /= np.sum(self.q)

    def track(self, image):
        self.position = mean_shift(img=image, init_pos=(self.position[1], self.position[0]), g=self.inverse_kernel, kernel=self.kernel, target_size=self.size, q=self.q, eps=self.eps, max_iter=self.number_iters, use_hsv=self.use_hsv)
        self.position = (self.position[1], self.position[0])
        
        x0 = max(int(self.position[1] - self.size[0] / 2), 0)
        y0 = max(int(self.position[0] - self.size[1] / 2), 0)

        if x0 + self.size[0] >= image.shape[1]:
            x0 = max(0, image.shape[1] - self.size[0] - 1)
        if y0 + self.size[1] >= image.shape[0]:
            y0 = max(0, image.shape[0] - self.size[1] - 1)

        bbox = [x0, y0, self.size[0], self.size[1]]

        if self.use_hsv:
            q_new = extract_histogram_hsv(image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]], self.number_bins, weights=self.kernel)
        else:
            q_new = extract_histogram(image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]], self.number_bins, weights=self.kernel)
        q_new /= np.sum(q_new)
        self.q = self.alpha * q_new + (1 - self.alpha) * self.q
        self.q /= np.sum(self.q)
        return bbox

class MSParams():
    def __init__(self, number_bins=16, sigma=0.5, alpha=0.025, number_iters=20, eps=1, use_hsv=False):
        self.number_bins = number_bins
        self.sigma = sigma
        self.alpha = alpha
        self.number_iters = number_iters
        self.eps = eps
        self.use_hsv = use_hsv
