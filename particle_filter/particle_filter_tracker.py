import cv2
import numpy as np
from kalman import kf_matrices

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

class PFTrackerParams():
    def __init__(self, motion_model="NCV", N=150, q=0.1, use_hsv=False):
        self.motion_model = motion_model
        self.N = N
        self.q = q
        self.use_hsv = use_hsv

class PFTracker(Tracker):

    def __init__(self, params=PFTrackerParams()):
        self.motion_model = params.motion_model
        self.N = params.N
        self.q = params.q
        self.alpha = 0.0005
        self.use_hsv = params.use_hsv
    
    def name(self):
        return "PFTracker"
    
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
        
        self.kernel = create_epanechnik_kernel(width=width, height=height, sigma=1)

        self.position = (left + width // 2, top + height // 2)
        template = image[top:top+self.kernel.shape[0], left:left+self.kernel.shape[1]]
        self.size = (template.shape[1], template.shape[0])

        if self.use_hsv:
            self.template_hist = extract_histogram_hsv(template, nbins=16, weights=self.kernel)
        else:
            self.template_hist = extract_histogram(template, nbins=16, weights=self.kernel)

        self.template_hist /= np.sum(self.template_hist)
        self.template_hist = self.template_hist.astype(np.float32)

        self.Fi, self.Q, self.H, self.R = kf_matrices(self.motion_model, q_param=self.q * min(self.size[0], self.size[1]), r_param=1)

        if self.motion_model == "RW":
            self.x = np.array([self.position[0], self.position[1]], dtype=np.float32)
        elif self.motion_model == "NCV":
            self.x = np.array([self.position[0], self.position[1], 0, 0], dtype=np.float32)
        elif self.motion_model == "NCA":
            self.x = np.array([self.position[0], self.position[1], 0, 0, 0, 0], dtype=np.float32)

        self.particles = sample_gauss(self.x, self.Q, self.N)
        self.weights = np.ones(self.N, dtype=np.float32) / self.N

        return self.particles

    def track(self, image):
        # resample particles according to their weights
        idx = np.random.choice(self.N, size=self.N, p=self.weights)
        self.particles = self.particles[idx]
        self.weights = np.ones(self.N, dtype=np.float32) / self.N

        # project particles through the motion model
        noise = sample_gauss(np.zeros(self.Q.shape[0]), self.Q, self.N)
        # move each particle
        if self.motion_model == "RW":
            self.particles += noise
        elif self.motion_model == "NCV":
            self.particles = np.matmul(self.Fi, self.particles.T).T + noise
        elif self.motion_model == "NCA":
            self.particles = np.matmul(self.Fi, self.particles.T).T + noise

        # recalculate weights on new positions
        for i in range(self.N):
            x = int(self.particles[i, 0])
            y = int(self.particles[i, 1])

            if x + self.size[0] / 2 >= image.shape[1] or y + self.size[1] / 2 >= image.shape[0]:
                x = image.shape[1] - self.size[0] / 2 - 1
                y = image.shape[0] - self.size[1] / 2 - 1

            top = int(max(y- self.size[1] / 2, 0))
            left = int(max(x - self.size[0] / 2, 0))
            bottom = top  + self.size[1]
            right = left + self.size[0]

            if bottom > image.shape[0]:
                bottom = image.shape[0] - 1
                top = bottom - self.size[1]
            if right > image.shape[1]:
                right = image.shape[1] - 1
                left = right - self.size[0]
            
            patch = image[top:bottom, left:right]
            
            if self.use_hsv:
                hist = extract_histogram_hsv(patch, nbins=16, weights=self.kernel)
            else:
                hist = extract_histogram(patch, nbins=16, weights=self.kernel)

            hist /= np.sum(hist)
            hist = hist.astype(np.float32)

            hel_dist = cv2.compareHist(self.template_hist, hist, cv2.HISTCMP_HELLINGER)
            # print(hel_dist)
            self.weights[i] = np.exp(-0.5 * hel_dist**2 / 0.2**2)

        self.weights /= np.sum(self.weights)

        # calculate new position of the target as the weighted mean of the particles
        self.x = np.sum(self.weights * self.particles.T, axis=1)
        self.x = self.x.flatten().astype(np.float32)

        self.position = (self.x[0], self.x[1])

        # update target visual model
        top = int(max(self.position[1] - self.size[1] / 2, 0))
        left = int(max(self.position[0] - self.size[0] / 2, 0))
        bottom = top  + self.size[1]
        right = left + self.size[0]

        if bottom > image.shape[0]:
            bottom = image.shape[0] - 1
            top = bottom - self.size[1]
        if right > image.shape[1]:
            right = image.shape[1] - 1
            left = right - self.size[0]

        bbox = [left, top, self.size[0], self.size[1]]
        new_target_patch = image[top:bottom, left:right]
        if self.use_hsv:
            new_target_hist = extract_histogram_hsv(new_target_patch, nbins=16, weights=self.kernel)
        else:
            new_target_hist = extract_histogram(new_target_patch, nbins=16, weights=self.kernel)
        new_target_hist /= np.sum(new_target_hist)

        self.template_hist = (1 - self.alpha) * self.template_hist + self.alpha * new_target_hist
        self.template_hist = self.template_hist.astype(np.float32)
        
        return bbox, self.particles
