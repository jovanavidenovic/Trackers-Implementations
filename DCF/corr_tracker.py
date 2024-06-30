import numpy as np
import cv2

class CorrTrackerParams():
    def __init__(self, alpha=0.15, filter_lambda=3, search_region_factor=1.15, gaussian_size=3, simplify=False):
        self.alpha = alpha
        self.filter_lambda = filter_lambda
        self.search_region_factor = search_region_factor
        self.gaussian_size = gaussian_size
        self.simplify = simplify

class CorrTracker():

    def __init__(self, params=CorrTrackerParams()):
        self.alpha = params.alpha
        self.filter_lambda = params.filter_lambda
        self.search_region_factor = params.search_region_factor
        self.gaussian_size = params.gaussian_size
        self.simplify = params.simplify

    def __preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = (image - np.mean(image)) / np.std(image)
        return image

    def name(self):
        if self.simplify:
            return 'CFT'
        else:
            return 'MOSSE'
    
    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
            
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.__preprocess_image(image)
        self.window_size = int(max(region[2], region[3]) * self.search_region_factor)
        left = int(max(region[0], 0))
        top = int(max(region[1], 0))
        right = int(min(region[0] + region[2], image.shape[1] - 1))
        bottom = int(min(region[1] + region[3], image.shape[0] - 1))
        width = right - left
        height = bottom - top
        
        if left + self.window_size > image.shape[1]:
            self.window_size = image.shape[1] - left - 1
        if top + self.window_size > image.shape[0]:
            self.window_size = image.shape[0] - top - 1

        self.window_size = (self.window_size, self.window_size)

        self.position = (top + height // 2, left + width // 2) # (y, x)
        self.template = image[top:top+height, left:left+width]        
        self.target_size = (self.template.shape[1], self.template.shape[0]) # (width, height)

        # extract correlation filter
        # Gaussian peak
        self.G = create_gauss_peak(self.window_size, self.gaussian_size)
        self.window_size = self.G.shape # (height, width)
        self.G = np.fft.fft2(self.G)
        # feature patch
        x0_F = max(int(self.position[1] - self.window_size[0] / 2), 0)
        y0_F = max(int(self.position[0] - self.window_size[1] / 2), 0)

        self.F = image[y0_F:y0_F+self.window_size[0], x0_F:x0_F+self.window_size[1]]
        self.cos_window = create_cosine_window(self.F.shape)
        self.F = np.multiply(self.F, self.cos_window)
        self.F = np.fft.fft2(self.F)
        self.F_cc = np.conj(self.F)

        self.numerator = np.multiply(self.G, self.F_cc)
        self.denominator = np.multiply(self.F, self.F_cc) + self.filter_lambda
        self.H = np.divide(self.numerator, self.denominator)


    def track(self, image):
        image = self.__preprocess_image(image)
        x0 = max(int(self.position[1] - self.window_size[0] / 2), 0)
        y0 = max(int(self.position[0] - self.window_size[1] / 2), 0)
        
        if x0 + self.window_size[1] > image.shape[1]:
            x0 = image.shape[1] - self.window_size[1] - 1
        if y0 + self.window_size[0] > image.shape[0]:
            y0 = image.shape[0] - self.window_size[0] - 1

        # extract search region
        search_region = image[y0:y0+self.window_size[0], x0:x0+self.window_size[1]]

        search_region = np.multiply(search_region, self.cos_window)
        search_region = np.fft.fft2(search_region)

        # calculate response
        response = np.multiply(self.H, search_region)
        response = np.fft.ifft2(response)

        max_idx = np.argmax(response)
        y_max_idx, x_max_idx = np.unravel_index(max_idx, response.shape)

        if x_max_idx > self.window_size[1] / 2:
            x_max_idx -= self.window_size[1]
        if y_max_idx > self.window_size[0] / 2:
            y_max_idx -= self.window_size[0] 
        
        # update position
        self.position = (self.position[0] + y_max_idx, self.position[1] + x_max_idx)  

        # update filter
        x0_F = max(int(self.position[1] - self.window_size[0] / 2), 0)
        y0_F = max(int(self.position[0] - self.window_size[1] / 2), 0)
        if x0_F + self.window_size[1] > image.shape[1]:
            x0_F = image.shape[1] - self.window_size[1] - 1
        if y0_F + self.window_size[0] > image.shape[0]:
            y0_F = image.shape[0] - self.window_size[0] - 1

        F_new = image[y0_F:y0_F+self.window_size[0], x0_F:x0_F+self.window_size[1]]
        F_new = np.multiply(F_new, self.cos_window)
        F_new = np.fft.fft2(F_new)
        F_new_cc = np.conj(F_new)

        if self.simplify:
            H_new = np.divide(np.multiply(self.G, F_new_cc), np.multiply(F_new, F_new_cc) + self.filter_lambda)   
            self.H = (1 - self.alpha) * self.H + self.alpha * H_new 
        else:
            numerator_new = np.multiply(self.G, F_new_cc)
            denominator_new = np.multiply(F_new, F_new_cc) + self.filter_lambda

            self.numerator = (1 - self.alpha) * self.numerator + self.alpha * numerator_new
            self.denominator = (1 - self.alpha) * self.denominator + self.alpha * denominator_new
            
            self.H = np.divide(self.numerator, self.denominator)

        bbox = [self.position[1] - self.target_size[0] // 2, self.position[0] - self.target_size[1] // 2, self.target_size[0], self.target_size[1]]
        return bbox
