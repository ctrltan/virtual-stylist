import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random


class FeatureExtractor:


    def get_silhouette(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments)

        log_silhouette_moments = (-1 * np.sign(hu_moments) * np.log(np.abs(hu_moments))).flatten()

        return log_silhouette_moments


    def find_euclidean_distance(self, pixel, centroid):
        distance = math.sqrt((pixel[0] - centroid[0])**2 + (pixel[1] - centroid[1])**2 + (pixel[2] - centroid[2])**2)
        return distance
        
        
    def initialise_centroids(self, image, k):
        height, width, channels = image.shape

        samples = 2

        rows = int(k / 2)

        row_range = height // (rows + 1)

        initial_centroids = []

        for x in range(rows):
            row_num = x + 1
            index = (row_num * row_range) - 1
            row = [tuple(pixel[:3]) for pixel in image[index] if pixel[3] == 255]
            if len(row) >= samples:
                row_samples = random.sample(row, samples)
                initial_centroids.extend(row_samples)
        
        return initial_centroids

        
    def prominent_colours_clustering(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        image = cv2.resize(image, (2240, 3360))

        height, width = image.shape[:2]

        scale = 1/9

        image = cv2.resize(image, (int(width * scale), int(height * scale)))

        image = cv2.GaussianBlur(image, (15, 15), sigmaX=0)

        pixel_array = image.reshape((-1, 4))

        pixel_array = [pixel for pixel in pixel_array if pixel[3] == 255]

        trim_size = (len(pixel_array) // 3) * 3
        pixel_array = pixel_array[:trim_size]

        pixel_array = (np.array(pixel_array).reshape((-1, 3))).tolist()


        k = 10

        initial_centroids = self.initialise_centroids(image, k)


        prev_clusters = []

        num_centroids = len(initial_centroids)
        if num_centroids == k:
            clusters = [[] for x in range(k)]
        else:
            clusters = [[] for x in range(num_centroids)]


        for pixel in pixel_array:
            distances = [self.find_euclidean_distance(pixel, centroid) for centroid in initial_centroids]
            shortest_distance_index = np.argmin(distances)
            
            clusters[shortest_distance_index].append(pixel)

        max_iterations = 60
        iterations = 1

        while iterations <= max_iterations:

            new_centroids = []

            for pixel_cluster in clusters:
                cluster_len = len(pixel_cluster)

                red_channel_arr = [red[0] for red in pixel_cluster]
                green_channel_arr = [green[1] for green in pixel_cluster]
                blue_channel_arr = [blue[2] for blue in pixel_cluster]

                red_channel_mean = int(sum(red_channel_arr) / cluster_len)
                green_channel_mean = int(sum(green_channel_arr) / cluster_len)
                blue_channel_mean = int(sum(blue_channel_arr) / cluster_len)

                centroid = (red_channel_mean, green_channel_mean, blue_channel_mean)
                
                new_centroids.append(centroid)
            
            
            if num_centroids == k:
                clusters = [[] for x in range(k)]
            else:
                clusters = [[] for x in range(num_centroids)]

            for pixel in pixel_array:
                distances = [self.find_euclidean_distance(pixel, centroid) for centroid in new_centroids]
                shortest_distance_index = np.argmin(distances)
            
                clusters[shortest_distance_index].append(pixel)
            
            iterations+=1
        
        cluster_indexes = (np.argsort([len(cluster) for cluster in clusters])[::-1])

        three_largest_clusters = [clusters[index] for index in cluster_indexes]
        
        colours = []
        for cluster in three_largest_clusters:
            cluster_len = len(cluster)

            red_channel_arr = [red[0] for red in cluster]
            green_channel_arr = [green[1] for green in cluster]
            blue_channel_arr = [blue[2] for blue in cluster]

            red_channel_mean = int(sum(red_channel_arr) / cluster_len)
            green_channel_mean = int(sum(green_channel_arr) / cluster_len)
            blue_channel_mean = int(sum(blue_channel_arr) / cluster_len)

            colours.append((red_channel_mean, green_channel_mean, blue_channel_mean))
        
        colours = [colour for colour in colours if 255 not in colour and 254 not in colour]

        return colours[:3]


    def bounding_box(self, image, rate):
        max_height, max_width = image.shape[:2]

        min_height, min_width = (0, 0)

        top_border_found = False
        top_border = min_height
        while top_border_found == False:
            top_border += rate
            row = image[top_border, :, :]
            top_border_found = np.any(row > 0)
        
        bottom_border_found = False
        bottom_border = max_height
        while bottom_border_found == False:
            bottom_border -= rate
            row = image[bottom_border, :, :]
            bottom_border_found = np.any(row > 0)
        
        left_border_found = False
        left_border = min_width
        while left_border_found == False:
            left_border += rate
            col = image[:, left_border, :]
            left_border_found = np.any(col > 0)
        
        right_border_found = False
        right_border = max_width
        while right_border_found == False:
            right_border -= rate
            col = image[:, right_border, :]
            right_border_found = np.any(col > 0)
        
        cropped_image = image[top_border:bottom_border, left_border:right_border, :]

        return cropped_image


    def get_item_region(self, outfit, item_label):
        outfit = self.bounding_box(outfit, 3)
        height = outfit.shape[0]
        #print(height)

        if item_label == 'tops' or item_label == 'jackets':
            start = int(1/7 * height)
            end = int(11/20 * height)
            region = outfit[start:end, :, :]
            return region
        
        if item_label == 'skirts':
            start = int(31/70 * height)
            end = int(24/35 * height)
            region = outfit[start:end, :, :]
            return region

        if item_label == 'trousers' or item_label == 'long_skirts':
            start = int(7/17 * height)
            #end = int(395/304 * height)
            end = int(385/395 * height)
            region = outfit[start:end, :, :]
            return region

        if item_label == 'dresses':
            start = int(1/7 * height)
            end = int(24/35 * height)
            region = outfit[start:end, :, :]
            return region


    def find_similar_item(self, clothing_item, outfit):
        label = os.path.basename(os.path.dirname(clothing_item))

        clothing_item = cv2.imread(clothing_item, cv2.IMREAD_UNCHANGED)
        outfit = cv2.imread(outfit, cv2.IMREAD_UNCHANGED)

        c_height, c_width = clothing_item.shape[:2]
        o_height, o_width = outfit.shape[:2]
        outfit = cv2.resize(outfit, (2240, 3360))
        scale = 1/9
        clothing_item = cv2.resize(clothing_item, (int(c_width * scale), int(c_height * scale)))
        outfit = cv2.resize(outfit, (int(o_width * scale), int(o_height * scale)))

        clothing_item = cv2.cvtColor(clothing_item, cv2.COLOR_BGRA2RGBA)
        outfit = cv2.cvtColor(outfit, cv2.COLOR_BGRA2RGBA)

        #print(label)

        region = self.get_item_region(outfit, label)

        gray_clothing_item = cv2.cvtColor(clothing_item, cv2.COLOR_RGBA2GRAY)
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGBA2GRAY)


        sift = cv2.SIFT_create()

        keypoints1, descriptors1 = sift.detectAndCompute(gray_clothing_item, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray_region, None)


        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = bf.match(descriptors1, descriptors2)

        similarity = (len(matches) / len(keypoints1)) * 100

        return similarity

    
    


