import os
import cv2
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# tennisball_path = r'C:\Users\stu-boock\Documents\hog-image-processing\tennisball'
# shuttlecock_path = r'C:\Users\stu-boock\Documents\hog-image-processing\tennisball'

tennisball_path = r'C:\Users\zenit\Documents\repositories\hog-image-processing\tennisball'
shuttlecock_path = r'C:\Users\zenit\Documents\repositories\hog-image-processing\tennisball'

def sort_images():
    all_images = {'tennisball':[], 'shuttlecock':[]}

    tennisball_files = os.listdir(tennisball_path)
    shuttlecock_files = os.listdir(shuttlecock_path)

    for filename in tennisball_files:
        all_images['tennisball'].append(filename)

    for filename in shuttlecock_files:
            all_images['shuttlecock'].append(filename)

    return all_images

def process_images(images):
    processed_images = {'tennisball':[], 'shuttlecock':[]}

    for label, filenames in images.items():
        if label == 'tennisball':
            image_dir = tennisball_path
        elif label == 'shuttlecock':
            image_dir = shuttlecock_path

        for filename in filenames:
            image_path_full = os.path.join(image_dir, filename)
            image = cv2.imread(image_path_full, cv2.IMREAD_GRAYSCALE)
            image_resized = cv2.resize(image, (100, 100))
            if(label == 'tennisball'):
                processed_images['tennisball'].append(image_resized)
            elif(label == 'shuttlecock'):
                processed_images['shuttlecock'].append(image_resized)

    return processed_images

def extract_image_hog(processed_image):
    fd, hog_image = hog(processed_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return fd

def extracting_images(processed_images):
    X = []
    y = []

    for label, images_list in processed_images.items():
        for image in images_list:
            features = extract_image_hog(image)
            if features is not None:
                X.append(features)
                if label == 'tennisball':
                    y.append(0)
                elif label == 'shuttlecock':
                    y.append(1)

    return X, y


if __name__ == "__main__":
    images = sort_images()
    processed_images = process_images(images)
    X, y = extracting_images(processed_images)

    print(X)
    print(y)