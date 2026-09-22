import os
import cv2
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



if __name__ == "__main__":
    images = sort_images()
    processed_images = process_images(images)

    plt.figure(figsize=(4, 4))
    plt.imshow(processed_images['tennisball'][0], cmap='gray')
    plt.show()