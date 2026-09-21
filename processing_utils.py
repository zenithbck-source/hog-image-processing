import os

def sort_images():
    all_images = {'tennisball':[], 'shuttlecock':[]}

    tennisball_drive_path = r'C:\Users\stu-boock\Documents\hog-image-processing\tennisball'
    tennisball_files = os.listdir(tennisball_drive_path)

    shuttlecock_drive_path = r'C:\Users\stu-boock\Documents\hog-image-processing\shuttlecock'
    shuttlecock_files = os.listdir(shuttlecock_drive_path)

    return tennisball_files, shuttlecock_files


if __name__ == "__main__":
    tennis, shuttle = sort_images()
    print(tennis)
    print(shuttle)