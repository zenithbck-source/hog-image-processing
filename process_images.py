from processing_utils import sort_images, process_images, extracting_images
from models import splitting_data, KNN_model, SVM_model, show_testing


images = sort_images()
processed_images = process_images(images)
X, y, hog_images = extracting_images(processed_images)
X_train, X_test, y_train, y_test, idx_train, idx_test = splitting_data(X, y)

y_pred_knn = KNN_model(X_train, X_test, y_train, y_test)
y_pred_svm = SVM_model(X_train, X_test, y_train, y_test)

show_testing(hog_images, idx_test, y_pred_knn, y_pred_svm)
    