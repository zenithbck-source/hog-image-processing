import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from processing_utils import sort_images, process_images, extracting_images
import math

def splitting_data(X, y):
    X = np.array(X)
    y = np.array(y)
    indices = np.arange(len(X))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, idx_train, idx_test

def KNN_model(X_train, X_test, y_train, y_test):

    # Parameters for KNN Classifier (n_neighbors, weights)
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
        'weights': ['uniform', 'distance'] # controls how much each K vote counts
    }

    # Generate all combinations of the above parameters and cross-validate (10-fold) on the training set
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=10, scoring='accuracy')
    knn_grid.fit(X_train, y_train)

    # Use KNN Classifier with the best parameters found above (GridSearchCV already retrained it on the full training set)
    best_knn = knn_grid.best_estimator_
    y_pred_knn = best_knn.predict(X_test) # Predict labels for the held-out test set

    # Evaluate the model
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"Best parameters (KNN): {knn_grid.best_params_}")
    print(f"Accuracy (KNN): {accuracy_knn * 100:.2f}%")
    print(f"Predictions (KNN): {y_pred_knn}")

    return y_pred_knn

def SVM_model(X_train, X_test, y_train, y_test):

    # Possible parameters for SVM Classifier (kernel, C, Gamma)
    svc_param_grid = {
        'kernel': ['linear', 'rbf'],   # whether the boundary is a straight line or a curve
        'C': [0.01, 0.1, 1, 10, 100],  # controls how strict the SVM is (high C: hug support vectors tightly, low C: wider, generalised boundary)
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]  # only used when kernel='rbf', controls each point's reach (high gamma: affects nearby area, low gamma: affects wider area)
    }

    # Generate all combinations of the above parameters and cross-validate (10-fold) on the training set
    svm_grid = GridSearchCV(SVC(), svc_param_grid, cv=10, scoring='accuracy')
    svm_grid.fit(X_train, y_train)

    # Use SVM Classifier with the best parameters found above (GridSearchCV already retrained it on the full training set)
    best_svm = svm_grid.best_estimator_
    y_pred_svm = best_svm.predict(X_test)  # Predict labels for the held-out test set

    # Evaluate the model
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Best parameters (SVM): {svm_grid.best_params_}")
    print(f"Accuracy (SVM): {accuracy_svm * 100:.2f}%")
    print(f"Predictions (SVM): {y_pred_svm}")

    return y_pred_svm

def show_testing(hog_images, idx_test, y_pred_knn, y_pred_svm, images_per_row=5):
    num_images = len(idx_test)
    num_rows = math.ceil(num_images / images_per_row)

    fig = plt.figure(figsize=(images_per_row * 3, num_rows * 3 + 1))

    for plot_position, original_index in enumerate(idx_test):
        plt.subplot(num_rows, images_per_row, plot_position + 1)
        plt.imshow(hog_images[original_index], cmap='gray')
        plt.axis('off')
        plt.title(f"{y_pred_knn[plot_position]} | {y_pred_svm[plot_position]}", fontsize=8)

    legend_elements = [
        Patch(facecolor='none', edgecolor='none', label='Title format: KNN | SVM'),
        Patch(facecolor='none', edgecolor='none', label='0 = Tennisball, 1 = Shuttlecock')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=1)

    # plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave room at the bottom for the legend
    plt.show()


if __name__ == "__main__":
    images = sort_images()
    processed_images = process_images(images)
    X, y, hog_images = extracting_images(processed_images)
    X_train, X_test, y_train, y_test, idx_train, idx_test = splitting_data(X, y)

    y_pred_knn = KNN_model(X_train, X_test, y_train, y_test)
    y_pred_svm = SVM_model(X_train, X_test, y_train, y_test)

    show_testing(hog_images, idx_test, y_pred_knn, y_pred_svm)
    