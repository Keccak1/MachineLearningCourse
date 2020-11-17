# using mnist dataset, optimize classifier to distinguish hand written digits
# all vs one classification - train ten classifiers,
# each of which will classify specific value against the rest

# download train and test data from
# http://yann.lecun.com/exdb/mnist/index.html and extract each archive
# if previous exercise have been done correctly,
# training should boil down only to appropriate data conversion
# (set data to X and y)

from matplotlib import pyplot as plt
import numpy as np


def main():
    # how to read mnist https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
    training_data, training_labels = read_data_set('./train-images-idx3-ubyte', './train-labels-idx1-ubyte', 60000)
    training_data = training_data/255.0 # can be normalized to converge faster

    # check if data have been loaded correctly
    example_idx = 156
    print(training_labels[example_idx])
    plt.imshow(training_data[example_idx])

    # training data must be vectorized
    # from [n_examples, height, width] to [n_examples, n_features]
    training_data.shape = [training_data.shape[0], training_data.shape[1] * training_data.shape[2]]

    # lets start with '2' vs rest
    y = np.array(training_labels == 2, dtype=np.float32)
    X = np.concatenate((np.ones((1, len(y))), training_data.T))

    theta = np.zeros((X.shape[0], 1))

    #---------------------

    alpha = 0.00001
    iter = 10

    while iter >= 0:
        # optimize theta

        iter -= 1
        pass

    # now we need to read test set and reshape it similarly to training set
    test_data, test_labels = read_data_set('./t10k-images-idx3-ubyte', './t10k-labels-idx1-ubyte', 10000)
    test_data = test_data / 255.0
    test_data.shape = [test_data.shape[0], test_data.shape[1] * test_data.shape[2]]
    y = np.array(test_labels == 2, dtype=np.float32)
    X = np.concatenate((np.ones((1, len(y))), test_data.T))

    # calculate hypothesis for test data
    h = 1.0 / (1 + np.exp(-theta.T @ X))
    y_pred = np.round(h.flatten()) # and classify it

    # calculate TP...
    # TP =
    # FP =
    # TN =
    # FN =

    # and the sensitivity and positive predictivity
    # se = TP / (TP+FN)
    # pp = TP / (TP+FP)

    # print('se: {}, pp: {}'.format(se, pp))


def read_data_set(images_path, labels_path, num_images):
    image_size = 28
    # training set
    with open(images_path, 'rb') as f:
        f.read(16)
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        training_data = data.reshape(num_images, image_size, image_size)
    # labels
    with open(labels_path, 'rb') as f:
        f.read(8)
        buf = f.read(num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        training_labels = data.reshape(num_images)
    return training_data, training_labels


if __name__ == "__main__":
    main()
