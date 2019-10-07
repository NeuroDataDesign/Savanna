import numpy as np
import time
import logging
from savanna.utils.dataset import get_dataset

def print_items(fraction_of_train_samples, number_of_train_samples, best_accuracy, time_taken, cnn_model, cnn_config):
    """
    Prints logging info while training cnns.
    """
    if cnn_model:
        logging.info("CNN Config: " + str(cnn_config))
    logging.info("Train Fraction: " + str(fraction_of_train_samples))
    logging.info("# of Train Samples: " + str(number_of_train_samples))
    logging.info("Accuracy: " + str(best_accuracy))
    logging.info("Experiment Runtime: " + str(time_taken) + "\n")

def run_experiment(experiment, experiment_name, rf_type="shared", cnn_model=None, cnn_config={}):
    """
    Runs an experiment

    Parameters
    ----------
    experiment : function
        The function to call during the experiment
    experiment_name : string
        The name of the experiment.
    rf_type : string, optional default="shared"
        Specifies the type of rf procedure to use.
    cnn_model : string or None, optional (default=None)
        Specifies the model to use.

    Returns
    -------
    list
        Accuracy of the function as n grows.
    """
    DATA_PATH = "savanna/datasets"

    MIN_TRAIN_SAMPLES = 10
    MAX_TRAIN_SAMPLES = 100
    N_TRIALS = 20
    RUN_RF = True
    RUN_CNN = False
    DATASET_NAME = "FashionMNIST" 
    CHOOSEN_CLASSES = [0, 3]
    BATCH_SIZE = 8
    EPOCHS = 10

    global fraction_of_train_samples_space, numpy_data, pytorch_data, train_indices_all_trials

    logging.info("##################################################################")
    logging.info("acc vs n_samples: " + experiment_name + "\n")


    numpy_data = dict()
    (numpy_data["train_images"], numpy_data["train_labels"]), (numpy_data["test_images"], numpy_data["test_labels"]) = \
        get_dataset(DATA_PATH, DATASET_NAME, is_numpy=True)

    pytorch_data = dict()
    pytorch_data["trainset"], pytorch_data["testset"] = get_dataset(
        DATA_PATH, DATASET_NAME, is_numpy=False)

    class_wise_train_indices = [np.argwhere(
        numpy_data["train_labels"] == class_index).flatten() for class_index in CHOOSEN_CLASSES]

    total_train_samples = sum([len(ci) for ci in class_wise_train_indices])

    MIN_TRAIN_FRACTION = MIN_TRAIN_SAMPLES / total_train_samples
    MAX_TRAIN_FRACTION = MAX_TRAIN_SAMPLES / total_train_samples
    fraction_of_train_samples_space = np.geomspace(MIN_TRAIN_FRACTION, MAX_TRAIN_FRACTION, num=10)

    # progressive sub sampling for each trial, over fraction of train samples
    # constructs from previously sampled indices and adds on to them as frac progresses
    train_indices_all_trials = list()
    for n in range(N_TRIALS):
        train_indices = list()
        for frac in fraction_of_train_samples_space:
            sub_sample = list()
            for i, class_indices in enumerate(class_wise_train_indices):
                if not train_indices:
                    num_samples = int(len(class_indices) * frac + 0.5)
                    sub_sample.append(np.random.choice(class_indices, num_samples, replace=False))
                else:
                    num_samples = int(len(class_indices) * frac + 0.5) - len(train_indices[-1][i])
                    sub_sample.append(np.concatenate([train_indices[-1][i], np.random.choice(
                        list(set(class_indices) - set(train_indices[-1][i])), num_samples, replace=False)]).flatten())
            train_indices.append(sub_sample)
        train_indices = [np.concatenate(t).flatten() for t in train_indices]
        train_indices_all_trials.append(train_indices)

    number_of_train_samples_space = [len(i) for i in train_indices_all_trials[0]]

    IMG_SHAPE = numpy_data["train_images"].shape[1:]


    acc_vs_n_all_trials = list()

    for trial_number, train_indices in zip(range(N_TRIALS), train_indices_all_trials):
        logging.info("Trial " + str(trial_number+1) + "\n")
        acc_vs_n = list()
        for fraction_of_train_samples, sub_train_indices in zip(fraction_of_train_samples_space, train_indices):
            if not cnn_model:
                start = time.time()
                accuracy, time_tracker = experiment(DATASET_NAME, numpy_data,
                                                    CHOOSEN_CLASSES, sub_train_indices, rf_type)
                end = time.time()
            else:
                start = time.time()
                accuracy, time_tracker = experiment(DATASET_NAME, cnn_model, pytorch_data,
                                                    CHOOSEN_CLASSES, sub_train_indices, cnn_config)
                end = time.time()
            time_taken = (end - start)

            print_items(fraction_of_train_samples, len(sub_train_indices),
                        accuracy, time_taken, cnn_model, cnn_config)

            acc_vs_n.append((accuracy, time_taken, time_tracker))

        acc_vs_n_all_trials.append(acc_vs_n)

    logging.info("##################################################################")

    return acc_vs_n_all_trials

def get_title_and_results_path(dataset_name, choosen_classes, min_train_samples, max_train_samples):
    """
    Gets results path for test logging
    """
    if dataset_name == "CIFAR10":
        '''CIFAR10'''
        CIFAR10_MAP = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
                       4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        TITLE = " vs ".join([CIFAR10_MAP[i].capitalize() +
                             " (" + str(i) + ")" for i in choosen_classes])
        RESULTS_PATH = "results/cifar10/" + "vs".join([str(i) for i in choosen_classes]) + "/" + str(
            min_train_samples) + "_to_" + str(max_train_samples) + "/"
    elif dataset_name == "SVHN":
        '''SVHN'''
        TITLE = " vs ".join([str(i) for i in choosen_classes])
        RESULTS_PATH = "results/svhn/" + "vs".join([str(i) for i in choosen_classes]) + "/" + str(
            min_train_samples) + "_to_" + str(max_train_samples) + "/"
    elif dataset_name == "FashionMNIST":
        '''FashionMNIST'''
        FashionMNIST_MAP = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        TITLE = " vs ".join([FashionMNIST_MAP[i] + " (" + str(i) + ")" for i in choosen_classes])
        RESULTS_PATH = "results/fashion_mnist/" + "vs".join([str(i) for i in choosen_classes]) + "/" + str(
            min_train_samples) + "_to_" + str(max_train_samples) + "/"

    return TITLE, RESULTS_PATH


def get_number_of_train_samples_space(data, choosen_classes, min_train_samples, max_train_samples):
    class_wise_train_indices = [np.argwhere(data[0][1] == class_index).flatten()
                                for class_index in choosen_classes]

    total_train_samples = sum([len(ci) for ci in class_wise_train_indices])

    MIN_TRAIN_FRACTION = min_train_samples / total_train_samples
    MAX_TRAIN_FRACTION = max_train_samples / total_train_samples
    fraction_of_train_samples_space = np.geomspace(MIN_TRAIN_FRACTION, MAX_TRAIN_FRACTION, num=10)

    train_indices = list()
    for frac in fraction_of_train_samples_space:
        sub_sample = list()
        for i, class_indices in enumerate(class_wise_train_indices):
            if not train_indices:
                num_samples = int(len(class_indices) * frac + 0.5)
                sub_sample.append(np.random.choice(class_indices, num_samples, replace=False))
            else:
                num_samples = int(len(class_indices) * frac + 0.5) - len(train_indices[-1][i])
                sub_sample.append(np.concatenate([train_indices[-1][i], np.random.choice(
                    list(set(class_indices) - set(train_indices[-1][i])), num_samples, replace=False)]).flatten())
        train_indices.append(sub_sample)
    train_indices = [np.concatenate(t).flatten() for t in train_indices]

    number_of_train_samples_space = [len(i) for i in train_indices]

    return number_of_train_samples_space
