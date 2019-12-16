import matplotlib.pyplot as plt
from .data_process import scaled_data_train,scaled_data_validation,scaled_data_test


def train_data_viewer():
    for i in scaled_data_train:
        print(i)

def validation_data_viewer():
    for i in scaled_data_validation:
        print(i)

def test_data_viewer():
    for i in scaled_data_test:
        print(i)

def plot_loss(loss_tra,loss_val):
    plt.plot(loss_tra,label="Train")
    plt.plot(loss_val,label="Validation")
    plt.legend(loc="upper right")
    plt.title("Loss")
    plt.show()

def plot_acc(acc_tra,acc_val):
    plt.plot(acc_tra,label="Train")
    plt.plot(acc_val,label="Validation")
    plt.legend(loc="upper right")
    plt.title("Accuracy")
    plt.show()
