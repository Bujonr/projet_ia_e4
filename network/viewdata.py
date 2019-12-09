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
