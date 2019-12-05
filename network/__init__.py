import tensorflow as tf
#print(tf.__version__)
from .data_process import final_data
from .data_process import final_labels

final_data = final_data
final_labels = final_labels

print(final_data.shape)
print(final_labels.shape)
