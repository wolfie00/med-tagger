# (Medical) Image Tagger

This repository provides a basic model for (medical & imbalanced) multi-label image classification. 

--- 

<p align="center">
<img src="image.png" width=100% height=100% 
class="center">
</p>

## Instructions

All the parameters are specified by a single Python dictionary that can be passed to the constructor of the class.
The data has to be in a .csv format like this:

<table>
  <tr>
    <th>image</th>
    <th>labels</th>
  </tr>
  <tr>
    <td>image1</td>
    <td>tag1;tag2;tag3</td>
  </tr>
  <tr>
    <td>image2</td>
    <td>tag4;tag1</td>
  </tr>
</table>

The program will read the .csv files into a Python dictionary. The two columns must be separated by a `\t`. Alternatively, you can load a JSON file directly as a `dict`. Next, the list with the categories (labels) can either be a Python list passed to the `configuration` or a `.txt` file with a label per line.
For the loss function binary cross-entropy, Focal loss, ASL loss and soft F1 loss are supported. For Focal and ASL losses, you have to specify the respective parameters of each loss.

## Example of Use

```python
from tagcxn import TagCXN

import tensorflow as tf

configuration = {
    'data': {
        'train_data_path': '...',  # .csv file or json file
        'val_data_path': '...',  # .csv file or json file
        'test_data_path': '...',  # .csv file or json file
        'skip_head': False,
        'img_size': (224, 224, 3),
        'tags': ['...', '...', '...']  # this can also be a path to a .txt file with the labels
    },
    'model': {
        # any instance of tf.keras.Model with its output being the representation of the image (not logits)
        # if a HuggingFace model is selected, data_format must be 'channels_first'
        'backbone': tf.keras.applications.EfficientNetV2B0(weights='imagenet',
                                                           include_top=False, ),
        'preprocessor': tf.keras.applications.efficientnet_v2,  # accompanying preprocessor
        'data_format': 'channels_last'
    },
    'model_parameters': {
        'pooling': 'avg',
        'repr_dropout': 0.2,
        'mlp_hidden_layers': [512, 512],  # two hidden layers
        'mlp_dropout': 0.2,
        'use_sam': False,
    },
    'training_parameters': {
        'loss': {
            'name': 'bce'
        },
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'patience_early_stopping': 3,
        'patience_reduce_lr': 1,
        'checkpoint_path': '...'  # path to save the best model
    },
    'save_results': True,
    'results_path': 'xxxx.csv'  # path to the results file

}

t = TagCXN(
    configuration=configuration
)

t.run()
```

## TO-DOs

## References

[1]  
[2]
[3]
[4]
[5] 
[6]
[7] soft F1 loss: https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d

