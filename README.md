# (Medical) Image Tagger

This repository provides a basic model for (medical & imbalanced) multi-label image classification. 

--- 

<p align="center">
<img src="image.png" width=100% height=100% 
class="center">
</p>

## Example of Use

All the parameters are specified by a single Python dictionary that can be passed to the constructor of the class.


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
