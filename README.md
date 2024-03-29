# (Medical) Image Tagger

This repository provides a basic model for (medical & imbalanced) multi-label image classification written in Keras/TensorFlow 2. 

--- 

<p align="center">
<img src="image.png" width=100% height=100% 
class="center">
</p>

---

## Summary 

A basic multi-label image classification model. Uses an encoder and a classification head. The classification threshold is also tuned using the validation data. The metric used for evaluation is the F1 score averaged over the test instances ('samples' averaging) based on the [ImageCLEFmedical 2022 Caption](https://www.imageclef.org/2022/medical/caption). You can change this based on your preferences. 

## Dependencies

- TensorFlow 2
- Keras
- TensorFlow Addons

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

The program will read the `.csv` files into a Python dictionary. If there are headers in your file, you have to use `skip_head=True`. The two columns must be separated by a split token, the default is `\t`. Alternatively, you can load a JSON file `{image1: 'tag1;tag2', ...}` directly as a `dict`. 

Next, the list with the categories (labels) can either be a Python list passed to the `configuration` or a `.txt` file with a label per line.
For the loss function binary cross-entropy, Focal loss[5], ASL loss[6] and soft F1 loss[7] are supported. For Focal and ASL losses, you have to specify the respective parameters of each loss.

## Example of Use

If you have the images stored in a single folder, pass the same folder path to the first 3 arguments.

```python
from tagcxn import TagCXN

import tensorflow as tf

configuration = {
    'data': {
        'train_images_folder': '...',  # path to folder of training images
        'val_images_folder': '...',  # path to folder of validation images
        'test_images_folder': '...',  # path to folder of testing images
        'train_data_path': '...',  # .csv file or json file
        'val_data_path': '...',  # .csv file or json file
        'test_data_path': '...',  # .csv file or json file
        'skip_head': False,
        'img_size': (224, 224, 3),
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

- [ ] Add Union and Intersection ensembles  
- [ ] k-NN model
- [ ] Add code for fine-tuning 

## References

[1] Charalampakos M.Sc. Thesis ["Exploring Deep Learning Methods for Medical Image Tagging", 2022](http://nlp.cs.aueb.gr/theses/f_charalampakos_msc_thesis.pdf)

```bibtex
@unpublished{Charalampakos2022,
  author = "F. Charalampakos",
  title = "Exploring Deep Learning Methods for Medical Image Tagging",
  year = "2022",
  note = "M.Sc. thesis, Department of Informatics, Athens University of Economics and Business}
}
 ``` 
<br />

[2] Charalampakos et al. ["AUEB NLP Group at ImageCLEFmedical Caption 2022" Proceedings of the Conference and Labs of the Evaluation Forum (CLEF 2022), 2022](http://ceur-ws.org/Vol-3180/paper-101.pdf) 

```bibtex
@inproceedings{Charalampakos2022,
  title="{AUEB NLP Group at ImageCLEFmedical Caption 2022}",
  author="F. Charalampakos and G. Zachariadis and J. Pavlopoulos and V. Karatzas and C. Trakas and I. Androutsopoulos",
  booktitle="CLEF2022 Working Notes",
  series = "{CEUR} Workshop Proceedings",
  publisher = "CEUR-WS.org",
  year="2022",
  pages = "1355-1373",
  address="Bologna, Italy"
}
```
<br />

[3] Karatzas et al. ["AUEB NLP Group at ImageCLEFmed Caption 2020" Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2020), 2020](http://nlp.cs.aueb.gr/pubs/AUEB_NLP_Group_at_ImageCLEFmed_Caption_2020.pdf) 

```bibtex
@inproceedings{Karatzas2020,
  title="{AUEB NLP Group at ImageCLEFmed Caption 2020}",
  author="B. Karatzas and V. Kougia and J. Pavlopoulos and I. Androutsopoulos",
  booktitle="CLEF2020 Working Notes",
  series = "{CEUR} Workshop Proceedings",
  publisher = "CEUR-WS.org",
  year="2020",
  address="Thessaloniki, Greece"
}
```
<br />

[4] Kougia et al. ["AUEB NLP Group at ImageCLEFmed Caption 2019" Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2019), 2019](http://nlp.cs.aueb.gr/pubs/paper_136.pdf)

```bibtex
@inproceedings{Kougia2019,
  title="{AUEB NLP Group at ImageCLEFmed Caption 2019}",
  author="V. Kougia and J. Pavlopoulos and I. Androutsopoulos",
  booktitle="CLEF2019 Working Notes",
  series = "{CEUR} Workshop Proceedings",
  publisher = "CEUR-WS.org",
  year="2019",
  address="Lugano, Switzerland"
}
```
<br />

[5] Focal loss: [Lin et. al "Focal Loss for Dense Object Detection" Proceedings of the IEEE International Conference on Computer Vision (ICCV),  2017](https://arxiv.org/pdf/1708.02002.pdf)

```bibtex
@article{Lin2017Focal,
  title={Focal Loss for Dense Object Detection},
  author={T. Lin and P. Goyal and R. B. Girshick and K. He and P. Doll{\'a}r},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  year={2017},
  address = "Venice, Italy",
  pages={2999-3007}
}
```
<br />

[6] ASL loss: [Ridnik et. al "Asymmetric Loss For Multi-Label Classification" Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf)

```bibtex
@article{Ridnik2021,
  title={Asymmetric Loss For Multi-Label Classification},
  author={T. Ridnik and E. Ben Baruch and N. Zamir and A. Noy and I. Friedman and M. Protter and L. Zelnik-Manor},
  journal={2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021},
  address="Online",
  pages={82-91}
}
```
<br />

[7] [ASL loss implementation](https://github.com/SmilingWolf/SW-CV-ModelZoo/blob/main/Losses/ASL.py) <br />
[8] [soft F1 loss](https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d) <br />

[9] SAM: [Foret et al. "Sharpness-aware Minimization for Efficiently Improving Generalization" ICLR 2021](https://iclr.cc/virtual/2021/poster/2782) 

```bibtex
@inproceedings{Foret21,
title={Sharpness-aware Minimization for Efficiently Improving Generalization},
author={P. Foret and A. Kleiner and H. Mobahi and B. Neyshabur},
booktitle={International Conference on Learning Representations},
year={2021},
address = {Online}
}
```
<br />

