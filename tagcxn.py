import gc
import json
import os

import numpy as np
from tqdm import tqdm

import refactor_utilities as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf


class TagCXN:

    def __init__(self, train_data_path, val_data_path, test_data_path,
                 img_size, backbone, preprocessor):
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.img_size = img_size
        self.train_data, self.val_data, self.test_data = dict(), dict(), dict()
        self.train_img_index, self.train_concepts_index = dict(), dict()
        self.val_img_index, self.val_concepts_index = dict(), dict()
        self.test_img_index = dict()
        self.tags_list = list()

        self.model = None

    def init_structures(self, tags, skip_head=False):
        if '.csv' in self.train_data_path:
            self.train_data = self.load_csv_data(self.train_data_path, skip_head=skip_head)
        else:
            self.train_data = self.load_json_data(self.train_data_path)
        if '.csv' in self.val_data_path:
            self.val_data = self.load_csv_data(self.val_data_path, skip_head=skip_head)
        else:
            self.val_data = self.load_json_data(self.val_data_path)
        if '.csv' in self.test_data_path:
            self.test_data = self.load_csv_data(self.test_data_path, skip_head=skip_head)
        else:
            self.test_data = self.load_json_data(self.test_data_path)

        print('Number of training examples:', len(self.train_data), 'Number of validation examples:',
              len(self.val_data), 'Number of testing examples:',
              len(self.test_data))

        self.train_img_index, self.train_concepts_index = utils.create_index(self.train_data)
        self.val_img_index, self.val_concepts_index = utils.create_index(self.val_data)
        self.test_img_index, _ = utils.create_index(self.test_data)

        self.tags_list = self.load_tags(tags)

    @staticmethod
    def load_csv_data(file_name, skip_head=False):
        """
        loads .csv file into a Python dictionary.
        :param file_name: the path to the file (string)
        :param skip_head: whether to skip the first row of the file (if there is a header) (boolean)
        :return: data dictionary (dict)
        """
        data = dict()
        with open(file_name, 'r') as f:
            if skip_head:
                next(f)
            for line in f:
                image = line.replace('\n', '').split('\t')
                concepts = image[1].split(';')
                if image[0]:
                    data[str(image[0] + '.jpg')] = ';'.join(concepts)
        print('Data loaded from:', file_name)
        return data

    @staticmethod
    def load_json_data(file_name):
        """
        loads the data of JSON format into a Python dictionary
        :param file_name: the path to the file (string)
        :return: data dictionary (dict)
        """
        print('Data loaded from:', file_name)
        return json.load(open(file=file_name, mode='r'))

    @staticmethod
    def load_tags(tags):
        """
        loads the tags list
        :param tags: the tags as a list or as a text file
        :return: the tags list
        """
        if not isinstance(tags, list):
            return [line.strip() for line in open(tags, 'r')]
        return tags

    def build_model(self, pooling, repr_dropout=0., mlp_hidden_layers=None,
                    mlp_dropout=0., use_sam=False, batch_size=None):
        """
        builds the Keras model
        :param pooling: global pooling method (string)
        :param repr_dropout: whether to apply dropout to the encoder's representation (rate != 0) (float)
        :param mlp_hidden_layers: a list containing the
        number of units of the MLP head. Leave None for plain linear (list)
        :param mlp_dropout: whether to apply dropout to the MLPs layers (rate != 0) (float)
        :param use_sam: whether to use SAM optimization (boolean)
        :param batch_size: the batch size of training (int)
        :return: Keras model
        """
        inp = tf.keras.layers.Input(shape=self.img_size, name='input')
        x = self.backbone(self.backbone.input, training=False)
        encoder = tf.keras.Model(inputs=self.backbone.input, outputs=x, name='backbone')
        z = encoder(inp)
        if pooling == 'avg':
            z = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(z)
        elif pooling == 'max':
            z = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(z)
        else:
            z = utils.GeM(name='gem_pool')(z)

        if repr_dropout != 0.:
            z = tf.keras.layers.Dropout(rate=repr_dropout, name='repr_dropout')
        for i, units in enumerate(mlp_hidden_layers):
            z = tf.keras.layers.Dense(units=units, activation='relu', name=f'MLP-layer-{i}')(z)
            if mlp_dropout != 0.:
                z = tf.keras.layers.Dropout(rate=mlp_dropout, name=f'MLP-dropout-{i}')(z)

        z = tf.keras.layers.Dense(units=len(self.tags_list), activation='sigmoid', name='LR')(z)
        model = tf.keras.Model(inputs=inp, outputs=z, name='TagCXN')
        if use_sam:
            assert batch_size // 4 == 0  # this must be divided exactly due to tf.split in the implementation of SAM.
            model = tf.keras.models.experimental.SharpnessAwareMinimization(
                model=model, num_batch_splits=(batch_size // 4), name='TagCXN_w_SAM'
            )
        return model

    def train(self, train_parameters):
        """
        method that trains the model
        :param train_parameters: model and training hyper-parameters
        :return: a Keras history object
        """
        batch_size = train_parameters.get('batch_size')
        self.model = self.build_model(pooling=train_parameters.get('pooling'),
                                      repr_dropout=train_parameters.get('repr_dropout'),
                                      mlp_hidden_layers=train_parameters.get('mlp_hidden_layers'),
                                      mlp_dropout=train_parameters.get('mlp_dropout'),
                                      use_sam=train_parameters.get('use_sam'), batch_size=batch_size)
        # loss = None
        if train_parameters.get('loss', {}).get('name') == 'bce':
            loss = tf.keras.losses.BinaryCrossentropy()
        elif train_parameters.get('loss', {}).get('name') == 'focal':
            loss = tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=True, alpha=train_parameters.get('loss', {}).get('focal_alpha'),
                gamma=train_parameters.get('loss', {}).get('focal_gamma')
            )
        elif train_parameters.get('loss', {}).get('name') == 'asl':
            loss = utils.AsymmetricLoss(
                gamma_neg=train_parameters.get('loss', {}).get('asl_gamma_neg'),
                gamma_pos=train_parameters.get('loss', {}).get('asl_gamma_pos'),
                clip=train_parameters.get('loss', {}).get('asl_clip')
            )
        else:
            loss = utils.loss_1mf1_by_bce

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=train_parameters.get('learning_rate')),
            loss=loss
        )

        early_stopping = utils.ReturnBestEarlyStopping(monitor='val_loss',
                                                       mode='min',
                                                       patience=train_parameters.get('patience_early_stopping'),
                                                       restore_best_weights=True, verbose=1)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1,
                                                         patience=train_parameters.get('patience_reduce_lr'))

        print('\nTraining...')
        history = self.model.fit(
            self.train_generator(list(self.train_img_index), batch_size, self.tags_list),
            steps_per_epoch=np.ceil(len(self.train_img_index) / batch_size),
            validation_data=self.val_generator(list(self.val_img_index), batch_size, self.tags_list),
            validation_steps=np.ceil(len(self.val_img_index) / batch_size),
            callbacks=[early_stopping, reduce_lr], verbose=1, epochs=train_parameters['epochs']
        )
        print('\nEnd of training...')

        if train_parameters.get('checkpoint_path') is not None:
            self.model.save(train_parameters.get('checkpoint_path'))

        gc.collect()

        return history

    def train_generator(self, ids, batch_size, train_tags):
        """
        generator for training data
        :param ids: indices for each training sample in a batch (list)
        :param batch_size: batch size (int)
        :param train_tags: list of tags
        :return: yields a batch of data
        """
        batch = list()
        while True:
            np.random.shuffle(ids)
            for i in ids:
                batch.append(i)
                if i != len(ids):  # if not in the end of the list
                    if len(batch) == batch_size:
                        yield utils.load_batch(batch, self.train_img_index, self.train_concepts_index,
                                               self.train_data_path, train_tags,
                                               self.preprocessor, size=self.img_size)

                        batch *= 0
                else:
                    yield utils.load_batch(batch, self.train_img_index, self.train_concepts_index,
                                           self.train_data_path, train_tags, self.preprocessor, size=self.img_size)
                    batch *= 0

    def val_generator(self, ids, batch_size, train_tags):
        """
        generator for validation data
        :param ids: indices for each validation sample in a batch (list)
        :param batch_size: batch size (int)
        :param train_tags: list of tags
        :return: yields a batch of data
        """
        batch = list()
        while True:
            np.random.shuffle(ids)
            for i in ids:
                batch.append(i)
                if i != len(ids):
                    if len(batch) == batch_size:
                        yield utils.load_batch(batch, self.val_img_index,
                                               self.val_concepts_index, self.val_data_path,
                                               train_tags, self.preprocessor, size=self.img_size, )
                        batch *= 0
                else:
                    yield utils.load_batch(batch, self.val_img_index, self.val_concepts_index,
                                           self.val_data_path, train_tags, self.preprocessor, size=self.img_size, )
                    batch *= 0

    def test_generator(self, ids, index, batch_size, t='val'):
        """
        generator for testing data
        :param ids: indices for each testing sample in a batch (list)
        :param index: data index (dict)
        :param batch_size: batch size (int)
        :param t: flag for validation or testing (string)
        :return:
        """
        batch = list()
        while True:
            # np.random.shuffle(ids)
            for i in ids:
                batch.append(i)
                if i != len(ids):
                    if len(batch) == batch_size:
                        yield utils.load_test_batch(batch, index, self.val_data_path
                                                    if t == 'val' else self.test_data_path,
                                                    self.preprocessor, size=self.img_size)
                        batch *= 0
                else:
                    yield utils.load_test_batch(batch, index, self.val_data_path
                                                if t == 'val' else self.test_data_path,
                                                self.preprocessor, size=self.img_size)
                    batch *= 0

    def train_tune_test(self, configuration):
        """
        core logic of the file --> train, tune and test
        :param configuration: the dictionary containing the configuration
        :return: a test score float, the test results in a dictionary format and a textual summary
        """
        self.init_structures(tags=configuration['tags'],
                             skip_head=configuration['data']['skip_head'])

        train_parameters = configuration['train_parameters']
        train_parameters.update(configuration['model_parameters'])

        training_history = self.train(train_parameters=train_parameters)

        bs = list(utils.divisor_generator(len(self.val_img_index)))[1]
        val_predictions = self.model.predict(self.test_generator(list(self.val_img_index),
                                                                 self.val_img_index, bs),
                                             verbose=1,
                                             steps=np.ceil(len(self.val_img_index) / bs))
        print(val_predictions.shape)
        best_threshold, val_score = self.tune_threshold(predictions=val_predictions,
                                                        not_bce=False)
        del val_predictions
        bs = list(utils.divisor_generator(len(self.test_img_index)))[1]
        test_predictions = self.model.predict(self.test_generator(list(self.test_img_index),
                                                                  self.test_img_index, bs, t='test'),
                                              verbose=1,
                                              steps=np.ceil(len(self.test_img_index) / bs))
        print(test_predictions.shape)
        test_score, test_results = self.test(best_threshold=best_threshold,
                                             predictions=test_predictions, )
        del test_predictions

        s = ('Development score = ' + str(test_score) +
             ' with threshold = ' + str(best_threshold) + ' and validation score = ' + str(val_score))

        return test_score, test_results, best_threshold, s

    def run(self, configuration):
        """
        basic run method
        :param configuration: a configuration dictionary containing several parameters
        :return: a dictionary of checkpoint paths alongside with scores and thresholds
        """
        thresholds_map = dict()
        test_scores = list()
        info = list()

        test_score, test_results, best_threshold, txt = self.train_tune_test(
            configuration=configuration
        )
        test_scores.append(test_score)
        if configuration['train_parameters']['checkpoint_path'] is not None:
            thresholds_map[configuration['train_parameters']['checkpoint_path']] = [best_threshold, test_score]
        info.append(txt)
        for i in range(len(info)):
            print(info[i])
        s = 'Mean dev score was: ' + str(sum(test_scores) / len(test_scores)) + '\n\n\n'
        print(s)
        info *= 0
        test_scores *= 0
        #
        if configuration.get('save_results'):
            print('\n\nSaving results...\n')
            with open(configuration.get('results_path'), 'w') as out_test:
                for result in test_results:
                    out_test.write(result + '\t' + test_results[result] + '\n')
            print('Results saved!')

        # pickle.dump(thresholds_map, open(str(self.backbone_name) + '_map.pkl', 'wb'))
        # pickle.dump(thresholds_map, open('temp_map.pkl', 'wb'))
        return thresholds_map

    def tune_threshold(self, predictions, not_bce=False):
        """
        method that tunes the classification threshold
        :param predictions: array of validation predictions (NumPy array)
        :param not_bce: flag for not bce losses (boolean)
        :return: best threshold and best validation score
        """
        print('\nGot predictions for validation set of split')
        print('\nTuning threshold for split #{}...')
        # steps = 100
        init_thr = 0.1
        if not_bce:
            init_thr = 0.3
        f1_scores = dict()
        recalls = dict()
        precisions = dict()
        print('Initial threshold:', init_thr)
        for i in tqdm(np.arange(init_thr, 1, .01)):
            threshold = i
            # print('Current checking threshold =', threshold)
            y_pred_val = dict()
            for j in range(len(predictions)):
                predicted_tags = list()

                # indices of elements that are above the threshold.
                for index in np.argwhere(predictions[j] >= threshold).flatten():
                    predicted_tags.append(self.tags_list[index])

                # string with ';' after each tag. Will be split in the f1 calculations.
                # print(len(predicted_tags))
                y_pred_val[list(self.val_data.keys())[j]] = ';'.join(predicted_tags)
                # print(y_pred_val)

            f1_scores[threshold], p, r, _ = utils.evaluate_f1(self.val_data, y_pred_val, test=True)
            recalls[threshold] = r
            precisions[threshold] = p

        # get key with max value.
        best_threshold = max(f1_scores, key=f1_scores.get)
        print('The best F1 score on validation data' +
              ' is ' + str(f1_scores[best_threshold]) +
              ' achieved with threshold = ' + str(best_threshold) + '\n')

        # print('Recall:', recalls[best_threshold], ' Precision:', precisions[best_threshold])
        return best_threshold, f1_scores[best_threshold]

    def test(self, best_threshold, predictions):
        """
        method that performs the evaluation on the test data
        :param best_threshold: the tuned classification threshold (float)
        :param predictions: array of test predictions (NumPy array)
        :return: test score and test results dictionary
        """
        print('\nStarting evaluation on test set...')
        y_pred_test = dict()

        for i in tqdm(range(len(predictions))):
            predicted_tags = list()
            # bt = best_threshold
            for j in range(len(self.tags_list)):
                if predictions[i, j] >= best_threshold:
                    predicted_tags.append(str(self.tags_list[j]))

            # string! --> will be split in the f1 function

            # final_tags = list(set(set(predicted_tags).union(set(most_frequent_tags))))
            # temp = ';'.join(final_tags)
            temp = ';'.join(predicted_tags)
            y_pred_test[list(self.test_data)[i]] = temp

        f1_score, p, r, _ = utils.evaluate_f1(self.test_data, y_pred_test, test=True)
        print('\nThe F1 score on the test set is: {}\n'.format(f1_score))
        # print('Precision score:', p)
        # print('Recall score:\n', r)
        # pickle.dump(y_pred_test, open(f'my_test_results_split_{split}.pkl', 'wb'))
        return f1_score, y_pred_test
