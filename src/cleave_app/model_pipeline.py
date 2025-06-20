import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomBrightness, RandomZoom, GaussianNoise, RandomContrast
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D




class CustomModel:
    '''
    Class is used to define custom model using pre-trained MobileNetV2 model.
    '''
    def __init__(self, train_ds, test_ds):
      self.train_ds = train_ds
      self.test_ds = test_ds

    def build_pretrained_model(self, image_shape, param_shape, unfreeze_from =None):
      '''
      Utilize pretrained CNN to supplement small dataset

      Parameters:
      ------------------------------------------------

      image_shape: tuple
        - dimensions of image
      param_shape: tuple
        - dimension of features

      Returns: tf.keras.Model
        - returns model to train
      '''
      pre_trained_model = MobileNetV2(input_shape=image_shape, include_top=False, weights="imagenet")
      pre_trained_model.trainable =  unfreeze_from is not None
      if unfreeze_from is not None:
        for layer in pre_trained_model.layers[:unfreeze_from]:
            layer.trainable = False

      # Data augmentation pipeline
      data_augmentation = Sequential([
            #RandomFlip(mode="HORIZONTAL_AND_VERTICAL"), doesn't represent real-world scenario
            RandomRotation(factor=(0.1)),
            RandomBrightness(factor=(0.1)),
            RandomZoom(height_factor=0.1, width_factor=0.1),
            GaussianNoise(stddev=0.01),
            RandomContrast(0.1)
        ])
      # CNN for images
      image_input = Input(shape=image_shape)
      x = data_augmentation(image_input)
      x = pre_trained_model(x)
      x = GlobalAveragePooling2D()(x)
      x = Dropout(0.1, name="dropout")(x)

      # Numerical featuers section
      params_input = Input(shape=param_shape)
      y = Dense(64, name="dense_param1", activation='relu')(params_input)
      y = Dense(24, name="dense_param2", activation='relu')(y)
      y = BatchNormalization()(y)
      

      combined = Concatenate()([x, y])
      z = Dense(48, name="dense_combined", activation='relu')(combined)
      z = BatchNormalization()(z)
      z = Dropout(0.2, name="dropout_combined")(z)
      z = Dense(5, name="output_layer", activation='softmax')(z)

      model = Model(inputs=[image_input, params_input], outputs=z)
      model.summary()
      return model

    def compile_model(self, image_shape, param_shape, learning_rate=0.001, metrics=['accuracy'], unfreeze_from=None,):
      '''
      Compile model after calling build_model function

      Parameters:
      -------------------------------------
      image_shape: tuple
          - dimensions of images
      param_shape: tuple
          - dimensions of parameters
      learning_rate: float
          - learning rate for training model
        metrics: list
          - metrics to monitor during training
          - default: accuracy

      Returns:
      tf.keras.Model
          - Mode to be trained
      '''
      # Adaptive Moment Estimation optimizer
      # Set learning rate and then compile model
      # Loss functions is categorical_crossentropy for multi-class classification
      #model = build_model((image_shape), (param_shape))
      model = self.build_pretrained_model(image_shape, param_shape, unfreeze_from)
      optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
      model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
      return model
    
    def create_checkpoints(self, checkpoint_filepath="/content/drive/MyDrive/checkpoints.keras", monitor="val_accuracy", mode="max", save_best_only=True):
      '''
      Create model checkpoints to avoid losing data while training

      Parameters:
      --------------------------------------

      checkpoint_filepath: str
        - path to save model checkpoints
        - default: /content/drive/MyDrive/checkpoints.keras
      monitor: str
        - metric to monitor during training
        - deafault: val_accuracy
      mode: str
        - max, min, avg
        - method to determine stoppping point of metric
        - default: max
      save_best_only: boolean
        - to determine if only best model shold be saved
        - deafault: True

      Returns: tf.callback.ModelCheckpoint
        - checkpoint to use during training
      '''
      model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        verbose=1
      )
      return model_checkpoint_callback
    
    def reduce_on_plateau(self, patience=3, mode='auto', factor=2.0, monitor='val_accuracy'):
      '''
      Create reduced learning rate callback if monitored value stops improving

      Parameters:
      ------------------------------------------

      patience: int
        - number of epochs before reducing learning rate
        - default: 3
      mode: str
        - min, max, auto, method to monitor
        - default: auto
      factor: float
        - factor by which to reduce learning rate
        - default: 2.0
      monitor: str
        - value to monitor before reducing learning rate
        - default: val_accuracy
      
      Returns: tf.keras.callbacks.ReduceLROnPlateau
        - Reduce Learning rate callback
      '''
      reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        patience=patience,
        mode=mode,
        factor=factor,
        min_lr=1e-7
      )
      return reduce_lr
    
    def create_early_stopping(self, patience=3, mode='max', monitor="val_accuracy"):
      '''
      Create early stopping callback to monitor training success and prevent overfitting.

      Parameters:
      ----------------------------------------

      patience: int
        - number of epochs to stop when monitor plateus
        - default: 3
      mode: str
        - max, min, avg
        - method to track monitor
        - default: max
      monitor: str
        - metric to monitor during training
        - default: val_accuracy
      
      Returns: tf.callbacks.EarlyStopping
        - early stopping callback
      '''
      es_callback = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode = mode,
        restore_best_weights=True,
        verbose=1
      )
      return es_callback

    def create_tensorboard_callback(self, log_dir, histogram_freq=1):
      return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq)
    
    def train_model(self, model, checkpoints=None, epochs=5, initial_epoch=0, early_stopping=None, reduce_lr=None, tensorboard=None, history_file=None, model_file=None):
      '''
      Train model with possible callbacks to prevent overfitting

      Parameters:
      -----------------------------------------

      model: tf.keras.Model
        - model to be trained
      checkpoints: tf.keras.callback.Checkpoints
        - checkpoints to save model
        - default: None
      epochs: int
        - number of training epochs to pass though
        - default: 5
      early_stopping: tf.keras.callback.EarlyStopping
        - early stopping callback to prevent overfitting
        - default: None
      reduce_lr: tf.keras.callbacks.ReduceLROnPlateau
        - reduce learning rate callback
        - default: None
      tensorboard: tf.keras.callbacks.TensorBoard
        - creates access to tensorboard option in google colab
      history_file: str
        - file to save history to
      model_file: str
        - file to save model to

      Returns: tf.keras.Model
        - trained model
      '''
      callbacks = []
      if early_stopping:
        callbacks.append(early_stopping)
      if checkpoints:
        callbacks.append(checkpoints)
      if tensorboard:
        callbacks.append(tensorboard)
      if reduce_lr:
        callbacks.append(reduce_lr) 

      if callbacks:
        history = model.fit(self.train_ds, epochs=epochs, initial_epoch=initial_epoch,
                    validation_data=(self.test_ds), callbacks=callbacks)
      else:
        print("Training without callbacks")
        history = model.fit(self.train_ds, epochs=epochs, initial_epoch=initial_epoch,
                    validation_data=(self.test_ds))
      if history_file:
        df = pd.DataFrame(history.history)
        df.to_csv(f"{history_file}", index=False)
      else:
        print("History not saved")
      if model_file:
        model.save(f'{model_file}')
      else:
        print("Model not saved")
      return history
    
    @staticmethod
    def train_kfold( datasets, image_shape, param_shape, learning_rate, metrics = ['accuracy', 'precision', 'recall'], checkpoints=None, epochs=5, initial_epoch=0, early_stopping=None, history_file=None, model_file=None):
      kfold_histories = []
      k_models = []
      train_datasets = [i[0] for i in datasets]
      test_datasets = [i[1] for i in datasets]

      callbacks=[]

      if early_stopping:
        callbacks.append(early_stopping)
      if checkpoints:
        callbacks.append(checkpoints)

      for fold, (train_ds, test_ds) in enumerate(zip(train_datasets, test_datasets)):
        print(f"\n=== Training fold {fold + 1} ===")

        custom_model = CustomModel(train_ds, test_ds)
        model = custom_model.compile_model(image_shape=image_shape, param_shape=param_shape, learning_rate=learning_rate, metrics=metrics)

        if callbacks:
          history = model.fit(train_ds, epochs=epochs, initial_epoch=initial_epoch,
                    validation_data=(test_ds), callbacks=callbacks)
        else:
          print("Training without callbacks")
          history = model.fit(train_ds, epochs=epochs, initial_epoch=initial_epoch,
                    validation_data=(test_ds))
          
        kfold_histories.append(history)
        k_models.append(model)
          
        if history_file:
          df = pd.DataFrame(history.history)
          df.to_csv(f"{history_file}_fold{fold+1}.csv", index=False)
        else:
          print("History not saved")
        if model_file:
          model.save(f'{model_file}_fold{fold+1}.keras')
        else:
         print("Model not saved")

      return k_models, kfold_histories
    
    @staticmethod
    def get_averages_from_kfold(kfold_histories):
      accuracy = []
      precision = []
      recall = []

      for history in kfold_histories:
        accuracy.append(max(history.history['accuracy']))
        precision.append(max(history.history['precision']))
        recall.append(max(history.history['recall']))

      avg_accuracy = np.mean(accuracy)
      avg_precision = np.mean(precision)
      avg_recall = np.mean(recall)

      print(f"Average Accuracy: {avg_accuracy:.2f}")
      print(f"Average Precision: {avg_precision:.2f}")
      print(f"Average Recall: {avg_recall:.2f}")


    def plot_metric(self, title, metric_1, metric_2, metric_1_label, metric_2_label, x_label, y_label):
      '''
      Plotting function for one metric

      Parameters:
      ----------------------------------------------

      title: str
        - title for plot
      metric_1, metric_2: strs
        - metrics to be plotted vs. each other
      metric_1_label, metric_2_label: strs
        - labels for each metric to plot
      x_label, y_label: strs
        - labels for graph axes
      '''

      plt.title(title)
      plt.plot(metric_1, label=metric_1_label)
      plt.plot(metric_2, label=metric_2_label)
      plt.xlabel(x_label)
      plt.ylabel(y_label)
      plt.legend(loc="lower right")
      plt.show()

class BuildMLPModel(CustomModel):

    def __init__(self, cnn_model_path, train_ds, test_ds):
        super().__init__(train_ds, test_ds)
        self.cnn_model = tf.keras.models.load_model(cnn_model_path)
        self.image_input = self.cnn_model.input[0]
        self.feature_output = self.cnn_model.get_layer('dropout').output
       

    def build_pretrained_model(self, param_shape):
        '''
        Build model

        Returns:
        tf.keras.Model
            - Model to be trained
            
        '''
        # Pre-trained base model
        x = Dense(64, name="first_dense_layer", activation='relu')(self.feature_output)
        x = Dense(32, name="second_dense_layer", activation='relu')(x)
        feature_input = Input(shape=param_shape, name='feature_input')  # Features
        #angle_input = Input(shape=(1,), name='angle_input')  # New input
        y = Dense(16, name="third_dense_layer", activation='relu')(feature_input)
        #y = Dense(16, activation='relu')(angle_input

        combined = Concatenate()([x, y])
        z = Dense(64, activation='relu')(combined)
        output = Dense(1, name='tension_output')(z)
        # Use angle input for 250LA
        regression_model = Model(inputs=[self.image_input, feature_input], outputs=output)
        regression_model.summary()
        return regression_model
    
    def compile_model(self, param_shape, learning_rate=0.001):
      '''
      Compile model after calling build_model function

      Parameters:
      -------------------------------------
      image_shape: tuple
          - dimensions of images
      param_shape: tuple
          - dimensions of parameters
      learning_rate: float
          - learning rate for training model

      Returns:
      tf.keras.Model
          - Mode to be trained
      '''
      # Adaptive Moment Estimation optimizer
      # Set learning rate and then compile model
      # Loss functions is mean squared error for regression
      model = self.build_pretrained_model(param_shape)
      optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
      model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
      return model
    
    def create_early_stopping(self, patience=3, mode='min', monitor="val_mae"):
      '''
      Create early stopping callback to monitor training success and prevent overfitting.

      Parameters:
      ----------------------------------------

      patience: int
        - number of epochs to stop when monitor plateus
        - default: 3
      mode: str
        - max, min, avg
        - method to track monitor
        - default: max
      monitor: str
        - metric to monitor during training
        - default: val_accuracy
      
      Returns: tf.callbacks.EarlyStopping
        - early stopping callback
      '''
      es_callback = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode = mode,
        restore_best_weights=True
      )
      return es_callback
    
    def create_checkpoints(self, checkpoint_filepath="/content/drive/MyDrive/mlp_checkpoints.keras", monitor="val_mae", mode="min", save_best_only=True):
      '''
      Create model checkpoints to avoid losing data while training

      Parameters:
      --------------------------------------

      checkpoint_filepath: str
        - path to save model checkpoints
        - default: /content/drive/MyDrive/checkpoints.keras
      monitor: str
        - metric to monitor during training
        - deafault: val_accuracy
      mode: str
        - max, min, avg
        - method to determine stoppping point of metric
        - default: max
      save_best_only: boolean
        - to determine if only best model shold be saved
        - deafault: True

      Returns: tf.callback.ModelCheckpoint
        - checkpoint to use during training
      '''
      model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        verbose=1
      )
      return model_checkpoint_callback