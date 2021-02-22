"""Module for training Neural Networks."""
import pandas as pd
from typing import Callable, Dict, Tuple
from tensorflow import keras
from tensorflow.keras.metrics import Recall
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px


class ICODeepTraining:
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        target_array: pd.Series, 
        dl_model: keras.Model, 
        ann_type: str, 
        size_array: int
    ):
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataset with the features to use for training.

        target_array : pd.Series
            Array with the targets for each sample in training dataset.

        dl_model: keras.Model
            NeuralNetwork model to use for training defined using Keras.

        ann_type : str
            Type of neural network architecture ('cnn', 'lstm' or None).

        size_array : int
            Size of each time series sample.

        Attributes
        ----------
        dataframe : pd.DataFrame
            Stored dataframe parameter.

        target_array : pd.Series
            Stored target_array parameter.
        
        X_train : pd.DataFrame
            Splitted dataframe attribute for training.

        y_train : pd.Series
            Splitted target_array attribute for training.

        X_validation : pd.DataFrame
            Splitted dataframe attribute for validation.

        y_validation : pd.Series
            Splitted target_array attribute for validation.

        dl_model : keras.Model
            Stored dl_model parameter used for training.
        
        ann_type : str
            Stored parameter ann_type.

        size_array : int
            Stored parameter size_array.

        history : keras.
            Training metadata used for evaluating model.

        df_validation_predictions : pd.DataFrame
            Dataframe with the predictions from the trained model.

        """
        self.dataframe = dataframe
        self.target_encoded = target_array
        self.X_train = None
        self.y_train = None
        self.X_validation = None
        self.y_validation = None
        self.dl_model = dl_model
        self.ann_type = ann_type
        self.size_array = size_array
        self.history = None
        self.df_validation_predictions = None

    def split_train_test(self, test_size=0.3):
        """Split dataset into train and test.

        Parameters
        ----------
        test_size : float (default=0.3)
            Proportion of samples to be assigned to test.

        """
        (
            self.X_train,
            self.X_validation,
            self.y_train,
            self.y_validation,
        ) = train_test_split(
            self.dataframe,
            self.target_encoded,
            test_size=test_size,
            random_state=161,
        )

        self.X_train = self.X_train.values.astype(float)
        self.X_validation = self.X_validation.values.astype(float)

        if self.ann_type in ('cnn', 'lstm'):
            self.X_train = self.X_train.reshape(
                (len(self.X_train), self.X_train.shape[1], 1)
            )
            self.X_validation = self.X_validation.reshape(
                (len(self.X_validation), self.X_validation.shape[1], 1)
            )

    def model_summary(self):
        """Summarize model metadata."""
        self.dl_model.summary()

    def train_network(
        self,
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
        callback=keras.callbacks.EarlyStopping(monitor='loss', patience=5),
        epochs=50,
        verbose=1,
        batch_size=32,
    ):
        """Perform training for neural network model.

        Parameters
        ----------
        loss : str (default)=binary_crossentropy')
            Loss function to use for training.

        optimizer : str (default='adam')
            The optimizing algorithm to use for training.

        metrics : List[str] (default=['accuracy'])
            Performance metrics to evaluate model.

        callback : keras.callbacks.EarlyStopping (default=EarlyStopping(
        monitor='loss', patience=5)
            Condition to stop training.

        epochs : int (default=50)
            Number of epochs to perform training.

        verbose : int (default=1)
            Wheter to print information when training.

        batch_size : int (default=32)
            Number of samples to use in each epoch for training.

        """
        self.dl_model.reset_states()
        self.dl_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.history = self.dl_model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            validation_data=(self.X_validation, self.y_validation),
            verbose=verbose,
            batch_size=batch_size,
            callbacks=[callback],
        )

    def plot_training(self, figsize: Tuple[int, int]=(1200, 800)):
        """Plot training results.

        Parameters
        ----------
        figsize : Tuple[int, int] (default=(1200, 800))
            Width and height of the plot in pixels.
        """

        df_training_metrics = pd.DataFrame(self.history.history)
        df_training_plotly = (
            pd.DataFrame(df_training_metrics.stack())
            .reset_index()
            .sort_values(by=['level_1', 'level_0'])
            .rename(
                columns={'level_0': 'epochs', 'level_1': 'metric', 0: 'values'}
            )
        )
        fig = px.line(
            df_training_plotly,
            x="epochs",
            y="values",
            color="metric",
            line_group="metric",
            hover_name="metric",
        )
        fig.show()

    def get_validation_predictions(self):
        """Define the outcome for the test dataframe using the trained model.
        """
        df_predictions = pd.DataFrame(self.y_validation)
        df_predictions['predictions'] = self.dl_model.predict(self.X_validation)[:, -1].round().astype(int)
        self.df_validation_predictions = df_predictions