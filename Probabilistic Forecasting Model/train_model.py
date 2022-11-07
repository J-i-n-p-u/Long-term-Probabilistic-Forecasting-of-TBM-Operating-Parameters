# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, \
    TimeDistributed, Lambda,BatchNormalization, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import numpy as np
import os
import math
from layers import GaussianLayer
from numpy.random import normal
from tqdm import tqdm
import pickle
from tensorflow.python.framework.ops import disable_eager_execution

#tf.compat.v1.disable_eager_execution()

def gaussian_likelihood(sigma):
    """Likelihood as per the paper."""

    def gaussian_loss(y_true, y_pred):
        """Updated from paper.
    
        See DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.
        """
        return tf.reduce_mean(
        tf.math.log(tf.math.sqrt(2 * math.pi))
        + tf.math.log(sigma)
        + tf.math.truediv(
            tf.math.square(y_true - y_pred), 2 * tf.math.square(sigma)
        )
        )

    return gaussian_loss



def build_deepar_model(model_paras):

    input_step = model_paras['input_step']
    input_dim = model_paras['input_dim']
    output_step = model_paras['output_step']
    output_dim = model_paras['output_dim']

    n_hidden = model_paras['n_hidden']
    model_name = model_paras['model_name']

    input_train = Input(shape=(input_step , input_dim))
    output_train = Input(shape=(output_step, output_dim))

    encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(
        n_hidden,
        # activation='elu', dropout=0.2, recurrent_dropout=0.2,
        return_sequences=False, return_state=True)(input_train)

    encoder_last_h1 = BatchNormalization(momentum=0.6)(encoder_last_h1)
    encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)


    decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
    decoder = LSTM(n_hidden,
                   # activation='elu',
                   dropout=0.2,
                   # recurrent_dropout=0.2,
                   return_state=False, return_sequences=True)(
                       decoder, initial_state=[encoder_last_h1, encoder_last_c])

    # print('decoder',type(decoder))
    loc, scale = GaussianLayer(output_train.shape[2], name="main_output")(decoder)

    model = Model(inputs=input_train, outputs=loc)

    adam = optimizers.Adam(lr=0.001)
    model.compile(
                    loss=gaussian_likelihood(scale),
                  optimizer=adam,
                  metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

    print(model.summary())

    if not os.path.exists('model_fig'):
        os.mkdir('model_fig')

    plot_model(
        model,
        to_file = f"model_fig/{model_name}.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
    )
    return model

def build_base_model(model_paras):

    input_step = model_paras['input_step']
    input_dim = model_paras['input_dim']
    output_step = model_paras['output_step']
    output_dim = model_paras['output_dim']
    n_hidden = model_paras['n_hidden']

    input_train = Input(shape=(input_step , input_dim))
    output_train = Input(shape=(output_step, output_dim))

    encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(
        n_hidden,
        # activation='elu', dropout=0.2, recurrent_dropout=0.2, 
        return_sequences=False, return_state=True)(input_train)

    encoder_last_h1 = BatchNormalization(momentum=0.6)(encoder_last_h1)
    encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

    decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
    decoder = LSTM(n_hidden,
                   # activation='elu', dropout=0.2, recurrent_dropout=0.2,
                   return_state=False, return_sequences=True)(
                       decoder, initial_state=[encoder_last_h1, encoder_last_c])

    out = TimeDistributed(Dense(output_train.shape[2]))(decoder)

    adam = optimizers.Adam(lr=0.01)
    model = Model(inputs=input_train, outputs=out)

    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
    print(model.summary())
    return model


def train_model(model, train, test, model_paras):
    train_x, train_y = train
    test_x, test_y = test
    model_name = model_paras['model_name']
    number_of_epochs = model_paras['number_of_epochs']
    batch_size = model_paras['batch_size']

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    check_pointer = ModelCheckpoint(
            'checkpoints/' + '%s_weights.{epoch:02d}-{val_mean_absolute_percentage_error:.2f}.h5' % model_name,
            monitor='val_mean_absolute_percentage_error',
            mode='min',
            save_best_only=True,
        save_weights_only = True
    )

    if not os.path.exists('tensorboard_logs'):
        os.mkdir('tensorboard_logs')

    tensorboard_logger = TensorBoard(
        log_dir='tensorboard_logs/', histogram_freq=0,
          write_graph=True, write_images=True
    )
    tensorboard_logger.set_model(model)

    if not os.path.exists('training_logs'):
        os.mkdir('training_logs')
    csv_logger = CSVLogger(filename=f'training_logs/{model_name}.csv')
    history = model.fit(
        x=train_x,
        y = train_y,
        validation_data=(test_x, test_y),
            # steps_per_epoch=step_size_train,
        epochs=number_of_epochs,
        batch_size = batch_size,
        callbacks=[
            check_pointer,
            # early_stopping_monitor,
                   csv_logger,
            tensorboard_logger],
#         class_weight =class_weight
    )

    return model


def plot_learning_curves_from_history_file(filename):
    matplotlib.rcParams['figure.figsize'] = (10, 6)
    history = pd.read_csv(filename)
    hv = history.values
    epoch=hv[:,0]
    acc=hv[:,1]
    loss=hv[:,2]
    val_acc=hv[:,3]
    val_loss=hv[:,4]
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(epoch,acc,epoch,val_acc)
    axes[0].set_title('model accuracy')
    axes[0].grid(which="Both")
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'test'], loc='lower right')
    axes[1].plot(epoch,loss,epoch,val_loss)
    axes[1].set_title('model loss')
    axes[1].grid(which="Both")
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['train', 'test'], loc='upper center')
    return fig


def predict(model, test_X):
    layer_name = 'main_output'
    intermediate_layer_model = Model(inputs=model.input,
                                      outputs=model.get_layer(layer_name).output)
    outputs = intermediate_layer_model.predict(test_X)

    # outputs = model.predict(test_X)
    y_pred = []
    for mu_s, sigma_s in tqdm(zip(outputs[0], outputs[1])):
        y_predi = []
        for mu, sigma in zip(mu_s.reshape(-1), sigma_s.reshape(-1)):
            sample = normal(
                loc=mu, scale=np.sqrt(sigma), size=1
            )
            y_predi.append(sample)
        y_predi = np.array(y_predi).reshape((model.output_shape[1], model.output_shape[2]))
        y_pred.append(y_predi)

    y_pred = np.array(y_pred)

    return y_pred


def transform_to_raw(y, norm_paras, flag):
    scale = norm_paras['max_paras'] - norm_paras['min_paras']
    bias = norm_paras['min_paras']
    if flag == 'F':
        y_raw = y * scale[1] + bias[1]
    else:
        y_raw = y * scale[2] + bias[2]
    return y_raw

def evaluate_results(test_y, pred_y, norm_paras, flag):

    test_y = transform_to_raw(test_y, norm_paras, flag)
    print(test_y.shape)
    pred_y = transform_to_raw(pred_y, norm_paras, flag)

    mae = np.mean(np.mean(abs(test_y-pred_y), axis=1), axis=0)

    mape = abs(test_y-pred_y) / test_y
    mape = np.mean(np.mean(mape, axis=1), axis=0)

    print(mae, mape)
    return (mae, mape)


if __name__ == "__main__":
    disable_eager_execution()

    # Load normalized train and test dataset
    open_file = open('train_xy_01_dict_2.pkl', "rb")
    train_xy_dict = pickle.load(open_file)
    open_file = open('test_xy_01_dict_2.pkl', "rb")
    test_xy_dict = pickle.load(open_file)
    # Load normalization parameters
    open_file = open('normal_paras_2.pkl', "rb")
    norm_paras = pickle.load(open_file)
    open_file.close()


    model_paras = dict(
            input_step = train_xy_dict['x'].shape[1],
            input_dim = train_xy_dict['x'].shape[2],
            output_step = train_xy_dict['F'].shape[1],
            output_dim = train_xy_dict['F'].shape[2],
            model_name = None,
            number_of_epochs = 10,
            batch_size = 128,
            n_hidden = 128,
            )


    flag = 'F'
    # flag = 'F'
    tf.compat.v1.experimental.output_all_intermediates(True)
    # model_paras['model_name'] = f'deepar_model_{flag}_40_v4'
    model_paras['model_name'] = f'deepar_model_{flag}_40_TBM2'
    model = build_deepar_model(model_paras)
    # model = build_base_model(model_paras)
    # model.load_weights('./checkpoints/deepar_model_T_40_v3_weights.29-18.69.h5')

    model = train_model(model, (train_xy_dict['x'], train_xy_dict[flag]), \
                        (test_xy_dict['x'], test_xy_dict[flag]), model_paras)

    # evaluate result
    # model = build_deepar_model(model_paras)
    # model.load_weights('./checkpoints/deepar_model_T_40_v2_weights.20-19.27.h5')
    # y_pred = model.predict(test_xy_dict['x'])
    # open_file = open(f"{cycle_type}_test_pred_{flag}.pkl", "wb")
    # pickle.dump(y_pred_raw, open_file)
    # open_file.close()
    # evaluate_results(test_xy_dict['F'], y_pred, norm_paras, flag)
    # layer_name = 'main_output'
    # intermediate_layer_model = Model(inputs=model.input,
    #                                   outputs=model.get_layer(layer_name).output)
    # outputs = intermediate_layer_model.predict(test_X)

















