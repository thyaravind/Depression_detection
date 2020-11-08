# %%

from keras.layers import ConvLSTM2D, Bidirectional, BatchNormalization, Conv3D, Cropping3D, ZeroPadding3D
from keras.models import Sequential
import pickle

#%% load spectrogram numpy arrays




# %%


for i in range(0,1500000,10000):



#%%

def ConvLSTM_Model(frames, channels, pixels_x, pixels_y, categories):
    trailer_input = Input(shape=(frames, channels, pixels_x, pixels_y)
                          , name='trailer_input')

    first_ConvLSTM = ConvLSTM2D(filters=20, kernel_size=(3, 3)
                                , data_format='channels_first'
                                , recurrent_activation='hard_sigmoid'
                                , activation='tanh'
                                , padding='same', return_sequences=True)(trailer_input)
    first_BatchNormalization = BatchNormalization()(first_ConvLSTM)
    first_Pooling = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first')(
        first_BatchNormalization)

    second_ConvLSTM = ConvLSTM2D(filters=10, kernel_size=(3, 3)
                                 , data_format='channels_first'
                                 , padding='same', return_sequences=True)(first_Pooling)
    second_BatchNormalization = BatchNormalization()(second_ConvLSTM)
    second_Pooling = MaxPooling3D(pool_size=(1, 3, 3), padding='same', data_format='channels_first')(
        second_BatchNormalization)

    outputs = [branch(second_Pooling, 'cat_{}'.format(category)) for category in categories]

    seq = Model(inputs=trailer_input, outputs=outputs, name='Model ')

    return seq


def branch(last_convlstm_layer, name):
    branch_ConvLSTM = ConvLSTM2D(filters=5, kernel_size=(3, 3)
                                 , data_format='channels_first'
                                 , stateful=False
                                 , kernel_initializer='random_uniform'
                                 , padding='same', return_sequences=True)(last_convlstm_layer)
    branch_Pooling = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first')(branch_ConvLSTM)
    flat_layer = TimeDistributed(Flatten())(branch_Pooling)

    first_Dense = TimeDistributed(Dense(512, ))(flat_layer)
    second_Dense = TimeDistributed(Dense(32, ))(first_Dense)

    target = TimeDistributed(Dense(1), name=name)(second_Dense)

    return target


# %%

model = Sequential()
model.add(BatchNormalization(input_shape=(None, None, None, 1)))
model.add(Conv3D(8,
                 kernel_size=(1, 5, 5),
                 padding='same',
                 activation='relu'))
model.add(Conv3D(8,
                 kernel_size=(3, 3, 3),
                 padding='same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(Bidirectional(ConvLSTM2D(16,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   return_sequences=True)))
model.add(Bidirectional(ConvLSTM2D(32,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   return_sequences=True)))
model.add(Conv3D(8,
                 kernel_size=(1, 3, 3),
                 padding='same',
                 activation='relu'))
model.add(Conv3D(1,
                 kernel_size=(1, 1, 1),
                 activation='sigmoid'))
model.add(Cropping3D((1, 2, 2)))  # avoid skewing boundaries
model.add(ZeroPadding3D((1, 2, 2)))
model.summary()
