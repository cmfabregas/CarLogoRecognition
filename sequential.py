from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D
from keras import optimizers

model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

model2 = Convolution1D(
    init = 'uniform',
    activation='linear',
    weights=None,
    border_mode='valid',
    subsample_length=1,
    W_regularizer=None, b_regularizer=None,
    W_constraint=None, b_constraint=None,
    input_dim=None, input_length=None
)