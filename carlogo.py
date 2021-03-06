from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications

#flickr = flickrapi.FlickrAPI('0a0e20275249094ff865b34e5fce9722','f9b02cabe43e291d',format='parsed-json')

#weights_path = 'vgg16_weights.h5'
#top_model_weights_path = 'first_try.h5'

img_width, img_height = 250, 250 #img demensions

train_data_dir ='data/train'    #Data and where it will be trained
validation_data_dir = 'data/validation'
nb_train_samples = 16
nb_validation_samples = 20
epochs = 50
batch_size = 16

#model = applications.VGG16(weights='imagenet', include_top=False)
#print('Model loaded.')

if K.image == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(250, 250, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    follow_links='True')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    follow_links='True')

model.fit_generator(
    train_generator,
    steps_per_epoch= nb_train_samples // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps= nb_validation_samples // batch_size)

