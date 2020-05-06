# organize dataset into a useful structure
import pandas as pd
import sys
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD

# dataset_home = 'dataset_dogs_vs_cats/'
# subdirs = ['train/', 'test/']
# for subdir in subdirs:
# 	labeldirs = ['dogs/', 'cats/']
# 	for labldir in labeldirs:
# 		newdir = dataset_home + subdir + labldir
# 		makedirs(newdir, exist_ok=True)
# seed(1)
# val_ratio = 0.25
# src_directory = 'train/'
# for file in listdir(src_directory):
# 	src = src_directory + '/' + file
# 	dst_dir = 'train/'
# 	if random() < val_ratio:
# 		dst_dir = 'test/'
# 	if file.startswith('cat'):
# 		dst = dataset_home + dst_dir + 'cats/'  + file
# 		copyfile(src, dst)
# 	elif file.startswith('dog'):
# 		dst = dataset_home + dst_dir + 'dogs/'  + file
# 		copyfile(src, dst)

def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model


model = define_model()
datagen = ImageDataGenerator(rescale=1.0/255.0)

train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
	class_mode='binary', batch_size=64, target_size=(200, 200))
test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
	class_mode='binary', batch_size=64, target_size=(200, 200))

history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
	validation_data=test_it, validation_steps=len(test_it), epochs=30, verbose=2)

acc = model.evaluate_generator(test_it, steps = len(test_it), verbose = 2)
print("Accuracy is: ", acc)

print("Saving the model")
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')

test_path = 'test1/'
valgen = ImageDataGenerator(rescale = 1.0/255.0)
predict_it = valgen.flow_from_directory('test/', class_mode= None, batch_size = 64, shuffle = False, target_size=(200, 200))

predictions = model.predict_generator(predict_it, steps = len(predict_it), verbose = 2)

print("Printing History")
print("=========================================")
print(history.history)
print("=========================================")

print("Printing raw predictions")
print("========================================")
print(predictions)
print("========================================")

# Mapping the results with the image
print("Saving the data to the csv file")
counter = range(1, len([predictions]) + 1)
solution = pd.DataFrame({"id": counter, "label":list(predictions)})
cols = ['label']
for col in cols:
    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)
solution.to_csv("dogsVScats.csv", index = False)
print("END")

def summarize_diagnostics():
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color = "blue", label = "train")
	pyplot.plot(history.history['val_loss'], color = "orange", label = "test")
	pyplot.subplot(212)
	pyplot.title("Accuracy")
	pyplot.plot(history.history['accuracy'], color = 'green', label = "train")
	pyplot.plot(history.history['val_accuracy'], color = 'red', label = "test")
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

summarize_diagnostics()

