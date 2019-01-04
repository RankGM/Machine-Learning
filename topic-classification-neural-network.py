import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer


#========TREAT INPUT=======
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)
word_index = reuters.get_word_index()
num_classes = max(y_train)+1
tokenizer = Tokenizer(num_words = 10000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#========INSPECTION=======
#print(x_train[0])
#print(y_train[0])


#========MODEL=======
batch_size = 32
epochs = 5

model = Sequential()
model.add(Dense(512, input_shape=(10000,)))
model.add(Activation('relu'))
model.add(Dropout(0.9))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])


history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)
score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose = 1)

print(f'Test loss:{score[0]}')
print(f'Test accuracy:{score[1]}')