import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from keras.layers import Dense, Flatten
from keras.datasets import mnist #база данных рукописных изображений

# загрузка выборки в тестовые и обучаемые выборки
# x_train - изображения цифр обучающей выборки
# y_train - вектор соответствующих значений цифры

# x_test - изображения цифр тестовой выборки
# y_test - вектор соответствующих значений цифры для тестовой выборки
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
# по итогу значения находятся в диапазоне от 0 до 1
x_train = x_train/255
x_test = x_test/255

# формат выходных значений в векторы по категориям
y_train_cat = keras.utils.np_utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.np_utils.to_categorical(y_test, 10)

# отображение первых 25 изображений из обучающей выборки
# plt.figure(figsize=(10, 5))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#
# plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)), # входной слой 28х28 пикселей изображение
    Dense(128, activation='relu'), # скрытый слой, из 128 нейронов, функция активации ReLu
    Dense(10, activation='softmax')# выходной слой, 10 нейронов
])

print(model.summary()) # вывод структуры НС в консоль

# Компиляция НС с оптимизацией по ADAM и критеорием - категориальной кросс-энтропией
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Запуск процесса обучения: 80% - обучающая выборка, 20% - выборка валидации
# на вход подаются: трен.выборка входа, выборка выхода, размер батча, колво эпох, разбиение обучающей выборки
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

model.evaluate(x_test, y_test_cat)

# Проверка распознавания цифр
n = 0
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res) # выводит список из 10 элементов от 0 до 9, определяет, какому элементу максимально относится
print("Распознанная цифра: ", np.argmax(res))

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Распознавание тестовой выборки
pred = model.predict(x_test) # прогон тестовой выборки, получим значение на выходе
pred = np.argmax(pred, axis=1) # выделение максимального значения

print(pred.shape)

print(pred[:20]) # вывод того, что предсказала сеть, первые 20
print(y_test[:20]) # вывод того, что должно быть, первые 20

# Выделение неверных результатов
mask = pred == y_test #формирование маски с помощью булевого условия, где равны = true, где нет = false
print(mask[:10])
x_false = x_test[~mask] # выделяем только неверные результаты распозанвания
p_false = pred[~mask]

print(x_false.shape) # вывод количества изображений из 10000, которые были распознаны неверно

# вывод первых неправильно распознанных значений
for i in range(5):
    print("Значение сети: " + str(p_false[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()


