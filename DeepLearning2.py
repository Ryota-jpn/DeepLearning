import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers

# CIFAR-10から画像データを取得
from keras.datasets import cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(f"学習データ（問題画像）：{x_train.shape}")
print(f"テストデータ（問題画像）：{x_test.shape}")

y_train, y_test = y_train.flatten(), y_test.flatten()

cat_train = x_train[np.where(y_train==3)]
dog_train = x_train[np.where(y_train==5)]
cat_test = x_test[np.where(y_test==3)]
dog_test = x_test[np.where(y_test==5)]

print("ネコの学習データ：", len(cat_train))
print("イヌの学習データ：", len(dog_train))
print("ネコのテストデータ：", len(cat_test))
print("イヌのテストデータ：", len(dog_test))

def disp_testdata(xdata, namedata):
    plt.figure(figsize=(12,10))
    for i in range(20):
        plt.subplot(4,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(xdata[i])
        plt.xlabel(namedata)
    plt.show()

disp_testdata(cat_train, "ネコ")
disp_testdata(dog_train, "イヌ")

# 取得した学習用画像に猫と犬を設定
class_names = ["ネコ", "イヌ"]

x_train = np.concatenate((cat_train, dog_train))
x_test = np.concatenate((cat_test, dog_test))

y_train = np.concatenate((np.full(5000, 0), np.full(5000, 1)))
y_test = np.concatenate((np.full(1000, 0), np.full(1000, 1)))

np.random.seed(1)
np.random.shuffle(x_test)
np.random.seed(1)
np.random.shuffle(y_test)

plt.figure(figsize=(12,10))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])
    plt.xlabel(class_names[y_test[i]])
plt.show()

# 学習画像を基に学習モデルを作成
model = keras.models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (5, 5), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(2, activation="softmax")) #2
model.summary(line_length=120)

# 作成した学習モデルを基に学習させる
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=30, #30
                    validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"テストデータの正解率は{test_acc:.2%}です。")

# 学習結果を確認
param = [["正解率", "accuracy", "val_accuracy"],
         ["誤差", "loss", "val_loss"]]
plt.figure(figsize=(10,4))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(param[i][0])
    plt.plot(history.history[param[i][1]], "o-")
    plt.plot(history.history[param[i][2]], "o-")
    plt.xlabel("学習回数")
    plt.legend(["訓練", "テスト"], loc="best")
    if i == 0:
        plt.ylim([0, 1])
plt.show()

# 学習用画像を増幅させる
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range = 30,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
)
g = datagen.flow(x_test, y_test, shuffle = False)
g_imgs1 = []
x_g, y_g = g.next()
g_imgs1.extend(x_g)

g = datagen.flow(x_test, y_test, shuffle = False)
g_imgs2 = []
x_g, y_g = g.next()
g_imgs2.extend(x_g)

plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(3, 6, i+1)
    plt.imshow(x_test[i], cmap = "Greys")
    plt.title(class_names[y_g[i]])

for i in range(6):
    plt.subplot(3, 6, i+7)
    plt.imshow(g_imgs1[i])

for i in range(6):
    plt.subplot(3, 6, i+13)
    plt.imshow(g_imgs2[i])

plt.show()

# 増幅データで再度学習させる
history = model.fit(datagen.flow(x_train, y_train), epochs = 30,
                    validation_data = (x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"テストデータの正解率は{test_acc:.2%}です。")

# 学習結果の確認（2回目）
param = [["正解率", "accuracy", "val_accuracy"],
         ["誤差", "loss", "val_loss"]]
plt.figure(figsize=(10,4))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(param[i][0])
    plt.plot(history.history[param[i][1]], "o-")
    plt.plot(history.history[param[i][2]], "o-")
    plt.xlabel("学習回数")
    plt.legend(["訓練", "テスト"], loc = "best")
    if i == 0:
        plt.ylim([0,1])
plt.show()

# 識別用画像を渡して識別結果を確認
pre = model.predict(x_test)

plt.figure(figsize=(12,10))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])
    index = np.argmax(pre[i])
    pct = pre[i][index]
    ans = ""
    if index != y_test[i]:
        ans = "x--o[" + class_names[y_test[i]] + "]"
    lbl = f"{class_names[index]}（{pct:.0%}）{ans}"
    plt.xlabel(lbl)
plt.show()