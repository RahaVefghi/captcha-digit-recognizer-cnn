import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow
import cv2
from tensorflow.keras.callbacks import EarlyStopping
import glob
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau


def preprocess_image(img):
    """پیش‌پردازش برای کاهش نویز کپچا"""
    # حذف نویز نقطه‌ای/پس‌زمینه
    img = cv2.medianBlur(img, 3)

    # افزایش کنتراست بین رقم و پس‌زمینه
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # حذف نویزهای کوچیک باقیمانده (morphological opening)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return img


def load_data():
    data = []
    labels = []

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "digit-images")
    files = glob.glob(os.path.join(dataset_dir, "*", "*"))
    total = len(files)

    if total == 0:
        print(f"[WARNING] No digit images found in '{dataset_dir}'. Please place digit-images folder here.")
        return None, None, None, None, 0, None

    IMG_SIZE = 64

    for i, item in enumerate(files):
        img = cv2.imread(item)
        if img is None:
            print(f"[WARNING] could not read {item}, skipped")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_image(img)
        img = img / 255.0

        label = item.split("\\")[1]

        data.append(img)
        labels.append(label)

        if i % 100 == 0:
            print(f"[INFO] {i}/{total} processed")

    data = np.array(data)
    data = data.reshape(data.shape[0], IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print("[INFO] classes found:", le.classes_)
    print("[INFO] number of classes:", num_classes)

    import collections
    print("[INFO] class distribution:", collections.Counter(labels))

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels_enc,
        test_size=0.2,
        stratify=labels_enc,
        random_state=42
    )

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, num_classes, le


def training(x_train, y_train, x_test, y_test, num_classes):
    net = models.Sequential([
        layers.Input(shape=(64, 64, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu", kernel_regularizer=l2(0.0005)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    net.summary()

    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    net.compile(optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"])

    datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.08
    )

    from sklearn.utils.class_weight import compute_class_weight
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weights = dict(enumerate(class_weights_arr))
    print("[INFO] class weights:", class_weights)

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(
        "digit_best.keras", monitor="val_accuracy", save_best_only=True, verbose=1
    )

    H = net.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=100,
        validation_data=(x_test, y_test),
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    net.save("digit.keras")
    return H, net


def show_plots(H):
    plt.style.use("ggplot")
    plt.plot(H.history["accuracy"], label="train accuracy")
    plt.plot(H.history["val_accuracy"], label="test accuracy")
    plt.plot(H.history["loss"], label="train loss")
    plt.plot(H.history["val_loss"], label="test loss")
    plt.xlabel("epochs")
    plt.ylabel("accuracy/loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, num_classes, le = load_data()
    if x_train is not None:
        H, net = training(x_train, y_train, x_test, y_test, num_classes)
        show_plots(H)