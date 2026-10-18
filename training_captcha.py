import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import glob
import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_image(img):
    """Preprocess captcha digit crop: median blur, adaptive thresholding, and morphological opening."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise background specks
    img = cv2.medianBlur(img, 3)

    # Adaptive binarization to enhance digit strokes against variable backgrounds
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Remove residual small salt noise
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return img


def load_data(max_samples_per_class=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "digit-images")

    # Fallback to shared course folder if not present locally
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        fallback = os.path.abspath(os.path.join(base_dir, "..", "iris dataset", "digit-images"))
        if os.path.exists(fallback):
            dataset_dir = fallback

    if not os.path.exists(dataset_dir):
        print(f"[WARNING] No digit images found at '{dataset_dir}'.")
        print("Please place 'digit-images' folder with class subdirectories (1-9) to train.")
        return None, None, None, None, 0, None

    files = glob.glob(os.path.join(dataset_dir, "*", "*"))
    total = len(files)
    if total == 0:
        print(f"[WARNING] No image files found in '{dataset_dir}'.")
        return None, None, None, None, 0, None

    print(f"[INFO] Found {total} images in '{dataset_dir}'")
    IMG_SIZE = 64

    data = []
    labels = []
    class_counts = {}

    for i, item in enumerate(files):
        # Extract class label reliably from folder name regardless of OS path separators
        label = os.path.basename(os.path.dirname(item))

        if max_samples_per_class is not None:
            if class_counts.get(label, 0) >= max_samples_per_class:
                continue

        img = cv2.imread(item)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_image(img)
        img = img / 255.0

        data.append(img)
        labels.append(label)
        class_counts[label] = class_counts.get(label, 0) + 1

        if (i + 1) % 1000 == 0:
            print(f"[INFO] {i + 1}/{total} images loaded")

    data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    num_classes = len(le.classes_)

    print(f"[INFO] Classes detected: {list(le.classes_)} ({num_classes} total)")
    print(f"[INFO] Sample distribution: {dict(collections.Counter(labels))}")

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels_enc,
        test_size=0.2,
        stratify=labels_enc,
        random_state=42
    )

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, num_classes, le


def build_model(num_classes):
    """Builds a regularized CNN with intermediate dropouts to prevent overfitting."""
    net = models.Sequential([
        layers.Input(shape=(64, 64, 1)),

        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_regularizer=l2(0.0005)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])
    return net


def training(x_train, y_train, x_test, y_test, num_classes, epochs=30):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    best_model_path = os.path.join(base_dir, "digit_best.keras")
    final_model_path = os.path.join(base_dir, "digit.keras")

    net = build_model(num_classes)
    net.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    net.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        shear_range=0.06
    )

    y_train_labels = np.argmax(y_train, axis=1)
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weights = dict(enumerate(class_weights_arr))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    H = net.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=epochs,
        validation_data=(x_test, y_test),
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    net.save(final_model_path)
    print(f"[INFO] Best model saved to: {best_model_path}")
    print(f"[INFO] Final model saved to: {final_model_path}")
    return H, net


def show_plots(H):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(base_dir, "training_plot.png")

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(H.history["accuracy"], label="Train Acc")
    plt.plot(H.history["val_accuracy"], label="Val Acc")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(H.history["loss"], label="Train Loss")
    plt.plot(H.history["val_loss"], label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[INFO] Training curves plot saved to: {plot_path}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, num_classes, le = load_data()
    if x_train is not None:
        H, net = training(x_train, y_train, x_test, y_test, num_classes, epochs=30)
        show_plots(H)