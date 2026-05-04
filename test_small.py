import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only

import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import MobileNetV3Small


# =========================
# Metrics
# =========================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# =========================
# Model
# =========================
def build_mobilenetv3_unet(input_shape=(224, 224, 1)):
    inputs = layers.Input(shape=input_shape)

    x3 = layers.Concatenate()([inputs, inputs, inputs])
    encoder = MobileNetV3Small(
        input_tensor=x3,
        include_top=False,
        weights="imagenet"
    )

    s1 = encoder.layers[4].output
    s2 = encoder.get_layer("expanded_conv_project_bn").output
    s3 = encoder.get_layer("expanded_conv_2_add").output
    s4 = encoder.get_layer("expanded_conv_5_add").output
    bridge = encoder.output

    def upsample_block(x, skip, filters):
        x = layers.Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding="same"
        )(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        return x

    d1 = upsample_block(bridge, s4, 256)
    d2 = upsample_block(d1, s3, 128)
    d3 = upsample_block(d2, s2, 64)
    d4 = upsample_block(d3, s1, 32)

    x = layers.Conv2DTranspose(
        16, (2, 2), strides=(2, 2), padding="same"
    )(d4)
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)

    model = models.Model(inputs, outputs, name="MobileNetV3_UNET")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[dice_coef, iou_coef]
    )
    return model


# =========================
# Data loading
# =========================
def load_test_data(
    test_dir,
    target_size=(224, 224),
    allowed_prefixes=("data1")
):
    img_dir = os.path.join(test_dir, "images")
    mask_dir = os.path.join(test_dir, "masks")

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError(f"cannot find {img_dir} or {mask_dir}")

    raw_images, raw_masks, filenames = [], [], []
    file_list = sorted(os.listdir(img_dir))

    for filename in file_list:
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        base_name = filename.rsplit(".", 1)[0]
        prefix = base_name.split("_")[0]

        if prefix not in allowed_prefixes:
            continue

        img_path = os.path.join(img_dir, filename)
        mask_filename = f"{base_name}_label.png"
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            print(f"⚠️ 找不到對應 mask，已跳過: {filename}")
            continue

        img = cv2.imdecode(
            np.fromfile(img_path, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.imdecode(
            np.fromfile(mask_path, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None or mask is None:
            print(f"⚠️ 讀取失敗，已跳過: {filename}")
            continue

        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        if mask.shape[:2] != target_size:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        raw_images.append(img)
        raw_masks.append(mask)
        filenames.append(filename)

    return np.array(raw_images), np.array(raw_masks), filenames


def preprocess_for_inference(imgs, masks):
    X = (imgs.astype(np.float32) / 255.0 - 0.5) / 0.5

    # binary mask: 0 or 1
    Y = (masks > 127).astype(np.float32)

    return np.expand_dims(X, axis=-1), np.expand_dims(Y, axis=-1)


def calc_per_image_metrics(y_true, y_pred, smooth=1e-6):
    dice_list = []
    iou_list = []

    for i in range(len(y_true)):
        gt = y_true[i].flatten().astype(np.float32)
        pred = y_pred[i].flatten().astype(np.float32)

        intersection = np.sum(gt * pred)
        gt_sum = np.sum(gt)
        pred_sum = np.sum(pred)

        dice = (2.0 * intersection + smooth) / (gt_sum + pred_sum + smooth)
        iou = (intersection + smooth) / (gt_sum + pred_sum - intersection + smooth)

        dice_list.append(dice)
        iou_list.append(iou)

    return np.array(dice_list), np.array(iou_list)


# =========================
# Main
# =========================
if __name__ == "__main__":
    BASE_DIR = "/mnt/c/Users/USER/NCKU/專題/auto_angle"
    MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_small_unet_last_train237_v456_test1.h5")
    TEST_DIR = BASE_DIR
    PRED_DIR = os.path.join(BASE_DIR, "predictions_output_small_last_train237_v456_test1_noearly")

    os.makedirs(PRED_DIR, exist_ok=True)

    print("Using CPU only.")
    print("reading testing data...")

    raw_imgs, raw_masks, filenames = load_test_data(
        TEST_DIR,
        allowed_prefixes=("data1")
    )

    print(f"total {len(raw_imgs)} images")

    if len(raw_imgs) == 0:
        raise ValueError("沒有讀到任何測試圖片，請檢查 images/masks 路徑和檔名。")

    X_test, Y_test = preprocess_for_inference(raw_imgs, raw_masks)

    print(f"building model and loading weights from {MODEL_PATH} ...")
    model = build_mobilenetv3_unet(input_shape=(224, 224, 1))
    model.load_weights(MODEL_PATH)

    # Warm-up: not included in timing
    print("\nWarming up...")
    _ = model.predict(X_test[:min(2, len(X_test))], verbose=0)

    # Only measure inference time
    print("\nTiming inference...")
    start_time = time.perf_counter()
    predictions = model.predict(X_test, verbose=0)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    per_image_time = total_time / len(X_test)

    pred_binary = (predictions > 0.5).astype(np.float32)

    dice_scores, iou_scores = calc_per_image_metrics(Y_test, pred_binary)
    

    print("\n===== Final Test Result =====")

    print(
        f"Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f} "
        f"(min={np.min(dice_scores):.4f}, max={np.max(dice_scores):.4f})"
    )

    print(
        f"IoU : {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f} "
        f"(min={np.min(iou_scores):.4f}, max={np.max(iou_scores):.4f})"
    )

    print("\nInference Time")
    print(f"Total inference time: {total_time:.4f} sec")
    print(f"Mean time per image : {per_image_time:.6f} sec")
    print(f"FPS                 : {1.0 / per_image_time:.2f}")

    print("\nGenerating predicted masks...")
    for i in range(len(raw_imgs)):
        filename = filenames[i]
        pred_mask = np.squeeze(pred_binary[i]).astype(np.uint8) * 255

        save_path = os.path.join(PRED_DIR, filename)
        cv2.imwrite(save_path, pred_mask)

    print(f"\npredicted masks saved to: {PRED_DIR}")