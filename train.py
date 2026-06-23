"""
train.py — train the CNN-BiLSTM-Attention model on the MAUS dataset.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config as cfg
from dataset_builder import build_dataset
from model import build_model


def train(data_dir=cfg.DATA_DIR, task="binary", epochs=cfg.EPOCHS):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = build_dataset(data_dir, task=task)

    n_classes = len(np.unique(np.concatenate([ytr, yva, yte])))
    model = build_model(
        input_len=Xtr.shape[1], n_channels=Xtr.shape[2],
        n_classes=n_classes, task=task,
    )
    model.summary()

    class_weights = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    ckpt_path = os.path.join(cfg.MODEL_DIR, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=epochs,
        batch_size=cfg.BATCH_SIZE,
        class_weight=class_weight_dict if task == "binary" else None,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(os.path.join(cfg.MODEL_DIR, "final_model.keras"))
    np.savez(os.path.join(cfg.OUTPUT_DIR, "test_set.npz"), X=Xte, y=yte)
    return model, history, (Xte, yte)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=cfg.RAW_DATA_DIR)
    p.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    args = p.parse_args()
    train(args.data_dir, args.task, args.epochs)
