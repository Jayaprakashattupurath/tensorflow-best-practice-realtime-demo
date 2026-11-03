import os
import numpy as np
import tensorflow as tf

# -------------------------------
# Best practice 1: reproducibility
# -------------------------------
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.keras")
NORM_PATH = os.path.join(MODEL_DIR, "norm_weights.npz")


def make_synthetic_data(n=5000):
    """
    Demo dataset for churn prediction:
    features = [monthly_spend, support_tickets, contract_months, logins_per_week]
    label: churn(1/0)
    """
    spend = np.random.normal(loc=60, scale=20, size=n).clip(5, 200)
    tickets = np.random.poisson(lam=2.2, size=n).clip(0, 15)
    contract = np.random.randint(1, 36, size=n)
    logins = np.random.normal(loc=5, scale=2, size=n).clip(0, 20)

    x = np.stack([spend, tickets, contract, logins], axis=1).astype("float32")

    # synthetic rule with noise
    risk_score = (
        0.02 * spend
        + 0.25 * tickets
        - 0.06 * contract
        - 0.18 * logins
        + np.random.normal(0, 0.8, size=n)
    )
    y = (risk_score > 1.5).astype("float32")
    return x, y


def build_model():
    # Best practice 2: keep preprocessing in model graph when possible
    norm = tf.keras.layers.Normalization(name="normalize")
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(4,)),
            norm,
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model, norm


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    x, y = make_synthetic_data(6000)
    split = int(len(x) * 0.8)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    # Best practice 3: tf.data pipeline for scalable input
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(1024)
        .batch(64)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64)

    model, norm = build_model()
    norm.adapt(x_train)  # Best practice 4: adapt ONLY on train split

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks, verbose=2)

    # Save normalization weights separately for transparent inspection/debugging
    np.savez(NORM_PATH, mean=norm.mean.numpy(), variance=norm.variance.numpy())
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved normalization stats to {NORM_PATH}")


if __name__ == "__main__":
    main()