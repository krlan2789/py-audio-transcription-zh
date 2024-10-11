import numpy as np
import tensorflow as tf
from termcolor import cprint

# Convert model weights to .bin
model = tf.keras.models.load_model("./audio_transcription_zh_base.h5")
weights = model.get_weights()

# Save weights as .bin file
with open("./audio_transcription_zh_base.bin", "wb") as f:
    for w in weights:
        f.write(w.tobytes())

cprint("Model saved as audio_transcription_zh_base.bin successfully!", "green")
