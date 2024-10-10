# import tensorflow as tf
# import tensorflow_io as tfio
# import numpy as np
# import json


# # Load character mappings
# def load_mappings():
#     with open("char_to_index.json", "r", encoding="utf-8") as f:
#         char_to_index = json.load(f)
#     with open("index_to_char.json", "r", encoding="utf-8") as f:
#         index_to_char = json.load(f)
#     return char_to_index, index_to_char


# char_to_index, index_to_char = load_mappings()
# max_length = 35  # Set this to the max length used in your training


# # Preprocess Audio
# def preprocess_audio(file_path):
#     audio = tfio.audio.AudioIOTensor(file_path)
#     audio_tensor = audio.to_tensor()
#     if len(audio_tensor.shape) > 1 and audio_tensor.shape[1] == 1:
#         audio_tensor = tf.squeeze(audio_tensor, axis=-1)
#     if len(audio_tensor.shape) > 1 and audio_tensor.shape[1] == 2:
#         audio_tensor = tf.reduce_mean(
#             audio_tensor, axis=1
#         )  # Average channels if there are two
#     if len(audio_tensor.shape) == 1:
#         audio_tensor = tf.expand_dims(audio_tensor, axis=-1)
#     return audio_tensor


# def preprocess_for_model(audio_tensor, max_length):
#     current_length = audio_tensor.shape[0]
#     if current_length < max_length:
#         padding = max_length - current_length
#         audio_tensor = tf.pad(audio_tensor, [[0, padding], [0, 0]], "CONSTANT")
#     else:
#         audio_tensor = audio_tensor[:max_length]
#     audio_tensor = tf.expand_dims(audio_tensor, axis=0)  # Add batch dimension
#     return audio_tensor


# def predict_transcription(model, file_path, max_length, char_to_index, index_to_char):
#     audio_tensor = preprocess_audio(file_path)
#     audio_tensor = preprocess_for_model(audio_tensor, max_length)
#     prediction = model.predict(audio_tensor)
#     transcription_indices = tf.argmax(prediction, axis=-1).numpy()[0]
#     transcription = "".join([index_to_char[str(i)] for i in transcription_indices])
#     return transcription


# # Load the model
# model = tf.keras.models.load_model("audio_transcription_zh_base.h5")

# # Test the Model
# # file_path = "./dataset/audio/001SHKBQ.2.ogg"
# file_path = "./dataset/audio/001SHKIR.2.ogg"
# transcription = predict_transcription(
#     model, file_path, max_length, char_to_index, index_to_char
# )
# print("Predicted Transcription:", transcription)

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import json


# Load character mappings
def load_mappings():
    with open("char_to_index.json", "r", encoding="utf-8") as f:
        char_to_index = json.load(f)
    with open("index_to_char.json", "r", encoding="utf-8") as f:
        index_to_char = json.load(f)
    return char_to_index, index_to_char


char_to_index, index_to_char = load_mappings()
max_length = 156  # Adjust this to the max length used in your training


# Preprocess Audio
def preprocess_audio(file_path):
    audio = tfio.audio.AudioIOTensor(file_path)
    audio_tensor = audio.to_tensor()
    if len(audio_tensor.shape) > 1 and audio_tensor.shape[1] == 1:
        audio_tensor = tf.squeeze(audio_tensor, axis=-1)
    if len(audio_tensor.shape) > 1 and audio_tensor.shape[1] == 2:
        audio_tensor = tf.reduce_mean(
            audio_tensor, axis=1
        )  # Average channels if there are two
    if len(audio_tensor.shape) == 1:
        audio_tensor = tf.expand_dims(audio_tensor, axis=-1)
    return audio_tensor


def preprocess_for_model(audio_tensor, max_length):
    current_length = audio_tensor.shape[0]
    if current_length < max_length:
        padding = max_length - current_length
        audio_tensor = tf.pad(audio_tensor, [[0, padding], [0, 0]], "CONSTANT")
    else:
        audio_tensor = audio_tensor[:max_length]
    audio_tensor = tf.expand_dims(audio_tensor, axis=0)  # Add batch dimension
    audio_tensor = tf.expand_dims(
        audio_tensor, axis=-1
    )  # Ensure it's shaped (1, 156, 1)
    return audio_tensor


def predict_transcription(model, file_path, max_length, char_to_index, index_to_char):
    audio_tensor = preprocess_audio(file_path)
    audio_tensor = preprocess_for_model(audio_tensor, max_length)
    prediction = model.predict(audio_tensor)
    transcription_indices = tf.argmax(prediction, axis=-1).numpy()[0]
    transcription = "".join([index_to_char[str(i)] for i in transcription_indices])
    return transcription


# Load the model
model = tf.keras.models.load_model("audio_transcription_zh_base.h5")

# Test the Model
file_path = "./dataset/audio/GDQNA4.4.ogg"
transcription = predict_transcription(
    model, file_path, max_length, char_to_index, index_to_char
)
print("Predicted Transcription:", transcription)
