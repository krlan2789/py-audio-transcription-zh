import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import json
from tensorflow.keras.layers import Input, LSTM, Dense


# Load Dataset
def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


dataset = load_dataset("dataset.json")


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
    return audio_tensor


audio_tensors = [preprocess_audio(item["audio_file"]) for item in dataset]
transcripts = [item["transcript"] for item in dataset]

# Extract unique characters from transcripts
transcripts_text = "".join(transcripts)
characters = sorted(set(transcripts_text))

# Create a mapping from characters to their index and vice versa
char_to_index = {char: index for index, char in enumerate(characters)}
index_to_char = {index: char for index, char in enumerate(characters)}

# Convert transcripts to indices
transcripts_indices = [
    [char_to_index[char] for char in transcript] for transcript in transcripts
]
max_length = max([len(transcript) for transcript in transcripts_indices])
transcripts_indices = tf.keras.preprocessing.sequence.pad_sequences(
    transcripts_indices, padding="post"
)


# Downsample audio tensors to match the transcription length
def downsample(audio, target_len):
    factor = len(audio) // target_len
    return audio[::factor]


# Ensure all audio tensors have the same number of time steps by padding or trimming
audio_tensors = [
    (
        tf.pad(
            downsample(audio, max_length),
            [[0, max_length - len(downsample(audio, max_length))], [0, 0]],
        )
        if len(audio) < max_length
        else audio[:max_length]
    )
    for audio in audio_tensors
]

# Ensure single feature per time step
audio_tensors = [
    tf.expand_dims(tf.squeeze(audio, axis=-1), axis=-1) for audio in audio_tensors
]

# Stack into a single tensor
audio_tensors = tf.stack(audio_tensors)

# Check the final shape of audio_tensors
print("Shape of audio_tensors after padding:", audio_tensors.shape)

# Ensure rank compatibility with target.ndim
transcripts_indices = np.array(transcripts_indices)

# Check shapes of data
print("Shape of audio_tensors:", audio_tensors.shape)
print("Shape of transcripts_indices:", transcripts_indices.shape)

# Model Architecture
input_audio = Input(shape=(max_length, 1), name="audio_input")
x = LSTM(128, return_sequences=True)(input_audio)
x = LSTM(128, return_sequences=True)(x)  # Ensure sequence output
output = Dense(len(characters), activation="softmax")(x)
model = tf.keras.Model(inputs=input_audio, outputs=output)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train Model
print("Starting training...")
history = model.fit(audio_tensors, transcripts_indices, epochs=10, batch_size=32)
print("Training completed.")

# Evaluate the Model
loss, accuracy = model.evaluate(audio_tensors, transcripts_indices)
print(f"Evaluation - Loss: {loss}, Accuracy: {accuracy}")

# Save Model
print("Saving model...")
model.save("./audio_transcription_zh_base.h5")
print("Model saved successfully.")

# Save character mappings
with open("char_to_index.json", "w", encoding="utf-8") as f:
    json.dump(char_to_index, f, ensure_ascii=False)
with open("index_to_char.json", "w", encoding="utf-8") as f:
    json.dump(index_to_char, f, ensure_ascii=False)
