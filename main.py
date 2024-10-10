import os
import json

audio_dir = "./dataset/audio/"
transcript_dir = "./dataset/transcripts/"
dataset = []

for audio_file in os.listdir(audio_dir):
    if audio_file.endswith(".ogg"):
        nameParts = audio_file.split('.')
        productCode = nameParts[0]
        transcriptIdx = int("0" + nameParts[1])
        transcript_file = productCode + ".txt"
        transcript = ''
        with open(os.path.join(transcript_dir, transcript_file), "r", encoding='utf-8') as f:
            content = f.read()
            transcriptParts = content.replace("\n*\n", "\n#\n").split('#')
            transcriptCount = len(transcriptParts) - 2
            mandarin = transcriptParts[transcriptIdx + 1].strip().split("\n")[1].replace("< ", "").replace(" >", "")
            for word in mandarin.split(' '):
                w = word
                if len(w) > 1:
                    w = word.split("(")[0]
                transcript += w
            print("%s -> %d/%d:\n%s\n" % (transcript_file, transcriptIdx, transcriptCount, transcript))
            # for idx in range(2, transcriptCount, 1):
        dataset.append(
            {
                "audio_file": os.path.join(audio_dir, audio_file),
                "transcript": transcript,
            }
        )

# Save the dataset to a JSON file
with open("dataset.json", 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print("Dataset created successfully!")