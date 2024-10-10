import json
import os
import re
import shutil

contents_dir = "./dataset/contents/"
audio_dir = "./dataset/audio/"
transcript_dir = "./dataset/transcripts/"


# Get txt files path
def get_txt_files(root_dir):
    file_paths = []
    file_paths_dest = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(root_dir):
        # Check if 'text' folder is exist
        if "text" in root.split(os.sep):
            if dirs:
                # Sort as alphanumeric
                dirs.sort(key=lambda x: int(x) if x.isdigit() else x)
                last_dir = dirs[-1]  # Get last folder after sorting
                subfolder_path = os.path.join(root, last_dir)

                # Get files from last folder
                for sub_root, _, sub_files in os.walk(subfolder_path):
                    idx = 1
                    for fl in sub_files:
                        extFile = fl.split(".")[1]
                        file = os.path.join(sub_root, fl)
                        file = file.replace("\\", "/")
                        dest = f"{re.sub(r"(/text/\d+/.*)", "", file)}.{extFile}".replace("contents/", "transcripts/")
                        file_paths.append(file)
                        file_paths_dest.append(dest)
                        idx += 1

                # Stop exploring another subfolder in this folder
                dirs.clear()

    return file_paths, file_paths_dest


# Get audio files path
def get_audio_files(root_dir):
    audio_file_paths = []
    audio_file_path_dest = []
    # Walk through the directory structure
    for root, dirs, files in os.walk(root_dir):
        # Check if 'audio' folder is exist
        if "audio" in root.split(os.sep):
            # Get all files in 'audio' folder
            idx = 1
            for fl in files:
                extFile = fl.split(".")[1]
                file = os.path.join(root, fl)
                file = file.replace("\\", "/")
                dest = (
                    f"{root}.{idx}.{extFile}".replace("\\", "/")
                    .replace("/audio.", ".")
                    .replace("/contents/", "/audio/")
                )
                audio_file_paths.append(file)
                audio_file_path_dest.append(dest)
                idx += 1

    return audio_file_paths, audio_file_path_dest


# Get list of txt files path
txt_list, txt_list_dest = get_txt_files(contents_dir)
# Get list of audio files path and destination
audio_list, audio_list_dest = get_audio_files(contents_dir)

# # Print results
# for index in range(len(txt_list)):
#     print(f"{txt_list[index]} -> {txt_list_dest[index]}")

# # Print results
# for index in range(len(audio_list)):
#     print(f"{audio_list[index]} -> {audio_list_dest[index]}")


# Move files
def move_files(from_paths, to_paths):
    if len(from_paths) is not len(to_paths):
        return

    # Memindahkan setiap file ke folder tujuan
    for idx in range(len(from_paths)):
        try:
            # Pastikan folder tujuan ada, jika tidak buat foldernya
            dest_dir = os.path.dirname(to_paths[idx])
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            shutil.copyfile(from_paths[idx], to_paths[idx])
            print(f"Success: {from_paths[idx]} copied to {to_paths[idx]}")
        except Exception as e:
            print(f"Failed to copy file {from_paths[idx]}: {e}")


# Move files
move_files(txt_list, txt_list_dest)
move_files(audio_list, audio_list_dest)



# -------------------------- Generate Dataset --------------------------


dataset = []

for audio_file in os.listdir(audio_dir):
    if audio_file.endswith(".ogg"):
        nameParts = audio_file.split(".")
        productCode = nameParts[0]
        transcriptIdx = int("0" + nameParts[1])
        transcript_file = productCode + ".txt"
        transcript = ""
        with open(
            os.path.join(transcript_dir, transcript_file), "r", encoding="utf-8"
        ) as f:
            content = f.read()
            if "#-" in content:
                content = content.split('#-')[0]
            content = content.replace("\n*\n", "\n#\n")
            transcriptParts = content.split("#")
            transcriptCount = len(transcriptParts) - 2
            # print(f"{transcriptIdx}/{transcriptCount} ({productCode}):\n{content}")
            mandarin = (
                transcriptParts[transcriptIdx + 1]
                .strip()
                .split("\n")[1]
                .replace("< ", "")
                .replace(" >", "")
            )
            for word in mandarin.split(" "):
                w = word
                if len(w) > 1:
                    w = word.split("(")[0]
                transcript += w
            print(
                "%s -> %d/%d:\n%s\n"
                % (transcript_file, transcriptIdx, transcriptCount, transcript)
            )
            # for idx in range(2, transcriptCount, 1):
        dataset.append(
            {
                "audio_file": os.path.join(audio_dir, audio_file),
                "transcript": transcript,
            }
        )

# Save the dataset to a JSON file
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print("Dataset created successfully!")


