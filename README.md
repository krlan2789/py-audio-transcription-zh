# Python Audio Transcription for Mandarin

- Linux / Mac OS
- Windows OS, run it using Docker
- Python, version 3.7 - 3.11
- Tensorflow, version 0.37.0 (use in this project)
- Tensorflow IO, version 2.16.2 (use in this project)
- librosa, version 0.10.2.post1 (use in this project)

## Linux and Mac OS

Go to project directory

```shell
cd path/to/python/project/dir/py-audio-transcription-zh
```

Install the required library

```shell
pip install tensorflow==2.16.2 tensorflow-io==0.37.0
```

## Using Docker

Run it with [this](https://hub.docker.com/r/krlan2789/python-tensorflow) Docker Image that contains required library :

```shell
docker pull krlan2789/python-tensorflow:1.0.1
docker run -it --rm -v path/on/your/machine/dir/py-audio-transcription-zh:path/on/docker/container/py-audio-trancription-zh krlan2789/python-tensorflow:1.0.1 bash

# Example
docker run -it --rm -v D:/Files/Documents/Python/Projects/py-audio-transcription-zh:/tmp/py-audio-trancription-zh krlan2789/python-tensorflow:1.0.1 bash
```

After entering the Docker Container terminal, run command below :

```shell
cd py-audio-trancription-zh
```

## Scripts

### Create Dataset

Run `main.py` to creating `dataset.json` file :

```shell
python main.py
```

### Train Model

Run `train.py` to train model and save to .h5 file :

```shell
python train.py
```

### Test Model

Run `transcribe.py` for testing the model :

```shell
python transcribe.py
```
