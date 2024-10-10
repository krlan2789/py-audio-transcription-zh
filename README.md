# Python Audio Transcription for Mandarin

- Linux / Mac OS
- Windows OS, run it using Docker
- Python, version 3.7 - 3.11
- Tensorflow, version 0.37.0 (use in this project)
- Tensorflow IO, version 2.16.2 (use in this project)

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

Run it with [this](https://hub.docker.com/r/krlan2789/python-tensorflow) Docker Image tha contain required library :

```shell
docker pull krlan2789/python-tensorflow
docker run -it --rm -v path/to/python/project/dir/py-audio-transcription-zh:/py-audio-trancription-zh krlan2789/python-tensorflow bash
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

Run `train.py` or `train_more.py` to train model and save to .h5 file :

```shell
python train_more.py # Or train.py
```

### Test Model

Run `transcribe.py` for testing the model :

```shell
python transcribe.py
```
