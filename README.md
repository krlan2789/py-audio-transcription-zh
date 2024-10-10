# Train Model

- Linux / Mac OS
- Windows OS, run it using Docker
- Python, version 3.7 - 3.11
- Tensorflow, version 0.37.0 (use in this project)
- Tensorflow IO, version 2.16.2 (use in this project)

## Linux and Mac OS

Install the required library

```shell
pip install tensorflow==2.16.2 tensorflow-io==0.37.0
```

## Using Docker

Run it with [this](https://hub.docker.com/r/krlan2789/python-tensorflow) Docker Image tha contain required library :

```shell
docker pull krlan2789/python-tensorflow
docker run -it --rm -v D:/Files/Documents/Python/Projects/py-audio-transcription-zh:/py-audio-trancription-zh krlan2789/python-tensorflow bash
```

## Running the Script

After entering the Docker Container terminal, run command below :

```shell
cd py-audio-trancription-zh
python train_more.py # Or train.py
```