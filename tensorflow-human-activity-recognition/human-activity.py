# TensorFlow and TF-Hub libraries.
from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed

logging.set_verbosity(logging.ERROR)

# Import libraries
import random
import re
import os
import tempfile
import ssl
import cv2
import numpy as np
import argparse

import imageio
from IPython import display

from urllib import request 

def dir_path(string):
  if os.path.isdir(string):
    return string
  else:
    raise NotADirectoryError(string)

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=dir_path)
args = vars(parser.parse_args())

# Get the kinetics-400 action labels from the GitHub repository.
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with request.urlopen(KINETICS_URL) as obj:
  labels = [line.decode("utf-8").strip() for line in obj.readlines()]

# Utilities to open video files using CV2
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

# Utilities to load the videos
def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)
      
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  sample_video = np.array(frames) / 255.0

  model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

  # load the tensorflow Inflated 3D CNN model pretrained on kinetic dataset from tensorflow hub
  i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

  # load the input to model
  logits = i3d(model_input)['default'][0]
  probabilities = tf.nn.softmax(logits)

  for i in np.argsort(probabilities)[::-1][:1]:
    print(os.path.basename(path), f"      {labels[i]:22}")

print("Result:[video-label]")
# video clips folder : https://github.com/vikashnitk/human-activity-recognition/tree/main/clips
for file in os.listdir(args["path"]):
  # Supported video formats
  ext = [".3g2", ".3gp", ".asf", ".asx", ".avi", ".flv", ".m2ts", ".mkv", ".mov", ".mp4", ".mpg", ".mpeg", ".rm", ".swf", ".vob", ".wmv"]
  if file.endswith(tuple(ext)):
    path=os.path.join(args["path"], file)
    load_video(path)