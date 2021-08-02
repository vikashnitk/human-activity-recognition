# Import libraries
import torch
import torchvision
import cv2
import argparse
import time
import numpy as np
import os

from urllib import request  
import albumentations as A

# define the transforms
transform = A.Compose([
    A.Resize(128, 171, always_apply=True),
    A.CenterCrop(112, 112, always_apply=True),
    A.Normalize(mean = [0.43216, 0.394666, 0.37645],
                std = [0.22803, 0.22145, 0.216989], 
                always_apply=True)
])

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
  class_names = [line.decode("utf-8").strip() for line in obj.readlines()]

# get the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the pytorch resnet 3D model pretrained on kinetic dataset from torchvision
model = torchvision.models.video.r3d_18(pretrained=True, progress=True)

# load the model onto the computation device
model = model.eval().to(device)

# Utilities to load the videos
def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  clips = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break

      image = frame.copy()
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = transform(image=frame)['image']
      clips.append(frame)
      
      if len(clips) == max_frames:
        break
  finally:
    cap.release()

  with torch.no_grad(): # we do not want to backprop any gradients
    input_frames = np.array(clips)
  # add an extra dimension        
    input_frames = np.expand_dims(input_frames, axis=0)
  # transpose to get [1, 3, num_clips, height, width]
    input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
  # convert the frames to tensor
    input_frames = torch.tensor(input_frames, dtype=torch.float32)
    input_frames = input_frames.to(device)
  # forward pass to get the predictions
    outputs = model(input_frames)
  # get the prediction index
    _, preds = torch.max(outputs.data, 1)
                
  # map predictions to the respective class names
    label = class_names[preds].strip()

    print(os.path.basename(path), f"      {label}")
    #print(f" {label}")


print("Result:[video-label]")
# video clips folder : https://github.com/vikashnitk/human-activity-recognition/tree/main/clips
for file in os.listdir(args["path"]):
  # Supported video formats
  ext = [".3g2", ".3gp", ".asf", ".asx", ".avi", ".flv", ".m2ts", ".mkv", ".mov", ".mp4", ".mpg", ".mpeg", ".rm", ".swf", ".vob", ".wmv"]
  if file.endswith(tuple(ext)):
    path=os.path.join(args["path"], file)
    load_video(path)


# python pytorch-human-activity-recognition.py --path /home/vikash/Desktop/videos/clips
