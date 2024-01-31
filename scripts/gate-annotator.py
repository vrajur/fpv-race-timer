""" Gate Annotator Tool

Description:
Tool to help annotate gates for the race. Helps with picking gates, formatting, and computing stats to ensure sufficient detection quality.


Input:
 - Video
Output:
 - Annotated video frames + customized detection thresholds
"""

import cv2
import os
import argparse
import math
import yaml

from ImageViewer import ImageViewer

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', dest='input', type=str, required=True, help='Path to video file')
parser.add_argument('--output', '-o', dest='output', type=str, required=False, help='Output directory to save results to')
# parser.add_argument('--flag', '-f', action='store_true', help='Flag parameter')
args = parser.parse_args()


def frames_exist(output_dir, frame_count):
  # Check if the expected number of frames already exist in the output directory
  existing_frames = len([name for name in os.listdir(output_dir) 
    if os.path.isfile(os.path.join(output_dir, name)) 
    and name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
  return existing_frames == frame_count

def try_load_save_frames(output_dir, config_name = "gates.yaml"):

  # Check if a config file exists in the output directory
  config_file = os.path.join(output_dir, config_name)
  config = None
  if os.path.isfile(config_file):
    with open(config_file, 'r') as f:
      config = yaml.load(f)

  save_frames = []

  if config is not None:
    # Print gate info:
    gates = [x['frame_num'] for x in config['gates']]
    num_gates = config['num_gates']
    num_invalid = config['num_invalid']
    print(f"Existing Config Found:\n\tGate Frames: {gates}\n\tNum Gates: {num_gates}\n\tNum Invalid: {num_invalid}")
    ret = input("Load Frames? [y/n]")
    if ret == 'y':
      save_frames = gates

  return save_frames

def generate_frames(video_path, output_dir=None):
  """
  Input: Video file
  Output: Directory of video frames and number of frames generated
  Steps:
    - Get frame name
    - Create directory if doesn't exist yet
    - Split video into frames
  """
  # If output directory is not specified, use the same location as the video file
  if output_dir is None:
    output_dir = os.path.splitext(video_path)[0] + "_frames"

  # Create output directory if it doesn't exist
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Open the video file
  cap = cv2.VideoCapture(video_path)

  # Get video properties
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  num_digits = math.floor(math.log10(frame_count)+1)

  print(f"Video FPS: {fps}")
  print(f"Total Frames: {frame_count}")


  # Check if frames already exist in the output directory
  if frames_exist(output_dir, frame_count):
    print("Expected number of frames already exist in the output directory. Skipping extraction.")

  else:
    # Loop through each frame and save it as an image
    for i in range(frame_count):
      ret, frame = cap.read()
      if not ret:
        break

      # Save the frame as an image file
      frame_filename = os.path.join(output_dir, "frame_"+f"{i + 1}".zfill(num_digits)+".png")
      cv2.imwrite(frame_filename, frame)

      print(f"Saved frame {i + 1}/{frame_count} as {frame_filename}")

    # Release the video capture object
    cap.release()

    print("Frames extraction completed.")

  return output_dir, frame_count


def main():
  frame_dir, num_frames = generate_frames(args.input, args.output)

  # save_frames = try_load_save_frames(frame_dir)

  viewer = ImageViewer(frame_dir)


if __name__ == '__main__':
  main()