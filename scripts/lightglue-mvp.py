from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
import matplotlib.pyplot as plt
from line_profiler import profile

import rospy
import rosbag
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import CompressedImage

import signal
import sys
import cv2
import numpy as np
import time
import yaml
import torch

import Utils
from GateDetector import GateDetector

import argparse

# USE_CUDA = False
device = 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', dest='input', type=str, required=True, help='Input video recording (video file)')
parser.add_argument('--gates', '-g', dest='gates', type=str, required=True, help='YAML configuration file for gates')
parser.add_argument('--frame_skip', '-f', dest='frame_skip', type=int, default=2, help='Speed up FPS by only processing every Nth frame')
parser.add_argument('--start', dest='start', type=int, default=0, help='Starting frame to skip to at the beginning')
parser.add_argument('--end', dest='end', type=int, default=sys.maxsize, help='Ending frame to end processing with')
parser.add_argument('--plot', '-p', action='store_true', help='Flag to display match results frame-by-frame')
parser.add_argument('--save', '-s', action='store_true', help='Save processing results to rosbag for debugging')
parser.add_argument('--verbose', '-v', action='store_true', help='Print additional debugging information to the screen')
args = parser.parse_args()


def init_pubs():
  rospy.init_node('xy_publisher', anonymous=True)
  pub1 = rospy.Publisher('telemetry', Int32MultiArray, queue_size=10)
  pub1 = rospy.Publisher('telemetry', Int32MultiArray, queue_size=10)
  pub2 = rospy.Publisher('frame/compressed', CompressedImage, queue_size=10)
  pub3 = rospy.Publisher('gate/compressed', CompressedImage, queue_size=10)
  # rate = rospy.Rate(1)  # Define the publishing rate (1 Hz in this example)
  return pub1, pub2, pub3


def publish(frame_num, match_count, gate_id, frame, gate_frame, match_threshold, FPS, pubs, bag=None):
  # Create a Int32MultiArray message to store x and y values
  telemetry_msg = Int32MultiArray()
  telemetry_msg.data = [frame_num, match_count, gate_id, int(match_threshold), int(FPS)]

  # Publish the x and y tuple
  pubs[0].publish(telemetry_msg)

  # Encode the OpenCV image to a compressed format
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Adjust quality as needed
  _, compressed_data = cv2.imencode('.jpg', frame, encode_param)

  # Create a CompressedImage message and fill in its data
  frame_msg = CompressedImage()
  frame_msg.format = 'jpeg'
  frame_msg.data = compressed_data.tostring()

  # Publish the CompressedImage message
  pubs[1].publish(frame_msg)

  # Publish gate frame image
  gate_msg = CompressedImage()
  gate_msg.format = 'jpeg'
  gate_msg.data = compressed_data.tostring()
  _, compressed_data = cv2.imencode('.jpg', gate_frame, encode_param)
  gate_msg.data = compressed_data.tostring()
  pubs[2].publish(gate_msg)

  # Write to bag file:
  if bag is not None:
    bag.write(pubs[0].name, telemetry_msg)
    bag.write(pubs[1].name, frame_msg)
    bag.write(pubs[2].name, gate_msg)

def load_config(config):
  with open(config, 'r') as f:
    return yaml.load(f)

def create_rosbag(video_file):
  rosbag_file = video_file[:-4] + ".bag"
  bag = rosbag.Bag(rosbag_file, 'w')
  return bag

# def signal_handler(sig, frame):
#   # Close the bag file when receiving an interrupt signal
#   if bag:
#       bag.close()
#   print('Bag file closed.')
#   sys.exit(0)


@profile
def main():

  # Load Config
  config = load_config(args.gates)

  # Create GateDetector
  detector = GateDetector(config)

  if detector.num_gates == 0:
    print("Configuration has no gates. Exiting.")
    exit(0)

  # Initializations
  pubs = init_pubs()
  ts = []
  match_counts = []
  frame_num = 0
  skip_step_size = args.frame_skip
  last_frame = None

  # Open video file
  cap = cv2.VideoCapture(args.input)

  # Check if the video file is opened successfully
  if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

  # Create bag file if save option set
  bag = None
  if args.save:
    bag = create_rosbag(video_file=args.input)

  # Read frames from video file
  while True:

    start_time = time.time()

    # Read next frame
    ret, frame = cap.read() 
  
    # Check if the frame is read successfully
    if not ret:
      print("End of video or error in reading frames.")
      break
    
    # Skip frames to maintain target FPS
    frame_num += 1
    if (frame_num % skip_step_size) != 0:
      continue

    # Skip to range start
    if frame_num < args.start:
      continue

    # End after range end
    if frame_num > args.end:
      break

    # Detect Gate:
    gate_detected = detector.detect(frame)

    # Compute FPS
    runtime = time.time() - start_time
    fps = round(1 / runtime, 2)
    amortized_fps = round(skip_step_size * fps, 2)

    # Publish Telemetry
    publish(
      frame_num=frame_num, 
      match_count=detector.telemetry['num_matches'],
      gate_id=detector.curr_gate_idx, 
      frame=frame,
      gate_frame = detector.gate.frame,
      match_threshold=detector.telemetry['threshold'],
      FPS=fps, 
      pubs=pubs,
      bag=bag
    ) 

    if gate_detected:
      print(f"DETECTED GATE {detector.curr_gate_idx} at frame {frame_num}")
      # detector.load_next_gate()
      if args.verbose:
        print(f"LOADED GATE {detector.curr_gate_idx}")
    
    if args.verbose:
      print(f"Frame: {frame_num} - Num Matches: {detector.telemetry['num_matches']} - Threshold: {detector.telemetry['threshold']}")
      print(f"Processing Time: {runtime}s ({fps} FPS) - Amortized FPS: {amortized_fps} FPS")

    if args.plot:
      if last_frame is None:
        last_frame = frame
      gate_frame = Utils.preprocess_numpy_image(last_frame)
      curr_frame = Utils.preprocess_numpy_image(frame)
      gate_frame_torch = numpy_image_to_torch(gate_frame).to(device)
      curr_frame_torch = numpy_image_to_torch(curr_frame).to(device)
      gate_feats = detector.extractor.extract(gate_frame_torch)
      curr_feats = detector.extractor.extract(curr_frame_torch)
      matches = detector.matcher({'image0': gate_feats, 'image1': curr_feats})


      axes = viz2d.plot_images([gate_frame, curr_frame])
      feats0, feats1, matches01 = [rbd(x) for x in [gate_feats, curr_feats, matches]]  # remove batch dimension

      kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
      m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
      viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
      # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
      plt.show()



      ### OLD

      # axes = viz2d.plot_images([detector.gate.frame, Utils.preprocess_numpy_image(frame)])

      # feats0, feats1, matches01 = [rbd(x) for x in [detector.gate.feats, detector.telemetry['feats'], detector.telemetry['matches']]]  # remove batch dimension

      # kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
      # if frame_num >= 412:
      #   import pdb; pdb.set_trace()
      # print(frame_num)
      # m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
      # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
      # # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
      # plt.show()

    if last_frame is not None:
      detector.load_dummy_gate(last_frame)
    last_frame = frame

  # Close RosBag if created
  if bag is not None:
    bag.close()
    print(f"Saved bag file: {bag.filename}")

  # Release the video capture object and close any open windows
  cap.release()


if __name__ == '__main__':
  main()