"""

GateEvaluator:
 
Inputs:
 - Video frame directory
 - List of gates
 - LightGlue configuration

Output:
 - List of valid gates + match count metadata
"""

from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch
import cv2
import torch
import os
import numpy as np
from pprint import pprint

import Utils

class GateEvaluator:

  def __init__(self, image_dir, image_files) -> None:
    self.image_dir = image_dir
    self.image_files = image_files
    self.test_batch_size = 5
    self.train_batch_size = 60
    self.feature_suppression_distance = 10


    self.extractor = SuperPoint(max_num_keypoints=2048, detection_threshold=0.01).eval().cuda()  # load the extractor
    self.matcher = matcher = LightGlue(features='superpoint', n_layers=5).eval().cuda()  # load the matcher 

  def load_image(self, image_file):
    full_path = os.path.join(self.image_dir, image_file)
    image = cv2.imread(full_path)
    image = Utils.preprocess_numpy_image(image)
    return numpy_image_to_torch(image).cuda()

  def quick_validity_check(self, gate_frames, next_gate=None):
    """Performs quick checks to ensure gate list is valid

    If not valid - prints error + description

    Checks:
    1. No gates are within the first 30 frames of the start
    2. Gates are at least 10 frames apart

    Args:
        gate_frames (list(int)): List of gate frames
        next_gate (optional, int): Potential next gate to add to the list

    Return:
      isValid (bool): whether list of gates is valid
    """

    if next_gate is not None:
      test_gates = gate_frames + [next_gate]
    else:
      test_gates = gate_frames

    threshold = self.test_batch_size + self.train_batch_size + 1 # 1 is because gate frame itself is excluded from batches
    if any([x<=threshold for x in test_gates]):
      print(f"No gates should be within the first {threshold} frames of the sequence)")
      return False

    if len(test_gates) <= 1:
      return True

    # Check if gates are too close together
    prev_gate = test_gates[0]
    min_proximity = self.train_batch_size
    for curr_gate in test_gates[1:]:
      if abs(curr_gate - prev_gate) < min_proximity:
        print(f"Gates {prev_gate} and {curr_gate} are too close to each other (within {min_proximity} frames of each other)")
        return False
      prev_gate = curr_gate


    return True
  
  def count_matches(self, ref_feats, image_batch):
    """Count Number of Matches for Batch of Images

    Args:
        ref_feats (_type_): Features from reference image ('image0')
        image_batch (list(str)): List of image files (without image directory)

    Returns:
        match_counts (list(int)): List of match counts for each image in the batch
    """
    # Count Batch Match Counts:
    match_counts = []
    for ii, test_frame in enumerate(image_batch):
      print(f"Matching frame {ii}/{len(image_batch)}")
      
      frame = self.load_image(test_frame)
      feats1 = self.extractor.extract(frame)
      Utils.filter_keypoints(feats1, self.feature_suppression_distance)

      # Find Matches
      matches01 = self.matcher({'image0': ref_feats, 'image1': feats1})
      num_matches = matches01['matches'][0].shape[0]
      match_counts.append(num_matches)
    
    return match_counts

  def full_validity_check(self, gate_frames):
    """

    For each gate:
    - Find previous 5 frames
    - Find the previous 1 second of frames (30 frames) prior to those frames (or last gate)
    - Match both sets of frames against gate frame
      - Count matches for each frame and measure mean and stdev
      - Assert that mean of first 5 frames is signficantly different than number of matches for 30 frames prior

    Args:
      gate_frames (_type_): _description_

    Return:
      gate_eval_data (list(int)): Metadata around number of matches for each gate. If -1 then gate is invalid
    """
    gate_eval_data = []
    num_invalid = 0
    

    for ii, gate in enumerate(gate_frames):

      assert (gate-self.test_batch_size-self.train_batch_size-1 >= 0)

      print(f"Evaluating Gate {gate} (Image: {self.image_files[gate]}) [{ii+1}/{len(gate_frames)}]")
    
      # Test batch (5 frames)
      test_idx0 = gate - self.test_batch_size
      test_idx1 = gate 
      test_batch = self.image_files[test_idx0:test_idx1]
      
      # print(test_batch)
      # print(self.image_files[gate])

      # Train Batch (30 frames)
      train_idx0 = gate - self.test_batch_size - self.train_batch_size
      train_idx1 = gate - self.test_batch_size
      train_batch = self.image_files[train_idx0:train_idx1]
      
      # print(train_batch)
      # print(self.image_files[gate])

      # Load Reference Image:
      ref_frame = self.load_image(self.image_files[gate])
      feats0 = self.extractor.extract(ref_frame)

      # Count Test Batch Match Counts:
      print("Running Test Batch")
      test_match_counts = self.count_matches(feats0, test_batch)

      # Count Train Batch Match Counts:
      print("Running Train Batch")
      train_match_counts = self.count_matches(feats0, train_batch)

      # Compute mean and standard dev for each batch:
      test_mean = float(np.mean(test_match_counts))
      test_std = float(np.std(test_match_counts))

      train_mean = float(np.mean(train_match_counts))
      train_std = float(np.std(train_match_counts))

      print(f"Train Stats:\n\tMean: {train_mean}\n\tStDev: {train_std}")
      print(f"Test Stats:\n\tMean: {test_mean}\n\tStDev: {test_std}")
      
      gate_match_threshold = train_mean + 2.5*train_std

      data = {
        "frame_num": gate,
        "frame": self.image_files[gate],
        "threshold": gate_match_threshold,
        "invalid": False,
        "train_mean": train_mean,
        "train_std": train_std,
        "test_mean": test_mean,
        "test_std": test_std,
        "train_match_counts": train_match_counts,
        "test_match_counts": test_match_counts,
      }

      if test_mean < gate_match_threshold:
        print(f"Gate {gate} is invalid (Image: {self.image_files[gate]}).")
        data['invalid'] = True
        num_invalid += 1

      gate_eval_data.append(data)
    
    print_results = [(x['frame'], x['threshold'], x['invalid']) for x in gate_eval_data]
    pprint(print_results)
    print(f"Gate Evaluation Complete [{num_invalid} invalid gates detected]")

    return gate_eval_data, num_invalid

  def batch_attempt_scrap(self, gate, test_batch):

      # Evaluate Test Batch
      ref_frame = cv2.imread(self.image_files[gate])
      ref_frame = Utils.preprocess_numpy_image(ref_frame)
      ref_frame_torch = numpy_image_to_torch(ref_frame).cuda()

      feats0_batched = self.extractor.extract(ref_frame_torch)
      for key in feats0_batched.keys():
        feats0_batched[key] = torch.cat([feats0_batched[key]] * self.test_batch_size)

      feats1_batched = None
      for frame in test_batch:
        frame1 = cv2.imread(frame)
        frame1 = Utils.preprocess_numpy_image(frame1)
        frame1_torch = numpy_image_to_torch(frame1).cuda()

        feats1 = self.extractor.extract(frame1_torch)
        if feats1_batched is None:
          feats1_batched = feats1
        else:
          for key in feats1.keys():
            feats1_batched[key] = torch.cat([feats1_batched[key], feats1[key]])

      matches01 = self.matcher({'image0': feats0_batched, 'image1': feats1_batched})


      

      