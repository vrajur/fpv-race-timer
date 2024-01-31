import os
import cv2
import collections
from line_profiler import profile

import numpy as np

from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch

import Utils

class Gate:

  def __init__(self, data, img_dir, device='cuda') -> None:
    self.device = device
    self.data = data

     # Load Match Threshold
    self.threshold = data['threshold']

    # Load Frame
    self.frame = Utils.preprocess_numpy_image(
      cv2.imread(os.path.join(img_dir, data['frame']))
    )

    # Load Distributions
    self.test_mean = data['test_mean']
    self.test_std = data['test_std']
    self.train_mean = data['train_mean']
    self.train_std = data['train_std']

    # Load Torch Frame
    self.frame_torch = numpy_image_to_torch(self.frame).to(self.device)

    # Load Features
    self.feats = None

    # Load Validity
    self.valid = not data['invalid']

  def use_dummy_image(self, image):
    self.frame = Utils.preprocess_numpy_image(image)
    self.frame_torch = numpy_image_to_torch(self.frame).to(self.device)
    self.feats = None
    self.valid = True


class GateDetector:

  def __init__(self, config, device='cuda') -> None:
    self.config = config
    self.device = device
    self.extractor = SuperPoint(**config['extractor']).eval().to(self.device)
    self.matcher = LightGlue(features='superpoint', **config['matcher']).eval().to(self.device)
    self.num_gates = config['num_gates']
    self.train_batch_size = config['train_batch_size']
    self.feature_suppression_distance = config['feature_suppression_distance']

    self.match_counts = collections.deque(maxlen=self.train_batch_size)

    self.curr_gate_idx = None
    self.gate = None
    self.load_gate(0)

    self.telemetry = {
      "num_matches": 0,
      "threshold": 0
    }

  def load_gate(self, idx):
    # Set Gate Index
    self.curr_gate_idx = idx

    # Load Gate
    gate_data = self.config['gates'][idx]
    img_dir = self.config['image_dir']

    self.gate = Gate(gate_data, img_dir)
        
    # Extract Features
    self.gate.feats = self.extractor.extract(self.gate.frame_torch)

    # Check validity
    if not self.gate.valid:
      print(f"WARNING GATE {idx} is INVALID - LOADING NEXT GATE")
      self.load_next_gate()
    
    # Clear Match Counts:
    self.match_counts.clear()

    return self.gate.valid

  def load_next_gate(self):
    next_idx = max((self.curr_gate_idx + 1) % self.num_gates, 0)
    self.load_gate(next_idx)

  def load_dummy_gate(self, image):
    self.load_gate(0)
    self.gate.use_dummy_image(image)
    self.gate.feats = self.extractor.extract(self.gate.frame_torch)

  # @profile
  # def filter_keypoints(self, features, radius = 10):
  #   filtered_features = features


  #   # For each keypoint find distance to all other keypoints
  #   """
  #   features.key() = dict_keys(['keypoints', 'keypoint_scores', 'descriptors', 'image_size'])
  #   features['keypoints'].shape = torch.Size([1, 512, 2])
  #   features['keypoints'][0,:,:].shape = torch.Size([512, 2])

  #   create a tensor of shape[512, 512, 2]

  #   """
  #   kpts = features['keypoints'][0,:,:].tolist()

  #   keep_idxs = self.remove_close_points(kpts, min_distance=radius)

  #   for k in ["keypoints", "keypoint_scores", "descriptors"]:
  #     print(f"Key: {k}, Shape: {filtered_features[k].shape}")
  #     filtered_features[k] = filtered_features[k][:, keep_idxs, ...]
  #     print(f"Key: {k}, Shape: {filtered_features[k].shape}")
  #   # import pdb; pdb.set_trace()

  #   # keep_idxs = list(range(kpts.shape[0]))
  #   # out = []
  #   # for ii in keep_idxs:
  #   #   kpt = kpts[ii,:]
  #   #   import pdb; pdb.set_trace()
  #   #   other_kpts = kpts[keep_idxs[ii+1:], :]
  #   #   dists = other_kpts - kpt
  #   #   keep_idxs = filter()
  #   #   rm_idxs = (torch.linalg.vector_norm(dists, dim=1) < radius).tolist()
  #   #   out.append(ii)

      


  #   return filtered_features

  # @profile
  # def remove_close_points(self, points, min_distance):
  #   """
  #   Modify a list of 2D points such that there are no points within a minimum distance of any other points.
  #   Keep the points that are earlier in the list, and keep track of the removed indices.

  #   Parameters:
  #   - points: List of 2D points, each represented as [x, y].
  #   - min_distance: Minimum distance threshold. Points closer than this distance will be removed.

  #   Returns:
  #   - List of indices that were kept from the original list.
  #   """
  #   modified_points = [points[0]]  # Keep the first point as it's always included
  #   removed_indices = []
  #   keep_idxs = []

  #   for i in range(1, len(points)):
  #     x, y = points[i]
  #     keep_point = True

  #     for j in range(len(modified_points)):
  #       last_x, last_y = modified_points[j]

  #       # Calculate Euclidean distance
  #       distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
  #       # print(f"computing distance between {points[i]} and {modified_points[j]}. Distance: {distance}")

  #       # If the point is too close to any other point, mark it for removal
  #       if distance < min_distance:
  #         # print(f"Removing point {points[i]}")
  #         keep_point = False
  #         removed_indices.append(i)
  #         break

  #     # Keep the point if it's not too close to any other point
  #     if keep_point:
  #       # print(f"Keeping point {points[i]}")
  #       modified_points.append(points[i])
  #       keep_idxs.append(i)

  #   return keep_idxs

  @profile
  def detect(self, frame):

    # Process Frame:
    frame = Utils.preprocess_numpy_image(frame)
    frame_torch = numpy_image_to_torch(frame).to(self.device)

    # Extract Features
    feats = self.extractor.extract(frame_torch)
    # feats = Utils.filter_keypoints(feats, self.feature_suppression_distance)

    # Find Matches
    matches = self.matcher({'image0': self.gate.feats, 'image1': feats})

    # Count Matches
    num_matches = matches['matches'][0].shape[0]
    self.match_counts.append(num_matches)

    # Compute threshold:
    threshold = self.gate.threshold
    offset = 5
    if len(self.match_counts) > offset:
      mean = np.array(self.match_counts).mean()
      # std = np.std(self.match_counts)
      # threshold = mean + 2.5 * std
      threshold = self.gate.test_mean - 1 * self.gate.test_std - mean
    
    threshold = int(threshold)


    # Store debug information
    self.telemetry["num_matches"] = num_matches
    self.telemetry["feats"] = feats
    self.telemetry["matches"] = matches
    self.telemetry["threshold"] = threshold
    
    return num_matches >= threshold
    




  
