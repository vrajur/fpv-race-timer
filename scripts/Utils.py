import cv2
import numpy as np
from line_profiler import profile
from scipy.spatial import cKDTree


def crop_numpy_image(img, w=0, h=100):

  height = img.shape[0]
  width = img.shape[1]

  cropped = img[h:height-h, w:width-w, ...]
  
  return cropped

def resize_numpy_image(img, scale=0.5):
  return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def preprocess_numpy_image(img):
  return resize_numpy_image(crop_numpy_image(img))


@profile
def filter_keypoints(features, radius = 20):
  
  # Initializations
  filtered_features = features
  kpts = features['keypoints'][0,:,:].tolist()

  # Remove nearby keypoints
  keep_idxs = remove_close_points(kpts, min_distance=radius)
  for k in ["keypoints", "keypoint_scores", "descriptors"]:
    filtered_features[k] = filtered_features[k][:, keep_idxs, ...]

  # Print results
  num_kpts_before = len(kpts)
  num_kpts_after = len(keep_idxs)
  print(f"Filtered keypoints from {num_kpts_before} to {num_kpts_after}")

  return filtered_features

@profile
def remove_close_points(points, min_distance):
  """
  Modify a list of 2D points such that there are no points within a minimum distance of any other points.
  Keep the points that are earlier in the list, and keep track of the removed indices.

  Parameters:
  - points: List of 2D points, each represented as [x, y].
  - min_distance: Minimum distance threshold. Points closer than this distance will be removed.

  Returns:
  - List of indices that were kept from the original list.
  """
  modified_points = [points[0]]  # Keep the first point as it's always included
  removed_indices = []
  keep_idxs = []

  for i in range(1, len(points)):
    x, y = points[i]
    keep_point = True

    for j in range(len(modified_points)):
      last_x, last_y = modified_points[j]

      # Calculate Euclidean distance
      distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
      # print(f"computing distance between {points[i]} and {modified_points[j]}. Distance: {distance}")

      # If the point is too close to any other point, mark it for removal
      if distance < min_distance:
        # print(f"Removing point {points[i]}")
        keep_point = False
        removed_indices.append(i)
        break

    # Keep the point if it's not too close to any other point
    if keep_point:
      # print(f"Keeping point {points[i]}")
      modified_points.append(points[i])
      keep_idxs.append(i)

  return keep_idxs


def remove_close_points_fast(points, min_distance):
  """
  Find points within a certain distance of a specific point using cKDTree.

  Parameters:
  - points: Array of 2D points, shape (N, 2).
  - query_point: The point around which to search for neighbors.
  - distance_threshold: Maximum distance to consider for neighbors.

  Returns:
  - List of indices of points within the specified distance.
  """

  keep_idxs = [0]

  pts = np.array(points)
  num_pts = pts.shape[0]
  kdtree = cKDTree(points)

  

  for ii in range(1, num_pts):

    pt = pts[ii,:]
    rm_idxs = kdtree.query_ball_point(pt, min_distance)


  return keep_idxs



def remove_close_points_fast1(points, min_distance):

  keep_idxs = []

  # Reshape points to (N, 1, 2) and (1, N, 2) for broadcasting
  pts = np.array(points)
  num_pts = pts.shape[0]
  points_reshaped = pts[:, np.newaxis, :]
  points_transposed = pts[np.newaxis, :, :]

  # Use Einstein summation notation to compute squared differences
  dists_sq = np.einsum('ijk,ijk->ij', points_reshaped - points_transposed, points_reshaped - points_transposed)

  M_keep = dists_sq >= min_distance**2

  """
  Find first ii for each column jj where distance is not far enough
    UT.argmax(axis=0)
  Set M[ii>ii0, jj] to False (excluded)
  Repeat for all jj

  """



  UT = np.triu(M_keep, 0)

  """
  Keep Indices: 
  - Should keep the ones that are greater > min_dist**2 for all points in row above the 0th diagonal
    - Sum of elements in row ii should exactly equal ii-1 because every point jj < ii should be true
      - True mean keep which means that the point jj is farther than min dist from point ii
  -If sum is less than ii-1 (meaning there are nearby points) --> then keep point ii, but dont keep jj
    - 
  """
  keep_idxs = UT.sum(axis=1) >= (np.arange(num_pts-1, -1, -1))

  # Ground truth checking:
  dists_sq2 = np.zeros([num_pts, num_pts])
  for ii in range(num_pts):
    for jj in range(num_pts):
      p0 = pts[ii,:]
      p1 = pts[jj,:]
      d = np.sum((p1-p0)**2)
      dists_sq2[ii, jj] = d

  print(f"Total Residual: {(dists_sq2-dists_sq).sum()}")

  import matplotlib.pyplot as plt 
  # plt.subplots(1,3)
  # ax1 = plt.subplot(1,3,1)
  # plt.imshow(dists_sq>=min_distance**2)
  plt.imshow(UT)
  plt.title("Fast Distance Squared Matrix")
  plt.colorbar()
  # ax2 = plt.subplot(1,3,2, sharex=ax1, sharey=ax1)
  # plt.imshow(dists_sq2>=min_distance**2)
  # plt.title("Slow Distance Squared Matrix")
  # plt.colorbar()
  # ax2 = plt.subplot(1,3,3, sharex=ax1, sharey=ax1)
  # plt.imshow(dists_sq2-dists_sq)
  # plt.title("Residuals")
  # plt.colorbar()

  plt.ion()
  plt.show()
  import pdb; pdb.set_trace()

  

  return keep_idxs



