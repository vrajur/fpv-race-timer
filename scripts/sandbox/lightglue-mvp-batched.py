from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
import matplotlib.pyplot as plt
from line_profiler import profile
import torch


import rospy
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np

def crop_numpy_image(img, w=100, h=100, plot=False):

  height = img.shape[0]
  width = img.shape[1]

  cropped = img[h:height-h, w:width-w, ...]

  if plot:
    plt.imshow(img)
    plt.show()
    plt.imshow(cropped)
    plt.show()
  
  return cropped

def resize_numpy_image(img, scale=0.5):
  return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def preprocess_numpy_image(img):
  return resize_numpy_image(crop_numpy_image(img))


def init_pubs():
  rospy.init_node('xy_publisher', anonymous=True)
  pub1 = rospy.Publisher('frame_match_count', Int32MultiArray, queue_size=10)
  pub2 = rospy.Publisher('frame/compressed', CompressedImage, queue_size=10)
  # rate = rospy.Rate(1)  # Define the publishing rate (1 Hz in this example)
  return pub1, pub2


def publish(frame_num, match_count, frame, pubs):
  # Create a Int32MultiArray message to store x and y values
  msg = Int32MultiArray()
  msg.data = [frame_num, match_count]

  # Publish the x and y tuple
  pubs[0].publish(msg)


  # Encode the OpenCV image to a compressed format
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Adjust quality as needed
  _, compressed_data = cv2.imencode('.jpg', frame, encode_param)

  # Create a CompressedImage message and fill in its data
  ros_compressed_image = CompressedImage()
  ros_compressed_image.format = 'jpeg'
  ros_compressed_image.data = compressed_data.tostring()

  # Publish the CompressedImage message
  pubs[1].publish(ros_compressed_image)

def convert_batch_to_tensor(feature_batch):
  return None


@profile
def main():
  # Open reference gate image
  ref_frame = cv2.imread("/media/vinay/Shared Storage/Outputs/Metazoic/recordings/Gates/3.jpg")
  ref_frame = preprocess_numpy_image(ref_frame)
  ref_frame_torch = numpy_image_to_torch(ref_frame).cuda()
  
  # FPV frame
  my_guess = 0
  frame_nums = range(1, 2000)
  fpv_frame_sources = [f"/home/vinay/data/Outputs/Metazoic/recordings/tinyhawk-fpv-testlaps-2023-11-09-FRAMES/frame{x:05d}.jpg" for x in frame_nums]

  pubs = init_pubs()  

  # Initializations
  extractor = SuperPoint(max_num_keypoints=256, detection_threshold=0.0).eval().cuda()  # load the extractor
  match_conf = {
    'width_confidence': -1,  # for point pruning
    'depth_confidence': -1,  # for early stopping,
    'n_layers': 5,
    'flash': True,
  }
  matcher = LightGlue(features='superpoint', **match_conf).eval().cuda()  # load the matcher 
  # matcher.compile(mode="reduce-overhead")
  match_counts = []
  ts = []

  do_plot = False

  curr_batch_size = 0
  max_batch_size = 30
  feats1_batched = None

  feats0_batched = extractor.extract(ref_frame_torch)
  for key in feats0_batched.keys():
    feats0_batched[key] = torch.cat([feats0_batched[key]] * max_batch_size)

  for fpv_frame_source in fpv_frame_sources:
    print(fpv_frame_source)
    fpv_frame = cv2.imread(fpv_frame_source)
    fpv_frame = preprocess_numpy_image(fpv_frame)
    fpv_frame_torch = numpy_image_to_torch(fpv_frame).cuda()

    # Compare frames
    # Estimate Relative Pose https://robotics.stackexchange.com/questions/14456/determine-the-relative-camera-pose-given-two-rgb-camera-frames-in-opencv-python
    
    #   Extract feature keypoints
    feats1 = extractor.extract(fpv_frame_torch)

    if curr_batch_size < max_batch_size:

      if feats1_batched is None:
        feats1_batched = feats1
      else:
        for key in feats1.keys():
          feats1_batched[key] = torch.cat([feats1_batched[key], feats1[key]])
      curr_batch_size += 1
      print(f"Building batch: {curr_batch_size}/{max_batch_size}")

    if curr_batch_size >= max_batch_size:
      #   Find Matches
      matches01 = matcher({'image0': feats0_batched, 'image1': feats1_batched})

      #   Generate points0 array and points1 array (size [N,2])
      import pdb; pdb.set_trace()
      feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
      matches = matches01['matches'] # indices with shape (K,2)
      points0 = feats0['keypoints'][matches[..., 0]] # coordinates in image 0, shape (K,2)
      points1 = feats1['keypoints'][matches[..., 1]] # coordinates in image 1, shaep (K,2)

      # #   Use cv2.findEssentialMat() (provide intrinsics)
      # cameraMatrix = np.eye(3)
      # ref_pts = points0.detach().cpu().numpy()
      # fpv_pts = points1.detach().cpu().numpy()
      # E, mask = cv2.findEssentialMat(points1=ref_pts, points2=fpv_pts, cameraMatrix=cameraMatrix)

      # #   Use cv2.recoverPose()
      # points, R, t, mask = cv2.recoverPose(points1=ref_pts, points2=fpv_pts, E=E)

      # Count number of matches
      # import pdb; pdb.set_trace()
      num_matches = matches.shape[0]
      match_counts.append(num_matches)
      # ts.append(t.flatten())
      # print(f"Frame: {fpv_frame_source[-9:-4]} - Num Matches: {num_matches} - t: {ts[-1]}")
      print(f"Frame: {fpv_frame_source[-9:-4]} - Num Matches: {num_matches}")
      publish(int(fpv_frame_source[-9:-4]), num_matches, fpv_frame, pubs)

      curr_batch_size = 0
      feats1_batched = None
    

    #   Plot results
    if do_plot:
      axes = viz2d.plot_images([ref_frame,fpv_frame])

      kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
      m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
      viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
      viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

      # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
      # viz2d.plot_images([ref_frame, fpv_frame])
      # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)

      # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
      # viz2d.plot_images([image0, image1])
      # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
      plt.show()

  plt.plot(frame_nums, match_counts)
  plt.axvline(my_guess, color='k')
  plt.show()


  # ts = np.array(ts)
  # plt.plot(frame_nums, ts[:,0], label='X')
  # plt.plot(frame_nums, ts[:,1], label='Y')
  # plt.plot(frame_nums, ts[:,2], label='Z')
  # plt.legend()
  # plt.show()


  # import pdb; pdb.set_trace()
if __name__ == '__main__':
  main()