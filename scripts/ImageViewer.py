import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import os
import yaml

from GateEvaluator import GateEvaluator



class ImageViewer:
  def __init__(self, image_dir, save_dir=None):
    self.image_dir = image_dir
    self.save_dir = self.image_dir if save_dir is None else save_dir
    self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    self.image_files.sort()
    self.index = 0
    self.saved_frames = []
    self.gate_evaluator = GateEvaluator(image_dir, self.image_files)

    self.config_filename = "gates.yaml"

    self.viz_frames = []
    self.viz_gates = False
    self.last_index = self.index

    if not self.image_files:
      print("No images found in the specified directory.")
      return

    self.try_load_save_frames()

    self.fig, self.ax = plt.subplots(figsize=(10, 13))
    
    # Add navigation buttons
    self.axprev = plt.axes([0.7, 0.01, 0.1, 0.05])
    self.axnext = plt.axes([0.81, 0.01, 0.1, 0.05])
    self.axslider = plt.axes([0.1, 0.01, 0.5, 0.05])
    
    self.bprev = Button(self.axprev, 'Previous')
    self.bprev.on_clicked(self.prev_image)
    
    self.bnext = Button(self.axnext, 'Next')
    self.bnext.on_clicked(self.next_image)
    
    self.slider = Slider(self.axslider, 'Scrub', 0, len(self.image_files) - 1, valinit=0)
    self.slider.on_changed(self.scrub_image)

    # Add new buttons
    self.axsave = plt.axes([0.1, 0.1, 0.1, 0.05])
    self.axdelete = plt.axes([0.25, 0.1, 0.1, 0.05])
    self.axexport = plt.axes([0.4, 0.1, 0.1, 0.05])
    self.axgateview = plt.axes([0.55, 0.1, 0.1, 0.05])

    self.bsave = Button(self.axsave, 'Save Frame')
    self.bsave.on_clicked(self.save_frame)
    
    self.bdelete = Button(self.axdelete, 'Del Last Frame')
    self.bdelete.on_clicked(self.delete_last_frame)

    self.bexport = Button(self.axexport, 'Export')
    self.bexport.on_clicked(self.export_gates)
    
    self.bgateview = Button(self.axgateview, 'Gate View')
    self.bgateview.on_clicked(self.view_gates)

    # Add frame input textbox and jump button
    self.axframeinput = plt.axes([0.7, 0.1, 0.1, 0.05])
    self.axjump = plt.axes([0.8, 0.1, 0.1, 0.05])

    self.textbox = TextBox(self.axframeinput, '', initial='1')
    self.textbox.on_submit(self.jump_to_frame)

    self.bjump = Button(self.axjump, 'Jump')
    self.bjump.on_clicked(self.jump_to_frame) 

    # Display the initial image
    self.display_image()

    # Connect key press events
    self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    plt.show()


  def try_load_save_frames(self):

    # Check if a config file exists in the output directory
    config_file = os.path.join(self.save_dir, self.config_filename)
    config = None
    if os.path.isfile(config_file):
      with open(config_file, 'r') as f:
        config = yaml.load(f)

    if config is not None:
      # Print gate info:
      gates = [x['frame_num'] for x in config['gates']]
      num_gates = config['num_gates']
      num_invalid = config['num_invalid_gates']
      print(f"Existing Config Found:\n\tGate Frames: {gates}\n\tNum Gates: {num_gates}\n\tNum Invalid: {num_invalid}")
      ret = input("Load Frames? [y/n]\n")
      if ret.strip().lower() == 'y':
        self.saved_frames = gates
        print("Loaded saved frames")


  def display_image(self):

    if self.viz_gates:
      image_path = os.path.join(self.image_dir, self.image_files[self.saved_frames[self.index]])
    else:
      image_path = os.path.join(self.image_dir, self.image_files[self.index])
    image = plt.imread(image_path)
    self.ax.imshow(image)

    # Check if the new value is different from the current index
    if int(self.slider.val) != self.index:
      self.slider.set_val(self.index)
    
    if self.viz_gates or self.index in self.saved_frames:
      self.fig.set_facecolor("green")
    else:
      self.fig.set_facecolor("white")
  
    if self.viz_gates:
      plt.title(f"Gate {self.index + 1}/{self.slider.valmax+1} [Image: {self.saved_frames[self.index] + 1}]")
    else:
      plt.title(f"Image {self.index + 1}/{self.slider.valmax+1}")
    plt.draw()

  def update_display(self):
    self.display_image()

  def prev_image(self, event):
    if self.index > 0:
      self.index -= 1
      self.update_display()

  def next_image(self, event):
    if self.index < self.slider.valmax:
      self.index += 1
      self.update_display()

  def scrub_image(self, val):
    self.index = int(val)
    self.update_display()

  def on_key_press(self, event):
    if event.key == 'left':
      self.prev_image(event)
    elif event.key == 'right':
      self.next_image(event)

  def jump_to_frame(self, text):
    try:
      target_frame = int(text)
      if 1 <= target_frame <= len(self.image_files):
        self.index = target_frame - 1
        self.update_display()
      else:
        print("Invalid frame number. Please enter a number between 1 and the total number of frames.")
    except ValueError:
      print("Invalid input. Please enter a valid frame number.")

  def save_frame(self, event):

    if self.viz_gates:
      print("Not saving gates while gate view is activated")
      return

    gate_frame = self.index
    if self.gate_evaluator.quick_validity_check(self.saved_frames, gate_frame):
      # Save the current frame to the internal array
      self.saved_frames.append(gate_frame)
      print(f"Frame {self.index + 1} saved.")
      self.update_display()
    self.print_saved_frames()

  def delete_last_frame(self, event):
    # Delete the last saved frame
    if self.saved_frames:
      del self.saved_frames[-1]
      print("Last saved frame deleted.")
    else:
      print("No saved frames to delete.")
    self.print_saved_frames()
    self.update_display()

  def print_saved_frames(self):
    # Print the list of saved frames
    frames = [x + 1 for x in self.saved_frames]
    print(f"Saved Frames [{len(frames)}]: {frames}")

  def view_gates(self, event):

    # Load State
    self.viz_gates = not self.viz_gates

    if self.viz_gates and len(self.saved_frames) == 0:
      self.viz_gates = False
      print("Not toggling viz to view gates: No gate frames currently saved")
      return

    # Update slider
    if self.viz_gates:
      self.slider.valmax = len(self.saved_frames) - 1
    else:
      self.slider.valmax = len(self.image_files) - 1

    # Update index
    tmp = self.index
    self.index = self.last_index
    self.last_index = tmp

    if self.viz_gates:
      print("Toggled Viz: Viewing Gate Frames")
    else:
      print("Toggled Viz: Viewing Image Frames")
      
    self.update_display()

  def export_gates(self, event):

    eval_data = []
    eval_data, num_invalid = self.gate_evaluator.full_validity_check(self.saved_frames)

    config = {
      "image_dir": self.image_dir,
      "matcher": vars(self.gate_evaluator.matcher.conf),
      "extractor": vars(self.gate_evaluator.extractor.conf),
      "gates": eval_data,
      "num_gates": len(eval_data),
      "num_invalid_gates": num_invalid,
      "train_batch_size": self.gate_evaluator.train_batch_size,
      "test_batch_size": self.gate_evaluator.test_batch_size,
      "feature_suppression_distance": self.gate_evaluator.feature_suppression_distance
    }

    print("Exporting Gate Data")
    print(yaml.dump(config))
    
    # Create save directory if doesn't exist
    if not os.path.exists(self.save_dir):
      os.makedirs(self.save_dir)

    # Save config as yaml file
    config_file = os.path.join(self.save_dir, self.config_filename)
    with open(config_file, 'w') as f:
      yaml.dump(config, f)

    print(f"Saved file to {config_file}")
    

