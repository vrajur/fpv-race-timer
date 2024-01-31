from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image
from time import time
from pathlib import Path
import torch
images = Path('/home/vinay/git/metazoic/external/LightGlue/assets')
device=torch.device('cuda')

extractor = SuperPoint(max_num_keypoints=256, detection_threshold=0.0).eval().to(device)  # load the extractor
match_conf = {
    'width_confidence': -1,  # for point pruning
    'depth_confidence': -1,  # for early stopping,
    'n_layers': 5,
    'flash': True,
}
matcher = LightGlue(features='superpoint', **match_conf).eval().to(device)

image0 = load_image(images / 'DSC_0411.JPG')
image1 = load_image(images / 'DSC_0410.JPG')

# batched
n = 30
feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))
for key in feats0.keys():
    feats0[key] = torch.cat([feats0[key]] * n)
    feats1[key] = torch.cat([feats1[key]] * n)
for i in range(10):
    last_time=time()
    with torch.inference_mode():
        pred1 = matcher({'image0': feats0, 'image1': feats1})
    current_time=time()
    print(current_time-last_time)


# non-batched
feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))
for i in range(10):
    torch.cuda.synchronize()
    last_time=time()
    preds = []
    for i in range(n):
        t = time()
        with torch.inference_mode():
            preds.append(matcher({'image0': feats0, 'image1': feats1}))
    torch.cuda.synchronize()
    current_time=time()
    print(current_time-last_time)
