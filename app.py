import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

torch.set_grad_enabled(False) # dissable gradient computation since we are not training the model

# Load extractor and matcher
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)


images_path = Path("assets")

# Transform image into a PyTorch Tensor, for LightGlue Neural Network
image0 = load_image(images_path / "group4_1.jpg")
image1 = load_image(images_path / "group4_2.jpg")

# Display images to ensure they are loaded correctly
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
# Adjust the dimensions from C,H,W to H,W,C
plt.imshow(image0.permute(1, 2, 0).cpu().numpy())  
plt.title('Image 0')
plt.subplot(1, 2, 2)
plt.imshow(image1.permute(1, 2, 0).cpu().numpy())
plt.title('Image 1')
plt.savefig("assets/images/loaded_images.png")
plt.show()

# viz2d.plot_images([image0, image1])


# Extract features from both images
# Extractor is superpoint, this one extract the features from the images
feats0 = extractor.extract(image0.to(device)) # .to(device) moves the tensor to the device (CPU/GPU)
# Both the neural network and the input data needs to be in the same device
feats1 = extractor.extract(image1.to(device))

# Extractor Neural Network adds the batch dimension
matches01 = matcher({"image0": feats0, "image1": feats1}) # This is lightglue matching the features
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]] # Remove batch dimension [1, C, H, W] into [C, H, W]

# Extract keypoints and matches
kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"] # Matches are align based on the keypoint's descriptors
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]] # slices the first column of the matches array and retrive the cordinates of the keypoint

# Display the number of matches
print(f'Number of matches: {matches.shape[0]}') # Returns the amount of rows for this array

# Visualize keypoints
kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"]) # Visualize the points that were pruned
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
plt.savefig("assets/images/fetures.png")
print(kpts0.shape[0])
print(kpts1.shape[0])

#Visualize the matches
axes = viz2d.plot_images([image0, image1]) # It allows to visualize 2D data
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2) # Visualizes the matches
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
viz2d.add_text(1, "franck somarriba", fs=20)
plt.savefig("assets/images/matched_keypoints.png")


#Show images
plt.show()

