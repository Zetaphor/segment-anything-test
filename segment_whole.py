import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

model_paths = {
  'vit_h': 'models/sam_vit_h_4b8939.pth',
  'vit_l': 'models/sam_vit_l_0b3195.pth',
  'vit_b': 'models/sam_vit_b_01ec64.pth'
}

MODEL_TYPE = 'vit_h'
IMAGE = './test_images/wren.png'
PROMPT = 'character'

image = cv2.imread(IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

device = "cuda"

sam = sam_model_registry[MODEL_TYPE](checkpoint=model_paths[MODEL_TYPE])
sam.to(device=device)

# Mask whole image
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

def generate_mask_image(anns):
    if len(anns) == 0:
        return np.zeros_like(image)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    mask_img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    mask_img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        mask_img[m] = color_mask
    return mask_img

mask_image = generate_mask_image(masks)

plt.figure(figsize=(20,20))
plt.imshow(mask_image)
plt.axis('off')

# Save the figure to a file
plt.savefig(f"masks_image-{MODEL_TYPE}.png", bbox_inches='tight', pad_inches=0, dpi=300)

# Close the figure after saving to free up memory
plt.close()
