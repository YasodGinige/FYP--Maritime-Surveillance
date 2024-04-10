import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import sys


Images=np.load('anchorImages.npy',allow_pickle=True).reshape(-1,192,192,3)
features=np.load('anchor_grads.npy',allow_pickle=True).reshape(-1,192,192,3)
print(features.shape)
#sys.exit()
features=np.abs(features)
max_val=np.max(features)
features=features*255/max_val


num_images=2
indexes=random.sample(range(0, len(features)-1), num_images)


selected_images=Images[indexes]
selected_features=features[indexes]
print(selected_features[0])
plt.imshow(selected_features[0])
plt.savefig("f_image.png")

print('.')
plt.imshow(selected_images[0])

print("####### plot images and features #########")
# for i in range(1,num_images+1):
#     plt.subplot(num_images, 2, (i-1)*2+1)
#     plt.imshow(selected_images[i-1])

#     plt.subplot( num_images, 2, (i-1)*2+2)
#     plt.imshow(selected_features[i-1])

# plt.show()
plt.savefig("image.png")
