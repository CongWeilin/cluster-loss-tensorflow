import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import manifold
import pickle

feature_embeddings = np.load('feature_embedding.npy') 

'''images is cifar test set, shaped (10000,32,32,3)'''
'''https://psu.box.com/s/urgcz6ikanw6mbaxin2vws4fvthw73jw'''
images = np.load('cifar.npy') 

# use tsne to cluster images in 2 dimensions
tsne = manifold.TSNE()
reduced = tsne.fit_transform(feature_embeddings)
reduced_transformed = reduced - np.min(reduced, axis=0)
reduced_transformed /= np.max(reduced_transformed, axis=0)
image_xindex_sorted = np.argsort(np.sum(reduced_transformed, axis=1))

plot_number=10000
image_width=32
merged_width = int(np.ceil(np.sqrt(plot_number))*image_width)
merged_image = np.zeros((merged_width, merged_width,3))

for counter, index in enumerate(image_xindex_sorted):
    # set location
    a = np.ceil(reduced_transformed[counter, 0] * (merged_width-image_width-1)+1)
    b = np.ceil(reduced_transformed[counter, 1] * (merged_width-image_width-1)+1)
    a = int(a - np.mod(a-1,image_width) + 1)
    b = int(b - np.mod(b-1,image_width) + 1)

    img = images[counter]
    merged_image[a:a+image_width, b:b+image_width,:] = img
    
plt.imshow(merged_image)
plt.show()
plt.imsave('tsne.jpg',merged_image)