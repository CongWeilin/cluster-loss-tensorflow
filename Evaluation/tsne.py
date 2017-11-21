import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import datasets, manifold

digits = datasets.load_digits(n_class=5)
X = digits.data
y = digits.target

# use tsne to cluster images in 2 dimensions
tsne = manifold.TSNE()
reduced = tsne.fit_transform(X)
reduced_transformed = reduced - np.min(reduced, axis=0)
reduced_transformed /= np.max(reduced_transformed, axis=0)
image_xindex_sorted = np.argsort(np.sum(reduced_transformed, axis=1))

# drawing all images in a merged image
plot_number=X.shape[0]
image_width = 8

merged_width = int(np.ceil(np.sqrt(plot_number))*image_width)
merged_image = np.zeros((merged_width, merged_width))

for counter, index in enumerate(image_xindex_sorted):
    # set location
    a = np.ceil(reduced_transformed[counter, 0] * (merged_width-image_width-1)+1)
    b = np.ceil(reduced_transformed[counter, 1] * (merged_width-image_width-1)+1)
    a = int(a - np.mod(a-1,image_width) + 1)
    b = int(b - np.mod(b-1,image_width) + 1)

    img = X[counter].reshape((image_width, image_width))
    merged_image[a:a+image_width, b:b+image_width] = img

plt.imshow(merged_image, cmap ='gray')
plt.show()