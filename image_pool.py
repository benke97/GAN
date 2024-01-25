import random
import torch


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size, image_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
            image_size (tuple) -- the size of each image (C, H, W)
        """
        self.pool_size = pool_size
        self.image_size = image_size
        if self.pool_size > 0:
            self.images = torch.empty((pool_size, *image_size))
            self.num_imgs = 0

    def query(self, images):
        if True:
            return images

        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        
        return_images = []
        for image in images:
            if self.num_imgs < self.pool_size:
                self.images[self.num_imgs] = image.data
                self.num_imgs += 1
                return_images.append(image.data)
            else:
                p = random.uniform(0, 1)
                if p > 1:  # Adjust this threshold as needed
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image.data
                    return_images.append(tmp)
                else:
                    return_images.append(image.data)

        return torch.stack(return_images)