from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def preprocess_image(image, height, width,
                     is_training=False,
                     bbox=None,
                     fast_mode=True,
                     normalize=False):
  image = tf.image.resize_image_with_crop_or_pad(image, height, width)
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  tf.summary.image('input_image', tf.expand_dims(image, 0))
  if normalize:
    # Normalize to range [-1, 1]
    image = 2.0 * (image - 0.5)
  else:
    # Recover original image dtype
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.to_float(image)
  return image