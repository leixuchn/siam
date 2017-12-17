#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Various transforms for video augmentation"""

import math
import random
import numbers

import collections
from PIL import Image, ImageOps

class Compose(object):
  """Composes several transforms together.
  Args:
      transforms (List[Transform]): list of transforms to compose.
  Example:
      >>> transforms.Compose([
      >>>     transforms.CenterCrop(10),
      >>>     transforms.ToTensor(),
      >>> ])
  """

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, example):
    for t in self.transforms:
      example = t(example)
    return example


class Map(object):
  def __init__(self, transform):
    self.transform = transform

  def __call__(self, example):
    return [self.transform(elem) for elem in example]


class IMap(object):
  def __init__(self, transform):
    self.transform = transform

  def __call__(self, example):
    new_example = []
    state = self.transform.getstate()
    for elem in example:
      self.transform.setstate(state)
      new_example.append(self.transform(elem))

    return new_example

class FixGray(object):
  def __init__(self, batch_size, gray_ratio=0.25):
    self.gray_ratio = gray_ratio
    self.batch_size = batch_size
    self.state = 0

  def getstate(self):
    return self.state

  def setstate(self, state):
    self.state = state

  def __call__(self, img):
    if self.state < int(self.gray_ratio * self.batch_size):
      L, = img.convert('L').split()
      img = Image.merge('RGB', [L, L, L])
    self.state = (self.state + 1) % self.batch_size
    return img

class RandomGray(object):
  def __init__(self, gray_ratio=0.25):
    self.gray_ratio = gray_ratio

  def getstate(self):
    return random.getstate()

  def setstate(self, state):
    random.setstate(state)

  def __call__(self, img):
    if random.random() < self.gray_ratio:
      L, = img.convert('L').split()
      img = Image.merge('RGB', [L, L, L])

    return img

class RandomStretch(object):
  def __init__(self, max_stretch=0.05, interpolation=Image.BILINEAR):
    self.max_stretch = max_stretch
    self.interpolation = interpolation

  def getstate(self):
    return random.getstate()

  def setstate(self, state):
    random.setstate(state)

  def __call__(self, img):
    scale = 1 + random.uniform(-self.max_stretch, self.max_stretch)
    w, h = img.size
    tw, th = int(round(w * scale)), int(round(h * scale))
    return img.resize((tw, th), self.interpolation)


class RandomResolution(object):
  def __init__(self, downsample_ratio=0.3, min_downsample=0.3, interpolation=Image.BILINEAR):
    self.downsample_ratio = downsample_ratio
    self.min_downsample = min_downsample
    self.interpolation = interpolation

  def getstate(self):
    return random.getstate()

  def setstate(self, state):
    random.setstate(state)

  def __call__(self, img):
    if random.random() < self.downsample_ratio:
      scale = random.uniform(self.min_downsample, 1.0)
      w, h = img.size
      tw, th = int(round(w * scale)), int(round(h * scale))
      # downsample
      img = img.resize((tw, th), self.interpolation)
      # upsample
      img = img.resize((w, h), self.interpolation)
    return img

class Scale(object):
  """Rescales the input PIL.Image to the given 'size'.
  If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
  If 'size' is a number, it will indicate the size of the smaller edge.
  For example, if height > width, then image will be
  rescaled to (size * height / width, size)
  size: size of the exactly size or the smaller edge
  interpolation: Default: PIL.Image.BILINEAR
  """

  def __init__(self, size, interpolation=Image.BILINEAR):
    assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
    self.size = size
    self.interpolation = interpolation

  def __call__(self, img):
    if isinstance(self.size, int):
      w, h = img.size
      if (w <= h and w == self.size) or (h <= w and h == self.size):
        return img
      if w < h:
        ow = self.size
        oh = int(self.size * h / w)
        return img.resize((ow, oh), self.interpolation)
      else:
        oh = self.size
        ow = int(self.size * w / h)
        return img.resize((ow, oh), self.interpolation)
    else:
      return img.resize(self.size, self.interpolation)


class CenterCrop(object):
  """Crops the given PIL.Image at the center to have a region of
  the given size. size can be a tuple (target_height, target_width)
  or an integer, in which case the target will be of a square shape (size, size)
  """

  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, img):
    w, h = img.size
    th, tw = self.size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomCrop(object):
  """Crops the given PIL.Image at a random location to have a region of
  the given size. size can be a tuple (target_height, target_width)
  or an integer, in which case the target will be of a square shape (size, size)
  """

  def __init__(self, size, padding=0):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size
    self.padding = padding

  def getstate(self):
    return random.getstate()

  def setstate(self, state):
    random.setstate(state)

  def __call__(self, img):
    if self.padding > 0:
      img = ImageOps.expand(img, border=self.padding, fill=0)

    w, h = img.size
    th, tw = self.size
    if w == tw and h == th:
      return img

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomSizedCrop(object):
  """
  size: size of the smaller edge
  interpolation: Default: PIL.Image.BILINEAR
  """

  def __init__(self, size, interpolation=Image.BILINEAR, area_range=(0.92, 1.0), aspect_range=(0.97, 1.03), max_attempts=100):
    self.size = size
    self.interpolation = interpolation
    self.area_range = area_range
    self.aspect_range = aspect_range
    self.max_attempts = max_attempts

  def getstate(self):
    return random.getstate()

  def setstate(self, state):
    random.setstate(state)

  def __call__(self, img):
    for attempt in range(self.max_attempts):
      area = img.size[0] * img.size[1]
      target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
      aspect_ratio = random.uniform(self.aspect_range[0], self.aspect_range[1])

      w = int(round(math.sqrt(target_area * aspect_ratio)))
      h = int(round(math.sqrt(target_area / aspect_ratio)))

      if random.random() < 0.5:
        w, h = h, w

      if w <= img.size[0] and h <= img.size[1]:
        x1 = random.randint(0, img.size[0] - w)
        y1 = random.randint(0, img.size[1] - h)

        img = img.crop((x1, y1, x1 + w, y1 + h))
        print('{},{},{},{}'.format(x1, y1, x1 + w, y1 + w))
        assert (img.size == (w, h))
        return img.resize((self.size, self.size), self.interpolation)

    # Fallback
    crop = RandomCrop(self.size)
    return crop(img)