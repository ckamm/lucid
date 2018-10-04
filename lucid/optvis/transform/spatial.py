# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tranformations you might want neural net visualizations to be robust to.

This module provides a variety of functions which stochastically transform a
tensorflow tensor. The functions are of the form:

  (config) => (tensor) => (stochastic transformed tensor)

"""

import tensorflow as tf
import numpy as np
import uuid

from lucid.optvis import param
from lucid.optvis.transform.utils import compose, angle2rads, rand_select
from lucid.optvis.transform.operators import _parameterized_flattened_homography


def pad(w, mode="REFLECT", constant_value=0.5):
    def inner(image_t):
        if constant_value == "uniform":
            constant_value_ = tf.random_uniform([], 0, 1)
        else:
            constant_value_ = constant_value
        return tf.pad(
            image_t,
            [(0, 0), (w, w), (w, w), (0, 0)],
            mode=mode,
            constant_values=constant_value_,
        )

    return inner


def crop_or_pad_to_shape(target_shape):
    """Ensures output has a specified shape.
    Will crop down or enlarge the passed tensor, but not scale its contents.
    May extend in future to allow REFLECT padding, for now use built in.

    Args:
      target_shape: scalar or tuple defining either the square length or the
        [x,y] shape that the result should have
    """
    if len(target_shape) != 2:
        raise ValueError("Target shape must be a list of length 2: [w,h].")

    for value in target_shape:
        if value <= 0:
            raise ValueError("Target width or height must be positive.")

    def inner(image_t):
        return tf.image.resize_image_with_crop_or_pad(image_t, *target_shape)

    return inner


def jitter(d, seed=None):
    def inner(image_t):
        image_t = tf.convert_to_tensor(image_t, preferred_dtype=tf.float32)
        t_shp = tf.shape(image_t)
        crop_shape = tf.concat([t_shp[:-3], t_shp[-3:-1] - d, t_shp[-1:]], 0)
        crop = tf.random_crop(image_t, crop_shape, seed=seed)
        shp = image_t.get_shape().as_list()
        mid_shp_changed = [
            shp[-3] - d if shp[-3] is not None else None,
            shp[-2] - d if shp[-3] is not None else None,
        ]
        crop.set_shape(shp[:-3] + mid_shp_changed + shp[-1:])
        return crop

    return inner


# 2D only version
def scale(scales, seed=None):
    def inner(image_t):
        image_t = tf.convert_to_tensor(image_t, preferred_dtype=tf.float32)
        scale = rand_select(scales, seed=seed)
        shp = tf.shape(image_t)
        scale_shape = tf.cast(scale * tf.cast(shp[-3:-1], "float32"), "int32")
        return tf.image.resize_bilinear(image_t, scale_shape)

    return inner


def rotate(angles, units="degrees", interpolation="BILINEAR", seed=None):
    def inner(image_t):
        image_t = tf.convert_to_tensor(image_t, preferred_dtype=tf.float32)
        angle = rand_select(angles, seed=seed)
        angle_rad = angle2rads(angle, units)
        return tf.contrib.image.rotate(image_t, angle_rad, interpolation=interpolation)

    return inner

def default_homography_parameters(shape, seed=None):
    """Returns parameters for homography() that create a random transform
    that usually results in only slight adjustment of the image"""
    d = np.max(shape)
    return dict(
      translation1_x = tf.truncated_normal([], stddev=3, seed=seed),
      translation1_y = tf.truncated_normal([], stddev=3, seed=seed),
      rotationAngleInRadians = angle2rads(tf.truncated_normal([], stddev=2.5, seed=seed)),
      shearingAngleInRadians = angle2rads(tf.truncated_normal([], stddev=2.5, seed=seed)),
      shear_x = tf.truncated_normal([], stddev=1e-2, seed=seed),
      shear_y = tf.truncated_normal([], stddev=1e-2, seed=seed),

      # intuition for vanishing_point:
      # - values above roughly 1/(size/2) make no sense because one edge of the
      #   image will be blown away to infinity
      # - values below roughly 1/(size/2)/(d/2) will displace pixels by
      #   around one unit
      vanishing_point_x = tf.truncated_normal([], stddev=1.0/(shape[0]/2)/(d/2), seed=seed),
      vanishing_point_y = tf.truncated_normal([], stddev=1.0/(shape[1]/2)/(d/2), seed=seed),

      translation2_x = tf.truncated_normal([], stddev=2, seed=seed),
      translation2_y = tf.truncated_normal([], stddev=2, seed=seed),
    )

def null_homography_parameters():
    """Returns parameters for homography() that make it an identity-transform"""
    return dict(
      translation1_x = 0.0,
      translation1_y = 0.0,
      rotationAngleInRadians = 0.0,
      shearingAngleInRadians = 0.0,
      shear_x = 0.0,
      shear_y = 0.0,
      vanishing_point_x = 0.0,
      vanishing_point_y = 0.0,
      translation2_x = 0.0,
      translation2_y = 0.0,
    )

def homography(parameters=None, seed=None, interpolation="BILINEAR"):
    """Most general 2D transform that can replace all our spatial transforms.
    Consists of an affine transformation + a perspective projection.

    By default, when parameters is None, a random homography based on
    default_homography_parameters() is performed.
    """

    def inner(image_t):
        shape_xy = tf.shape(image_t)[1:3]
        if parameters is None:
          parameters = default_homography_parameters(shape_xy, seed)

        transform_t = tf.py_func(
            _parameterized_flattened_homography,
            [
                parameters['translation1_x'],
                parameters['translation1_y'],
                parameters['rotationAngleInRadians'],
                parameters['shearingAngleInRadians'],
                parameters['shear_x'],
                parameters['shear_y'],
                parameters['vanishing_point_x'],
                parameters['vanishing_point_y'],
                parameters['translation2_x'],
                parameters['translation2_y'],
                shape_xy,
            ],
            [tf.float32],
            stateful=False,
        )[0]
        transform_t.set_shape([8])
        # print(transform_t.eval())
        # print([t.eval() for t in result])
        transformed_t = tf.contrib.image.transform(
            image_t, transform_t, interpolation=interpolation
        )
        return transformed_t

    return inner
