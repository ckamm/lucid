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

"""Linear algebra helpers that create matrix operators/transformations."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

def _rotation(angleInRadians):
    cos = tf.cos(angleInRadians)
    sin = tf.sin(angleInRadians)
    return tf.convert_to_tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=tf.float32)

def _scaling(angle, x, y):
    rotate = _rotation(angle)
    unrotate = _rotation(-angle)
    basescale = tf.convert_to_tensor([[x, 0, 0], [0, y, 0], [0, 0, 1]], dtype=tf.float32)
    return tf.matmul(unrotate, tf.matmul(basescale, rotate))

def _translation(x, y):
    return tf.convert_to_tensor([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=tf.float32)

def _projection(x, y):
    return tf.convert_to_tensor([[1, 0, 0], [0, 1, 0], [x, y, 1]], dtype=tf.float32)

def _homography(
    initial_translation, rotation, scaling, projection, final_translation, shape
):
    center = _translation(shape[0] // 2, shape[1] // 2)
    uncenter = _translation(-shape[0] // 2, -shape[1] // 2)

    return (
        tf.matmul(final_translation,
        tf.matmul(center,
        tf.matmul(projection,
        tf.matmul(rotation,
        tf.matmul(scaling,
        tf.matmul(uncenter,
                  initial_translation))))))
    )


def _parameterized_flattened_homography(
    translation1_x,
    translation1_y,
    rotationAngleInRadians,
    scalingAngleInRadians,
    scaling_x,
    scaling_y,
    vanishing_point_x,
    vanishing_point_y,
    translation2_x,
    translation2_y,
    shape_xy,
):
    initial_translate = _translation(translation1_x, translation1_y)
    rotate = _rotation(rotationAngleInRadians)
    scale = _scaling(scalingAngleInRadians, scaling_x, scaling_y)
    project = _projection(vanishing_point_x, vanishing_point_y)
    final_translate = _translation(translation2_x, translation2_y)
    matrix = _homography(
        initial_translate, rotate, scale, project, final_translate, shape_xy
    )

    #
    # conform to tf.contrib.image.transform interface
    #

    # in image.transform the first index is the y coordinate
    # TODO: Maybe this ordering is normal for our inputs too?
    flip = tf.constant([0, 1, 0, 1, 0, 0, 0, 0, 1], shape=(3, 3), dtype=tf.float32)
    matrix = tf.matmul(flip, tf.matmul(matrix, flip))

    # invert, since actually the inverse transformation will be done
    matrix = tf.linalg.inv(matrix)

    # it expects the lower right corner to be 1 - the transformation is
    # invariant to scalar multiplication so just divide by it
    matrix = matrix / matrix[2, 2]

    # and select the 8 values it needs
    return tf.to_float(tf.reshape(matrix, [-1])[:8])
