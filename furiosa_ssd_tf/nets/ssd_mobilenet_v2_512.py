# Copyright 2016 Paul Balanca. All Rights Reserved.
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
"""Definition of 512 VGG-based SSD network.

This model was initially introduced in:
SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325

Two variants of the model are defined: the 300x300 and 512x512 models, the
latter obtaining a slightly better accuracy on Pascal VOC.

Usage:
    with slim.arg_scope(ssd_vgg.ssd_vgg()):
        outputs, end_points = ssd_vgg.ssd_vgg(inputs)
@@ssd_vgg
"""
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common
from nets.ssd_mobilenet_v2_512_ops import depthwise_conv2d as tf_layers_depthwise_conv2d

slim = tf.contrib.slim

# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    """Implementation of the SSD VGG-based 512 network.

    The default features layers with 512x512 image input are:
      conv4 ==> 64 x 64
      conv7 ==> 32 x 32
      conv8 ==> 16 x 16
      conv9 ==> 8 x 8
      conv10 ==> 4 x 4
      conv11 ==> 2 x 2
      conv12 ==> 1 x 1
    The default image size used to train this network is 512x512.
    """
    default_params = SSDParams(
        img_shape=(512, 512),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['expanded_conv_14/expansion_output', 'conv_2/output', 'expanded_conv_18/output',
                     'expanded_conv_19/output', 'expanded_conv_20/output', 'expanded_conv_21/output'],
        feat_shapes=[(32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
        anchor_size_bounds=[0.10, 0.90],
        anchor_sizes=[(20.48, 51.2),
                      (51.2, 133.12),
                      (133.12, 215.04),
                      (215.04, 296.96),
                      (296.96, 378.88),
                      (378.88, 460.8),
                      (460.8, 542.72)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1. / 3],
                       [2, .5, 3, 1. / 3],
                       [2, .5, 3, 1. / 3],
                       [2, .5, 3, 1. / 3],
                       [2, .5, 3, 1. / 3],
                       [2, .5, 3, 1. / 3]],
        anchor_steps=[16, 32, 64, 128, 256, 512],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
    )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_512_mobilenet_v2'):
        """Network definition.
        """
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    # ======================================================================= #
    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        # if clipping_bbox is not None:
        #     rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def layer_shape(layer):
    """Returns the dimensions of a 4D layer tensor.
    Args:
      layer: A 4-D Tensor of shape `[height, width, channels]`.
    Returns:
      Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if layer.get_shape().is_fully_defined():
        return layer.get_shape().as_list()
    else:
        static_shape = layer.get_shape().with_rank(4).as_list()
        dynamic_shape = tf.unstack(tf.shape(layer), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(512, 512)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (512 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * 0.04, img_size * 0.1]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        shape = l.get_shape().as_list()[1:4]
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors,), dtype=dtype)
    w = np.zeros((num_anchors,), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i + di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i + di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based SSD 512.
# =========================================================================== #
def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.nn.softmax,
            data_format='channels_last',
            reuse=None,
            scope='SSD_MobilenetV2'):
    """SSD net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'SSD_MobilenetV2', [inputs], reuse=reuse):
        with tf.variable_scope('MobilenetV2_base'):
            net = conv_bn_block(inputs=inputs, filters=32, kernel_size=3, strides=(2, 2),
                                name='conv_1', end_points=end_points,
                                data_format=data_format)

            reuse = True if 'ssd_net' in locals() else None
            is_training = False

            # expanded conv 16 x 1
            net = inverted_bottleneck_conv_bn_block(inputs=net, expansion_ratio=1,
                                                    reuse=reuse, is_training=is_training,
                                                    out_channels=16, subsample=False, name='expanded_conv_1',
                                                    end_points=end_points,
                                                    data_format=data_format)

            expansion_ratio = 6
            # expanded conv 24 x 2
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    24, True, name='expanded_conv_2', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    24, False, name='expanded_conv_3', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            # expanded conv 32 x 3
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    32, True, name='expanded_conv_4', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    32, False, name='expanded_conv_5', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    32, False, name='expanded_conv_6', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)

            # expanded conv 64 x 4
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    64, True, name='expanded_conv_7', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    64, False, name='expanded_conv_8', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    64, False, name='expanded_conv_9', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    64, False, name='expanded_conv_10', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)

            # expanded conv 96 x 3
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    96, False, name='expanded_conv_11', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    96, False, name='expanded_conv_12', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    96, False, name='expanded_conv_13', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)

            # expanded conv 160 x 3
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    160, True, name='expanded_conv_14', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    160, False, name='expanded_conv_15', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    160, False, name='expanded_conv_16', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)

            # expanded conv 320 x 1
            net = inverted_bottleneck_conv_bn_block(net, expansion_ratio,
                                                    320, False, name='expanded_conv_17', end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)

            net = conv_bn_block(net, 1280, 1, (1, 1), 'conv_2', end_points=end_points, data_format=data_format)

        # add extra feature layers
        with tf.variable_scope('Extra_feature_layers'):
            net = inverted_bottleneck_conv_bn_block(inputs=net, expansion_ratio=0.2,
                                                    out_channels=512, subsample=True,
                                                    name='expanded_conv_18',
                                                    extra_layer=True, end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, 0.25,
                                                    256, True, 'expanded_conv_19', extra_layer=True,
                                                    end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, 0.5,
                                                    256, True, 'expanded_conv_20', extra_layer=True,
                                                    end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)
            net = inverted_bottleneck_conv_bn_block(net, 0.25,
                                                    128, True, 'expanded_conv_21', extra_layer=True,
                                                    end_points=end_points,
                                                    reuse=reuse, is_training=is_training,
                                                    data_format=data_format)

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        with tf.variable_scope('Multibox_headers'):
            for i, layer in enumerate(feat_layers):
                p, l = multibox_head(end_points[layer],
                                     num_classes,
                                     anchor_sizes[i],
                                     anchor_ratios[i],
                                     ind=i, data_format=data_format)
                predictions.append(prediction_fn(p))
                logits.append(p)
                localisations.append(l)

    return predictions, localisations, logits, end_points


ssd_net.default_image_size = 512

BN_MOMENTUM = 0.9
BN_EPSILON = 1e-5
USE_FUSED_BN = True

_conv_bn_initializer = tf.glorot_uniform_initializer


def inverted_bottleneck_conv_bn_block(inputs, expansion_ratio, out_channels,
                                      subsample, name, end_points, data_format,
                                      reuse, is_training, extra_layer=False):
    _bn_axis = -1 if data_format == 'channels_last' else 1

    with tf.variable_scope(name):

        in_name = 'input'
        inputs = tf.identity(inputs, name=in_name)

        layer = name + '/' + in_name
        end_points[layer] = inputs

        scope_name1 = 'expand'
        scope_name2 = 'depthwise'
        scope_name3 = 'linear_project'

        if extra_layer:
            scope_name1 = 'shrink'
            scope_name3 = 'nonlinear_project'

        if data_format == 'channels_first':
            in_channels = inputs.get_shape().as_list()[1]
        else:
            in_channels = inputs.get_shape().as_list()[-1]

        if expansion_ratio != 1:
            with tf.variable_scope(scope_name1):

                outputs = tf.layers.conv2d(inputs=inputs, filters=int(expansion_ratio * in_channels),
                                           kernel_size=1, strides=(1, 1), padding='same',
                                           data_format=data_format, activation=None, use_bias=False,
                                           kernel_initializer=_conv_bn_initializer(),
                                           bias_initializer=None,
                                           #                                                 name='{}_1'.format(name),
                                           reuse=reuse)
                outputs = tf.layers.batch_normalization(inputs=outputs, axis=_bn_axis,
                                                        momentum=BN_MOMENTUM, epsilon=BN_EPSILON, fused=USE_FUSED_BN,
                                                        name='BatchNorm'.format(name), reuse=reuse,
                                                        training=is_training)
                outputs = tf.nn.relu6(features=outputs, name='Relu6'.format(name))

            if extra_layer:
                out_name = 'shrink_output'
            else:
                out_name = 'expansion_output'

            outputs = tf.identity(outputs, name=out_name)
            layer = name + '/' + out_name
            end_points[layer] = outputs
        else:
            outputs = inputs

        with tf.variable_scope(scope_name2):

            strides = (2, 2) if subsample else (1, 1)

            outputs = tf_layers_depthwise_conv2d(inputs=outputs,
                                                 kernel_size=3, strides=strides, padding='same',
                                                 data_format=data_format, activation=None, use_bias=False,
                                                 depthwise_initializer=_conv_bn_initializer(),
                                                 bias_initializer=None,
                                                 #                                                       name='{}_2'.format(name),
                                                 reuse=reuse)

            outputs = tf.layers.batch_normalization(inputs=outputs, axis=_bn_axis,
                                                    momentum=BN_MOMENTUM, epsilon=BN_EPSILON, fused=USE_FUSED_BN,
                                                    name='BatchNorm'.format(name), reuse=reuse,
                                                    training=is_training)

            outputs = tf.nn.relu6(features=outputs, name='Relu6'.format(name))

        out_name = 'depthwise_output'
        outputs = tf.identity(outputs, name=out_name)
        layer = name + '/' + out_name
        end_points[layer] = outputs

        with tf.variable_scope(scope_name3):

            outputs = tf.layers.conv2d(inputs=outputs, filters=out_channels,
                                       kernel_size=1, strides=(1, 1), padding='same',
                                       data_format=data_format, activation=None, use_bias=False,
                                       kernel_initializer=_conv_bn_initializer(),
                                       bias_initializer=None,
                                       #                                             name='{}_3'.format(name),
                                       reuse=reuse)

            outputs = tf.layers.batch_normalization(inputs=outputs, axis=_bn_axis,
                                                    momentum=BN_MOMENTUM, epsilon=BN_EPSILON, fused=USE_FUSED_BN,
                                                    name='BatchNorm'.format(name), reuse=reuse,
                                                    training=is_training)
            if extra_layer:
                outputs = tf.nn.relu6(features=outputs, name='Relu6'.format(name))
            else:
                outputs = tf.identity(outputs, name='Identity')

        is_residual = in_channels == out_channels

        if is_residual and not subsample:
            outputs = tf.add(inputs, outputs)

        out_name = 'output'
        outputs = tf.identity(outputs, name=out_name)
        layer = name + '/' + out_name
        end_points[layer] = outputs

        return outputs


def conv_bn_block(inputs, filters, kernel_size, strides, name,
                  end_points, data_format=None, reuse=None, is_training=None):
    _bn_axis = -1 if data_format == 'channels_last' else 1

    with tf.variable_scope(name):
        in_name = 'input'
        inputs = tf.identity(inputs, name=in_name)

        layer = name + '/' + in_name
        end_points[layer] = inputs

        outputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                   padding='same',
                                   data_format=data_format, activation=None, use_bias=False,
                                   kernel_initializer=_conv_bn_initializer(),
                                   bias_initializer=None,
                                   #                                         name='{}_1'.format(name),
                                   reuse=reuse)

        outputs = tf.layers.batch_normalization(inputs=outputs, axis=_bn_axis,
                                                momentum=BN_MOMENTUM, epsilon=BN_EPSILON, fused=USE_FUSED_BN,
                                                name='BatchNorm'.format(name), reuse=reuse,
                                                training=is_training)

        outputs = tf.nn.relu6(features=outputs, name='Relu6'.format(name))

        out_name = 'output'
        outputs = tf.identity(outputs, name=out_name)

        layer = name + '/' + out_name
        end_points[layer] = outputs

    return outputs


def multibox_head(inputs, num_classes, sizes, ratios=[1], data_format='channels_last', ind=None):
    net = inputs

    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    with tf.variable_scope('Box_predictor_{}'.format(ind + 1)):
        # Location.
        num_loc_pred = num_anchors * 4
        loc_pred = tf.layers.conv2d(net, num_loc_pred, (3, 3), use_bias=True,
                                    name='loc_predictor_{}'.format(ind + 1), strides=(1, 1),
                                    padding='same', data_format=data_format, activation=None,
                                    kernel_initializer=tf.glorot_uniform_initializer(),
                                    bias_initializer=tf.zeros_initializer())
        loc_pred = tf.reshape(loc_pred, tensor_shape(loc_pred, 4)[:-1] + [num_anchors, 4])

        # Class prediction.
        num_cls_pred = num_anchors * num_classes

        cls_pred = tf.layers.conv2d(net, num_cls_pred, (3, 3), use_bias=True,
                                    name='cls_predictor_{}'.format(ind + 1), strides=(1, 1),
                                    padding='same', data_format=data_format, activation=None,
                                    kernel_initializer=tf.glorot_uniform_initializer(),
                                    bias_initializer=tf.zeros_initializer())
        cls_pred = tf.reshape(cls_pred, tensor_shape(cls_pred, 4)[:-1] + [num_anchors, num_classes])
    return cls_pred, loc_pred


def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    with tf.name_scope(scope, 'ssd_losses'):
        l_cross_pos = []
        l_cross_neg = []
        l_loc = []
        for i in range(len(logits)):
            dtype = logits[i].dtype
            with tf.name_scope('block_%i' % i):
                # Determine weights Tensor.
                pmask = gscores[i] > match_threshold
                fpmask = tf.cast(pmask, dtype)
                n_positives = tf.reduce_sum(fpmask)

                # Select some random negative entries.
                # n_entries = np.prod(gclasses[i].get_shape().as_list())
                # r_positive = n_positives / n_entries
                # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                # Negative mask.
                no_classes = tf.cast(pmask, tf.int32)
                predictions = slim.softmax(logits[i])
                nmask = tf.logical_and(tf.logical_not(pmask),
                                       gscores[i] > -0.5)
                fnmask = tf.cast(nmask, dtype)
                nvalues = tf.where(nmask,
                                   predictions[:, :, :, :, 0],
                                   1. - fnmask)
                nvalues_flat = tf.reshape(nvalues, [-1])
                # Number of negative entries to select.
                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
                n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
                max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                n_neg = tf.minimum(n_neg, max_neg_entries)

                val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                minval = val[-1]
                # Final negative mask.
                nmask = tf.logical_and(nmask, -nvalues > minval)
                fnmask = tf.cast(nmask, dtype)

                # Add cross-entropy loss.
                with tf.name_scope('cross_entropy_pos'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=gclasses[i])
                    loss = tf.losses.compute_weighted_loss(loss, fpmask)
                    l_cross_pos.append(loss)

                with tf.name_scope('cross_entropy_neg'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=no_classes)
                    loss = tf.losses.compute_weighted_loss(loss, fnmask)
                    l_cross_neg.append(loss)

                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * fpmask, axis=-1)
                    loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                    loss = tf.losses.compute_weighted_loss(loss, weights)
                    l_loc.append(loss)

        # Additional total losses...
        with tf.name_scope('total'):
            total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
            total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
            total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
            total_loc = tf.add_n(l_loc, 'localization')

            # Add to EXTRA LOSSES TF.collection
            tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
            tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
            tf.add_to_collection('EXTRA_LOSSES', total_cross)
            tf.add_to_collection('EXTRA_LOSSES', total_loc)
