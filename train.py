"""
Simple U-Net implementation in TensorFlow

Objective: detect vehicles

y = f(X)

X: image (256, 256, 3)
y: mask (256, 256, 1)
   - binary image
   - background is masked 0
   - vehicle is masked 255

Loss function: maximize IOU

    (intersection of prediction & grount truth)
    -------------------------------
    (union of prediction & ground truth)

Notes:
    In the paper, the pixel-wise softmax was used.
    But, I used the IOU because the datasets I used are
    not labeled for segmentations

Original Paper:
    https://arxiv.org/abs/1505.04597
"""
import time
import os
import pandas as pd
import tensorflow as tf
from utils import decode_labels, decode_images

def image_augmentation(image, mask):
    """Returns (maybe) augmented images

    (1) Random flip (left <--> right)
    (2) Random flip (up <--> down)
    (3) Random brightness
    (4) Random hue

    Args:
        image (3-D Tensor): Image tensor of (H, W, C)
        mask (3-D Tensor): Mask image tensor of (H, W, 1)

    Returns:
        image: Maybe augmented image (same shape as input `image`)
        mask: Maybe augmented mask (same shape as input `mask`)
    """
    concat_image = tf.concat([image, mask], axis=-1)

    maybe_flipped = tf.image.random_flip_left_right(concat_image)
    maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)

    image = maybe_flipped[:, :, :-1]
    mask = maybe_flipped[:, :, -1:]

    # image = tf.image.random_brightness(image, 0.7)
    # image = tf.image.random_hue(image, 0.3)

    return image, mask


def get_image_mask(queue, augmentation=True):
    """Returns `image` and `mask`

    Input pipeline:
        Queue -> CSV -> FileRead -> Decode JPEG

    (1) Queue contains a CSV filename
    (2) Text Reader opens the CSV
        CSV file contains two columns
        ["path/to/image.jpg", "path/to/mask.jpg"]
    (3) File Reader opens both files
    (4) Decode JPEG to tensors

    Notes:
        height, width = 256, 256

    Returns
        image (3-D Tensor): (256, 256, 3)
        mask (3-D Tensor): (256, 256, 1)
    """
    text_reader = tf.TextLineReader(skip_header_lines=0)
    _, csv_content = text_reader.read(queue)

    image_path, mask_path = tf.decode_csv(
        csv_content, record_defaults=[[""], [""]])

    image_file = tf.read_file(image_path)
    mask_file = tf.read_file(mask_path)

    image = tf.image.decode_jpeg(image_file, channels=3)
    image.set_shape([256, 256, 3])
    image = tf.cast(image, tf.float32)

    mask = tf.image.decode_jpeg(mask_file, channels=1)
    mask.set_shape([256, 256, 1])
    mask = tf.cast(mask, tf.float32)
    mask = mask / (tf.reduce_max(mask) + 1e-7)

    if augmentation:
        image, mask = image_augmentation(image, mask)

    # image = tf.image.per_image_standardization(image)

    return image, mask


def conv_conv_pool(input_,
                   n_filters,
                   training,
                   flags,
                   name,
                   pool=True,
                   activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
                name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upconv_concat(inputA, input_B, n_filter, flags, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D(inputA, n_filter, flags, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))


def upconv_2D(tensor, n_filter, flags, name):
    """Up Convolution `tensor` by 2 times

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations

    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
        name="upsample_{}".format(name))


def make_unet(X, training, flags=None):
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    # net = X / 127.5 - 1
    mean, _ = tf.nn.moments(X, [1, 2], keep_dims=True)
    net = tf.subtract(X, mean)

    conv1, pool1 = conv_conv_pool(net, [8, 8], training, flags, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, flags, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, flags, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, flags, name=4)
    conv5 = conv_conv_pool(
        pool4, [128, 128], training, flags, name=5, pool=False)

    up6 = upconv_concat(conv5, conv4, 64, flags, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, flags, name=6, pool=False)

    up7 = upconv_concat(conv6, conv3, 32, flags, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, flags, name=7, pool=False)

    up8 = upconv_concat(conv7, conv2, 16, flags, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, flags, name=8, pool=False)

    up9 = upconv_concat(conv8, conv1, 8, flags, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, flags, name=9, pool=False)

    return tf.layers.conv2d(
        conv9,
        1, (1, 1),
        name='final',
        activation=tf.nn.sigmoid,
        padding='same')


def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)


def make_train_op(y_pred, y_true):
    """Returns a training operation

    Loss function = - IOU(y_pred, y_true)

    IOU is

        (the area of intersection)
        --------------------------
        (the area of two boxes)

    Args:
        y_pred (4-D Tensor): (N, H, W, 1)
        y_true (4-D Tensor): (N, H, W, 1)

    Returns:
        train_op: minimize operation
    """
    # loss = -IOU_(y_pred, y_true)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    tf.summary.scalar("Loss", loss)

    global_step = tf.train.get_or_create_global_step()

    optim = tf.train.AdamOptimizer()
    return optim.minimize(loss, global_step=global_step)


def read_flags():
    """Returns flags"""

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs")

    parser.add_argument("--batch_size", default=80, type=int, help="Batch size")

    parser.add_argument(
        "--logdir", default="logdir", help="Tensorboard log directory")

    parser.add_argument(
        "--reg", type=float, default=0.1, help="L2 Regularizer Term")

    parser.add_argument(
        "--ckdir", default="models", help="Checkpoint directory")

    parser.add_argument(
        "--exp_dir", default="exp", help="Independent experimental environment")

    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU number")

    flags = parser.parse_args()
    return flags


def main(flags):
    exp_dir = flags.exp_dir.rstrip("/")

    train = pd.read_csv(exp_dir + "/train.csv")
    n_train = train.shape[0]

    test = pd.read_csv(exp_dir + "/eval.csv")
    n_test = test.shape[0]

    current_time = time.strftime("%m_%d_%H_%M_%S")
    train_logdir = "%s/%s/%s/%s" % (exp_dir, flags.logdir, current_time, "train")
    test_logdir = "%s/%s/%s/%s" % (exp_dir, flags.logdir, current_time, "eval")

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name="X")
    y = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="y")
    mode = tf.placeholder(tf.bool, name="mode")

    pred = make_unet(X, mode, flags)

    # f1
    pred_label = tf.cast(tf.greater(pred, 0.5), dtype=tf.float32)
    H, W, _ = pred_label.get_shape().as_list()[1:]

    pred_flat = tf.reshape(pred_label, [-1, H * W])
    true_flat = tf.reshape(y, [-1, H * W])

    intersection = pred_flat * true_flat
    pre = tf.reduce_sum(intersection) / (tf.reduce_sum(pred_flat) + 1e-7)
    rec = tf.reduce_sum(intersection) / (tf.reduce_sum(true_flat) + 1e-7)
    f1_score = 2 / (1 / pre + 1 / rec + 1e-7)
    tf.summary.scalar("precision", pre)
    tf.summary.scalar("rec", rec)
    tf.summary.scalar("f1_score", f1_score)

    tf.add_to_collection("inputs", X)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred)

    tf.summary.histogram("Predicted Mask", pred)

    transform_pred = tf.cast(tf.greater(pred, 0.5), tf.int32)
    transform_pred = tf.py_func(decode_labels, [transform_pred], tf.uint8)
    transform_y = tf.py_func(decode_labels, [tf.cast(y, tf.int32)], tf.uint8)
    transform_X = tf.py_func(decode_images, [X], tf.uint8)
    tf.summary.image("Predicted Image",
                     tf.concat(values=[transform_X,
                                       transform_y,
                                       transform_pred],
                               axis=2), max_outputs=1)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = make_train_op(pred, y)

    IOU_op = IOU_(pred, y)
    IOU_op = tf.Print(IOU_op, [IOU_op])
    tf.summary.scalar("IOU", IOU_op)

    train_csv = tf.train.string_input_producer([exp_dir + '/train.csv'])
    test_csv = tf.train.string_input_producer([exp_dir + '/eval.csv'])
    train_image, train_mask = get_image_mask(train_csv)
    test_image, test_mask = get_image_mask(test_csv, augmentation=False)

    X_batch_op, y_batch_op = tf.train.shuffle_batch(
        [train_image, train_mask],
        batch_size=flags.batch_size,
        capacity=flags.batch_size * 5,
        min_after_dequeue=flags.batch_size * 2,
        allow_smaller_final_batch=True)

    X_test_op, y_test_op = tf.train.batch(
        [test_image, test_mask],
        batch_size=flags.batch_size,
        capacity=flags.batch_size * 2,
        allow_smaller_final_batch=True)

    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_summary_writer = tf.summary.FileWriter(test_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=None)
        # if os.path.exists(flags.ckdir) and tf.train.checkpoint_exists(
        #         flags.ckdir):
            # latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
            # saver.restore(sess, latest_check_point)
            # pass

        # else:
        #     try:
        #         os.rmdir(flags.ckdir)
        #     except IOError:
        #         pass
        #     os.mkdir(flags.ckdir)

        try:
            global_step = tf.train.get_global_step(sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for epoch in range(flags.epochs):

                for step in range(0, n_train, flags.batch_size):

                    X_batch, y_batch = sess.run([X_batch_op, y_batch_op])

                    _, step_iou, step_summary, global_step_value = sess.run(
                        [train_op, IOU_op, summary_op, global_step],
                        feed_dict={X: X_batch,
                                   y: y_batch,
                                   mode: True})

                    train_summary_writer.add_summary(step_summary,
                                                     global_step_value)

                # total_iou = 0
                    if step % (n_train // n_test) == 0:
                # for step in range(0, n_test, flags.batch_size):
                        X_test, y_test = sess.run([X_test_op, y_test_op])
                        step_iou, step_summary = sess.run(
                            [IOU_op, summary_op],
                            feed_dict={X: X_test,
                                       y: y_test,
                                       mode: False})

                    # total_iou += step_iou * X_test.shape[0]

                        test_summary_writer.add_summary(step_summary, global_step_value)

                saver.save(sess, "{}/model-{:02d}.ckpt".format(exp_dir + os.sep + flags.ckdir, epoch))

        finally:
            coord.request_stop()
            coord.join(threads)
            # saver.save(sess, "{}/model.ckpt".format(flags.ckdir))


if __name__ == '__main__':
    flags = read_flags()
    os.environ['CUDA_VISIBLE_DEVICES'] = '%s' % flags.gpu
    main(flags)
