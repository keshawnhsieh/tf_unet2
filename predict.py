import tensorflow as tf
import numpy as np
import os
from utils import read_img, write_img
from smooth_tiled_predictions import predict_img_with_smooth_windowing

def read_flags():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image", help="Image to predict")

    parser.add_argument(
        "model", help="Model to use")

    parser.add_argument(
        "--gpu", type=int, default=1, help="GPU number")

    return parser.parse_args()

def main(flags):
    model = flags.model
    image = flags.image

    saver = tf.train.import_meta_graph(model + ".meta")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    saver.restore(sess, model)
    X, mode = tf.get_collection("inputs")
    pred = tf.get_collection("outputs")[0]

    print('loading image')
    proj, geotrans, input_img = read_img(image)
    input_img = input_img[:, :, :3]
    input_img = np.asarray(input_img, dtype='uint8')

    label_pred = predict_img_with_smooth_windowing(
        input_img,
        window_size=256,
        subdivisions=2,
        batch_size=256,
        pred_func=(
            lambda img: sess.run(pred, feed_dict={X: img, mode: False})
        )
    )
    label_pred = label_pred[:, :, 0]
    label_pred[np.where(label_pred >= 0.5)] = 1
    label_pred[np.where(label_pred < 0.5)] = 0
    label_pred.astype(np.uint8)

    prd_name = "%s_m%s_prd.tif" % (image.split('.')[0], model.split('.')[0][-2:])
    write_img(prd_name, proj, geotrans, label_pred)

if __name__ == '__main__':
    flags = read_flags()
    os.environ['CUDA_VISIBLE_DEVICES'] = '%s' % flags.gpu
    main(flags)