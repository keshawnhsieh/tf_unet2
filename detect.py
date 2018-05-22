import tensorflow as tf
import numpy as np
import os
from PIL import Image
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
        "save_dir", help="Where to save result")

    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU number")

    return parser.parse_args()

def mk_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def main(flags):
    wz = 256

    model = flags.model
    abs_path, _ = os.path.split(os.path.abspath(__file__))
    model_map = {
        "1m": abs_path + "/model/1m/model-99.ckpt",
        "0.3m": abs_path + "/model/0.3m/model-99.ckpt",
        "0.5m": abs_path + "/model/0.5m/model-99.ckpt",
    }
    model = model_map[model]

    image = flags.image
    save_dir = flags.save_dir.rstrip('/')

    saver = tf.train.import_meta_graph(model + ".meta")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    saver.restore(sess, model)
    X, mode = tf.get_collection("inputs")
    pred = tf.get_collection("outputs")[0]

    # print('loading image')
    proj, geotrans, input_img = read_img(image)
    input_img = input_img[:, :, :3]
    input_img = np.asarray(input_img, dtype='uint8')

    label_pred = predict_img_with_smooth_windowing(
        input_img,
        window_size=wz,
        subdivisions=2,
        batch_size=256,
        pred_func=(
            lambda img: sess.run(pred, feed_dict={X: img, mode: False})
        )
    )
    label_pred = label_pred[:, :, 0]
    label_pred[np.where(label_pred >= 0.5)] = 1
    label_pred[np.where(label_pred < 0.5)] = 0
    label_pred = label_pred.astype(np.uint8)

    # full size image
    tmp_dir = "%s/product_big" % save_dir
    mk_dir(tmp_dir)
    prd_name = "%s/%s" % \
               (tmp_dir, image.split('/')[-1])
    write_img(prd_name, proj, geotrans, label_pred)

    # tiny size patch
    tmp_dir1 = "%s/original_small" % save_dir
    tmp_dir2 = "%s/product_small" % save_dir
    mk_dir(tmp_dir1)
    mk_dir(tmp_dir2)
    _, w = input_img.shape[:2]
    y_range = w // wz
    for i in range(200):
        x = i // y_range * wz
        y = i % y_range * wz

        patch = input_img[x: x + wz, y: y + wz]
        pt_name = "%s/%s_%s.png" % \
                  (tmp_dir1, image.split('/')[-1].split('.')[0], i)
        Image.fromarray(patch).save(pt_name)

        patch = label_pred[x: x + wz, y: y + wz]
        patch = np.asarray(patch * 255, dtype=np.uint8)
        pt_name = "%s/%s_%s.png" % \
                  (tmp_dir2, image.split('/')[-1].split('.')[0], i)
        Image.fromarray(patch).save(pt_name)

if __name__ == '__main__':
    flags = read_flags()
    os.environ['CUDA_VISIBLE_DEVICES'] = '%s' % flags.gpu
    main(flags)