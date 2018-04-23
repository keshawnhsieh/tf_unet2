import numpy as np
from glob import glob
import tifffile as tif
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os

def read_flags():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_dir", help="Independent experimental environment")

    parser.add_argument(
        "data_dir", help="Raw data directory")

    parser.add_argument(
        "eval_image", help="Image used for evaluation")

    parser.add_argument(
        "--size", type=int, default=256, help="Size of image")

    parser.add_argument(
        "--threshold", type=float, default=0.05, help="The lower limit of the effective area")

    parser.add_argument(
        "--random_seed", type=int, default=1234, help="Random seed")

    return parser.parse_args()

def get_roi(array, x, y, size):
    return array[x:x + size, y:y + size]

def check_pass(label, threshold):
    s = np.sum(label, axis=(0, 1))
    h, w = label.shape[:2]
    return float(s) / (h * w) > threshold

def build(image,
          label,
          exp_dir,
          size=256,
          scale=2,
          threshold=0.05):
    def save_png(image, name):
        Image.fromarray(image).save(name)

    obj = image.split("/")[-1]
    image = tif.imread(image)
    label = tif.imread(label)
    print("loaded image and label %s" % obj)

    h, w = image.shape[:2]
    n = int((h // size) * (w // size) * scale)
    dsp_list = []
    for _ in tqdm(range(n)):
        while 1:
            xs = np.random.randint(h - size)
            ys = np.random.randint(w - size)

            if [xs, ys] in dsp_list:
                continue

            roi_label = get_roi(label, xs, ys, size)
            if check_pass(roi_label, threshold):
                break

        roi_image = get_roi(image, xs, ys, size)
        image_name = "%s/data/%s_%s_%s.png" % (exp_dir, obj.split(".")[0], xs, ys)
        save_png(roi_image, image_name)

        label_name = "%s/data/%s_%s_%s_mask.png" % (exp_dir, obj.split(".")[0], xs, ys)
        save_png(roi_label, label_name)
        dsp_list.append([image_name, label_name])

    return dsp_list

def main(flags):
    # set random seed
    np.random.seed(flags.random_seed)

    # glob file
    data_dir = flags.data_dir.rstrip("/")
    image_dir = data_dir + "/image"
    label_dir = data_dir + "/label"
    image_list = glob(image_dir + "/*.tif")
    label_list = glob(label_dir + "/*.tif")
    eval_image = [image for image in image_list if flags.eval_image in image]
    if eval_image == []:
        print("Evaluation image is not in directory")
        exit(0)
    eval_image = eval_image[0]
    eval_label = eval_image.replace("image", "label")
    image_list.remove(eval_image)
    label_list.remove(eval_label)

    # create experiment directory
    exp_dir = flags.exp_dir.strip("/")
    if os.path.exists(exp_dir):
        print("%s directory exits" % exp_dir)
        exit(0)
    else:
        # for nas
        os.mkdir("/home/nas/extra_cache/unet_exp/%s" % exp_dir)
        os.system("ln -s /home/nas/extra_cache/unet_exp/%s %s" % (exp_dir, exp_dir))
        # os.mkdir(exp_dir)
        os.mkdir(exp_dir + "/data")
        os.mkdir(exp_dir + "/models")

    dps_list = []
    for image, label in zip(image_list, label_list):
        dps_list.extend(build(image,
                              label,
                              exp_dir,
                              size=flags.size,
                              threshold=flags.threshold))

    # export train.csv
    tmp = pd.DataFrame(dps_list)
    tmp.to_csv(exp_dir + "/train.csv", header=False, index=False)

    dps_list = build(eval_image,
                     eval_label,
                     exp_dir,
                     size=flags.size,
                     threshold=0.0)

    # export eval.csv
    tmp = pd.DataFrame(dps_list)
    tmp.to_csv(exp_dir + "/eval.csv", header=False, index=False)

if __name__ == '__main__':
    main(read_flags())