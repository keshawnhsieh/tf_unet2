from PIL import Image
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gdal

# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def decode_images(image, num_images=1):
    n, h, w, c = image.shape
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(3):
        scalar = MinMaxScaler()
        # scalared = scalar.fit_transform(image[0, :, :, i]) * 255.0
        scalared = image[0, :, :, i]
        outputs[0, :, :, i] = scalared.astype(np.uint8)

    return outputs

def read_img(file_name):
    image = gdal.Open(file_name)

    im_width = image.RasterXSize
    im_height = image.RasterYSize

    im_geotrans = image.GetGeoTransform()
    im_proj = image.GetProjection()
    im_data = image.ReadAsArray(0, 0, im_width, im_height)
    if len(im_data.shape) == 3:
        im_data = im_data.transpose(1, 2, 0)

    del image
    return im_proj, im_geotrans, im_data

def write_img(file_name, im_proj, im_geotrans, im_data):
    if 'uint8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
        # print 'uint8'
    elif 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_data = im_data.transpose(2, 0, 1)
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        file_name,
        im_width,
        im_height,
        im_bands,
        datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset