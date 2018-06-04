#Juancarlos/Lily

import os
import csv
import const
import tensorflow as tf
from io import BytesIO
from PIL import Image


# ################################### Taken from tensor-flow object detection tutorial ############################### #
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
# #################################################################################################################### #

#construct feature function
#parameters
#   box: the coordinates points of the box
#   encoded_img: image
#   encoded_img_height: height of the image
#   encoded_img_width: width of the image
#   img_name: file name of the image
def construct_feature(box, encoded_img, encoded_img_height, encoded_img_width, img_name):
    encoded_img_format, encoded_img_class, encoded_img_label = b'jpg', b'ball', 1
    #encoded_img_format: 'jpg' turned into a bytes literal. 'jpg' is the format of the images
    #encoded_img_class: 'ball' turned into a bytes literal. 'ball' is the name of the box
    #encoded_img_label: 1 because there is 1 box
    encoded_img_name = img_name.encode('utf-8')
    #img_name: actual name of the image is encoded in 'utf-8'

    #arrays
    x_min, x_max, y_min, y_max = [], [], [], []
    classes, labels = [], []

    x_mi, x_mx, y_mi, y_mx = box[0] / encoded_img_width, box[2] / encoded_img_width, \
                             box[1] / encoded_img_height, box[3] / encoded_img_height
    x_mi = 1 if x_mi > 1 else x_mi
    x_mx = 1 if x_mx > 1 else x_mx
    y_mi = 1 if y_mi > 1 else y_mi
    y_mx = 1 if y_mx > 1 else y_mx

    x_min.append(x_mi)
    x_max.append(x_mx)
    y_min.append(y_mi)
    y_max.append(y_mx)
    classes.append(encoded_img_class)
    labels.append(encoded_img_label)

    if len(x_min) == len(x_max) == len(y_min) == len(y_max):
        example = tf.train.Example(features=tf.train.Features(feature={
            const.HEIGHT_KEY: int64_feature(encoded_img_height),
            const.WIDTH_KEY: int64_feature(encoded_img_width),
            const.FILENAME_KEY: bytes_feature(encoded_img_name),
            const.SOURCE_KEY: bytes_feature(encoded_img_name),
            const.ENCODED_IMAGE_KEY: bytes_feature(encoded_img),
            const.FORMAT_KEY: bytes_feature(encoded_img_format),
            const.XMIN_KEY: float_list_feature(x_min),
            const.XMAX_KEY: float_list_feature(x_max),
            const.YMIN_KEY: float_list_feature(y_min),
            const.YMAX_KEY: float_list_feature(y_max),
            const.CLASS_KEY: bytes_list_feature(classes),
            const.LABEL_KEY: int64_list_feature(labels)
        }))
        return example
    else:
        raise ValueError(img_name)

#loads image into a format that is suitable for encoding into a tf record file
#parameters:
#   image_path: the full path to the .jpg images. calls this method multiple time from look_through_images() .jpg image
#it returns the encoded image, height and width.
def load_encoded_image(image_path):
    """
    Loads an image in a format suitable for encoding into a tf record file.

    :param image_path: the full path to a .jpg image
    :return: the encoded image, its height, its width
    """
    with tf.gfile.GFile(image_path, 'rb') as fid: #put .jpg image into variable fid
        encoded_jpg = fid.read() #reads the image and puts it inside encoded_jpg
    encoded_jpg_io = BytesIO(encoded_jpg)
    #A stream implementation using an in-memory bytes buffer.
    img = Image.open(encoded_jpg_io) #Opens and identifies the given image file.

    width, height = img.size #puts image wight and height inside its respective variable
    return encoded_jpg, height, width #return


#load_csv function;
#paramters:
#   path_to_csv: path of where the .csv files are at.
def load_csv(path_to_csv):
    # key = image_name, val = [xmin, ymin, xmax, ymax]
    data_map = {}
    with open(path_to_csv) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            #for every row in the reader, it saves the actual filename, xmin, ymin, xmax, and ymax into their
            #respective variable name
            image_name, xmin, ymin, xmax, ymax = row['filename'], row['xmin'], row['ymin'], row['xmax'], row['ymax']
            #once the data is read, put the data in a hashmap.
            data_map[image_name] = [int(xmin), int(ymin), int(xmax), int(ymax)]
    return data_map #return the hashmap


#look through the images function
#parameters:
#   directory: directory of where the images are in
#   data_map: hashmap of either train or test
#   record_name: the name of the output tfrecord file
def look_through_images(directory, data_map, record_name):
    #We use filter to only save .jps images into image_filter. It saves all the .jpg images.
    image_filter = filter(lambda f: f.endswith('.jpg'), os.listdir(directory))
    counter=0;

    """for file in image_filter:
       print(file)"""
    #TFRecordWriter class writes records to a TFRecord File
    writer = tf.python_io.TFRecordWriter(os.path.join('./', record_name)) #writer will write to the tfRecord file

    count = 0
    for file in image_filter: #look through image_filter files; the .jpg images.
        box = data_map.get(file, None) #using the image_filter list of images, we find its associated value (xmin, ymin etc)
        if box is None:
            #if images aren't found in the hashmap, we print an error message
            print('Error! Box for {} was not found!'.format(file))
            count += 1 #print number of images not found
        else:
            img, height, width = load_encoded_image(os.path.join(directory, file)) #else if image is found inside hashmap, call load_encoded_image
            if img is not None: #if the image is found in the load_encoded_image function then we do this..
                example = construct_feature(box, img, height, width, file)
                writer.write(example.SerializeToString())
    writer.flush()
    writer.close()
    return count


n = 0
#load csv function gets called with parameter of where the .csv files are at.
train = load_csv('./data/train.csv') #train hashmap
test = load_csv('./data/test.csv') #test hashmap

n += look_through_images(directory='./images/train/', data_map=train, record_name='train.tfrecord')
n += look_through_images(directory='./images/test/', data_map=test, record_name='test.tfrecord')
print('ERROR FILES: {}'.format(n))
