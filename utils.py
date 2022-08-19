import requests
from constants import BASIC_HEADER
import pathlib
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

def save_file(response, directory, index):
    file = open(f"{directory}/{index}.jpg", "wb")
    file.write(response)
    file.close()

def get_image(url):
    response = requests.get(f'{url}', headers=BASIC_HEADER)
    image = response.content

    return image    

def count_files():
    directory = pathlib.Path("data")
    total_bikes = len(list(directory.glob("train/bike/*.jpg")))
    total_motorcycle = len(list(directory.glob("train/motorcycle/*.jpg")))
    
    return total_bikes, total_motorcycle

def normalize_imgs(bytes, img_size, channels):
    try:
      image = tf.io.decode_image(bytes,channels=channels)
      image = tf.image.resize(image, img_size)
      image = tf.cast(image, tf.float32) / 255.
      return image
    except:
      return None
