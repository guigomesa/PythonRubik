

import urllib.request

model_url = 'https://tfhub.dev/tensorflow/coco-ssd/1?tf-hub-format=compressed'
model_filename = 'model.h5'

urllib.request.urlretrieve(model_url, model_filename)