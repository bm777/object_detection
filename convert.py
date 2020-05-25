from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from yolov3.models import YoloV3
from yolov3.utils import load_darknet_weights
import tensorflow as tf

flags.DEFINE_string('weights', './data/yolov3.wiegthts', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to output')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def main(_argv):
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	if len(physical_devices) > 0:
		tf.config.experimental.set_memory_growth(physical_devices[0], True)

	yolo = YoloV3(classes=FLAGS.num_classes)
	yolo.summary()
	logging.info("model created")

	load_darknet_weights(yolo, FLAGS.weights, False) # False for absence of yolo-TinY
	logging.info("weights loaded")

	img = np.random.random((1,320,320,3)).astype(np.float32)
	output = yolo(img)
	logging.info("sanity check passed")

	yolo.save_weights(FLAGS.output)
	logging.info("weights saved")



if __name__ == '__main__':
	try:
		app.run(main)
	except SystemExit:
		pass