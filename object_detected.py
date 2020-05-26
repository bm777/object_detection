import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3.models import YoloV3


from yolov3.dataset import transform_images #for image processing
from yolov3.utils import draw_outputs # for drw on images

# ------ flags or args----------
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4', 'path to video file or number for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def main(_argv):
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	for physical_device in physical_devices:
		tf.config.experimental.set_memory_growth(physical_device, True)

	# import weights
	yolo = YoloV3(classes=FLAGS.num_classes)
	yolo.load_weights(FLAGS.weights)
	logging.info("weights loaded")

	# Import classes
	class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
	logging.info("classes loaded")

	# list of time for procces on each frame
	times = []

	# try to load webcam or a video file 
	try:
		vid = cv2.VideoCapture(int(FLAGS.video))
	except:
		vid = cv2.VideoCapture(FLAGS.video)

	out = None

	if FLAGS.output:
		#by default VideoCapture returns float instead of int
		width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(vid.get(cv2.CAP_PROP_FPS))
		codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
		out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

	while True:
		_, img = vid.read()
		if img is None:
			logging.info("Empty Frame")
			time.sleep(0.1)
		img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_in = tf.expand_dims(img_in, 0)
		img_in = transform_images(img_in, FLAGS.size)
		
		t1 = time.time()
		boxes, scores, classes, nums = yolo.predict(img_in)
		for i in num:
			print(i)
		t2 = time.time()
		times.append(t2-t1)
		times = times[-20:]

		img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
		img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0,30),
													cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (1, 0, 255), 2)

		if FLAGS.output:
			out.write(img)
		cv2.imshow('Output', img)
		if cv2.waitKey(1) == ord('q'):
			break

	vid.release()
	cv2.destroyALlWindows()

# the main function
if __name__ == '__main__':
	try:
		app.run(main)
	except:
		pass