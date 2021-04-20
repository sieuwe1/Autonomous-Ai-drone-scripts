
import jetson.inference
import jetson.utils

import argparse
import sys

from segnet_utils import *

input = None
output = None
net = None 
buffers = None

def initialize_detector():
	global input, output, net, buffers
	# parse the command line
	parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
									formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage() +
									jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

	parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
	parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
	parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
	parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
	parser.add_argument("--visualize", type=str, default="mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
	parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
	parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
	parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")

	is_headless = [""] #["--headless"]

	try:
		opt = parser.parse_known_args()[0]
	except:
		print("")
		parser.print_help()
		sys.exit(0)

	# load the segmentation network
	net = jetson.inference.segNet(opt.network, sys.argv)

	# set the alpha blending value
	net.SetOverlayAlpha(opt.alpha)

	# create buffer manager
	buffers = segmentationBuffers(net, opt)

	# create video sources & outputs
	input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
	output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

# process frames until user exits
def get_detections():

	# capture the next image
	img_input = input.Capture()

	# allocate buffers for this size image
	buffers.Alloc(img_input.shape, img_input.format)

	# process the segmentation network
	net.Process(img_input, ignore_class="background,aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,pottedplant,sheep,sofa,train,tvmonitor")

	print(net.Mask)

	# generate the overlay
	#if buffers.overlay:
	#	net.Overlay(buffers.overlay, filter_mode="point")

	# generate the mask
	#if buffers.mask:
	#	net.Mask(buffers.mask, filter_mode="point")

	# composite the images
	#if buffers.composite:
	#	jetson.utils.cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
	#	jetson.utils.cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

	# render the output image
	#output.Render(buffers.output)

	#output_img = jetson.utils.cudaToNumpy(buffers.output)
	#return output_img, net.GetNetworkFPS()