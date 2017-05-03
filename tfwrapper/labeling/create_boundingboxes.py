import os
import sys

from boundingbox import parse_bboxes
from boundingbox import write_bboxes
from boundingbox import BoundingBox
from boundingbox import BoundingBoxVisualizer

def create_boundingbox(filename, visualizer=None):
	if not visualizer:
		visualizer = BoundingBoxVisualizer(interactive=True)

	topleft, bottomright = visualizer.display(filename)
	if topleft and bottomright:
		min_x, min_y = topleft
		max_x, max_y = bottomright
		
		return BoundingBox(min_x, min_y, max_x, max_y, filename=os.path.basename(filename))
	else:
		return BoundingBox(0, 0, 0, 0, filename=os.path.basename(filename))

def create_boundingboxes(folder, output_file=None):
	if not output_file:
		output_file = os.path.join(folder, 'bboxes.txt')

	bboxes = parse_bboxes(output_file)
	existing = [bbox.filename for bbox in bboxes]
	visualizer = BoundingBoxVisualizer(interactive=True)

	for filename in os.listdir(folder):
		if filename.lower().endswith('.jpg') and filename not in existing:
			src = os.path.join(folder, filename)
			bbox = create_boundingbox(src, visualizer)
			if bbox is not None:
				bboxes.append(bbox)
		if visualizer.abort:
			break

	write_bboxes(bboxes, output_file)

if __name__ == '__main__':
	folder = sys.argv[1]
	create_boundingboxes(folder)