import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import distance

def parse_bbox(s):
	try:
		filename = s.split(',')[0]
		min_x, min_y, max_x, max_y = [float(x) for x in s.strip().split('(')[1][:-1].split(',')]

		return BoundingBox(min_x, min_y, max_x, max_y, filename=filename)
	except Exception as e:
		print(str(e))
		return None

def parse_bboxes(filename):
	bboxes = []

	if os.path.isfile(filename):
		with open(filename, 'r') as f:
			for line in f.readlines():
				bbox = parse_bbox(line)
				if bbox:
					bboxes.append(bbox)

	return bboxes

def write_bboxes(bboxes, filename):
	with open(filename, 'w') as f:
		for bbox in bboxes:
			if bbox.filename:
				f.write(bbox.filename + ',')
				coords = [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y]
				f.write('(' + ','.join([str(x) for x in coords]) + ')')
				f.write('\n')

class BoundingBox():
	def __init__(self, min_x, min_y, max_x, max_y, filename=None, img=None):
		self.min_x = min_x
		self.min_y = min_y
		self.max_x = max_x
		self.max_y = max_y
		self.filename = filename
		self.img = img

class BoundingBoxVisualizer():
	topleft = None
	bottomright = None
	circles = []
	rect = None
	abort = False
	press = None

	def __init__(self, min_x=None, min_y=None, max_x=None, max_y=None, interactive=False):
		if min_x and min_y:
			self.topleft = (min_x, min_y)
		if max_x and max_y:
			self.bottomright = (max_x, max_y)
		self.interactive = interactive

	def clear(self):
		for circle in self.circles:
			circle.remove()
		self.circles = []

		if self.rect:
			self.rect.remove()
		self.rect=None

	def draw_circles(self):
		if self.topleft:
			circle = plt.Circle(self.topleft, radius=10, fc='g')
			self.ax.add_patch(circle)
			self.circles.append(circle)

		if self.bottomright:
			circle = plt.Circle(self.bottomright, radius=10, fc='g')
			self.ax.add_patch(circle)
			self.circles.append(circle)

	def draw_rect(self):
		if self.topleft and self.bottomright:
			self.rect = mpatches.Rectangle(self.topleft, 
				self.bottomright[0] - self.topleft[0],
				self.bottomright[1] - self.topleft[1],
				fill=False,
				color='g')
			self.ax.add_patch(self.rect)

	def redraw(self):
		self.clear()
		self.draw_circles()
		self.draw_rect()

		plt.draw()
		plt.show()

	def onclick(self, event):
		print('ONCLICK ' + str(event.button))
		if event.button == 1:
			x, y = event.xdata, event.ydata

			if not self.topleft:
				self.topleft = (x, y)
			elif not self.bottomright:
				self.bottomright = (x, y)
			elif distance.euclidean((x, y), self.topleft) < distance.euclidean((x, y), self.bottomright):
				self.topleft = (x, y)
			else:
				self.bottomright = (x, y)

			self.redraw()
		else:
			if event.button == 2:
				self.abort = True

			plt.close()

	def display(self, filename):
		self.fig = plt.figure()
		self.ax = plt.gca()
		self.topleft = None
		self.bottomright = None

		img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
		plt.imshow(img)
		self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		plt.show()

		return self.topleft, self.bottomright