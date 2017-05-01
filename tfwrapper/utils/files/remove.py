import os

def remove_dir(root):
	for filename in os.listdir(root):
		target = os.path.join(root, filename)
		if os.path.isdir(target):
			remove_dir(root=target)
		else:
			os.remove(target)

	os.rmdir(root)