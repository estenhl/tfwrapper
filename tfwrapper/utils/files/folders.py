import os


def safe_mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def remove_dir(root):
    for filename in os.listdir(root):
        target = os.path.join(root, filename)
        if os.path.isdir(target):
            remove_dir(root=target)
        else:
            os.remove(target)

    os.rmdir(root)