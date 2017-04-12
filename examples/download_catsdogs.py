

from tfwrapper.datasets import catsdogs


catsdogs.download_cats_and_dogs()

container = catsdogs.create_container(max_images=20)

print(len(container.labels))