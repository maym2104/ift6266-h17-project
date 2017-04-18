import os
import theano

# Settings
theano.config.floatX = 'float32'
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SAVED_DIR = os.path.join(ROOT_DIR, 'saved')
DATA_DIR = os.path.join(ROOT_DIR, 'coco')
DATASET_AUGMENTATION = True         # Use dataset augmentation (disables AUTO_RESIZE)
LOAD_DATASET_IN_MEMORY = False       # If True, load entire resized dataset in memory, otherwise load each batch in memory from disk
MAX_HEIGHT = 64                     # Image will be resized to MAX_WIDTH x MAX_HEIGHT
MAX_WIDTH = 64
AUTO_RESIZE = True                  # If true, image will be resized to MAX_W x MAX_H, otherwise image will be cropped
MAX_ITEMS_PER_DATASET = -1          # If not -1, only keep first 'x' items in train, valid, and test dataset (to debug algorithm)
MAX_TRAINING_ITEMS = -1
MAX_VALIDATION_ITEMS = -1
MAX_TEST_ITEMS = -1
SAVE_MODEL_TO_DISK = True
SAMPLES_TO_GENERATE = -1            # Number of test samples to generate per epoch (-1 to disable)
GPU_AVAILABLE_MEMORY = 12*(2**30)   # 12 GB
