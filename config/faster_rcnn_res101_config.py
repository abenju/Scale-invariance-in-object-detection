
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 10
BATCH_SIZE = 4
NUM_CLASSES = 21
CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
DETECTION_THRESHOLD = 0.3

MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]

BACKBONE = 'resnet101'

MODEL_SAVE_PATH = "state/faster_rcnn_{}.pt".format(BACKBONE)

OUTPUT_PATH = "output/"
SAVE_DIR = "output/"
