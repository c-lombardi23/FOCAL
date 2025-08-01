IMAGE_DIMS = [224, 224]  # default image size for transfer learning
IMAGE_SIZE = [224, 224, 3]  # defuault image size with 3 channels

# Required dataframe structure
REQ_COLUMNS = [
    "CleaveAngle",
    "CleaveTension",
    "ScribeDiameter",
    "Misting",
    "Hackle",
    "ImagePath",
    "Diameter",
    "FiberType",
]
# Required CNN features
FEATURES_CNN = [
    "CleaveAngle",
    "CleaveTension",
    "ScribeDiameter",
    "Misting",
    "Hackle",
]
# Required MLP features
FEATURE_MLP = [
    "CleaveAngle",
    "ScribeDiameter",
    "Misting",
    "Hackle",
]
# Required prediction features
PRED_FEATURES = [
    "CleaveAngle",
    "CleaveTension",
    "ScribeDiameter",
    "Misting",
    "Hackle",
]
