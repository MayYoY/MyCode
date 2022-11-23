from torchvision import transforms


class PreprocessConfig:
    # for PURE
    input_path = "D:\papers\Video\pure_test"
    output_path = "D:\papers\Video\my_code\processed\data"
    record_path = "D:\papers\Video\my_code\processed\\record.csv"

    W = 72
    H = 72
    detector = "D:\papers\Video\\rPPG-Toolbox-main\dataset\haarcascade_frontalface_default.xml"
    DYNAMIC_DETECTION = False
    DYNAMIC_DETECTION_FREQUENCY = 300
    CROP_FACE = True
    LARGE_FACE_BOX = True
    LARGE_BOX_COEF = 1.5

    INTERPOLATE = True  # for PURE

    DATA_TYPE = ["Standardize"]  # Raw / Difference / Standardize
    LABEL_TYPE = "Standardize"

    DO_CHUNK = True
    CHUNK_LENGTH = 300
    CHUNK_STRIDE = -1


class SCAMPSConfig:
    record_path = "D:\papers\Video\\test_cache\\record.csv"
    Fs = 30
    batch_size = 2
    trans = None


class LoadConfig:
    record_path = "D:\papers\Video\my_code\processed\\record.csv"
    Fs = 30
    batch_size = 2
    trans = None
