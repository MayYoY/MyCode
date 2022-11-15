from torchvision import transforms


class PreprocessConfig:
    input_path = "D:\papers\Video\my_code\\ubfc_rppg_test\dataset2"
    output_path = "D:\papers\Video\my_code\\ubfc_rppg_test\processed\data"
    record_path = "D:\papers\Video\my_code\\ubfc_rppg_test\processed\\record.csv"

    W = 72
    H = 72
    detector = "D:\papers\Video\\rPPG-Toolbox-main\dataset\haarcascade_frontalface_default.xml"
    DYNAMIC_DETECTION = False
    DYNAMIC_DETECTION_FREQUENCY = 300
    CROP_FACE = True
    LARGE_FACE_BOX = True
    LARGE_BOX_COEF = 1.5

    DATA_TYPE = ["Standardize"]  # Raw / Difference / Standardize
    LABEL_TYPE = "Standardize"

    DO_CHUNK = True
    CHUNK_LENGTH = 300


class SCAMPSConfig:
    record_path = "D:\papers\Video\\test_cache\\record.csv"
    Fs = 30
    batch_size = 2
    trans = None


class UBFCConfig:
    record_path = "D:\papers\Video\my_code\\ubfc_rppg_test\processed\\record.csv"
    Fs = 30
    batch_size = 2
    trans = None
