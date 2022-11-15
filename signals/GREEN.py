from .utils import getRGB


def GREEN(frames):
    rgb_data = getRGB(frames)  # 1 x C x T
    BVP = rgb_data[:, 1, :]
    BVP = BVP.reshape(-1)
    return BVP
