
import numpy as np
import cv2
import random


def normalize_staining(img):
    """
    Adopted from "Classification of breast cancer histology images using Convolutional Neural Networks",
    Teresa Araújo , Guilherme Aresta, Eduardo Castro, José Rouco, Paulo Aguiar, Catarina Eloy, António Polónia,
    Aurélio Campilho. https://doi.org/10.1371/journal.pone.0177544
    Performs staining normalization.
    # Arguments
        img: Numpy image array.
    # Returns
        Normalized Numpy image array.
    """
    Io = 240
    beta = 0.15
    alpha = 1
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = img.shape
    img = img.reshape(h * w, c)
    OD = -np.log((img.astype("uint16") + 1) / Io)
    ODhat = OD[(OD >= beta).all(axis=1)]
    W, V = np.linalg.eig(np.cov(ODhat, rowvar=False))

    Vec = -V.T[:2][::-1].T  # desnecessario o sinal negativo
    That = np.dot(ODhat, Vec)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])

    HE = HE.T
    Y = OD.reshape(h * w, c).T

    C = np.linalg.lstsq(HE, Y)
    maxC = np.percentile(C[0], 99, axis=1)

    C = C[0] / maxC[:, None]
    C = C * maxCRef[:, None]
    Inorm = Io * np.exp(-np.dot(HERef, C))
    Inorm = Inorm.T.reshape(h, w, c).clip(0, 255).astype("uint8")

    return Inorm


def hematoxylin_eosin_aug(img, low=0.7, high=1.3, seed=None):
    """
    "Quantification of histochemical staining by color deconvolution"
    Arnout C. Ruifrok, Ph.D. and Dennis A. Johnston, Ph.D.
    http://www.math-info.univ-paris5.fr/~lomn/Data/2017/Color/Quantification_of_histochemical_staining.pdf

    Performs random hematoxylin-eosin augmentation

    # Arguments
        img: Numpy image array.
        low: Low boundary for augmentation multiplier
        high: High boundary for augmentation multiplier
    # Returns
        Augmented Numpy image array.
    """
    D = np.array([[1.88, -0.07, -0.60],
                  [-1.02, 1.13, -0.48],
                  [-0.55, -0.13, 1.57]])
    M = np.array([[0.65, 0.70, 0.29],
                  [0.07, 0.99, 0.11],
                  [0.27, 0.57, 0.78]])
    Io = 240

    h, w, c = img.shape
    OD = -np.log10((img.astype("uint16") + 1) / Io)
    C = np.dot(D, OD.reshape(h * w, c).T).T
    r = np.ones(3)
    r[:2] = np.random.RandomState(seed).uniform(low=low, high=high, size=2)
    img_aug = np.dot(C, M) * r

    img_aug = Io * np.exp(-img_aug * np.log(10)) - 1
    img_aug = img_aug.reshape(h, w, c).clip(0, 255).astype("uint8")
    return img_aug

def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)

def HorizontalFlip(img):
    img = cv2.flip(img, 1)
    return img


def VerticalFlip(img):
    img = cv2.flip(img, 0)
    return img

def RandomRotate(img):
    factor = random.randint(0, 4)
    img = np.rot90(img, factor)
    return img

def RandomContrast(img):
    limit=.1

    alpha = 1.0 + limit * random.uniform(-1, 1)

    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    maxval = np.max(img[..., :3])
    dtype = img.dtype
    img[:, :, :3] = clip(alpha * img[:, :, :3] + gray, dtype, maxval)
    return img

def RandomBrightness(img):

    limit=.1

    alpha = 1.0 + limit * random.uniform(-1, 1)

    maxval = np.max(img[..., :3])
    dtype = img.dtype
    img[..., :3] = clip(alpha * img[..., :3], dtype, maxval)
    return img

def CenterCrop(img):
    limit=.1
    height, width = (450,450)
    h, w, c = img.shape
    dy = (h - height) // 2
    dx = (w - width) // 2

    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2, x1:x2]

    return img

def RandomHueSaturationValue(img):

    hue_shift_limit=(-10, 10)
    sat_shift_limit=(-25, 25)
    val_shift_limit=(-25, 25)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
    h = cv2.add(h, hue_shift)
    sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
    s = cv2.add(s, sat_shift)
    val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
    v = cv2.add(v, val_shift)
    img = cv2.merge((h, s, v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img
