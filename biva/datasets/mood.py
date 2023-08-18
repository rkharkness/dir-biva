from monai.transforms import *
from monai.utils import ensure_tuple

from biva.numpy_dataset import Numpy2dDataSet

def get_mood_datasets(root, transform, device='cpu'):
    return Numpy2dDataSet(root, transforms=transform), Numpy2dDataSet(root, transforms=transform), Numpy2dDataSet(root, transforms=transform)
