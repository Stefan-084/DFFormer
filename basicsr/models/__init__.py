from basicsr.utils import get_root_logger

from .image_restoration_model import ImageCleanModel


def create_model(opt):
    model_type = opt["model_type"]
    if model_type != "ImageCleanModel":
        raise ValueError(f"Unsupported model type: {model_type}")
    model = ImageCleanModel(opt)
    logger = get_root_logger()
    logger.info(f"Model [{model.__class__.__name__}] is created.")
    return model
