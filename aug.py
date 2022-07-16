from typing import List

import albumentations as albu
from torchvision import transforms

def get_transforms(size: int, scope: str = 'geometric', crop='random'):
    augs = {'strong': albu.Compose([albu.HorizontalFlip(),
                                    albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=.4),
                                    albu.ElasticTransform(),
                                    albu.OpticalDistortion(),
                                    albu.OneOf([
                                        albu.CLAHE(clip_limit=2),
                                        albu.IAASharpen(),
                                        albu.IAAEmboss(),
                                        albu.RandomBrightnessContrast(),
                                        albu.RandomGamma()
                                    ], p=0.5),
                                    albu.OneOf([
                                        albu.RGBShift(),
                                        albu.HueSaturationValue(),
                                    ], p=0.5),
                                    ]),
            'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            'geometric': albu.Compose([albu.HorizontalFlip(),
                                     albu.VerticalFlip(),
                                     albu.RandomRotate90(),
                                     ]),
            'None': None
            }

    aug_fn = augs[scope]
    crop_fn = {'random': albu.RandomCrop(size, size, always_apply=True),
               'center': albu.CenterCrop(size, size, always_apply=True)}[crop]

    pipeline = albu.Compose([aug_fn, crop_fn], additional_targets={'target': 'image'})


    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process


def get_normalize():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    def process(a, b):
        image = transform(a).permute(1, 2, 0) - 0.5
        target = transform(b).permute(1, 2, 0) - 0.5
        return image, target

    return process






