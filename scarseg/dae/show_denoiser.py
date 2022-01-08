import os

import numpy as np
from matplotlib import pyplot as plt

from scarseg.dae.denoiser import Denoiser
from scarseg.generators import DAEGenerator

if __name__ == "__main__":
    model = Denoiser("/home/y4tsu/PycharmProjects/3d_unet/checkpoint/denoiser_05", 8)
    gen = DAEGenerator(
        "/media/y4tsu/ml-fast/cmr/",
        os.path.join("/media/y4tsu/ml-fast/cmr/", "2D", "val", "transverse"),
        1,
        [144, 144],
        {
            "0": "bg",
            "1": "lv lumen",
            "2": "lv myo",
            "3": "scar",
            "4": "rv lumen",
            "5": "rv myo",
            "6": "pap",
            "7": "aorta",
        },
        dataset="val",
        shuffle=True,
        use_cropper=False,
        zoom_aug=False,
    )
    x, y = gen.__getitem__(0)
    plt.imshow(np.argmax(x[0, ...], axis=-1), cmap="gray")
    plt.suptitle("x")
    plt.show()

    y_pred = model.denoise(np.squeeze(x))
    plt.imshow(np.argmax(y_pred, axis=-1), cmap="gray")
    plt.suptitle("y_pred")
    plt.show()

    plt.imshow(np.argmax(y[0, ...], axis=-1), cmap="gray")
    plt.suptitle("y")
    plt.show()
