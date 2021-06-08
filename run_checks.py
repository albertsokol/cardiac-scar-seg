import numpy as np
import os
from readers import NIIReader


if __name__ == '__main__':
    reader = NIIReader(slice_20=False)
    counter = 0

    # Check that all label images contain a label for all the segmentation types
    for g in ['train', 'val', 'test']:
        image_fnames = [
            os.path.join('/media/y4tsu/ml_data/cmr/3D/', g, 'label', x)
            for x in sorted(os.listdir(f'/media/y4tsu/ml_data/cmr/3D/{g}/label'))
        ]

        for x in image_fnames:
            img = reader.read(x)
            u, c = np.unique(img, return_counts=True)
            print(u, c)
            if u.shape[0] != 5:
                counter += 1
                print(f'{x} does not have 5 classes labelled')

    print(f'{counter} images did not have a RV label')
