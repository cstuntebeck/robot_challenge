# Kontaktabzug: 8 Bilder je Video als ein JPG (Claas, 23.11.)
import sys, numpy as np, imageio.v2 as iio
from PIL import Image
for src in sys.argv[1:]:
    r = iio.get_reader(src)
    n = r.count_frames()
    idx = np.linspace(0, n - 1, 8).astype(int)
    tiles = []
    for i in idx:
        f = r.get_data(int(i))
        im = Image.fromarray(f).resize((400, int(400 * f.shape[0] / f.shape[1])))
        tiles.append(np.asarray(im))
    rows = [np.concatenate(tiles[k:k + 4], axis=1) for k in (0, 4)]
    out = src.replace('.mp4', '_sheet.jpg').replace('/videos/', '/videos/sheets/')
    Image.fromarray(np.concatenate(rows, axis=0)).save(out, quality=70)
    print('OK', out, n)
