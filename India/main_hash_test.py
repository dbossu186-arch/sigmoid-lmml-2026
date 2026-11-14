from PIL import Image
import imagehash


    h1 = imagehash.dhash(Image.open("train/img_000004.jpg"))
    h2 = imagehash.dhash(Image.open("train/img_000005.jpg"))
    print(h1 - h2)   # distanța Hamming: 0 = identice, mic = foarte asemănătoare