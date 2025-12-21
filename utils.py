
## this file includes : 
## 1. plotting function
## 2. THINGS -> NSD category alignment dictionary


## --------------------------------------------------------------------------------- ##
## plotting all images directly under a given path, and output as a tiled image 
## plot(path, ROWS1 = None, COLS1 = None)

import sys
from pathlib import Path
from typing import List, Optional, Tuple
import datetime

from PIL import Image


def get_current_time_info():
    now = datetime.datetime.now()
    standard_format = now.strftime("%m-%d_%H-%M-%S")
    return standard_format


# 固定配置：根据实际需求修改以下常量
IMAGE_DIR = None
ROWS = 5
COLS = 4
OUTPUT_PATH = Path(f"display_rt_{get_current_time_info()}.png")
RESIZE_TO: Optional[Tuple[int, int]] = None


def ensure_config() -> None:
    if not IMAGE_DIR.exists() or not IMAGE_DIR.is_dir():
        print(f"目录不存在：{IMAGE_DIR}", file=sys.stderr)
        raise SystemExit(1)


def collect_images(directory: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)


def load_and_resize(path: Path, target_size):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    return img


def tile_images(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    if not images:
        raise ValueError("没有可用的图像")
    w, h = images[0].size
    canvas = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x0, y0 = c * w, r * h
        canvas.paste(img, (x0, y0))
    for idx in range(len(images), rows * cols):
        r, c = divmod(idx, cols)
        x0, y0 = c * w, r * h
        canvas.paste(Image.new("RGB", (w, h), color=(255, 255, 255)), (x0, y0))
    return canvas


def plot(path, save_path = "", ROWS1 = None, COLS1 = None):
    global IMAGE_DIR, ROWS, COLS, OUTPUT_PATH
    IMAGE_DIR = Path(path)
    file_name = path.split('/')[-1]
    OUTPUT_PATH = Path(save_path + f"display_rt_{get_current_time_info()}_{file_name}.png")
    ensure_config()

    images_paths = collect_images(IMAGE_DIR)
    if len(images_paths) == 0:
        return 
    if ROWS1 != None:
        ROWS = ROWS1
        COLS = COLS1
    else:
        cc = 12
        COLS = 2
        while cc <= len(images_paths):
            cc *= 2
            COLS += 2
        ROWS = math.ceil(len(images_paths) / COLS)
    need = ROWS * COLS

    target_size = RESIZE_TO
    selected = images_paths[:]
    images = [load_and_resize(p, target_size) for p in selected]


    if target_size is None:
        base_w, base_h = images[0].size
        for img in images[1:]:
            if img.size != (base_w, base_h):
                print("检测到尺寸不一致，请统一配置 RESIZE_TO。", file=sys.stderr)
                return 1

    grid = tile_images(images, ROWS, COLS)
    grid.save(OUTPUT_PATH)
    print(f"保存完成：{OUTPUT_PATH}")



## --------------------------------------------------------------------------------- ##
## THINGS-EEG2 -> NSD category alignment

## old-ver
""" 
dict0 = {
0 : [[0],       [0, 1, 2, 4, 8, 13, 17, 26, 35]], # animals
1 : [[1],       [39]], # birds
2 : [[2, 3],    [40]], # body parts / cloth
3 : [[8],       [3]], # electronic devices
4 : [[6, 9],    [6, 15, 32]], # food
5 : [[10],      [18]], # fruits
6 : [[11, 12],  [12, 19, 28]], # indoor scenes
7 : [[20],      [37]], # plants
8 : [[21],      [10, 11, 21, 25, 29, 34, 38]], # sports
9 : [[23],      [9]], # toys
10 : [[25],     [5, 20, 27, 24, 33, 30]] # vehicle
}
"""

## new-ver
# i : [[a1, a2, ..], [b1, b2, ..]] # i is the id (no real meaning); a's are THINGS-EEG2 cluster labels; b's are NSD cluster labels
dict0 = { 
# 哺乳动物
0 : [[3],   [0, 1, 2, 4, 8, 13, 17, 26, 35]],
# 鸟类
1 : [[4],   [16, 39]],
# toys、马
2 : [[33], [8, 9]],

# 饼类
3 : [[15, 17, 18], [15, 31, 32]],
# 甜品
4 : [[14, 15, 16], [31, 32]],
# 其它加工性的吃的
5 : [[7, 12], [31, 32]],
# 蔬菜
6 : [[19, 23], [31, 32]],
# fruits
7 : [[8], [18]],
# 糖类
8 : [[9], [6]],

# 带棍棒的球类运动相关
9 : [[21], [10, 11, 25, 29]],

# 室内场合
10 : [[10, 13, 20, 27, 28, 41, 42, 45, 46], [12, 19, 28]],
11 : [[37], [28]],
12 : [[10, 24, 29, 39, 43, 44], [12, 19]],

# 陆地交通工具
13 : [[5, 26, 32], [5, 20, 24]],

# 空中交通工具
14 : [[36], [33]],

# 水上交通工具
15 : [[26, 35], [30]],
# 冰雪运动
16 : [[26], [10, 21, 34, 38]],

# plants
17 : [[23], [37]],

# clock
18 : [[10], [36]],

# 户外装置
19 : [[34, 40, 41, 47], [14, 16, 22, 23]],

# 电子设备
20 : [[24], [3]],
# 工具
21 : [[10, 37, 38, 45, 46, 48], [7]],

# 关于人
# 衣服 装备
22 : [[6, 25, 29, 38], [14, 31, 40]],
# 器官
23 : [[11, 30, 31], [31, 40]]
}




dict1 = { # NSD text label to cluster id
"zebra" : 0,
"bear" : 1,
"dog" : 2,
"computer" : 3,
"giraffe" : 4,
"bike" : 5,
"sweets" : 6,
"umbrella" : 7,
"horse" : 8,
"toy" : 9,
"sports" : 10,
"baseball" : 11,
"bedroom" : 12,
"cow" : 13,
"group of people" : 14,
"pizza" : 15,
"sky" : 16,
"elephant" : 17,
"fruits" : 18,
"living room" : 19,
"vehicle" : 20,
"surfer" : 21,
"hydrant" : 22,
"stop sign" : 23,
"train" : 24,
"tennis" : 25,
"cat" : 26,
"bus" : 27,
"bathroom" : 28,
"soccer" : 29,
"boat" : 30,
"person eating" : 31,
"food" : 32,
"airplane" : 33,
"skate" : 34,
"sheep" : 35,
"clocktower" : 36,
"flower" : 37,
"ski" : 38,
"bird" : 39,
"a person" : 40
}