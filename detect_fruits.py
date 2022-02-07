import json
from pathlib import Path
from typing import Dict

import click
import cv2 as cv
from tqdm import tqdm
import numpy as np



def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    img = cv.imread(img_path, cv.IMREAD_COLOR)

    #TODO: Implement detection method.
    def detect_contours(img):
        copy = img
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        dilation = cv.dilate(gray, (21, 21))
        contours, _ = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(copy, contours, -1, (0, 0, 255), 4)
        fruits = 0
        for contour in contours:
            if cv.contourArea(contour) > 1500:
                fruits += 1
        return fruits

    def recognize_orange(img):
        frame = img
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        l_b = np.array([0, 190, 173])
        u_b = np.array([21, 255, 255])
        mask = cv.inRange(hsv, l_b, u_b)
        res = cv.bitwise_and(frame, frame, mask=mask)
        kernel = np.ones((7, 7), np.uint8)
        dil = cv.dilate(res, kernel, 3)
        return dil

    def recognize_banana(img):
        frame = img
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        l_b = np.array([18, 81, 138])
        u_b = np.array([138, 255, 255])
        mask = cv.inRange(hsv, l_b, u_b)
        res = cv.bitwise_and(frame, frame, mask=mask)
        return res

    def recognize_apples(img):
        frame = img
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        l_b = np.array([0, 52, 0])
        u_b = np.array([14, 222, 255])
        l_b1 = np.array([146, 52, 0])
        u_b1 = np.array([255, 222, 255])

        mask = cv.inRange(hsv, l_b, u_b)
        mask1 = cv.inRange(hsv, l_b1, u_b1)
        mask = cv.bitwise_or(mask, mask1)

        res = cv.bitwise_and(frame, frame, mask=mask)
        return res

    banana = detect_contours(recognize_banana(img))
    orange = detect_contours(recognize_orange(img))
    apple = detect_contours(recognize_apples(img))

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
