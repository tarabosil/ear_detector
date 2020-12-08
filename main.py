import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import config

leftear_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_leftear.xml')
righrear_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_rightear.xml')
nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')


def train_data():

    images = os.listdir(config.path_to_train_data)

    for img_name in images:
        img = cv2.imread(f'{config.path_to_train_data}/{img_name}')
        img2 = cv2.imread(f'{config.path_to_annotated_data}/{img_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        viola_jones(img, img_name, img2, gray)


def viola_jones(img, img_name, img2, gray):
    """ Viola Jones algortihm for detecting ears """

    right_ears = righrear_cascade.detectMultiScale(
        gray,
        scaleFactor=config.right_step,
        minNeighbors=config.right_size,
        minSize=(20, 20),
        maxSize=(100, 100)
    )
    left_ears = leftear_cascade.detectMultiScale(
        gray,
        scaleFactor=config.left_step,
        minNeighbors=config.left_size,
        minSize=(20, 20),
        maxSize=(100, 100)
    )

    if config.use_nose_detection:

        nose = nose_cascade.detectMultiScale(
            gray,
            scaleFactor=1.01,
            minNeighbors=3,
            maxSize=(100, 100)
        )

        for (xn, yn, wn, hn) in nose:
            print("Nose: ", xn)

            right_found = False
            for (x, y, w, h) in right_ears:
                if not right_found and x > xn:
                    print("Righr: ", x)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    cv2.putText(img, 'Right ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    right_found = True
                else:
                    break
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), -1)
                # cv2.putText(img, 'Right ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            left_found = False
            for (x, y, w, h) in left_ears:
                if not left_found and x < xn:
                    print("Left: ", x)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    cv2.putText(img, 'Left ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                    left_found = True
                else:
                    break
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), -1)
                # cv2.putText(img, 'Left ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            break
    else:
        right_found = False
        for (x, y, w, h) in right_ears:
            if not right_found:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), -1)
                cv2.putText(img, 'Right ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                right_found = True
            else:
                break
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), -1)
            # cv2.putText(img, 'Right ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        left_found = False
        for (x, y, w, h) in left_ears:
            if not left_found:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), -1)
                cv2.putText(img, 'Left ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                left_found = True
            else:
                break
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), -1)
            # cv2.putText(img, 'Left ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    # cv2.imshow('monkey', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(f'data/test_viola/{img_name}', img)
    cv2.imwrite(f'data/test_viola_rect/{img_name}', img2)
    print(img_name)
    cv2.waitKey()
    cv2.destroyAllWindows()

    images = [img]

    return images


if __name__ == "__main__":
    train_data()