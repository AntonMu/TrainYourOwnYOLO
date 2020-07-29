"""
Python script to generate custom anchors for your dataset
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

src_path = os.path.join(get_parent_dir(0), "src")
sys.path.append(src_path)

keras_path = os.path.join(src_path, "keras_yolo3")
Data_Folder = os.path.join(get_parent_dir(1), "Data")
Image_Folder = os.path.join(Data_Folder, "Source_Images", "Training_Images")
VoTT_Folder = os.path.join(Image_Folder, "vott-csv-export")
YOLO_filename = os.path.join(VoTT_Folder, "data_train.txt")

anchors_dir = os.path.join(keras_path, "model_data")

FLAGS = None

if __name__ == "__main__":
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """

    parser.add_argument(
        "--annotation_file",
        type=str,
        default=YOLO_filename,
        help="Path to annotation file for Yolo. Default is " + YOLO_filename,
    )

    parser.add_argument(
        "--anchors_dir",
        type=str,
        default=anchors_dir,
        help="Path to store YOLO anchors. Default is " + anchors_dir,
    )

    parser.add_argument(
        "--tiny",
        default=False,
        action="store_true",
        help="Store the Anchors for yolo tiny. Default is False"
    )

    parser.add_argument(
        "--show_plot",
        default=False,
        action="store_true",
        help="Show the scatter plot of widths and heights of bounding boxes. Default is False",
    )

    FLAGS = parser.parse_args()

    with open(FLAGS.annotation_file) as f:
        lines = f.readlines()
    
    # retrieving the bounding box coordinates
    boxes = []
    for line in lines:
        splits = line.split(" ")
        annots = splits[1:]
        for annot in annots:
            annot = [int(x) for x in annot.split(',')]
            annot = annot[:-1]
            boxes.append([annot[2] - annot[0],  annot[3] - annot[1]])

    # converting to numpy array for easy computation
    boxes = np.array(boxes).reshape(-1, 2)

    # calculation of intersection of union => helper function for kmeans
    def iou(box, clusters):
        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])

        intersection = x * y
        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]

        iou_ = intersection/ (box_area + cluster_area - intersection)

        return iou_

    # kmeans
    def kmeans(boxes, k, dist = np.median):
        num_boxes = boxes.shape[0]

        dists = np.empty((num_boxes, k))
        last_clusters = np.zeros((num_boxes, ))

        np.random.seed()
        clusters = boxes[np.random.choice(num_boxes, k, replace=False)]

        while True:
            for num  in range(num_boxes):
                dists[num] = 1 - iou(boxes[num], clusters)

            nearest_clusters = np.argmin(dists, axis = 1)

            if (last_clusters == nearest_clusters).all():
                break

            for cluster in range(k):
                clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis = 0)

            last_clusters = nearest_clusters

        return clusters

    
    # arrange the anchors in terms of area 
    def arrange_anchors(anchors):
        areas = np.zeros((anchors.shape[0],))
        for i in range(anchors.shape[0]):
            areas[i] = anchors[i, 0] * anchors[i, 1]

        sort_ind = np.argsort(areas)

        sort_anchors = anchors[sort_ind, :]

        return sort_anchors

    
    if FLAGS.show_plot:
        plt.title("Distribution of bboxes")
        plt.scatter(boxes[:, 0], boxes[:, 1])
        plt.xlabel('widths')
        plt.ylabel('heights')
        plt.show()
    
    yolo_anchors = list(np.int32(arrange_anchors(kmeans(boxes, 9)).reshape(-1,)))
    yolo_tiny_anchors = list(np.int32(arrange_anchors(kmeans(boxes, 6)).reshape(-1,)))

    f = open(os.path.join(FLAGS.anchors_dir, "my_yolo_anchors.txt"), "w+")
    anchor_string = ','.join([str(x) for x in yolo_anchors])
    f.write(anchor_string)
    f.close()
    
    if FLAGS.tiny:
        f = open(os.path.join(FLAGS.anchors_dir, "my_tiny_yolo_anchors.txt"), "w+")
        anchor_string = ','.join([str(x) for x in yolo_tiny_anchors])
        f.write(anchor_string)
        f.close()