import sys, os
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import readline

readline.parse_and_bind("tab: complete")


def detect_logo(yolo, img, save_img=True, save_img_path="output"):
    try:
        image = Image.open(img)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except:
        print("File Open Error! Try again!")
        return None, None

    prediction, r_image = yolo.detect_image(image)

    if save_img:
        r_image.save(os.path.join(save_img_path, os.path.basename(img)))

    return prediction, r_image


FLAGS = None

if __name__ == "__main__":
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """
    parser.add_argument(
        "--model",
        type=str,
        dest="model_path",
        help="path to model weight file, default " + YOLO.get_defaults("model_path"),
    )

    parser.add_argument(
        "--anchors",
        type=str,
        dest="anchors_path",
        help="path to anchor definitions, default " + YOLO.get_defaults("anchors_path"),
    )

    parser.add_argument(
        "--classes",
        type=str,
        dest="classes_path",
        help="path to class definitions, default " + YOLO.get_defaults("classes_path"),
    )

    parser.add_argument(
        "--gpu_num",
        type=int,
        help="Number of GPU to use, default " + str(YOLO.get_defaults("gpu_num")),
    )
    parser.add_argument(
        "--image",
        default=False,
        action="store_true",
        help="Image detection mode, will ignore all positional arguments",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Image detection mode for each file specified in input txt, will ignore all positional arguments",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        dest="score",
        default=0.3,
        help="Model confidence threshold above which to show predictions",
    )
    parser.add_argument(
        "--iou_min",
        type=float,
        dest="iou_thr",
        default=0.45,
        help="IoU threshold for pruning object candidates with higher IoU than higher score boxes",
    )
    parser.add_argument(
        "--input",
        nargs="?",
        type=str,
        required=False,
        default="input",
        help="Image or video input path",
    )

    parser.add_argument(
        "--output",
        nargs="?",
        type=str,
        default="output",
        help="output path: either directory for single/batch image, or filename for video",
    )

    FLAGS = parser.parse_args()

    if not os.path.isdir("output"):
        os.makedirs("output")

    if FLAGS.image:
        """
        Image detection mode, either prompt user input or was passed as argument
        """
        print("Image detection mode")

        yolo = YOLO(**vars(FLAGS))
        if FLAGS.input == "input":
            while True:
                FLAGS.input = input("Input image filename (q to quit):")
                if FLAGS.input in ["q", "quit"]:
                    yolo.close_session()
                    exit()

                img = FLAGS.input
                prediction, r_image = detect_logo(
                    yolo, img, save_img=True, save_img_path=FLAGS.output
                )
                if prediction is None:
                    continue

        else:
            img = FLAGS.input
            prediction, r_image = detect_logo(
                yolo, img, save_img=True, save_img_path=FLAGS.output
            )

        yolo.close_session()

    elif "batch" in FLAGS:
        print("Batch image detection mode: reading " + FLAGS.batch)

        with open(FLAGS.batch, "r") as file:
            file_list = [line.split(" ")[0] for line in file.read().splitlines()]
        out_txtfile = os.path.join(FLAGS.output, "data_pred.txt")
        txtfile = open(out_txtfile, "w")

        yolo = YOLO(**vars(FLAGS))

        for img in file_list[:10]:
            prediction, r_image = detect_logo(
                yolo, img, save_img=True, save_img_path="output"
            )

            txtfile.write(img + " ")
            for pred in prediction:
                txtfile.write(",".join([str(p) for p in pred]) + " ")
            txtfile.write("\n")

        txtfile.close()
        yolo.close_session()
    elif "video" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.video, FLAGS.output)
    else:
        print("Must specify at least --image or --video.  See usage with --help.")
