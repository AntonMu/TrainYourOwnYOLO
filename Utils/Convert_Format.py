from os import path, makedirs
import pandas as pd
import numpy as np
import re
import os
from PIL import Image
from Get_File_Paths import GetFileList, ChangeToOtherMachine


def convert_vott_csv_to_yolo(
    vott_df,
    labeldict=dict(
        zip(
            ["Cat_Face"],
            [
                0,
            ],
        )
    ),
    path="",
    target_name="data_train.txt",
    abs_path=False,
):

    # Encode labels according to labeldict if code's don't exist
    if not "code" in vott_df.columns:
        vott_df["code"] = vott_df["label"].apply(lambda x: labeldict[x])
    # Round float to ints
    for col in vott_df[["xmin", "ymin", "xmax", "ymax"]]:
        vott_df[col] = (vott_df[col]).apply(lambda x: round(x))

    # Create Yolo Text file
    last_image = ""
    txt_file = ""

    for index, row in vott_df.iterrows():
        if not last_image == row["image"]:
            if abs_path:
                txt_file += "\n" + row["image_path"] + " "
            else:
                txt_file += "\n" + os.path.join(path, row["image"]) + " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                ]
            )
        else:
            txt_file += " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                ]
            )
        last_image = row["image"]
    file = open(target_name, "w")
    file.write(txt_file[1:])
    file.close()
    return True


def csv_from_xml(directory, path_name=""):
    # First get all images and xml files from path and its subfolders
    image_paths = GetFileList(directory, ".jpg")
    xml_paths = GetFileList(directory, ".xml")
    result_df = pd.DataFrame()
    if not len(image_paths) == len(xml_paths):
        print("number of annotations doesnt match number of images")
        return False
    for image in image_paths:
        target_filename = os.path.join(path_name, image) if path_name else image
        source_filename = os.path.join(directory, image)
        y_size, x_size, _ = np.array(Image.open(source_filename)).shape
        source_xml = image.replace(".jpg", ".xml")
        txt = open(source_xml, "r").read()
        y_vals = re.findall(r"(?:x>\n)(.*)(?:\n</)", txt)
        ymin_vals = y_vals[::2]
        ymax_vals = y_vals[1::2]
        x_vals = re.findall(r"(?:y>\n)(.*)(?:\n</)", txt)
        xmin_vals = x_vals[::2]
        xmax_vals = x_vals[1::2]
        label_vals = re.findall(r"(?:label>\n)(.*)(?:\n</)", txt)
        label_name_vals = re.findall(r"(?:labelname>\n)(.*)(?:\n</)", txt)
        df = pd.DataFrame()
        df["xmin"] = xmin_vals
        df["xmin"] = df["xmin"].astype(float) * x_size
        df["ymin"] = ymin_vals
        df["ymin"] = df["ymin"].astype(float) * y_size
        df["xmax"] = xmax_vals
        df["xmax"] = df["xmax"].astype(float) * x_size
        df["ymax"] = ymax_vals
        df["ymax"] = df["ymax"].astype(float) * y_size
        df["label"] = label_name_vals
        df["code"] = label_vals
        df["image_path"] = target_filename
        df["image"] = os.path.basename(target_filename)
        result_df = result_df.append(df)
    #     Bring image column first
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    result_df = result_df[cols]
    return result_df


def crop_and_save(
    image_df,
    target_path,
    target_file,
    one=True,
    label_dict={0: "house"},
    postfix="cropped",
):
    """Takes a vott_csv file with image names, labels and crop_boxes
    and crops the images accordingly

    Input csv file format:

    image   xmin ymin xmax ymax label
    im.jpg  0    10   100  500  house


    Parameters
    ----------
    df : pd.Dataframe
        The input dataframe with file_names, bounding box info
        and label
    source_path : str
        Path of source images
    target_path : str, optional
        Path to save cropped images
    one : boolean, optional
        if True, only the most central house will be returned

    Returns
    -------
    True if completed succesfully
    """
    if not path.isdir(target_path):
        makedirs(target_path)

    previous_name = ""
    counter = 0
    image_df.dropna(inplace=True)
    image_df["image_path"] = ChangeToOtherMachine(image_df["image_path"].values)

    def find_rel_position(row):
        current_name = row["image_path"]
        x_size, _ = Image.open(current_name).size
        x_centrality = abs((row["xmin"] + row["xmax"]) / 2 / x_size - 0.5)
        return x_centrality

    if one:
        centrality = []
        for index, row in image_df.iterrows():
            centrality.append(find_rel_position(row))
        image_df["x_centrality"] = pd.Series(centrality)
        image_df.sort_values(["image", "x_centrality"], inplace=True)
        image_df.drop_duplicates(subset="image", keep="first", inplace=True)
    new_paths = []
    for index, row in image_df.iterrows():
        current_name = row["image_path"]
        if current_name == previous_name:
            counter += 1
        else:
            counter = 0
        imageObject = Image.open(current_name)
        cropped = imageObject.crop((row["xmin"], row["ymin"], row["xmax"], row["ymax"]))
        label = row["label"]
        if type(label) == int:
            label = label_dict[label]
        image_name_cropped = (
            "_".join([row["image"][:-4], postfix, label, str(counter)]) + ".jpg"
        )
        new_path = os.path.join(target_path, image_name_cropped)
        cropped.save(new_path)
        new_paths.append(new_path.replace("\\", "/"))
        previous_name = current_name
    pd.DataFrame(new_paths, columns=["image_path"]).to_csv(target_file)
    return True


if __name__ == "__main__":
    # Prepare the houses dataset for YOLO
    labeldict = dict(
        zip(
            ["house"],
            [
                0,
            ],
        )
    )
    multi_df = pd.read_csv(
        "C:/Users/Anton/Documents/Insight/eq/EQ_new/Train_Housing_detector/2/vott-csv-export/Housing_cropping-export.csv"
    )
    multi_df.drop_duplicates(subset=None, keep="first", inplace=True)
    convert_vott_csv_to_yolo(
        multi_df,
        labeldict,
        path="/home/ubuntu/logohunter/data/houses/",
        target_name="data_train.txt",
    )

    # Prepare the windows dataset for YOLO
    path = "C:/Users/Anton/Documents/Insight/eq/EQ_new/Train_Window_Detector/base"
    csv_from_xml(path, "/home/ubuntu/logohunter/data/windows").to_csv(
        "C:/Users/Anton/Documents/Insight/eq/EQ_new/Train_Window_Detector/base/annotations.csv"
    )

    label_names = [
        "background",
        "facade",
        "molding",
        "cornice",
        "pillar",
        "window",
        "door",
        "sill",
        "blind",
        "balcony",
        "shop",
        "deco",
    ]
    labeldict = dict(zip(label_names, list(range(12))))
    convert_vott_csv_to_yolo(
        csv_from_xml(path, "/home/ubuntu/logohunter/data/windows"), labeldict
    )
