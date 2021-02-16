import cv2
import os
import pandas as pd
from controller import cvision

affect_net_expressions = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt'
}
raf_db_expression_mapping = {
    "Surprise": 1,
    "Fear": 2,
    "Disgust": 3,
    "Happy": 4,
    "Sad": 5,
    "Anger": 6,
    "Neutral": 7,
    "Contempt": -1,
    None: -42
}


def get_subgroup_df_row_generator(csv_path, subgroup_path):
    """
    Loads benchmark subgroup dataframe containing "file" (str), "gender" (int), "race" (int), "expression_id" (int)
    :param csv_path: path to sub-group .csv file.
    :return: DataFrame for sub-group.
    """
    col_names = ["file", "gender", "race", "expression_id"]
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        file_path = os.path.join(subgroup_path, row[col_names[0]])
        yield index, file_path, row[col_names[1]], row[col_names[2]], row[col_names[3]]


def predict_single_image(img_path):
    # Read an image
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Recognize a facial expression if a face is detected.
    # The boolean argument set to False indicates that the process runs on CPU
    fer = cvision.recognize_facial_expression(image, True, None, False)

    # Print list of emotions (individual classification from 9 convolutional branches and the ensemble classification)
    # Last prediction is ensemble prediction
    # print(fer.list_emotion)
    voted_emotion = fer.list_emotion[-1] if fer.list_emotion is not None else None
    return voted_emotion


def predict_expressions_for_subgroup(csv_path, subgroup_path):
    expressions = dict()
    for idx, image_path, gender, race, expression_id in get_subgroup_df_row_generator(csv_path, subgroup_path):
        print(image_path)

        pred_expression = predict_single_image(image_path)
        expressions[idx] = {
            "image_path": image_path,
            "gender": gender,
            "race": race,
            "expression_id": expression_id,
            "expression_id_esr": raf_db_expression_mapping[pred_expression],
        }
    df = pd.DataFrame.from_dict(expressions, orient="index")
    return df


def run_prediction(benchmark_dir, races, genders, expressions, base_dataset_name):
    for race in races:
        for gender in genders:
            for expression in expressions:
                images_path = os.path.join(benchmark_dir, race, gender, expression)
                csv_path = os.path.join(images_path, f"subgroup--{race}_{gender}_{expression}.csv")

                # predict
                subgroup_df = predict_expressions_for_subgroup(csv_path, images_path)

                # save results to .csv
                target_path = os.path.join(images_path,
                                           f"pred--{race}_{gender}_{expression}--esr-{base_dataset_name}-no-detection.csv")
                subgroup_df.to_csv(target_path, index=False)


if __name__ == '__main__':
    benchmark_dir = "/home/steffi/dev/data/FER-benchmark"
    races = ["Caucasian", "African-American", "Asian"]
    genders = ["Male", "Female"]
    expressions = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

    base_dataset_name = "affectnet"
    img_path = "/home/steffi/dev/independent_study/ESR/images/test_1331_aligned.jpg"
    run_prediction(benchmark_dir, races, genders, expressions, base_dataset_name)
