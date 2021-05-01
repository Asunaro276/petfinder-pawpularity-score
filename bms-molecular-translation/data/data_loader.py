import glob


def make_datapath_list(img_id_list):
    num_img = len(img_id_list)
    train_img_list = list()

    for i in range(num_img):
        img_path = glob.glob(f"train/**/{img_id_list[i]}.png", recursive=True)
        train_img_list.append(*img_path)
    return train_img_list


if __name__ == "__main__":
    import pandas as pd
    df_train = pd.read_csv("train_labels.csv").image_id[:50]
    print(make_datapath_list(df_train))
