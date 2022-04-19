from utils.pretrain_data import load_data

def main():

    cfg = {'cngt_clips_path': "D:/Thesis/datasets/cngt_train_clips.zip",
           'signbank_path': "D:/Thesis/datasets/NGT_Signbank_resized.zip",
           'window_size': 64,
           'window_stride': 32,
           'local': False}

    data_train = load_data(cfg, "train")


if __name__ == "__main__":
    main()