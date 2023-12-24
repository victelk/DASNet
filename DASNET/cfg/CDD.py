import os

BASE_PATH = '/mnt/tv_developers/users/victor/DASNet'
PRETRAIN_MODEL_PATH = os.path.join(BASE_PATH, 'pretrain')
DATA_PATH = '/mnt/tv_developers/users/victor/datasets/ChangeDetectionDataset/Real/subset'
TRAIN_DATA_PATH = DATA_PATH
TRAIN_LABEL_PATH = DATA_PATH
TRAIN_TXT_PATH = os.path.join(TRAIN_DATA_PATH, 'train.txt')
VAL_DATA_PATH = DATA_PATH
VAL_LABEL_PATH = DATA_PATH
VAL_TXT_PATH = os.path.join(VAL_DATA_PATH, 'val.txt')
TEST_DATA_PATH = DATA_PATH
TEST_LABEL_PATH = DATA_PATH
TEST_TXT_PATH = os.path.join(VAL_DATA_PATH, 'test.txt')
SAVE_PATH = os.path.join(BASE_PATH, 'save')
SAVE_CKPT_PATH = os.path.join(SAVE_PATH, 'ckpt')
if not os.path.exists(SAVE_CKPT_PATH):
    os.makedirs(SAVE_CKPT_PATH, exist_ok=True)
SAVE_PRED_PATH = os.path.join(SAVE_PATH, 'prediction')
if not os.path.exists(SAVE_PRED_PATH):
    os.mkdir(SAVE_PRED_PATH)
TRAINED_BEST_PERFORMANCE_CKPT = os.path.join(SAVE_CKPT_PATH, 'CDD_model_best.pth')
INIT_LEARNING_RATE = 1e-4
DECAY = 5e-5
MOMENTUM = 0.90
MAX_ITER = 40000
BATCH_SIZE = 1
THRESHS = [0.1, 0.3, 0.5]
THRESH = 0.1
LOSS_PARAM_CONV = 3
LOSS_PARAM_FC = 3
TRANSFROM_SCALES= (256, 256)
T0_MEAN_VALUE = (87.72, 100.2, 90.43)
T1_MEAN_VALUE = (120.24, 127.93, 121.18)
