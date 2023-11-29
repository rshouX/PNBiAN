from vocab import Vocabulary
import evaluation
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RUN_PATH = "./runs/runX/checkpoint/model_best.pth.tar"
DATA_PATH = "/root/autodl-tmp/"

evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="test",fold5=True)
