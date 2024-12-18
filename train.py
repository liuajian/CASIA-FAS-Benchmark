import argparse
import torch

from dassl.utils import setup_logger, set_random_seed
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import datasets.casia_data
import datasets.dg_data

import util.evaluator
import util.evaluator_cdi

import trainers.clip
import trainers.clip_cdi
import trainers.coop
import trainers.icpe

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.protocol:
        cfg.DATASET.PROTOCOL = args.protocol

    if args.preprocess:
        cfg.DATASET.PREPROCESS = args.preprocess

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    cfg.DATASET.IS_VIDEO = args.is_video
    cfg.DATASET.IS_FLEXIBLE = args.is_flexible
    cfg.TEST.FINAL_MODEL = 'best_val'  ## best_val  last_step

    if args.is_flexible:
        cfg.TEST.EVALUATOR = "FAS_CDI"
    else:
        cfg.TEST.EVALUATOR = "FAS"

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER = CN()
    cfg.TRAINER.GPU = [int(s) for s in args.gpu_ids.split(',')]
    cfg.TRAINER.T = args.temperature
    cfg.TRAINER.C = args.coefficient
    cfg.TRAINER.PREC = "amp"
    cfg.TRAINER.UPDATE = False
    ## Baseline
    cfg.TRAINER.CLIP = CN()
    cfg.TRAINER.CLIP.VERSION = args.version
    cfg.TRAINER.CLIP.PROMPT = args.prompt
    cfg.TRAINER.CLIP_CDI = CN()
    cfg.TRAINER.CLIP_CDI.VERSION = args.version
    cfg.TRAINER.CLIP_CDI.PROMPT = args.prompt
    cfg.TRAINER.CLIP_DG = CN()
    cfg.TRAINER.CLIP_DG.VERSION = args.version
    cfg.TRAINER.CLIP_DG.PROMPT = args.prompt
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16        # number of context vectors
    cfg.TRAINER.COOP.CTX_INIT = ""     # initialization words
    cfg.TRAINER.COOP.CSC = False       # class-specific context (False or True)
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = args.ctp

    ## TPAMI-24
    cfg.TRAINER.ICPE = CN()
    cfg.TRAINER.ICPE.IS_LEARN = False  # false: fixed prompt, True: prompt learning
    cfg.TRAINER.ICPE.N_DACTX = 16      # number of Domain-Agnostic context vectors
    cfg.TRAINER.ICPE.N_DSCTX = 16      # number of Domain-Specific context vectors
    cfg.TRAINER.ICPE.CSC = True        # class-specific context (False or True)
    cfg.TRAINER.ICPE.PROMPT = args.prompt
    cfg.TRAINER.ICPE.N_CCTX = 32       # number of content context vectors
    cfg.TRAINER.ICPE.N_SCTX = 32       # number of style context vectors
    cfg.TRAINER.ICPE.CTX_INIT = ""     # initialization words

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    # args.eval_only = True
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        _, threshold_val = trainer.test(split="val", threshold=None)
        trainer.test(split="test", threshold=threshold_val)
        return

    if not args.no_train:
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0", help="gup")
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--protocol", type=str, default="", help="protocol")
    parser.add_argument("--is_video", type=int, default='0', help="load image form video")
    parser.add_argument("--is_flexible", type=int, default='0', help="modal fusion or flexible modal FAS")
    parser.add_argument("--preprocess", type=str, default="resize_flip", help="preprocess")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--version", type=str, default="", help="version of trainer")
    parser.add_argument("--prompt", type=str, default="", help="type of text")
    parser.add_argument("--ctp", type=str, default="", help="class token position (end or middle)")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--temperature", type=str, default="", help="temperature")
    parser.add_argument("--coefficient", type=str, default="", help="coefficient")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
