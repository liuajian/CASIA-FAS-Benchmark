import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from util.utils_FAS import cross_entropy
_tokenizer = _Tokenizer()
# from trainers.trainer_intra import TrainerX

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, device, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)                           ## 101
        n_ctx = cfg.TRAINER.COOP.N_CTX                    ## 16
        ctx_init = cfg.TRAINER.COOP.CTX_INIT              ## ""
        dtype = clip_model.dtype                          ## torch.float16
        ctx_dim = clip_model.ln_final.weight.shape[0]     ## 512
        clip_imsize = clip_model.visual.input_resolution  ## 224
        cfg_imsize = cfg.INPUT.SIZE[0]                    ## 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)  ## [1, 77]
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)    ## [1, 77, 512]
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]  ### remove sot and eot: [n_ctx, 512]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized   [n_ctx, 512]
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)         ## [101, 77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)             ## [101, 77, 512]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        ### tokenized_prompts: [SOS(:1), Prompt(1:n_ctx+1), CLS&EOS(1 + n_ctx:)]
        self.register_buffer("token_prefix", embedding[:, :1, :])          # SOS       [101, 1, 512]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS  [101, 60, 512]
        self.n_cls = n_cls  ## 101
        self.n_ctx = n_ctx  ## 16
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION   ## end

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]

                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, device, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, device, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]  ## fp16

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dataset = build_dataset(self.cfg)
        self.train_loader_x = dataset.train_loader_x
        self.val_loader = dataset.val_loader
        self.test_loader = dataset.test_loader
        self.dataset = dataset
        self.lab2cname = dataset.lab2cname
        self.classnames = dataset.classnames
        self.templates = dataset.templates

    def build_model(self):
        cfg = self.cfg
        self.device = torch.device('cuda:%d' % cfg.TRAINER.GPU[0])

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess = clip.load(cfg.MODEL.BACKBONE.NAME, device=self.device)
        self.dtype = clip_model.dtype
        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.device, self.classnames, clip_model)

        # print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad_(cfg.TRAINER.UPDATE)
            if "text_encoder" in name:
                param.requires_grad_(False)
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CoOp", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        XY_R, XY_T, XY_L, XY_D = self.parse_batch_train(batch)  ## [32, 3, 224, 224] [32]

        prec = self.cfg.TRAINER.PREC
        if prec == "amp":
            with autocast():
                output = self.model(XY_R)
                loss = F.cross_entropy(output, XY_L)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(XY_R)
            loss = F.cross_entropy(output, XY_L)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": cross_entropy(output, XY_L)[0]
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        X_R, X_T, X_L, X_D = batch['X_R'].to(self.device), \
                             batch['X_T'], batch['X_L'].to(self.device), batch['X_D'].to(self.device)
        Y_R, Y_T, Y_L, Y_D = batch['Y_R'].to(self.device), \
                             batch['Y_T'], batch['Y_L'].to(self.device), batch['Y_D'].to(self.device)

        domain_num = int(max(X_D)) + 1
        split = int(Y_R.shape[0] / domain_num)
        X_Rs = torch.split(X_R, split_size_or_sections=split, dim=0)
        Y_Rs = torch.split(Y_R, split_size_or_sections=split, dim=0)
        X_Ts = [X_T[i:i + split] for i in range(0, len(X_T), split)]
        Y_Ts = [Y_T[i:i + split] for i in range(0, len(Y_T), split)]
        X_Ls = torch.split(X_L, split_size_or_sections=split, dim=0)
        Y_Ls = torch.split(Y_L, split_size_or_sections=split, dim=0)
        X_Ds = torch.split(X_D, split_size_or_sections=split, dim=0)
        Y_Ds = torch.split(Y_D, split_size_or_sections=split, dim=0)

        XY_R, XY_T, XY_L, XY_D = [], [], [], []
        for d in range(domain_num):
            XY_R.append(torch.cat([X_Rs[d], Y_Rs[d]], dim=0))
            XY_L.append(torch.cat([X_Ls[d], Y_Ls[d]], dim=0))
            XY_D.append(torch.cat([X_Ds[d], Y_Ds[d]], dim=0))
            XY_T += (X_Ts[d] + Y_Ts[d])
        XY_R = torch.cat(XY_R, dim=0)
        XY_L = torch.cat(XY_L, dim=0)
        XY_D = torch.cat(XY_D, dim=0)
        return XY_R, XY_T, XY_L, XY_D

    def parse_batch_test(self, batch):
        frame1, frame2, label, text1, text2 = \
            batch['frame1'].to(self.device), batch['frame2'].to(self.device), \
            batch['label'].to(self.device), batch['text1'], batch['text2']
        # frame1 = frame1.type(self.dtype)
        return frame1, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
