import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from functools import partial

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from clip import clip
from util.utils_FAS import cross_entropy
from trainers.trainer_fas import TrainerX

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

@TRAINER_REGISTRY.register()
class CLIP(TrainerX):
    """CLIP@V: Use only its image encoder V and discard the text encoder L.
       CLIP@VL: Use its image encoder V and the text encoder L.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]  ## fp16

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """

        dataset = build_dataset(self.cfg)
        self.train_loader_x = dataset.train_loader
        self.val_loader = dataset.dev_loader
        self.test_loader = dataset.test_loader
        self.lab2cname = dataset.lab2cname
        self.classnames = dataset.classnames
        self.templates = dataset.templates

    def build_model(self):
        cfg = self.cfg
        self.device = torch.device('cuda:%d' % cfg.TRAINER.GPU[0])
        self.version = cfg.TRAINER.CLIP.VERSION
        self.prompt = cfg.TRAINER.CLIP.PROMPT
        self.is_video = cfg.DATASET.IS_VIDEO
        self.n_cls = len(self.classnames)
        self.out_dir = cfg.OUTPUT_DIR
        self.data_root = cfg.DATASET.ROOT

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, self.preprocess = clip.load(cfg.MODEL.BACKBONE.NAME, device=self.device)
        self.dtype = clip_model.dtype

        print("Building custom CLIP@"+self.version)
        if 'VL' == self.version:
            self.model = clip_model
            if 'RN50' in cfg.MODEL.BACKBONE.NAME:
                embed_dim = 1024
            elif 'ViT-B' in cfg.MODEL.BACKBONE.NAME:
                embed_dim = 512
            self.logit_scale = clip_model.logit_scale
            self.text_encoder = TextEncoder(clip_model)

        elif 'V' == self.version:
            self.model = clip_model.visual
            if 'RN50' in cfg.MODEL.BACKBONE.NAME:
                self.model.attnpool = None
                embed_dim = 1024
            elif 'ViT-B' in cfg.MODEL.BACKBONE.NAME:
                self.model.proj = None
                embed_dim = 768

        self.model.head = nn.Linear(embed_dim, self.n_cls, bias=True)
        self.model.norm = partial(nn.LayerNorm, eps=1e-6)(embed_dim)

        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            self.model.float()

        # print("Turning on gradients in the image encoder")
        for name, param in self.model.named_parameters():
            if ('adapter' not in name) and ('head' not in name):
                param.requires_grad_(cfg.TRAINER.UPDATE)
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)
        # NOTE: only give xxx to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CLIP@"+cfg.TRAINER.CLIP.VERSION, self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None
        self.clip_model = clip_model
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        XY_R, XY_L = self.parse_batch_train(batch)

        with autocast():
            if 'VL' == self.version:
                # labels = torch.tensor(np.arange(XY_R.shape[0]), device=self.device)
                # logits_per_image, logits_per_text = self.model(XY_R, XY_T)
                # loss_i = F.cross_entropy(logits_per_image, labels)
                # loss_t = F.cross_entropy(logits_per_image, labels)
                # loss = (loss_i + loss_t) / 2.0
                if 'class' in self.prompt:
                    text_features = None
                elif 'engineering' in self.prompt:
                    text_features = self.engineering_templates(self.classnames, self.templates)
                elif 'ensembling' in self.prompt:
                    text_features = self.ensembling_templates(self.templates, self.classnames)

                logit = self.forward_VL(self.model, XY_R, text_features=text_features)
            elif 'V' == self.version:
                image_features = self.model(XY_R)
                image_features = self.model.norm(image_features.float())
                logit = self.model.head(image_features)

        loss = F.cross_entropy(logit, XY_L)
        self.optim.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update()

        loss_summary = {
            "loss": loss.item(),
            "acc": cross_entropy(logit, XY_L)[0]
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def model_inference(self, batch):
        logits = []
        with autocast():
            if 'V' == self.version:
                for idx, XY_R in enumerate(batch):
                    image_features = self.model(XY_R)
                    image_features = self.model.norm(image_features.float())
                    logit = self.model.head(image_features)
                    logits.append(logit)
            elif 'VL' == self.version:
                if 'class' in self.prompt:
                    text_features = None
                elif 'engineering' in self.prompt:
                    text_features = self.engineering_templates(self.classnames, self.templates)
                elif 'ensembling' in self.prompt:
                    text_features = self.ensembling_templates(self.classnames, self.templates)
                for idx, XY_R in enumerate(batch):
                    logit = self.forward_VL(self.model, XY_R, text_features=text_features)
                    logits.append(logit)
        logit = sum(logits) / len(logits)
        return logit

    def forward_VL(self, model, XY_R, text_features=None):
        logit_scale = self.logit_scale.exp()
        image_features = model.encode_image(XY_R)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if text_features is None:  ## class
            # Use label-conditioned prompt for each class
            prompts = [f"This is an image of {label} face." for label in self.classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = self.text_encoder(tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit = logit_scale * image_features @ text_features.t()
        return logit

    def engineering_templates(self, classnames, templates):
        classifier_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).to(self.device)           # tokenize
            class_embeddings = self.text_encoder(texts)            # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            classifier_weights.append(class_embedding)
        classifier_weights = torch.stack(classifier_weights, dim=1).to(self.device)
        return classifier_weights.t()

    def ensembling_templates(self, classnames, templates):
        num_temp = len(templates)
        mean_text_features = 0
        for i, temp in enumerate(templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = self.text_encoder(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
        return mean_text_features

    def parse_batch_train(self, batch):
        X_R, X_L, X_D = batch['X_R'].to(self.device), batch['X_L'].to(self.device), batch['X_D'].to(self.device)
        Y_R, Y_L, Y_D = batch['Y_R'].to(self.device), batch['Y_L'].to(self.device), batch['Y_D'].to(self.device)
        XY_R = torch.cat([X_R, Y_R], dim=0)
        XY_L = torch.cat([X_L, Y_L], dim=0)
        return XY_R, XY_L

    def parse_batch_test(self, batch):
        frame1, frame2, label, path = \
            batch['frame1'].to(self.device), batch['frame2'].to(self.device), \
            batch['label'].to(self.device), batch['path']
        return [frame1, frame2], label



