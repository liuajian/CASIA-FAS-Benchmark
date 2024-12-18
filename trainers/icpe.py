"""
ICPE-FAS: Instance and Category Prompts Engineering CLIP for DG FAS (TPAMI-24)
AJ
2024.12.15
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from dassl.utils import load_pretrained_weights
from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from clip import clip
from util.utils_FAS import cross_entropy
from trainers.trainer_fas import TrainerX
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

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
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, device, classnames, domainnames, clip_model):
        super().__init__()
        self.is_learn = cfg.TRAINER.ICPE.IS_LEARN
        self.n_cls = len(classnames)
        self.n_dom = len(domainnames)
        self.csc = cfg.TRAINER.ICPE.CSC
        dtype = clip_model.dtype  ## torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.vocab_size = clip_model.vocab_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ## CSPL
        domain_names = ["{}".format(domain) for domain in domainnames]
        self.n_dactx = cfg.TRAINER.ICPE.N_DACTX  ## 16
        self.n_dsctx = cfg.TRAINER.ICPE.N_DSCTX  ## 16

        if not self.is_learn:   ## fixed prompt
            prompts_mix = [f"This is an image of " + name + " face from " + domain + " domain."
                           for domain in domain_names for name in classnames]
            self.tokenized_prompts_mix = torch.cat([clip.tokenize(p) for p in prompts_mix]).to(device)   # [6, 77]
            with torch.no_grad():
                self.embedding_mix = clip_model.token_embedding(self.tokenized_prompts_mix).type(dtype)  # [6, 77, 512]

            prompts_cls = [f"This is an image of {label} face." for label in classnames]
            self.tokenized_prompts_cls = torch.cat([clip.tokenize(p) for p in prompts_cls]).to(device)   # [2, 77]
            with torch.no_grad():
                self.embedding_cls = clip_model.token_embedding(self.tokenized_prompts_cls).type(dtype)  # [2, 77, 512]

            prompts_dom = [f"from {label} domain." for label in domainnames]
            self.tokenized_prompts_dom = torch.cat([clip.tokenize(p) for p in prompts_dom]).to(device)   # [3, 77]
            with torch.no_grad():
                self.embedding_dom = clip_model.token_embedding(self.tokenized_prompts_dom).type(dtype)  # [3, 77, 512]
            print('Fixed Prompt: ', prompts_mix, prompts_cls, prompts_dom)
        else:                    ## prompt learning
            if cfg.TRAINER.ICPE.CSC:
                print("Initializing domain agnostic vectors with class-specific contexts")
                DA_ctx_vectors = torch.empty(self.n_cls, self.n_dactx, ctx_dim, dtype=dtype)
            else:
                print("Initializing domain agnostic vectors with class-generic context")
                DA_ctx_vectors = torch.empty(self.n_dactx, ctx_dim, dtype=dtype)          ## [16, 512]
            print("Initializing domain specific vectors with class-invariant contexts")
            DS_ctx_vectors = torch.empty(self.n_dom, self.n_dsctx, ctx_dim, dtype=dtype)  ## [3, 16, 512]
            nn.init.normal_(DA_ctx_vectors, std=0.02)    # to be optimized
            nn.init.normal_(DS_ctx_vectors, std=0.02)
            self.DA_ctx_vectors, self.DS_ctx_vectors = nn.Parameter(DA_ctx_vectors), nn.Parameter(DS_ctx_vectors)
            print("DA/DS_ctx_vectors size: {}/{}".format(DA_ctx_vectors.size(), DS_ctx_vectors.size()))
            print(f"Number of Domain-Agnostic/-Specific Context words (tokens): {self.n_dactx}/{self.n_dsctx}")

            self.cls_name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            self.dom_name_lens = [len(_tokenizer.encode(name)) for name in domain_names]
            prompt_prefix_mix = " ".join(["X"] * (self.n_dactx + self.n_dsctx))
            prompts_mix = [prompt_prefix_mix + " " + name + " " + domain + " " for domain in domain_names for name in classnames]
            self.tokenized_prompts_mix = torch.cat([clip.tokenize(p) for p in prompts_mix]).to(device)  # [6, 77]
            eos_list = self.tokenized_prompts_mix.argmax(dim=-1)
            with torch.no_grad():
                embedding_mix = clip_model.token_embedding(self.tokenized_prompts_mix).type(dtype)      # [6, 77, 512]
            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use those computed using the current class names
            self.register_buffer("token_prefix_mix", embedding_mix[:, :1, :])                        # SOS
            self.register_buffer("token_suffix_mix", embedding_mix[:, 1 + (self.n_dactx + self.n_dsctx):, :])  # CLS, EOS
            print('mix', prompts_mix)
            print('mix', self.tokenized_prompts_mix.shape, self.token_prefix_mix.shape, self.token_suffix_mix.shape)

            # Disentangling class feature
            cls_embeddings, tokenized_prompts_clss = 0, 0
            for i in range(0, 6, 2):
                cls_embedding, tokenized_prompts_cls = \
                self.Disentangling_cls(embedding_mix[i:i+2, :, :], self.tokenized_prompts_mix[i:i+2, :], eos_list[i:i+2])
                cls_embeddings += cls_embedding
                tokenized_prompts_clss += tokenized_prompts_cls
            cls_embedding, self.tokenized_prompts_cls = cls_embeddings/3.0, tokenized_prompts_clss/3.0  # [2, 77] [2, 77, 512]
            self.register_buffer("token_prefix_cls", cls_embedding[:, :1, :])                   # SOS
            self.register_buffer("token_suffix_cls", cls_embedding[:, 1 + (self.n_dactx):, :])  # CLS, EOS

            # Disentangling modal feature
            dom_embeddings, tokenized_prompts_doms = 0, 0
            for i in range(0, 2, 1):
                dom_embedding, tokenized_prompts_dom = \
                self.Disentangling_dom(embedding_mix[i:embedding_mix.shape[0]:2, :, :],
                self.tokenized_prompts_mix[i:embedding_mix.shape[0]:2, :], eos_list[i:embedding_mix.shape[0]:2])
                dom_embeddings += dom_embedding
                tokenized_prompts_doms += tokenized_prompts_dom
            dom_embedding, self.tokenized_prompts_dom = dom_embeddings/2.0, tokenized_prompts_doms/2.0  # [3, 77] [3, 77, 512]
            self.register_buffer("token_prefix_dom", dom_embedding[:, :1, :])                   # SOS
            self.register_buffer("token_suffix_dom", dom_embedding[:, 1 + (self.n_dsctx):, :])  # CLS, EOS

    def Disentangling_cls(self, cls_embedding, cls_prompt, cls_eos_list):
        embeddings, prompts = [], []
        for i in range(self.n_cls):
            prefix_i = cls_embedding[i:i + 1, :1, :]  # SOS
            da_ctx_i = cls_embedding[i:i + 1, 1:1 + self.n_dactx, :]  # CTX
            class_i = cls_embedding[i:i + 1, (1 + self.n_dactx + self.n_dsctx):
                                             (1 + self.n_dactx + self.n_dsctx + self.cls_name_lens[i]), :]
            end_of_i = cls_embedding[i:i + 1, cls_eos_list[i]:cls_eos_list[i] + 1, :]  # EOS
            suffix_i = cls_embedding[i:i + 1, cls_eos_list[i] + 1:, :]
            supp = cls_embedding.shape[1] - \
                   (prefix_i.shape[1] + da_ctx_i.shape[1] + class_i.shape[1] + end_of_i.shape[1] + suffix_i.shape[
                       1])
            embedding = torch.cat(
                [
                    prefix_i,  # 1
                    da_ctx_i,  # 16
                    class_i,  # 1
                    end_of_i,  # 1
                    suffix_i,  # 41
                    suffix_i[:, 0:supp, :]  # 17
                ],
                dim=1,
            )
            prompt = torch.cat(
                [
                    cls_prompt[i:i + 1, :1],
                    cls_prompt[i:i + 1, 1:1 + self.n_dactx],
                    cls_prompt[i:i + 1, (1 + self.n_dactx + self.n_dsctx):
                                        (1 + self.n_dactx + self.n_dsctx + self.cls_name_lens[i])],
                    cls_prompt[i:i + 1, cls_eos_list[i]:cls_eos_list[i] + 1],
                    cls_prompt[i:i + 1, cls_eos_list[i] + 1:],
                    cls_prompt[i:i + 1, cls_eos_list[i] + 1:][:, 0:supp]
                ],
                dim=1,
            )
            embeddings.append(embedding)
            prompts.append(prompt)
        cls_embedding = torch.cat(embeddings, dim=0)
        tokenized_prompts_cls = torch.cat(prompts, dim=0)
        return cls_embedding, tokenized_prompts_cls

    def Disentangling_dom(self, dom_embedding, dom_prompt, dom_eos_list):
        embeddings, prompts = [], []
        for i in range(self.n_dom):
            prefix_i = dom_embedding[i:i + 1, :1, :]  # SOS
            ms_ctx_i = dom_embedding[i:i + 1, 1:1 + self.n_dsctx, :]  # CTX
            domain_i = dom_embedding[i:i + 1, (1 + self.n_dactx + self.n_dsctx + self.cls_name_lens[0]):
                                              (1 + self.n_dactx + self.n_dsctx + self.cls_name_lens[0] +
                                               self.dom_name_lens[i]), :]
            end_of_i = dom_embedding[i:i + 1, dom_eos_list[i]:dom_eos_list[i] + 1, :]  # EOS
            suffix_i = dom_embedding[i:i + 1, dom_eos_list[i] + 1:, :]
            supp = dom_embedding.shape[1] - \
                   (prefix_i.shape[1] + ms_ctx_i.shape[1] + domain_i.shape[1] + end_of_i.shape[1] + suffix_i.shape[
                       1])
            embedding = torch.cat(
                [
                    prefix_i,
                    ms_ctx_i,
                    domain_i,
                    end_of_i,
                    suffix_i,
                    suffix_i[:, 0:supp, :]
                ],
                dim=1,
            )
            prompt = torch.cat(
                [
                    dom_prompt[i:i + 1, :1],
                    dom_prompt[i:i + 1, 1:1 + self.n_dsctx],
                    dom_prompt[i:i + 1, (1 + self.n_dactx + self.n_dsctx + self.dom_name_lens[0]):
                                        (1 + self.n_dactx + self.n_dsctx + self.dom_name_lens[0] +
                                         self.dom_name_lens[i])],
                    dom_prompt[i:i + 1, dom_eos_list[i]:dom_eos_list[i] + 1],
                    dom_prompt[i:i + 1, dom_eos_list[i] + 1:],
                    dom_prompt[i:i + 1, dom_eos_list[i] + 1:][:, 0:supp]
                ],
                dim=1,
            )
            embeddings.append(embedding)
            prompts.append(prompt)
        dom_embedding = torch.cat(embeddings, dim=0)
        tokenized_prompts_dom = torch.cat(prompts, dim=0)
        return dom_embedding, tokenized_prompts_dom

    def get_content(self, x):
        cls_token, pac_token = x[:, 0:1], x[:, 1:]
        pac_tmp = torch.unsqueeze(pac_token.transpose(1, 2), dim=-1)
        x = pac_tmp.reshape((pac_tmp.shape[0], pac_tmp.shape[1],) + tuple((14, 14)))
        mu = x.mean(dim=[2, 3], keepdim=True)  # compute instance mean
        var = x.var(dim=[2, 3], keepdim=True)  # compute instance variance
        sig = (var + 1e-6).sqrt()  # compute instance standard deviation
        x_normed = (x - mu) / sig
        pac_tmp = x_normed
        pac_token = pac_tmp.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token, pac_token), dim=1)
        return x

    def construct_prompts(self, ctx, prefix, suffix):
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        return prompts

    @autocast()
    def forward(self):
        ## CSPL
        if not self.is_learn:  ## fixed prompt
            prompts_mix, prompts_cls, prompts_dom = self.embedding_mix, self.embedding_cls, self.embedding_dom
            return prompts_mix, prompts_cls, prompts_dom
        else:  ## prompt learning
            prefix_mix, prefix_cls, prefix_dom = self.token_prefix_mix, self.token_prefix_cls, self.token_prefix_dom
            suffix_mix, suffix_cls, suffix_dom = self.token_suffix_mix, self.token_suffix_cls, self.token_suffix_dom
            DA_ctx_vectors, DS_ctx_vectors = self.DA_ctx_vectors, self.DS_ctx_vectors  # [16, 512] vs. [3, 16, 512]
            if DA_ctx_vectors.dim() == 2:  # class-generic  context
                DA_ctx_vectors_ = DA_ctx_vectors.unsqueeze(0).expand(self.n_dom, -1, -1)  # [16, 512]-[3, 16, 512]
                if not self.csc:
                    DA_ctx_vectors_ = DA_ctx_vectors_.unsqueeze(1).expand(-1, self.n_cls, -1,
                                                                          -1)  # [3, 16, 512]-[3, 2, 16, 512]
                    DA_ctx_vectors = DA_ctx_vectors.unsqueeze(0).expand(self.n_cls, -1, -1)  # [16, 512]-[2, 16, 512]
            else:  # class-specific context
                DA_ctx_vectors_ = DA_ctx_vectors.unsqueeze(0).expand(self.n_dom, -1, -1,
                                                                     -1)  # [2, 16, 512]-[3, 2, 16, 512]
            DS_ctx_vectors_ = DS_ctx_vectors.unsqueeze(1).expand(-1, self.n_cls, -1, -1)  # [3, 16, 512]-[3, 2, 16, 512]
            ## [3, 2, 16, 512], [3, 2, 16, 512]-> [3, 2, 32, 512]->[6, 32, 512]
            # CTX_vectors = torch.cat([DA_ctx_vectors_, DS_ctx_vectors_], dim=2).reshape(
            #     self.n_cls * self.n_dom, self.n_dactx + self.n_dsctx, self.ctx_dim)
            # prompts_mix = self.construct_prompts(CTX_vectors, prefix_mix, suffix_mix)
            prompts_cls = self.construct_prompts(DA_ctx_vectors, prefix_cls, suffix_cls)  # [2, 77, 512]
            prompts_dom = self.construct_prompts(DS_ctx_vectors, prefix_dom, suffix_dom)  # [3, 77, 512]

            return [DA_ctx_vectors_, DS_ctx_vectors_, prefix_mix, suffix_mix], prompts_cls, prompts_dom

class CustomCLIP(nn.Module):
    def __init__(self, cfg, device, classnames, domainnames, clip_model):
        super().__init__()
        self.is_learn = cfg.TRAINER.ICPE.IS_LEARN

        self.prompt_learner = PromptLearner(cfg, device, classnames, domainnames, clip_model)
        self.n_dactx, self.n_dsctx = self.prompt_learner.n_dactx, self.prompt_learner.n_dsctx
        self.n_cls, self.n_dom = self.prompt_learner.n_cls, self.prompt_learner.n_dom
        self.tokenized_prompts_mix = self.prompt_learner.tokenized_prompts_mix
        self.tokenized_prompts_cls = self.prompt_learner.tokenized_prompts_cls
        self.tokenized_prompts_dom = self.prompt_learner.tokenized_prompts_dom

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, image, label=None, domain=None):
        cls = self.image_encoder(image.type(self.dtype))
        if not self.is_learn:
            prompts_mix, prompts_cls, prompts_dom = self.prompt_learner()
        else:
            [DA_ctx_vectors_, DS_ctx_vectors_, prefix_mix, suffix_mix], prompts_cls, prompts_dom = self.prompt_learner()
            ### mix  shuffle DS_ctx_vectors_ with dim0
            ctx_dim = DS_ctx_vectors_.size(-1)  # 512
            idx = torch.randperm(DS_ctx_vectors_.shape[0])
            DS_ctx_vectors_ = DS_ctx_vectors_[idx, :].view(DS_ctx_vectors_.size())
            ## [3, 2, 16, 512], [3, 2, 16, 512]-> [3, 2, 32, 512]->[6, 32, 512]
            CTX_vectors = torch.cat([DA_ctx_vectors_, DS_ctx_vectors_], dim=2).reshape(  # [3, 2, 32, 512]-[6, 32, 512]
                self.n_cls * self.n_dom, self.n_dactx + self.n_dsctx, ctx_dim)
            prompts_mix = torch.cat([prefix_mix, CTX_vectors, suffix_mix], dim=1)

        ### Image_features ###
        image_features = cls / cls.norm(dim=-1, keepdim=True)
        image_features = image_features
        logit_scale = self.logit_scale.exp()
        ### mix
        text_features = self.text_encoder(prompts_mix, self.tokenized_prompts_mix)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_mix = logit_scale * image_features @ text_features.t()  # [bs, dim] [dim, 6]

        ### cls
        text_features = self.text_encoder(prompts_cls, self.tokenized_prompts_cls)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_cls = logit_scale * image_features @ text_features.t()  # [bs, dim] [dim, 2]

        ### dom
        text_features = self.text_encoder(prompts_dom, self.tokenized_prompts_dom)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_dom = logit_scale * image_features @ text_features.t()  # [bs, dim] [dim, 3]

        if self.prompt_learner.training:
            loss_mix = 0.0
            for d in range(self.n_dom):
                logit = logits_mix[:, d * self.n_cls:(d + 1) * self.n_cls]
                loss_mix += F.cross_entropy(logit, label)
            loss_mix /= self.n_dom
            loss_cls = F.cross_entropy(logits_cls, label)
            loss_dom = F.cross_entropy(logits_dom, domain)
            total_loss = loss_mix + loss_cls + loss_dom
            return logits_cls, total_loss
        return logits_cls

@TRAINER_REGISTRY.register()
class ICPE(TrainerX):
    """Category and Instance Prompts Engineering (ICPE).
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        """Create essential data-related attributes.
        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dataset = build_dataset(self.cfg)
        self.train_loader_x = dataset.train_loader
        self.val_loader = dataset.dev_loader
        self.test_loader = dataset.test_loader
        self.dataset = dataset
        self.lab2cname = dataset.lab2cname
        self.classnames = dataset.classnames
        self.domainnames = dataset.domainnames

    def build_model(self):
        cfg = self.cfg
        self.device = torch.device('cuda:%d' % cfg.TRAINER.GPU[0])
        self.prec = cfg.TRAINER.PREC
        self.prompt = cfg.TRAINER.ICPE.PROMPT

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess = clip.load(cfg.MODEL.BACKBONE.NAME, device=self.device)
        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        self.dtype = clip_model.dtype

        print("Building custom ICPE")
        self.model = CustomCLIP(cfg, self.device, self.classnames, self.domainnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad_(cfg.TRAINER.UPDATE)
            if "prompt_learner" in name:
                param.requires_grad_(True)
            if "adapter" in name:
                param.requires_grad_(True)
        # Double check
        print(f"Parameters to be updated:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("ICPE", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None
        self.clip_model = clip_model

    def forward_backward(self, batch):
        XY_R, XY_L, XY_D = self.parse_batch_train(batch)
        model, optim, scaler = self.model, self.optim, self.scaler

        if self.prec == "amp":
            with autocast():
                logit, loss = model(XY_R, XY_L, XY_D)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logit, loss = model(XY_R, XY_L, XY_D)
            optim.zero_grad()
            loss.backward()
            optim.step()

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
            for idx, XY_R in enumerate(batch):
                logit = self.model(XY_R)
                logits.append(logit)
        logit = sum(logits) / len(logits)
        return logit

    def parse_batch_train(self, batch):
        X_R, X_L, X_D = batch['X_R'].to(self.device), batch['X_L'].to(self.device), batch['X_D'].to(self.device)
        Y_R, Y_L, Y_D = batch['Y_R'].to(self.device), batch['Y_L'].to(self.device), batch['Y_D'].to(self.device)

        split = int(Y_R.shape[0] / 3)
        X_Rs = torch.split(X_R, split_size_or_sections=split, dim=0)
        Y_Rs = torch.split(Y_R, split_size_or_sections=split, dim=0)
        X_Ls = torch.split(X_L, split_size_or_sections=split, dim=0)
        Y_Ls = torch.split(Y_L, split_size_or_sections=split, dim=0)
        X_Ds = torch.split(X_D, split_size_or_sections=split, dim=0)
        Y_Ds = torch.split(Y_D, split_size_or_sections=split, dim=0)
        XY_R = torch.cat([torch.cat([X_Rs[0], Y_Rs[0]], dim=0),
                               torch.cat([X_Rs[1], Y_Rs[1]], dim=0),
                               torch.cat([X_Rs[2], Y_Rs[2]], dim=0)], dim=0)
        XY_L = torch.cat([torch.cat([X_Ls[0], Y_Ls[0]], dim=0),
                               torch.cat([X_Ls[1], Y_Ls[1]], dim=0),
                               torch.cat([X_Ls[2], Y_Ls[2]], dim=0)], dim=0)
        XY_D = torch.cat([torch.cat([X_Ds[0], Y_Ds[0]], dim=0),
                               torch.cat([X_Ds[1], Y_Ds[1]], dim=0),
                               torch.cat([X_Ds[2], Y_Ds[2]], dim=0)], dim=0)
        return XY_R, XY_L, XY_D

    def parse_batch_test(self, batch):
        frame1, frame2, label, path = \
            batch['frame1'].to(self.device), batch['frame2'].to(self.device), \
            batch['label'].to(self.device), batch['path']
        return [frame1, frame2], label



