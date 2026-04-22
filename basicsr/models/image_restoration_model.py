import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import torch
import torch.nn.functional as F

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module("basicsr.models.losses")
metric_module = importlib.import_module("basicsr.metrics")


class ImageCleanModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = define_network(deepcopy(opt["network_g"]))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt["path"].get("pretrain_network_g")
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key=self.opt["path"].get("param_key", "params"),
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]
        self.ema_decay = train_opt.get("ema_decay", 0)

        pixel_type = train_opt["pixel_opt"].pop("type")
        cri_pix_cls = getattr(loss_module, pixel_type)
        self.cri_pix = cri_pix_cls(**train_opt["pixel_opt"]).to(self.device)

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = [v for _, v in self.net_g.named_parameters() if v.requires_grad]

        optim_type = train_opt["optim_g"].pop("type")
        if optim_type == "Adam":
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt["optim_g"])
        elif optim_type == "AdamW":
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt["optim_g"])
        else:
            raise NotImplementedError(f"Unsupported optimizer: {optim_type}")
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data["lq"].to(self.device)
        self.gt = data["gt"].to(self.device)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)

        if isinstance(preds, list):
            self.output = preds[-1]
            loss_input = preds
        elif isinstance(preds, dict):
            loss_input = preds["img"]
            self.output = loss_input[-1] if isinstance(loss_input, list) else loss_input
        else:
            loss_input = preds
            self.output = preds

        l_pix = self.cri_pix(loss_input, self.gt)
        l_total = l_pix + 0.0 * sum(p.sum() for p in self.net_g.parameters())
        l_total.backward()

        if self.opt["train"].get("use_grad_clip", True):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(OrderedDict(l_pix=l_pix, l_total=l_total))

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get("scale", 1)
        _, _, h, w = self.lq.size()
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        self.nonpad_test(img)
        _, _, out_h, out_w = self.output.size()
        self.output = self.output[:, :, : out_h - mod_pad_h * scale, : out_w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        img = self.lq if img is None else img
        self.net_g.eval()
        with torch.no_grad():
            pred = self.net_g(img)
        if isinstance(pred, list):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred["img"]
            if isinstance(pred, list):
                pred = pred[-1]
        self.output = pred
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        rank, _ = get_dist_info()
        if rank == 0:
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        return 0.0

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt["val"]["metrics"].keys()}

        window_size = self.opt["val"].get("window_size", 0)
        test_fn = self.pad_test if window_size else self.nonpad_test

        cnt = 0
        max_num = self.opt["val"].get("max_num", 4000)
        for idx, val_data in enumerate(dataloader):
            if idx >= max_num:
                break

            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            if window_size:
                test_fn(window_size)
            else:
                test_fn()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]], rgb2bgr=rgb2bgr)
            gt_img = tensor2img([visuals["gt"]], rgb2bgr=rgb2bgr) if "gt" in visuals else None

            del self.lq
            del self.output
            if hasattr(self, "gt"):
                del self.gt
            torch.cuda.empty_cache()

            if save_img:
                if self.opt["is_train"]:
                    save_img_path = osp.join(self.opt["path"]["visualization"], img_name, f"{img_name}_{current_iter}.png")
                else:
                    save_img_path = osp.join(self.opt["path"]["visualization"], dataset_name, f"{img_name}.png")
                imwrite(sr_img, save_img_path)

            if with_metrics and gt_img is not None:
                opt_metric = deepcopy(self.opt["val"]["metrics"])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop("type")
                        self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop("type")
                        self.metric_results[name] += getattr(metric_module, metric_type)(
                            visuals["result"], visuals["gt"], **opt_
                        )
            cnt += 1

        current_metric = 0.0
        if with_metrics and cnt > 0:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name},\t"
        for metric, value in self.metric_results.items():
            log_str += f"\t# {metric}: {value:.4f}"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], "net_g", current_iter, param_key=["params", "params_ema"])
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)
