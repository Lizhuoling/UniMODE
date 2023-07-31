# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, List, Optional
import pdb
import numpy as np
import cv2
import copy
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.config import CfgNode as CN
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.data import MetadataCatalog

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
import maskrcnn_benchmark.structures.image_list as maskrcnn_ImageList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from cubercnn.modeling.detector3d.build import build_3d_detector

@META_ARCH_REGISTRY.register()
class OV_3D_Det(BaseModule):
    def __init__(self, cfg, priors=None):
        super().__init__()
        self.cfg = cfg
        self.fp16_enabled = True
        self.img_mean = torch.Tensor(self.cfg.MODEL.PIXEL_MEAN)[:, None, None]
        self.img_std = torch.Tensor(self.cfg.MODEL.PIXEL_STD)[:, None, None]

        self.thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
        self.dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id

        self.care_thing_classes = self.cfg.DATASETS.CATEGORY_NAMES
        self.local2global_cls_map = dict()
        for src_idx, care_cls in enumerate(self.care_thing_classes):
            tgt_idx = self.thing_classes.index(care_cls)
            self.local2global_cls_map[src_idx] = tgt_idx
        
        ### Build the GLIP model ###
        self.glip_cfg = copy.deepcopy(cfg.MODEL.GLIP_MODEL)
        self.glip_cfg.freeze()
        if self.glip_cfg.USE_GLIP:
            self.glip_model = build_detection_model(self.glip_cfg, cfg)

            if self.glip_cfg.GLIP_WEIGHT != "":
                checkpointer = DetectronCheckpointer(cfg, self.glip_model, save_dir="")
                _ = checkpointer.load(self.glip_cfg.GLIP_WEIGHT)
        
        # Prepare the content for GLIP detection.
        if cfg.MODEL.DETECTOR3D.OV_PROTOCOL:
            class_name_list = cfg.DATASETS.CATEGORY_NAMES + cfg.DATASETS.NOVEL_CLASS_NAMES
        else:
            class_name_list = cfg.DATASETS.CATEGORY_NAMES
        categories = {i:class_name for i, class_name in enumerate(class_name_list)}
        self.all_queries, self.all_positive_map_label_to_token = create_queries_and_maps_from_dataset(categories, self.glip_cfg)
        assert len(self.all_queries) == 1

        self.forward_once_flag = False

        ### Build the 3D Det model ###
        self.detector = build_3d_detector(cfg)

        #self.max_range = [9999, -9999, 9999, -9999, 9999, -9999]    # For debug

    @torch.no_grad()
    def forward_once(self):
        if not self.forward_once_flag:
            with torch.no_grad():
                # Produce the class name token embedding
                tokenized = self.glip_model.tokenizer.batch_encode_plus(self.all_queries, 
                                                            max_length=self.glip_cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                            padding='max_length' if self.glip_cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True,
                                                            return_tensors='pt',
                                                            truncation=True).to('cuda')
                tokenizer_input = {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask}
                language_dict_features = self.glip_model.language_backbone(tokenizer_input)
                class_name_token_emb = language_dict_features['embedded'][language_dict_features['masks'].bool()]

                # Transform the class name token embedding to the class name embedding
                self.class_name_emb = []
                all_positive_map_label_to_token = self.all_positive_map_label_to_token[0]
                for key, value in all_positive_map_label_to_token.items():
                    cls_token_emb = class_name_token_emb[value].mean(dim = 0, keepdim = True)
                    self.class_name_emb.append(cls_token_emb)
                self.class_name_emb = torch.cat(self.class_name_emb, dim = 0)
            
            self.forward_once_flag = True

    def pad_glip_outs(self, glip_outs):
        desired_box_num = self.cfg.MODEL.GLIP_MODEL.MODEL.ATSS.DETECTIONS_PER_IMG
        for glip_out in glip_outs:
            box_num = glip_out.bbox.shape[0]
            if box_num == desired_box_num:
                continue
            # Note: For unknown reason, the output box number by GLIP can be bigger than DETECTIONS_PER_IMG (such as 101 vs 100). I speculate this may be a bug in GLIP.
            elif box_num > desired_box_num:
                glip_out.bbox = glip_out.bbox[:desired_box_num]
                glip_out.extra_fields['scores'] = glip_out.extra_fields['scores'][:desired_box_num]
                glip_out.extra_fields['labels'] = glip_out.extra_fields['labels'][:desired_box_num]
                glip_out.extra_fields['cls_emb'] = glip_out.extra_fields['cls_emb'][:desired_box_num]
            else:
                pad_num = desired_box_num - box_num 
                glip_out.bbox = torch.cat((glip_out.bbox, glip_out.bbox.new_zeros(pad_num, 4)), dim = 0)
                glip_out.extra_fields['scores'] = torch.cat((glip_out.extra_fields['scores'], glip_out.extra_fields['scores'].new_zeros(pad_num,)), dim = 0)
                glip_out.extra_fields['labels'] = torch.cat((glip_out.extra_fields['labels'], glip_out.extra_fields['labels'].new_zeros(pad_num,)), dim = 0)
                glip_out.extra_fields['cls_emb'] = torch.cat((glip_out.extra_fields['cls_emb'], glip_out.extra_fields['cls_emb'].new_zeros(pad_num, glip_out.extra_fields['cls_emb'].shape[-1])), dim = 0)
        return glip_outs

    @auto_fp16(apply_to=('images',), out_fp32=True)
    def forward_glip(self, images, captions, positive_map_label_to_token):
        glip_outs = self.glip_model(images, captions=captions, positive_map=positive_map_label_to_token)
        return glip_outs

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):

        # For survey the data statistics. For debug
        '''for batch in batched_inputs:
            xyz = batch['instances'].gt_boxes3D[:, 6:9] # 9 numbers in gt_boxes3D: projected 2D center, depth, w, h, l, 3D center
            if xyz[:, 0].min() < self.max_range[0]:
                self.max_range[0] = xyz[:, 0].min()
            if xyz[:, 0].max() > self.max_range[1]:
                self.max_range[1] = xyz[:, 0].max()
            if xyz[:, 1].min() < self.max_range[2]:
                self.max_range[2] = xyz[:, 1].min()
            if xyz[:, 1].max() > self.max_range[3]:
                self.max_range[3] = xyz[:, 1].max()
            if xyz[:, 2].min() < self.max_range[4]:
                self.max_range[4] = xyz[:, 2].min()
            if xyz[:, 2].max() > self.max_range[5]:
                self.max_range[5] = xyz[:, 2].max()
            print('max_range:', self.max_range)
        return'''

        if self.cfg.MODEL.GLIP_MODEL.USE_GLIP:
            self.forward_once()
        else:
            self.class_name_emb = None

        images = self.preprocess_image(batched_inputs)
        ori_img_resolution = torch.Tensor([(img.shape[2], img.shape[1]) for img in images]).to(images.device)   # Left shape: (bs, 2)

        # OV 2D Det
        if self.glip_cfg.NO_GRADIENT:
            no_gradient_wrapper = torch.no_grad
        else:
            no_gradient_wrapper = pseudo_with

        if self.training:
            batched_inputs = self.remove_invalid_gts(batched_inputs)
        novel_cls_id_list = []
        if self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL:
            for novel_cls in self.cfg.DATASETS.NOVEL_CLASS_NAMES:
                novel_cls_id = self.thing_classes.index(novel_cls)
                novel_cls_id_list.append(novel_cls_id)

        if self.cfg.MODEL.GLIP_MODEL.GLIP_INITIALIZE_QUERY or (self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL and self.training and self.cfg.MODEL.DETECTOR3D.PETR.HEAD.DET2D_FROM == 'GLIP'):
            with no_gradient_wrapper():
                query_i = 0   # Only support one query sentence.

                captions = [self.all_queries[query_i] for ii in range(len(batched_inputs))]
                positive_map_label_to_token = self.all_positive_map_label_to_token[query_i]
                    
                if self.glip_model.training:
                    self.glip_model.eval()
                glip_outs = self.forward_glip(images.tensor, captions, positive_map_label_to_token)
                
                for cnt, glip_out in enumerate(glip_outs):
                    glip_out.extra_fields['cls_emb'] = self.extract_cls_emb(glip_out.extra_fields['labels'], positive_map_label_to_token)
                    #vis_2d_det(boxlist = glip_out, img = batched_inputs[cnt]['image'].permute(1, 2, 0).numpy(), class_names = self.cfg.DATASETS.CATEGORY_NAMES) 
            glip_results = dict()
            glip_outs = self.pad_glip_outs(glip_outs)
            glip_results['bbox'] = torch.stack([ele.bbox for ele in glip_outs], dim  = 0)   # Left shape: (bs, box_num, 4)
            glip_results['scores'] = torch.stack([ele.get_field('scores') for ele in glip_outs], dim  = 0)  # Left shape: (bs, box_num)
            glip_results['labels'] = torch.stack([ele.get_field('labels') for ele in glip_outs], dim  = 0)  # Left shape: (bs, box_num)
            glip_results['cls_emb'] = torch.stack([ele.get_field('cls_emb') for ele in glip_outs], dim  = 0)  # Left shape: (bs, box_num, cls_emb_len)
        else:
            glip_results = {}

        if self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL and self.training:
            novel_cls_gts = []
            if self.cfg.MODEL.DETECTOR3D.PETR.HEAD.DET2D_FROM == 'GLIP':
                novel_glip_box_flag = glip_results['labels'].new_zeros((glip_results['bbox'].shape[0], glip_results['bbox'].shape[1]), dtype = torch.bool)  # Left shape: (bs, num_glip_out)
                for novel_cls_id in novel_cls_id_list:
                    novel_glip_box_flag = novel_glip_box_flag | (glip_results['labels'] == novel_cls_id)
                novel_glip_box_flag = novel_glip_box_flag & (glip_results['scores'] > self.cfg.MODEL.GLIP_MODEL.GLIP_PSEUDO_CONF)   # Left shape: (bs, num_glip_out)
                for bs_idx, flag_one_bs in enumerate(novel_glip_box_flag):
                    novel_cls_gts.append({})
                    novel_cls_gts[bs_idx]['labels'] = glip_results['labels'][bs_idx][novel_glip_box_flag[bs_idx]].clone()
                    novel_cls_gts[bs_idx]['bbox'] = glip_results['bbox'][bs_idx][novel_glip_box_flag[bs_idx]].clone()
                    bs_reso_after_aug = ori_img_resolution[bs_idx]  # Left shape: (2,)
                    novel_cls_gts[bs_idx]['bbox'] = novel_cls_gts[bs_idx]['bbox'] / torch.cat((bs_reso_after_aug, bs_reso_after_aug), dim = -1)[None]   # Normalize the 2D box.
            elif self.cfg.MODEL.DETECTOR3D.PETR.HEAD.DET2D_FROM == 'GT2D':
                for bs_idx, batch_input in enumerate(batched_inputs):
                    novel_cls_gts.append({})
                    bs_gt_boxes = batch_input['instances'].get('gt_boxes').tensor.to(images.device)  # Left shape: (num_box, 4)
                    bs_gt_cls = batch_input['instances'].get('gt_classes').to(images.device)    # Left shape: (num_box,)
                    bs_novel_box_flag = bs_gt_boxes.new_zeros((bs_gt_boxes.shape[0],), dtype = torch.bool) # Left shape: (num_box,)
                    for novel_cls_id in novel_cls_id_list:
                        bs_novel_box_flag = bs_novel_box_flag | (bs_gt_cls == novel_cls_id)
                    novel_cls_gts[bs_idx]['labels'] = bs_gt_cls[bs_novel_box_flag]  # Left shape: (num_valid_box,)
                    bs_reso_after_aug = ori_img_resolution[bs_idx]  # Left shape: (2,)
                    novel_cls_gts[bs_idx]['bbox'] = bs_gt_boxes[bs_novel_box_flag] / torch.cat((bs_reso_after_aug, bs_reso_after_aug), dim = -1)[None]  # Left shape: (num_valid_box, 4)
        else:
            novel_cls_gts = None

        if self.training and self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL:
            batched_inputs = self.remove_novel_gts(batched_inputs, novel_cls_id_list)
        
        # 3D Det
        detector_out = self.detector(images, batched_inputs, glip_results, self.class_name_emb)
        
        if self.training:
            loss_dict = self.forward_train(detector_out, batched_inputs, ori_img_resolution, novel_cls_gts)
            return loss_dict
        else:
            return self.forward_inference(detector_out, batched_inputs, ori_img_resolution)
        
    def forward_train(self, detector_out, batched_inputs, ori_img_resolution, novel_cls_gts = None):
        return self.detector.loss(detector_out, batched_inputs, ori_img_resolution, novel_cls_gts)

    def forward_inference(self, detector_out, batched_inputs, ori_img_resolution, conf_thre = 0.01):
        inference_results = self.detector.inference(detector_out, batched_inputs, ori_img_resolution)
        
        return inference_results

    def extract_cls_emb(self, cls_idxs, positive_map_label_to_token):
        cls_emb_list = []
        for cls_idx in cls_idxs:
            if cls_idx.item() >= len(positive_map_label_to_token):
                cls_emb_list.append(torch.zeros_like(self.class_name_emb[0]))
            else:
                cls_emb = self.class_name_emb[cls_idx.item()]
                cls_emb_list.append(cls_emb)
        cls_embs = torch.stack(cls_emb_list, axis = 0)
        return cls_embs

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].cuda() for x in batched_inputs]
        images = [(x - self.img_mean.to(x.device)) / self.img_std.to(x.device) for x in images]

        images = ImageList.from_tensors(images)
        return images
    
    def remove_invalid_gts(self, batched_inputs):
        '''
            Description: Remove the labels the class IDs of which are -1.
        '''
        for batch_input in batched_inputs:
            cls_gts = batch_input['instances'].get('gt_classes')
            valid_gt_mask = (cls_gts != -1)
            for gt_key in ['gt_classes', 'gt_boxes', 'gt_boxes3D', 'gt_poses', 'gt_keypoints', 'gt_unknown_category_mask', 'gt_featreso_box2d_center', 'gt_featreso_2dto3d_offset']:
                if gt_key in ('gt_boxes', 'gt_keypoints'):
                    batch_input['instances']._fields[gt_key].tensor = batch_input['instances']._fields[gt_key].tensor[valid_gt_mask]
                else:
                    batch_input['instances']._fields[gt_key] = batch_input['instances']._fields[gt_key][valid_gt_mask]

        return batched_inputs

    def remove_novel_gts(self, batched_inputs, novel_cls_id_list):
        for bs_idx, batch_input in enumerate(batched_inputs):
            cls_gts = batch_input['instances'].get('gt_classes')
            valid_gt_mask = torch.ones_like(cls_gts, dtype = torch.bool)
            if self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL:
                for novel_cls_id in novel_cls_id_list:
                    valid_gt_mask = valid_gt_mask & (cls_gts != novel_cls_id)   # Remove the label instances that are in the novel class name list.

            for gt_key in ['gt_classes', 'gt_boxes', 'gt_boxes3D', 'gt_poses', 'gt_keypoints', 'gt_unknown_category_mask']:
                if gt_key in ('gt_boxes', 'gt_keypoints'):
                    batch_input['instances']._fields[gt_key].tensor = batch_input['instances']._fields[gt_key].tensor[valid_gt_mask]
                else:
                    batch_input['instances']._fields[gt_key] = batch_input['instances']._fields[gt_key][valid_gt_mask]
                    
        return batched_inputs
        

def pseudo_with():
    pass

def vis_2d_det(boxlist, img, class_names, conf_thre = 0.5):
    img = np.ascontiguousarray(img)
    box_locs = boxlist.bbox.cpu().numpy().astype(np.int32)
    box_labels = boxlist.extra_fields['labels'].cpu().numpy()
    box_confs = boxlist.extra_fields['scores'].cpu().numpy()

    valid_mask = (box_confs > conf_thre) & (box_labels < len(class_names))
    box_locs = box_locs[valid_mask]
    box_labels = box_labels[valid_mask]
    box_confs = box_confs[valid_mask]

    for cnt, box_loc in  enumerate(box_locs):
        box_label = box_labels[cnt]
        class_name = class_names[box_label]
        img = cv2.rectangle(img, box_loc[0:2], box_loc[2:4], (0, 255, 0))
        img = cv2.putText(img, "{:s}: {:.2f}".format(class_name, box_confs[cnt]), box_loc[0:2], cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
    cv2.imwrite('vis.png', img)
    pdb.set_trace()

def create_queries_and_maps_from_dataset(categories, cfg):
    
    labels = []
    label_list = []
    keys = list(categories.keys())
    keys.sort()
    for i in keys:
        labels.append(i)
        label_list.append(categories[i])

    if cfg.TEST.CHUNKED_EVALUATION != -1:
        labels = chunks(labels, cfg.TEST.CHUNKED_EVALUATION)
        label_list = chunks(label_list, cfg.TEST.CHUNKED_EVALUATION)
    else:
        labels = [labels]
        label_list = [label_list]
    
    all_queries = []
    all_positive_map_label_to_token = []

    for i in range(len(labels)):
        labels_i = labels[i]
        label_list_i = label_list[i]
        query_i, positive_map_label_to_token_i = create_queries_and_maps(
            labels_i, label_list_i, additional_labels = None, cfg = cfg)
        
        all_queries.append(query_i)
        all_positive_map_label_to_token.append(positive_map_label_to_token_i)
    print("All queries", all_queries)
    return all_queries, all_positive_map_label_to_token

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert(counter == len(lst))

    return all_

def create_queries_and_maps(labels, label_list, additional_labels = None, cfg = None):

    # Clean label list
    original_label_list = label_list.copy()
    label_list = [clean_name(i) for i in label_list]
    # Form the query and get the mapping
    tokens_positive = []
    start_i = 0
    end_i = 0
    objects_query = ""

    # sep between tokens, follow training
    separation_tokens = ". "
    use_caption_prompt = False
    caption_prompt = None
    
    for _index, label in enumerate(label_list):
        if use_caption_prompt:
            objects_query += caption_prompt[_index]["prefix"]
        
        start_i = len(objects_query)

        if use_caption_prompt:
            objects_query += caption_prompt[_index]["name"]
        else:
            objects_query += label
        
        end_i = len(objects_query)
        tokens_positive.append([(start_i, end_i)])  # Every label has a [(start, end)]
        
        if use_caption_prompt:
            objects_query += caption_prompt[_index]["suffix"]

        if _index != len(label_list) - 1:
            objects_query += separation_tokens
    
    if additional_labels is not None:
        objects_query += separation_tokens
        for _index, label in enumerate(additional_labels):
            objects_query += label
            if _index != len(additional_labels) - 1:
                objects_query += separation_tokens

    print(objects_query)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenized = tokenizer(objects_query, return_tensors="pt")
    elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
        from transformers import CLIPTokenizerFast
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                        from_slow=True, mask_token='ðŁĴĳ</w>')
        else:
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                        from_slow=True)
        tokenized = tokenizer(objects_query,
                              max_length=cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                              truncation=True,
                              return_tensors="pt")
    else:
        tokenizer = None
        raise NotImplementedError

    # Create the mapping between tokenized sentence and the original label
    positive_map_token_to_label, positive_map_label_to_token = create_positive_dict(tokenized, tokens_positive,
                                                                                        labels=labels)  # from token position to original label
    return objects_query, positive_map_label_to_token

def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j, iff token i is mapped to j label"""
    positive_map = defaultdict(int)

    # Additionally, have positive_map_label_to_tokens
    positive_map_label_to_token = defaultdict(list)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            for i in range(beg_pos, end_pos + 1):
                positive_map[i] = labels[j]  # because the labels starts from 1
                positive_map_label_to_token[labels[j]].append(i)
            # positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map, positive_map_label_to_token  # / (positive_map.sum(-1)[:, None] + 1e-6)

def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name
