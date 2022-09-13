# Original Copyright (c) 2019 Weijie Su. Licensed under the MIT License.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import _init_paths
import os
import argparse
import torch
import subprocess
import json
import random
import copy
import pandas as pd
import pickle as pkl
import numpy as np

from simmc2.function.config import config, update_config
from simmc2.function.test_dstc10 import test_net
from simmc2.evaluate_dst import evaluate_from_json

def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--ori-dialogue', type=str, help='original test dialogue file')
    parser.add_argument('--cudnn-off', help='disable cudnn', default=False, action='store_true')
    parser.add_argument('--savepath', type=str, default=None)

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)
    # if args.model_dir is not None:
    #     config.OUTPUT_PATH = os.path.join(args.model_dir, config.OUTPUT_PATH)

    config.GPUS = '0'
    config.TRAIN.AUTO_RESUME = False

    return args, config

def clean_cap(cap):
    obj_pool = ['rack', 'closet', 'wall', 'shel', 'carousel', 'table', 'mirror', 'cubicle', 'row', 'cupboard', 
            'floor stand', 'compartment', 'cabinet', 'cubby', 'cubbies', 'the display', 'slot', 'circular display',
                'mannequin', 'division', 'section']
    obj_pool = ['rack', 'closet', 'wall', 'shel', 'carousel', 'table', 'mirror', 'cubicle', 'row', 'cupboard', 
            'stand', 'compartment', 'cabinet', 'cubby', 'cubbies', 'display', 'slot', 'circular',
                'mannequin', 'division', 'section']
    for o in obj_pool:
        if o in cap:
            return o + ' [SEP] '
    return ''

def gen_cap(obj, cap):
    return obj['color'] + ' ' + obj['type'] + ' ' + clean_cap(cap)
    
def get_obj_bbox(scene, raw_cap, obj_meta_text, all_meta):
#     if scene in pos_info:
#         pinfo = pos_info[scene]
#     elif scene.replace('m_', '') in pos_info:
#         pinfo = pos_info[scene.replace('m_', '')]
#     else:
#         print(scene)
#         raise ValueError('Does find pos info')
    pinfo = []
    scene_path = '../SIMMC2_data/public/' + scene + '_scene.json'
    scene = json.load(open(scene_path, 'r'))['scenes']
    assert (len(scene) == 1), 'Multiple scenes'
    obj_rec = []
    box_rec = []
    seg_rec = []
    text_rec = []
    all_ids = []
    obj_cats = []
    cap_rec = []
    
    if len(pinfo) != 0:
        p = np.array(pinfo)
        x_std = np.std(p[:,0])
        y_std = np.std(p[:,1])
        if (x_std < 0.5) or (y_std < 0.5):
            add_pos = False
        else:
            add_pos = True
            
    # load raw caption
    scene_key = scene_path.split('/')[-1]
    if scene_key not in raw_cap:
        if scene_key.replace('m_', '') in raw_cap:
            scene_key = scene_key.replace('m_', '')
            
    if scene_key in raw_cap:
        if len(raw_cap[scene_key]) != 0:
            cap = raw_cap[scene_key]
        else:
            cap = [''] * len(scene[0]['objects'])
#             print(scene_key, 'does not have captions')
    else:
        cap = [''] * len(scene[0]['objects'])
#         print(scene_key, 'does not have captions')
            
    
    for idx, obj in enumerate(scene[0]['objects']):
        obj_rec.append('[OBJ{}]'.format(str(obj['index'])))
        [x, y, h, w] = obj['bbox']
        box_rec.append([x, y, x+w, y+h])
        seg_rec.append(None)
        text_rec.append(clean_cap(cap[idx]) + obj_meta_text[obj['prefab_path']])
        all_ids.append(obj['index'])
        obj_cats.append(obj_meta_text[obj['prefab_path']].split('.')[0])
        cap_rec.append(gen_cap(all_meta[obj['prefab_path']], cap[idx]))
#         obj_type.add(obj_meta_text[obj['prefab_path']].split('.')[0])
    return obj_rec, box_rec, seg_rec, text_rec, all_ids, obj_cats, pinfo, cap_rec

def get_pos_info(p):
    near_th = 1.5
    far_th = 6
    outlier_th = 10.
    dist = np.sqrt(p[0]**2 + p[1]**2)
    if dist < near_th:
        return 'front'
    elif (dist < outlier_th) and (dist > far_th):
        return 'far or back'
    else:
        return ''
    

def get_edge_feat(scene):
    scene = json.load(open('../SIMMC2_data/public/' + scene + '_scene.json', 'r'))['scenes'][0]
    num_of_objs = len(scene['objects'])
    edge_feat = np.zeros([num_of_objs, num_of_objs])
    # ['right', 'left', 'down', 'up']
    order_of_directions = {'right': 1, 'left': 2, 'down': 3, 'up': 4}
    oppo_inds = {'right':2, 'left':1, 'down':4, 'up':3}
    dt= pd.DataFrame(scene['objects']).sort_values('index').reset_index()
    search_ind = dt['index'].to_dict()
    search_ind = {v:k for k, v in search_ind.items()}
    
    for direction in scene['relationships']:
        dir_ind = order_of_directions[direction]
        for target in scene['relationships'][direction]:
            for neighbor in scene['relationships'][direction][target]:
                edge_feat[search_ind[int(target)], search_ind[int(neighbor)]] = dir_ind
                edge_feat[search_ind[int(neighbor)], search_ind[int(target)]] = oppo_inds[direction]
    return edge_feat


def generate_label_files(dialogue_file, source_folder='../SIMMC2_data', save_name='test_preprocessed.json'):
    dialogue_lst = json.load(open(dialogue_file, 'r'))['dialogue_data']

    obj_text_file = os.path.join(source_folder, 'data_processed/meta_text_all.json')
    obj_meta_text = json.load(open(obj_text_file, 'r'))
    # pos_info_file = '../data_processed/relative_pos.json'
    # pos_info = json.load(open(pos_info_file, 'r'))
    raw_cap_file = os.path.join(source_folder, 'raw_captions.json')
    raw_cap = json.load(open(raw_cap_file, 'r'))
    # obj_type = set()
    fashion_meta = json.load(open(os.path.join(source_folder, 'fashion_prefab_metadata_all.json'), 'r'))
    all_meta = json.load(open(os.path.join(source_folder, 'furniture_prefab_metadata_all.json'), 'r'))
    all_meta.update(fashion_meta)
    data = [] 
    data_ct = 0
    edge_mats = {}

    for dlg_ind, dlg in enumerate(dialogue_lst):
        ct_scene = 0
        cur_scene = '0'
        next_scene = sorted(dlg['scene_ids'].keys())[ct_scene + 1] if len(dlg['scene_ids'].keys()) > ct_scene + 1 else None
        text_rec = []
        prev_mentioned_id2turn = {}
        prev_mentioned_turn2id = {}
        
        obj_rec, box_rec, seg_rec, obj_info, all_ids, obj_cats, pinfo, cap_rec = get_obj_bbox(dlg['scene_ids'][cur_scene], raw_cap, obj_meta_text, all_meta)
        edge_feat = get_edge_feat(dlg['scene_ids'][cur_scene])
        accu_turn = 0
        for idx, turn in enumerate(dlg['dialogue']):
            prev_mentioned_turn2id[idx] = []
            # check scene
            if next_scene is not None:
                if idx == int(next_scene):
                    text_rec = []
                    obj_rec_old, box_rec_old, seg_rec_old, obj_info_old, all_ids_old, obj_cats_old, pinfo_old, cap_rec_old = copy.copy(obj_rec), copy.copy(box_rec), copy.copy(seg_rec), copy.copy(obj_info), copy.copy(all_ids), copy.copy(obj_cats), copy.copy(pinfo), copy.copy(cap_rec)
                    edge_feat_old = copy.copy(edge_feat)
                    old_scene = copy.copy(cur_scene)
                    obj_rec, box_rec, seg_rec, obj_info, all_ids, obj_cats, pinfo, cap_rec = get_obj_bbox(dlg['scene_ids'][next_scene], raw_cap, obj_meta_text, all_meta)
                    edge_feat = get_edge_feat(dlg['scene_ids'][next_scene])
                    ct_scene += 1
                    cur_scene = next_scene
                    next_scene = sorted(dlg['scene_ids'].keys())[ct_scene + 1] if len(dlg['scene_ids'].keys()) > ct_scene + 1 else None
                    prev_mentioned_id2turn_old = copy.copy(prev_mentioned_id2turn)
                    prev_mentioned_turn2id_old = copy.copy(prev_mentioned_turn2id)
                    prev_mentioned_id2turn = {}
                    prev_mentioned_turn2id = {}
                    prev_mentioned_turn2id[idx] = []
        # check obj reference
        # If it is a 'disambiguate' request, skip it
            
            # if turn['system_transcript_annotated']['act'] == 'REQUEST:DISAMBIGUATE':
            if 0:
#         if len(turn['transcript_annotated']['act_attributes']['objects']) == 0 or len(turn['system_transcript_annotated']['act_attributes']['objects']) == 0:
#         if len(turn['transcript_annotated']['act_attributes']['objects']) == 0 or turn['system_transcript_annotated']['act'] == 'REQUEST:DISAMBIGUATE':
                text_rec.append('[USER] ' + turn['transcript'] + ' [SYS] ' + turn['system_transcript'])
                if len(turn['system_transcript_annotated']['act_attributes']['objects']) != 0:
                    for m in turn['system_transcript_annotated']['act_attributes']['objects']:
                        if '[OBJ{}]'.format(str(m)) in prev_mentioned_id2turn:
                            prev_mentioned_turn2id[prev_mentioned_id2turn['[OBJ{}]'.format(str(m))]].remove('[OBJ{}]'.format(str(m)))
                        prev_mentioned_id2turn['[OBJ{}]'.format(str(m))] = idx 
                        prev_mentioned_turn2id[idx].append('[OBJ{}]'.format(str(m)))
            else:
                # add user querry
                qry = ('[QRY] ' + turn['transcript'])
                # add to data
                if 'transcript_annotated' in turn.keys():
                    mentioned = turn['transcript_annotated']['act_attributes']['objects']
                else:
                    mentioned = []
                mentioned = []

                data.append({'annot_id': data_ct, 'objects': obj_rec, 'img_fn': dlg['scene_ids'][cur_scene], 
                             'boxes': box_rec, 'segms': seg_rec, 'question': ' '.join(text_rec + [qry]), 'obj_meta': obj_info,
                             'answer_choices': mentioned, 'mentioned_lbl': copy.copy(prev_mentioned_id2turn),
                            '3d_pos': pinfo, 'prev_ref':0, 'turn_index': idx, 'dialogue_index': dlg_ind})
#                             'cap': cap_rec})
                edge_mats[dlg['scene_ids'][cur_scene]] = edge_feat
                data_ct += 1
                

                #update prev_mentioned dict
                
                mentioned_sys = turn['system_transcript_annotated']['act_attributes']['objects']
                # mentioned in current scene & previous scene
                mentioned_in_cur_sys = []
                mentioned_in_prev_sys = []
                for m in mentioned_sys:
                    ans = '[OBJ{}]'.format(str(m))
                    if ans in obj_rec:
                        mentioned_in_cur_sys.append(ans)
                    elif ans in obj_rec_old:
                        mentioned_in_prev_sys.append(ans)
                for m in mentioned_in_cur_sys:
                    if m in prev_mentioned_id2turn:
                        prev_mentioned_turn2id[prev_mentioned_id2turn[m]].remove(m)
                    prev_mentioned_id2turn[m] = idx 
                    prev_mentioned_turn2id[idx].append(m)
                for m in mentioned_in_prev_sys:
                    if m in prev_mentioned_id2turn_old:
                        prev_mentioned_turn2id_old[prev_mentioned_id2turn_old[m]].remove(m)
                    prev_mentioned_id2turn_old[m] = idx 
                    if idx in prev_mentioned_turn2id_old:
                        prev_mentioned_turn2id_old[idx].append(m)
                    else:
                        prev_mentioned_turn2id_old[idx] = [m]
                        
    #                 for p in flatten(prev_mentioned_old):
    #                     if p not in obj_rec_old:
    #                         print(p, prev_mentioned_old, obj_rec_old)
    #                         raise ValueError()
                    
                if 'system_transcript' in turn.keys():
                    text_rec.append('[USER] ' + turn['transcript'] + ' [SYS] ' + turn['system_transcript'])
                else:
                    text_rec.append('[USER] ' + turn['transcript'])
                if len(text_rec) >= 3:
                    text_rec = text_rec[1:]
                    removed_turn = sorted(list(prev_mentioned_turn2id.keys()))[0]
                    for oid in prev_mentioned_turn2id[removed_turn]:
                        del prev_mentioned_id2turn[oid]
                    del prev_mentioned_turn2id[removed_turn]
                    prev_mentioned_turn2id = {k-1:v for k, v in prev_mentioned_turn2id.items()}
                    prev_mentioned_id2turn = {k:v-1 for k, v in prev_mentioned_id2turn.items()}
    #                 prev_mentioned = prev_mentioned[1:]
    print('Num of data =', len(data))
    print('Temporary label file save to', os.path.join(source_folder, 'data_processed', save_name))
    json.dump({'version':1.0, 'data': data}, open(os.path.join(source_folder, 'data_processed', save_name), 'w'))
    pkl.dump(edge_mats, open(os.path.join(source_folder, 'data_processed/test_edge_mat_preprocessed.pkl'), 'wb'))



def main():
    args, config = parse_args()
    # generate label
    save_name = 'test_preprocessed.json'
    source_folder = '../SIMMC2_data'
    generate_label_files(args.ori_dialogue, source_folder=source_folder, save_name=save_name)
    config.DATASET.VAL_ANNOTATION_FILE = save_name
    config.DATASET.VAL_ADJ_PATH = os.path.join(source_folder, 'data_processed/test_edge_mat_preprocessed.pkl')
    test_net(args, config)

    # generate required format
    pred = json.load(open(args.savepath, 'r'))[0]
    ori = json.load(open(args.ori_dialogue, 'r'))['dialogue_data']
    label = json.load(open(os.path.join(config.DATASET.DATASET_PATH, config.DATASET.VAL_ANNOTATION_FILE), 'r'))['data']
    cur_dlg_index = None
    mentioned_objs_dlevel = []
    for ind, logits in enumerate(pred['answer_logits']):
        objs = label[ind]['objects']
        turn_index = label[ind]['turn_index']
        dlg_index = label[ind]['dialogue_index']
        if cur_dlg_index == None:
            cur_dlg_index = dlg_index
        if cur_dlg_index != dlg_index:
            mentioned_objs_dlevel = []
            cur_dlg_index = dlg_index
        ref_objs = [objs[i] for i in range(len(objs)) if logits[i] > 0.5]
        ref_objs = [int(i.replace('[OBJ', '').replace(']', '')) for i in ref_objs]
        mentioned_objs_dlevel += ref_objs
        ori[cur_dlg_index]['mentioned_object_ids'] = mentioned_objs_dlevel
        if 'transcript_annotated' not in ori[cur_dlg_index]['dialogue'][turn_index].keys():
            ori[cur_dlg_index]['dialogue'][turn_index]['transcript_annotated'] = {}
            ori[cur_dlg_index]['dialogue'][turn_index]['transcript_annotated']['act_attributes'] = {}
        ori[cur_dlg_index]['dialogue'][turn_index]['transcript_annotated']['act_attributes']['objects'] = ref_objs

    json.dump({"dialogue_data": ori}, open(args.savepath.replace('.json', '_dstc.json'), 'w'))
    d_true = json.load(open(args.ori_dialogue, 'r'))['dialogue_data']
    # d_pred = json.load(open(args.savepath.replace('.json', '_dstc.json'), 'r'))['dialogue_data']
    
    # report = evaluate_from_json(d_true, ori)
    # print(report)

if __name__ == '__main__':
    main()


