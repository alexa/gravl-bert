import json
import numpy as np
import cv2
import os
import textwrap

pred_file = '/home/ec2-user/danfengg/simmc_2stm_2/base_qa2r_4x16G_fp32_simmc2_test2015.json'
pred = json.load(open(pred_file, 'r'))[0]
label_file = '/home/ec2-user/simmc2/data_processed/dev_preprocessed_v7.json'
lbl = json.load(open(label_file, 'r'))['data']

def get_label(objects, answer_choices):
    lb = [0] * len(objects)
    for a in answer_choices:
        lb[objects.index(a)] = 1
    return lb


fp_cum = 0
fn_cum = 0
tp_cum = 0
rec = []
for n, lb in enumerate(lbl):
    p = np.asarray(pred['answer_logits'][n])
#     od = np.argsort(p)[-3]
#     p = ((p >= p[od]) * (p >0.4)).astype(int)
    p = ((p >0.5)).astype(int)
    l = np.asarray(get_label(lb['objects'], lb['answer_choices']))
    fp = (p == 1) * (l == 0)
    fp_inds = np.where(fp == 1)[0]
    fp_cum += len(fp_inds)
    tp = (p == 1) * (l == 1)
    tp_inds = np.where(tp == 1)[0]
    tp_cum += len(tp_inds)
    fn = (p == 0) * (l == 1)
    fn_inds = np.where(fn == 1)[0]
    fn_cum += len(fn_inds)
    if (len(fp_inds) == 0) and (len(fn_inds) == 0):
        continue
    rec.append({'question': lb['question'], 'img_fn': lb['img_fn'], 'fp_inds': list(fp_inds), 'fp_boxes': [lb['boxes'][i] for i in fp_inds],
               'fn_inds': list(fn_inds), 'fn_boxes': [lb['boxes'][i] for i in fn_inds],
               'tp_inds': list(tp_inds), 'tp_boxes': [lb['boxes'][i] for i in tp_inds]})


print(tp_cum/(fp_cum+tp_cum))
print(tp_cum/(fn_cum+tp_cum))

import cv2
import os
img_source = '/home/ec2-user/simmc2/data/simmc2_scene_images_dstc10_public'
save_folder = '/home/ec2-user/danfengg/simmc_2stm_2/visualizations'

def bbox_draw(img, boxes, tp, color):
    for i, b in enumerate(boxes):
        left, top, right, bottom = b
        cv2.rectangle(img,(left, top), (right, bottom), color, 2)
        cv2.putText(img, tp, (left, top - 12), 0, 1, color, 2)
    
for idx, r in enumerate(rec):
    img_path = os.path.join(img_source, r['img_fn'] + '.png')
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
    elif os.path.isfile(img_path.replace('m_', '')):
        img = cv2.imread(img_path.replace('m_', ''))
    else:
        raise ValueError()
    bbox_draw(img, r['tp_boxes'], 'tp', (0,255,0))
    bbox_draw(img, r['fp_boxes'], 'fp', (255,0,0))
    bbox_draw(img, r['fn_boxes'], 'fn', (0,255,0))
    qs = r['question'].split('[QRY]')[-1]
    cv2.putText(img, qs, (3,30), 0, 1, (0,0,0), 2)
    cv2.imwrite(os.path.join(save_folder, 'visualization_{}.png'.format(idx)), img)
    
    
    break
