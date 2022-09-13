import os
import glob
import json
tr_dialogue_file = '../SIMMC2_data/simmc2_dials_dstc10_train.json'
tr_dialogue = json.load(open(tr_dialogue_file, 'r'))['dialogue_data']
dev_dialogue_file = '../SIMMC2_data/simmc2_dials_dstc10_dev.json'
dev_dialogue = json.load(open(dev_dialogue_file, 'r'))['dialogue_data']
tr_dialogue = tr_dialogue + dev_dialogue
rec = {}
pool = ['rack', 'closet', 'wall', 'shel', 'carousel', 'table', 'mirror', 'cubicle', 'cupboard', 
            'floor stand', 'compartment', 'cabinet', 'cubby', 'cubbies', 'the display', 'slot', 'circular display',
           'mannequin', 'division', 'section']
pool_2 = ["I'll take", "I mean", "Could you", "Add", "I want", "Let me get", "Can you", "I'd like", "Please",
         'I meant']
def check_word(sen):
    for w in pool:
        if w in sen:
            return True
    return False
def del_word(sen):
    for w in pool_2:
        sen = sen.replace(w, '')
    return sen
for d in tr_dialogue:
    scene_rec = d['scene_ids']
    cur_scene = scene_rec['0']
    if len(scene_rec) != 0:
        next_scene_id = [i for i in scene_rec.keys() if i != 0][0]
    else:
        next_scene_id = '1000'
    for v in scene_rec.values():
        if v not in rec:
            rec[v] = {}
    for idx, turn in enumerate(d['dialogue']):
        if len(turn['transcript_annotated']['act_attributes']['objects']) == 1:
            if idx >= int(next_scene_id):
                cur_scene = scene_rec[next_scene_id]
                next_scene_id = '1000'
            for o in turn['transcript_annotated']['act_attributes']['objects']:
                if check_word(turn['transcript']):
                    rec[cur_scene][o] = del_word(turn['transcript'])
data = {'train': [], 'val': []}
image_source = '../SIMMC2_data/simmc2_scene_images_dstc10_public/'
scene_source = '../SIMMC2_data/public/'
image_lst = glob.glob(image_source + '*.png')
obj_ct = 0
for idx, imp in enumerate(rec):
    if idx < 900:
        split = 'train'
    else:
        split = 'val'
        
    scene_path = scene_source + imp + '_scene.json'
    img_path = image_source + imp + '.png'
    if not os.path.isfile(img_path):
        img_path = img_path.replace('m_', '')
#    img = Image.open(img_path)
#    width, height = img.size
    objs = json.load(open(scene_path, 'r'))['scenes'][0]['objects']
    boxes = []
    sentences = []
    
    for n, obj in enumerate(objs):
        if obj['index'] not in rec[imp]:
            continue
        bbox = [obj['bbox'][0], obj['bbox'][1], obj['bbox'][3], obj['bbox'][2]]
        boxes.append(bbox)
        
        sentence = rec[imp][obj['index']]
        sentences.append(sentence)
    if len(boxes) != 0:
        data[split].append({'img_path':img_path.split('/')[-1], 'boxes': boxes, 'captions': sentences})
    
json.dump(data, open('data3.json', 'w'))



