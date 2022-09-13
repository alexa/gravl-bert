
import argparse
import json
import random
import copy
import pandas as pd
import pickle as pkl
import numpy as np
import glob
import os
from PIL import Image
from nltk.corpus import cmudict

def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()  # use only the first one
    
def attr_to_sentence(k, attr, txt):

#     if k == 'availableSizes':
# #         return txt + 'The current available sizes are ' + ','.join(attr) + '. '
#         return txt + 'Available sizes ' + ','.join(attr) + '. '
    if k == 'price':
#         return txt + 'The price is ' + str(attr) + '. '
        return txt + 'Price ' + str(attr) + '. '
    elif k == 'customerReview':
#         return txt + 'The customer review score is ' + str(attr) + '. '
        return txt + 'Customer review score ' + str(attr) + '. '
    elif k == 'customerRating':
#         return txt + 'The customer rating score is ' + str(attr) + '. '
        return txt + 'Customer rating score ' + str(attr) + '. '
    elif k == 'materials':
#         return txt + 'It is made from ' + attr + '. '
        return txt + attr + '. '
    elif k in ['brand', 'size']:
#         return txt + 'The {} is {}. '.format(k, attr)
        return txt + '{} {}. '.format(k, attr)
    else:
        return txt

def check_region(bbox, size):
    quad_map = np.zeros(size)
    ct = 0
    x_intv = size[0] // 4 + 1
    y_intv = size[1] // 2 + 1
    for y in range(2):
        for x in range(4):
            quad_map[x*x_intv:(x+1)*x_intv, y*y_intv:(y+1)*y_intv] = ct
            ct += 1
    x, y, h, w = bbox
    area = quad_map[x:x+w, y:y+h]
    if x + w > size[0] or y + h > size[1]:
        print(x+h, y+w, size, bbox)
        raise ValueError('box exceeds border')
    return ['region ' + str(int(i)) for i in np.unique(area)]

def clean_cap(cap):
	obj_pool = ['rack', 'closet', 'wall', 'shel', 'carousel', 'table', 'mirror', 'cubicle', 'row', 'cupboard', 
			'floor stand', 'compartment', 'cabinet', 'cubby', 'cubbies', 'the display', 'slot', 'circular display',
		   'mannequin', 'division', 'section']
	for o in obj_pool:
		if o in cap:
			return o + '. '
	return ''

def get_obj_bbox(scene):
	pinfo = None
	# if scene in pos_info:
	#	 pinfo = pos_info[scene]
	# elif scene.replace('m_', '') in pos_info:
	#	 pinfo = pos_info[scene.replace('m_', '')]
	# else:
	#	 print(scene)
	#	 raise ValueError('Does find pos info')
	scene_path = os.path.join(scene_source, scene + '_scene.json')
	scene = json.load(open(scene_path, 'r'))['scenes']
	assert (len(scene) == 1), 'Multiple scenes'
	obj_rec = []
	box_rec = []
	seg_rec = []
	text_rec = []
	all_ids = []
	obj_cats = []
	
	# if len(pinfo) != 0:
	#	 p = np.array(pinfo)
	#	 x_std = np.std(p[:,0])
	#	 y_std = np.std(p[:,1])
	#	 if (x_std < 0.5) or (y_std < 0.5):
	#		 add_pos = False
	#	 else:
	#		 add_pos = True
			
	# load raw caption
	scene_key = scene_path.split('/')[-1]
	if scene_key not in raw_cap:
		if scene_key.replace('m_', '') in raw_cap:
			scene_key = scene_key.replace('m_', '')
			
	if scene_key in raw_cap:
		if len(raw_cap[scene_key]) != 0:
			cap = raw_cap[scene_key]
		else:
			cap = None
			print(scene_key, 'does not have captions')
	else:
		cap = None
		print(scene_key, 'does not have captions')
			

	for idx, obj in enumerate(scene[0]['objects']):
		obj_rec.append('[OBJ{}]'.format(str(obj['index'])))
		[x, y, h, w] = obj['bbox']
		box_rec.append([x, y, x+w, y+h])
		seg_rec.append(None)
#		 ids = obj_meta_text[obj['prefab_path']].find('region')
#		 text_rec.append(obj_meta_text[obj['prefab_path']][:ids-1])
#		 if len(pinfo) != 0 and add_pos:
#			 text_rec.append(obj_meta_text[obj['prefab_path']] + get_pos_info(pinfo[idx]))
#		 else:
		if cap is not None:
			text_rec.append(clean_cap(cap[idx]) + obj_meta_text[obj['prefab_path']])
		else:
			text_rec.append(obj_meta_text[obj['prefab_path']])
		all_ids.append(obj['index'])
		obj_cats.append(obj_meta_text[obj['prefab_path']].split('.')[0])
#		 obj_type.add(obj_meta_text[obj['prefab_path']].split('.')[0])
	return obj_rec, box_rec, seg_rec, text_rec, all_ids, obj_cats, pinfo

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
	scene = json.load(open(os.path.join(scene_source, scene + '_scene.json'), 'r'))['scenes'][0]
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



def flatten(lst):
	l = []
	for subl in lst:
		for e in subl:
			l.append(e)
	return list(set(l))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='devtest')
	parser.add_argument('--simmc_source', type=str, default='SIMMC2_data')
	parser.add_argument('--scene_source', type=str, default='SIMMC2_data/public')
	parser.add_argument('--save_path', type=str, default='SIMMC2_data/data_processed')
	args = parser.parse_args()

	mode = args.mode	
	dialogue_file = args.simmc_source + '/simmc2_dials_dstc10_{}.json'.format(mode)
	scene_source = args.scene_source

	if not os.path.exists(args.save_path):
		os.mkdir(args.save_path)

	'''
	################################
	Generate metadata
	################################
	'''
	print('Genearting metadata file...')
	print(args.simmc_source + '/simmc2_dials_dstc10_train.json')
	dialog_lst = json.load(open(args.simmc_source + '/simmc2_dials_dstc10_train.json', 'r'))['dialogue_data']
	dialog_lst_2 = json.load(open(args.simmc_source + '/simmc2_dials_dstc10_devtest.json', 'r'))['dialogue_data']
	dialog_lst_3 = json.load(open(args.simmc_source + '/simmc2_dials_dstc10_teststd_public.json', 'r'))['dialogue_data']
	dialog_lst = dialog_lst + dialog_lst_2 + dialog_lst_3
	fashion_meta = json.load(open(args.simmc_source + '/fashion_prefab_metadata_all.json', 'r'))
	all_meta = json.load(open(args.simmc_source + '/furniture_prefab_metadata_all.json', 'r'))
	all_meta.update(fashion_meta)
	obj_rec = []
	obj_text = {}
	tp = set()
	for dialog in dialog_lst:
		scene_lst = dialog['scene_ids']
		for scene in scene_lst:
	#		 if 'wayfair' in scene_lst[scene]:
	#			 continue
			scene_path = args.scene_source + '/' + scene_lst[scene] + '_scene.json'
			scene_meta = json.load(open(scene_path, 'r'))['scenes']
			img_path = args.simmc_source + '/simmc2_scene_images_dstc10_public/' + scene_lst[scene].replace('m_', '') + '.png'
			if os.path.isfile(img_path):
				size = Image.open(img_path).size
			else:
				size = Image.open(img_path.replace('part1', 'part2')).size
	#		 print('{} subscenes included'.format(len(scene_meta)))
			for subscene in scene_meta:
				for obj in subscene['objects']: 
					obj_meta = all_meta[obj['prefab_path']]
					if obj['prefab_path'] in obj_rec:
						continue
					else:
						obj_rec.append(obj['prefab_path'])
					obj_bbox = obj['bbox']
	#				 obj_pos = obj['position']
					text = ''
					for attr_k in obj_meta.keys():
						text = attr_to_sentence(attr_k, obj_meta[attr_k], text)
					
					obj_text[obj['prefab_path']] = text
	json.dump(obj_text, open(args.save_path + '/meta_text_all.json', 'w'))
	print('Metadata file generated!')

	'''
	################################
	Generate scenegraph
	################################
	'''
	print('Genearting scene graph...')
	scene_lst = glob.glob(args.scene_source + '/scene.json')
	for s in scene_lst:
		scene = json.load(open(s, 'r'))
		rec = []
		for k in scene.keys():
			subscene = scene[k]
			for subsubscene in subscene:
				print(subsubscene)
				num_of_objs = len(subsubscene['objects'])
				edge_feat = np.zeros([num_of_objs, num_of_objs, 4])
				# ['right', 'left', 'down', 'up']
				order_of_directions = {'right': 0, 'left': 1, 'down': 2, 'up': 3}
				dt= pd.DataFrame(subsubscene['objects']).sort_values('index').reset_index()
				search_ind = dt['index'].to_dict()
				search_ind = {v:k for k, v in search_ind.items()}
				
				for direction in subsubscene['relationships']:
					dir_ind = order_of_directions[direction]
					for target in subsubscene['relationships'][direction]:
						for neighbor in subsubscene['relationships'][direction][target]:
							edge_feat[search_ind[int(target)], search_ind[int(neighbor)], dir_ind] = 1
				rec.append(edge_feat)
		savepath = s.replace('scene.json', 'graph.npy')
		np.save(savepath, rec)
	print('Scenegraph generated!')

	'''
	################################
	Generate input files
	################################
	'''
	
	dialogue_lst = json.load(open(dialogue_file, 'r'))['dialogue_data']
	obj_text_file = args.save_path + '/meta_text_all.json'
	obj_meta_text = json.load(open(obj_text_file, 'r'))
	# pos_info_file = args.save_path + '/relative_pos.json'
	# pos_info = json.load(open(pos_info_file, 'r'))
	raw_cap_file = 'SIMMC2_data/raw_captions.json'
	raw_cap = json.load(open(raw_cap_file, 'r'))

	ct_wrong = 0
	ct_ans = 0
	ct_all = 0
	ct_neq = 0
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
		
		obj_rec, box_rec, seg_rec, obj_info, all_ids, obj_cats, pinfo = get_obj_bbox(dlg['scene_ids'][cur_scene])
		edge_feat = get_edge_feat(dlg['scene_ids'][cur_scene])
		accu_turn = 0
		for idx, turn in enumerate(dlg['dialogue']):

			prev_mentioned_turn2id[idx] = []
			# check scene
			if next_scene is not None:
				if idx == int(next_scene):
					text_rec = []
					obj_rec_old, box_rec_old, seg_rec_old, obj_info_old, all_ids_old, obj_cats_old, pinfo_old = copy.copy(obj_rec), copy.copy(box_rec), copy.copy(seg_rec), copy.copy(obj_info), copy.copy(all_ids), copy.copy(obj_cats), copy.copy(pinfo)
					edge_feat_old = copy.copy(edge_feat)
					old_scene = copy.copy(cur_scene)
					obj_rec, box_rec, seg_rec, obj_info, all_ids, obj_cats, pinfo = get_obj_bbox(dlg['scene_ids'][next_scene])
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
			if len(turn['transcript_annotated']['act_attributes']['objects']) == 0 or len(turn['system_transcript_annotated']['act_attributes']['objects']) == 0:
				# add user utterance
				# add system utterance
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
				mentioned = turn['transcript_annotated']['act_attributes']['objects']					
				not_mentioned = [i for i in all_ids if i not in mentioned]
				# mentioned in current scene & previous scene
				mentioned_in_cur = []
				mentioned_in_prev = []
				for m in mentioned:
					ans = '[OBJ{}]'.format(str(m))
					if ans in obj_rec:
						mentioned_in_cur.append(ans)
					elif ans in obj_rec_old:
						mentioned_in_prev.append(ans)

				if (len(mentioned_in_cur) != 0):
					mentioned_turns = [prev_mentioned_id2turn[i] if i in prev_mentioned_id2turn else idx for i in mentioned_in_cur]
					mentioned_turns = [1 if i in mentioned_turns else 0 for i in range(len(text_rec))]
					data.append({'annot_id': data_ct, 'objects': obj_rec, 'img_fn': dlg['scene_ids'][cur_scene], 
								 'boxes': box_rec, 'segms': seg_rec, 'question': ' '.join(text_rec + [qry]), 'obj_meta': obj_info,
								 'answer_choices': mentioned_in_cur, 'mentioned_turns': mentioned_turns, 'mentioned_lbl': copy.copy(prev_mentioned_id2turn),
								'3d_pos': pinfo, 'prev_ref':0, 'turn_index': idx, 'dialogue_index': dlg_ind})
					edge_mats[dlg['scene_ids'][cur_scene]] = edge_feat
					data_ct += 1
					ct_ans += len(mentioned_in_cur)
					ct_all += len(obj_rec)
					
				if (len(mentioned_in_prev) != 0):
					mentioned_turns = [prev_mentioned_id2turn_old[i] if i in prev_mentioned_id2turn_old else idx for i in mentioned_in_prev]
					mentioned_turns = [1 if i in mentioned_turns else 0 for i in range(len(text_rec))]
					data.append({'annot_id': data_ct, 'objects': obj_rec_old, 'img_fn': dlg['scene_ids'][old_scene], 
							 'boxes': box_rec_old, 'segms': seg_rec_old, 'question': ' '.join(text_rec + [qry]), 
							 'answer_choices': mentioned_in_prev, 'obj_meta': obj_info_old, 'mentioned_turns': mentioned_turns, 'mentioned_lbl': copy.copy(prev_mentioned_id2turn_old),
								'3d_pos':pinfo_old, 'prev_ref':1, 'turn_index': idx, 'dialogue_index': dlg_ind})
					edge_mats[dlg['scene_ids'][cur_scene]] = edge_feat
					data_ct += 1
					ct_ans += len(mentioned_in_prev)
					ct_all += len(obj_rec_old)
					prev_mentioned_turn2id_old[idx] = []

				#update prev_mentioned dict
	#			 if data_ct == 605 or data_ct == 606:
	#				 print(data[-1])
	#				 print(prev_mentioned_id2turn, prev_mentioned_turn2id)
				
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
					prev_mentioned_turn2id_old[idx].append(m)
						
	#				 for p in flatten(prev_mentioned_old):
	#					 if p not in obj_rec_old:
	#						 print(p, prev_mentioned_old, obj_rec_old)
	#						 raise ValueError()
					
				text_rec.append('[USER] ' + turn['transcript'] + ' [SYS] ' + turn['system_transcript'])
				if len(text_rec) >= 3:
					text_rec = text_rec[1:]
					removed_turn = sorted(list(prev_mentioned_turn2id.keys()))[0]
					for oid in prev_mentioned_turn2id[removed_turn]:
						del prev_mentioned_id2turn[oid]
					del prev_mentioned_turn2id[removed_turn]
					prev_mentioned_turn2id = {k-1:v for k, v in prev_mentioned_turn2id.items()}
					prev_mentioned_id2turn = {k:v-1 for k, v in prev_mentioned_id2turn.items()}
	#				 prev_mentioned = prev_mentioned[1:]

	json.dump({'version':1.0, 'data': data}, open(args.save_path + '/{}_preprocessed.json'.format(mode), 'w'))
	pkl.dump(edge_mats, open(args.save_path + '/{}_edge_mat_preprocessed.pkl'.format(mode), 'wb'))