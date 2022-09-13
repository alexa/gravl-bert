import json
import numpy
import torch
@torch.no_grad()
class save_false_cases():
	def __init__(self, output_path):
		self.output_path = output_path
		self.record = []

	def update(self, outputs):
		cls_logits = outputs['label_logits']
		if cls_logits.dim() == 1:
			cls_preds = (cls_logits > 0.5).long()
		else:
			cls_preds = cls_logits.argmax(dim=1)
		label = outputs['label'].view(-1)
		FP = (cls_preds == 1) * (label == 0)
		FN = (cls_preds == 0) * (label == 1)
		self.record.append({'FP': torch.where(FP == 1).cpu().numpy(), 'FN': torch.where(FN == 1).cpu().numpy()})

	def save(self):
		json.dump(self.record, open(self.output_path, 'w'))