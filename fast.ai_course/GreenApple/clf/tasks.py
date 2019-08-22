from __future__ import unicode_literals, absolute_import
from fastai.vision import load_learner, open_image


def predict(img_path):
	'''Takes in image path and run the stored model on
		the image on that path and returns the predicted class
	'''
	learn = load_learner('/home/chetan/Deep-Learning/fast.ai_course/GreenApple/clf/model',
						 file='export.pkl')
	img = open_image(img_path)
	print(img_path)
	pred_class, pred_idx, outputs = learn.predict(img)
	print(pred_class)
	return outputs
