__all__ = ['is_cat', 'learn', 'classify_image', 'categories', 'image', 'label', 'examples', 'intf']

from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

# Load model trainig on Kaggle
# https://www.kaggle.com/code/bruno5/saving-a-basic-fastai-model/notebook
learn = load_learner('model.pkl')

# Classify image
categories = ('Dog', 'Cat')
def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

# Gradio interface
image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['dog.jpeg', 'cat.jpeg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)