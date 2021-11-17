import streamlit as st
import tensorflow as tf
#import cv2
from PIL import Image, ImageOps
from skimage.transform import resize
import numpy as np
import time
import os, random


#loading the flower classification model
st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def load_model(model_file):
    model_main_path ='./saved_model/'
    model = tf.keras.models.load_model(model_main_path + model_file)
    return model

file_1 = 'flower_classification_step_model.hdf5'
model_1 = load_model(file_1)

file_2 = 'flower_classification_timeaug_model.hdf5'
model_2 = load_model(file_2)

def load_image_predict(image, model):
    IMAGE_SIZE = (331, 331)
    
    image = ImageOps.fit(image, IMAGE_SIZE, Image.ANTIALIAS)
    img = np.asarray(image)/255
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)

    return prediction



#Class name
CLASS= ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']



st.markdown(
   """
   <style>
   .reportview-container {
       background: url("https://www.wallpapertip.com/wmimgs/144-1445080_spring-flowers-wallpaper-mobile-for-free-wallpaper-pretty.jpg")
   }
    """,
    unsafe_allow_html=True
)

st.title('Flower Image Classifition App')
st.write('''
\n
Pick a flower image from validation set or upload your own image.

''')

options = st.sidebar.selectbox('Select an dataset', ('validation','user image'))
models = st.sidebar.selectbox('Select a model', ('model_1','model_2','model_combined'))
model_dict = {'model_1':model_1, 'model_2':model_2, 'model_combined':'model_combined'}
prediction = st.sidebar.button('Make a prediction')

#define  function to make prediction based on the selection from models
def show_prediction(file, model):
    #get the string of the model
    model_key = [key for key, value in model_dict.items() if value == model][0]
    #show the selected model 
    st.subheader('Model chosen: ' + model_key)
    image = Image.open(file)
    st.image(image, use_column_width = False, width =331)

    if model_key != 'model_combined':

        prob = load_image_predict(image, model)
        prediction_text ='This image most likely is '+CLASS[np.argmax(prob)]
        
    else:         
        
        prob_1 = load_image_predict(image, model_1)
        prob_2= load_image_predict(image, model_2)
        prediction_text = 'This image most likely is ' + CLASS[np.argmax((prob_1+prob_2)/2)]
    return prediction_text 

if options == 'user image':
    #upload a flower image
    file = st.sidebar.file_uploader('Please upload an flower image within CLASS', type=['jpeg','jpg','png'])
    if prediction:
        prediction_text = show_prediction(file, model_dict[models])
        time.sleep(1)
        st.subheader(prediction_text)


        
else:
    if prediction:    
	# randomly select a file from streamlit_image folder
        main_path = './streamlit_image/'
        filename = random.choice(os.listdir(main_path))
        image_path = os.path.join(main_path, filename)

        prediction_text = show_prediction(image_path, model_dict[models])
        time.sleep(1)
        st.subheader(prediction_text)
        time.sleep(1)
        actual_text = 'It is actually is ' + filename.replace('-','.').split('.')[0]
        st.subheader(actual_text)


class_button = st.sidebar.button('Click here for flower classes')
if class_button:
    st.write('''
        ###  Flower classes included in the App
    ''' 
    +str(CLASS))