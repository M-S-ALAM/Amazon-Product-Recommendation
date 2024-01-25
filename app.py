import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import streamlit as st
from ML_Algorithm.merge_result import Combine_results

st.set_page_config(page_title='Amazon Recommendation', page_icon='🏡', layout='centered')
st.title(':green[Amazon Apparel Recommendations]')

if all(key not in st.session_state.keys() for key in 'data'):
    st.session_state['data'] = None


def fetch_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


asin_number = st.text_input('Enter ASIN number')
select_algorithm = st.selectbox('Choose Algorithm',
                                options=['Bag of Word(BOW)', 'Term frequency-inverse document frequency(TF-IDF)',
                                         'Inverse document frequency(IDF)', 'Average Word 2 Vec',
                                         'IDF weighted Word 2 vec', 'Algorithm with brand and color',
                                         'Deep Learning(VGG16)', 'All the algorithm combined results'])
number_ = st.number_input('Number of recommendation', min_value=1, max_value=10)

submit_button = st.button(label='Prediction', type='primary')
data = pd.read_pickle('/home/shobot/Desktop/Project pro/Amazon Product Reviews/database/16k_apperal_data_preprocessed')
if submit_button:
    if asin_number is not None:
        indices = Combine_results(asin_number, number_, select_algorithm).Recommended_results()
        df_indices = list(data.index[indices])
        st.write(':red[Search Items]')
        #data = data.reset_index(drop=True)
        indexes = data[data['asin'] == asin_number].index
        col1, col2 = st.columns(2)
        with col1:
            img_url = data['medium_image_url'].loc[indexes[0]]
            image = fetch_image_from_url(img_url)
            st.image(image)
        with col2:
            st.write('ASIN :', data['asin'].loc[indexes[0]])
            st.write('Brand:', data['brand'].loc[indexes[0]])
            st.write('Title:', data['title'].loc[indexes[0]])
        st.write('='*88)
        st.write(':red[Recommended Items]')
        for i in range(len(df_indices)-1):
            col1, col2 = st.columns(2)
            with col1:
                img_url = data['medium_image_url'].loc[df_indices[i+1]]
                image = fetch_image_from_url(img_url)
                st.image(image)
            with col2:
                st.write('ASIN :', data['asin'].loc[df_indices[i+1]])
                st.write('Brand:', data['brand'].loc[df_indices[i+1]])
                st.write('Title:', data['title'].loc[df_indices[i+1]])
            st.write('='*88)
    else:
        st.write('Please! enter correct input')
