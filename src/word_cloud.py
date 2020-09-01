from image_loader import open_images
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from os import path, getcwd, listdir, walk
from PIL import Image
import numpy as np
import glob


def word_cloud(text, title, savefile):
    '''Create a wordcloud with from fruits and vegetable names'''
    fv_mask = np.array(Image.open(path.join(d, 'images/index.jpeg')))    
    wordcloud = WordCloud(mask=fv_mask, background_color='white', stopwords=ENGLISH_STOP_WORDS).generate_from_text(text)
    plt.figure(figsize=(10, 6), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontdict={"fontsize": 22})
    plt.tight_layout()
    plt.savefig(savefile, bbox_inches='tight')
    plt.show()
    return plt

if __name__ == '__main__':
    d = getcwd()
    all_fru_veg = listdir('data/fruits_vegetables/')    
    jt = ' '.join(all_fru_veg)
    word_cloud(jt, "Fruits And  Veggies", 'images/fv_world_cloud.png')