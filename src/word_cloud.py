from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from os import path, getcwd, listdir
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

def main():
    pass

if __name__ == '__main__':
    main()
    d = getcwd()
    all_fru_veg = listdir('data/Train/')
    jt = ' '.join(all_fru_veg)
    word_cloud(jt, title="Fruits And Veggies", savefile='images/fv_world_cloud.png')