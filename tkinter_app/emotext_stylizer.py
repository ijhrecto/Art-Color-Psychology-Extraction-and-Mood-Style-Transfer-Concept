import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

import tkinter as tk
from tkinter import StringVar, ttk
from PIL import ImageTk, Image

class Text2StyledImage():
    def __init__(self) -> None:
        self.df_emo = pd.read_csv(os.path.join("..//data//annotation_data", "annot_emo_paint.csv"))
        self.style_path = os.path.join("..//data//style//")
        
        # style transfer model
        model_path = os.path.join("..//data//model//", "magenta_arbitrary-image-stylization-v1-256_2")
        self.model = hub.load(model_path)

    def get_stylized_image(self, text_input, content_filename):
        # text 2 style transfer process
        prep_text = self.preprocess_text(text_input)
        style_filename = self.get_best_emo_match_img(prep_text)
        style_image = self.load_image(os.path.join(self.style_path, style_filename))
        content_image = self.load_image(content_filename)
        stylized_image_tensor = self.model(tf.constant(content_image), tf.constant(style_image))[0]
        output_image = self.tensor_to_image(stylized_image_tensor)
        
        return output_image

    def preprocess_text(self, text):
        # tokenization
        text_no_punc = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
        text_token = word_tokenize(text_no_punc)

        # stopwords
        stop_words=set(stopwords.words("english"))
        text_filtered=[]
        for w in text_token:
            if w not in stop_words:
                text_filtered.append(w)

        # stemmer
        stemmer = SnowballStemmer("english")
        stemmed_words=[]
        for t in text_filtered:
            stemmed_words.append(stemmer.stem(t))
        
        return stemmed_words

    def get_best_emo_match_img(self, text):
        # get the weight of each file
        # by finding the matching emotion words
        file_weight = {}
        for idx, row in self.df_emo.iterrows():
            pt = self.preprocess_text(row["emotion"])
            
            # get matching string elements
            matching = [s for s in pt if any(xs in s for xs in text)]

            # create dictionary for matched emo keywords
            match_weight = len(matching)
            file_weight.update({row["filename"]: match_weight})
        
        
        best_match = max(file_weight, key=file_weight.get)

        return best_match

    def load_image(self, image_path):
        max_dim = 512
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)#Detects the image to perform apropriate opertions
        img = tf.image.convert_image_dtype(img, tf.float32)#converts image to tensor dtype

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)# Casts a tensor to float32.

        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)

        return img[tf.newaxis, :]
    
    def tensor_to_image(self, tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

class EmoStylizerApp():
    def __init__(self) -> None:
        # root
        self.root = tk.Tk()
        # self.root.geometry("750x290")
        self.root.title("EmoStyler: Emotive Text to Stylized Image")
        # style=ttk.Style()

        # file paths
        self.content_dir = os.path.join("..", "data", "content")
        self.style_dir = os.path.join("..", "data", "style")

        # input image
        self.content_filename = os.path.join(self.content_dir, "input_add.jpg")
        self.content_image = ImageTk.PhotoImage(Image.open(self.content_filename).resize((245, 245)))
        self.frame_content = tk.Frame(self.root, padx=20, pady=20)
        self.content_image_box = ttk.Label(self.frame_content, image=self.content_image, width=245)
        self.button_import = ttk.Button(self.frame_content, text="Input Image", command=self.set_content_image)

        # plus & equals
        self.frame_plus = tk.Label(self.root,text="+", font="Arial 32") 
        self.frame_equals = tk.Label(self.root,text="=", font="Arial 32") # apply style  / image

        # input text
        self.frame_text = tk.Frame(self.root, pady=20, padx=20)
        self.text_input = tk.Text(self.frame_text, width=30, height=15, state="disabled")
        self.button_stylize = ttk.Button(self.frame_text, text="Apply Emotive Text", state="disabled")

        # output image
        self.output_image = os.path.join(self.content_dir, "empty.jpg")
        self.output_image = ImageTk.PhotoImage(Image.open(self.output_image).resize((245, 245)))      
        self.frame_output = tk.Frame(self.root, padx=20, pady=20)
        self.stylized_image_box = tk.Label(self.frame_output, image=self.output_image, height=245, width=245)
        self.button_save = ttk.Button(self.frame_output, text="Save Stylized Image", state="disabled")

        # update and loop
        self.update_UI()
        self.root.mainloop()

    def update_UI(self):
        # frames 
        self.frame_content.pack(side=tk.LEFT)
        self.frame_plus.pack(side=tk.LEFT)
        self.frame_text.pack(side=tk.LEFT)
        self.frame_equals.pack(side=tk.LEFT)
        self.frame_output.pack(side=tk.LEFT)

        # widgets
        self.content_image_box.grid(row=0, column=0)
        self.button_import.grid(row=1, column=0)
        self.text_input.grid(row=0, column=0)
        self.button_stylize.grid(row=1, column=0)
        self.stylized_image_box.grid(row=0, column=0)
        self.button_save.grid(row=1, column=0)
    
    def set_content_image(self):
        self.content_filename = tk.filedialog.askopenfilename(initialdir=self.content_dir, title="Select Image", 
                                    filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")) )
        self.content_image = ImageTk.PhotoImage(Image.open(self.content_filename).resize((245, 245)))
        self.content_image_box.config(image=self.content_image)
        self.content_image_box.image = self.content_image

        # update other widgets

        self.button_stylize.config(state="normal", command=self.stylize_the_image)
        self.text_input.config(state="normal")

        """ # turn off button when there is no text in the textbox
        text_string = self.text_input.get("1.0",tk.END)
        if text_string: # if it contains text
            self.text_input.config(state="normal")
        """
        
        self.update_UI()



    def stylize_the_image(self):
        # stylizing process
        st = Text2StyledImage()
        self.stylized_image = st.get_stylized_image(
            text_input=self.text_input.get("1.0",tk.END), 
            content_filename=self.content_filename)

        # update other widgets
        self.stylized_image_tk = ImageTk.PhotoImage(self.stylized_image.resize((245, 245)))
        self.stylized_image_box.config(image=self.stylized_image_tk)
        self.stylized_image_box.image = self.stylized_image_tk
        self.button_save.config(state="normal", command=self.save_image)
        self.update_UI()

    def save_image(self):
        # open dialog
        file = tk.filedialog.asksaveasfile(mode='w', defaultextension=".jpg", 
                    filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")) )
        if file:
            abs_path = os.path.abspath(file.name)
            # out = Image.alpha_composite(im, txt)
            self.stylized_image.save(abs_path) # saves the image to the input file name. 
        file.close()
        self.update_UI()


if __name__ == "__main__":

    EmoStylizerApp()
    # add messagebox while the style transfer is happening

    # improve the paint annotations from the color emotion recognition part
    # use color palette emotion research for the improvement of the painting annotations

    # improve the text extraction
    # get the emotions in the text input
    # then assign that as the image
    # get the image with most matching annotation
    # if there are multiple best match, choose randomly