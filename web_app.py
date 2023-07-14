import streamlit as st
import TableExtractor as te
import cv2
from img2table.ocr import EasyOCR
from img2table.document import Image
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


    
def table_extractor(path):
    table_extractor = te.TableExtractor(path)
    perspective_corrected_image = table_extractor.execute()
    
def use_ocr(path):
    doc = Image(path)
    ocr = EasyOCR(lang=["ru"])
    doc.to_xlsx('result/table.xlsx',
            ocr=ocr,
            implicit_rows=False,
            borderless_tables=False)
    
    
    
def main():
    st.markdown(
        """ 
        # Мой первый проект
        """
    )
    
    image_file = st.file_uploader(label='ЧТо то там', type='jpeg')
    if image_file is not None:
        st.image(image=image_file)
        with open('result/original.jpg', 'wb') as file:
            file.write(image_file.getbuffer())
    table_extractor('result/original.jpg')
    use_ocr('result/result.jpg')
    data = pd.read_excel('result/table.xlsx')
    st.dataframe(data=data)
    
    
    

if __name__== '__main__':
    main()