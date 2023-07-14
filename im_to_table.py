import TableExtractor as te
import cv2
from img2table.ocr import EasyOCR
from img2table.document import Image
from PIL import Image as PImage


path_to_image = "405/2.jpg"
table_extractor = te.TableExtractor(path_to_image)
perspective_corrected_image = table_extractor.execute()


doc = Image('result.jpg')

ocr = EasyOCR(lang=["ru"])

doc.to_xlsx('table.xlsx',
            ocr=ocr,
            implicit_rows=False,
            borderless_tables=False)
