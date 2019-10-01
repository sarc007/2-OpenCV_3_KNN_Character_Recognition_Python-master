import pytesseract
import cv2
from PIL import Image, ImageFilter,ImageOps
# img = Image.open("t1.jpg") # abcdefghijklmnopqrstuvwxyz
def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """

    text = pytesseract.image_to_string(Image.open(filename))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

print(ocr_core('25.jpg'))

# text = pytesseract.image_to_string(img, config="-c tessedit"
#                                               "_char_whitelist=0123456789:-"
#                                               " --psm 7"
#                                               " -l osd"
#                                               " ")
# print(text)

