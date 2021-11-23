from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
def similarity(img1,img2):
    # calling VGGFace
    model_name = "VGG-Face"
    model = DeepFace.build_model(model_name)
    result = DeepFace.verify(img1, img2)  # validate our images
    print(result)
similarity("image1.jpeg","image2.jpeg")