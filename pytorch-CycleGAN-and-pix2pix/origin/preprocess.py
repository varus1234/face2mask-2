import cv2
import numpy as np
from PIL import Image
import os


# 顔検出と切り抜き
def detect_face(image_picture):
    # img_name : str ('image.jpg')
    
    HAAR_FILE = os.path.join('origin', 'haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(HAAR_FILE)
    
    img_name = image_picture.split('/')[-1].split('.')[0]
    img = cv2.imread(image_picture)
    
    height, width = img.shape[:2]
    
    img_g = cv2.imread(image_picture,0)
    
    faces = cascade.detectMultiScale(img_g, minNeighbors=3, scaleFactor=1.1)
    
    # print(faces)
    
    faces_cut = []
    faces_position = []
    for face in faces:
        # for x,y,w,h in face:
            x, y, w, h = face[0], face[1], face[2], face[3]
            ex_w = round(w*0.2)
            x_min = x - ex_w
            x_max = x + w + ex_w
            
            ex_h = round(h*0.2)
            y_min = y - ex_h
            y_max = y + h + ex_h
            
            if x_min < 0: x_min = 0
            if x_max > width: x_max = width
            if y_min < 0: y_min = 0
            if y_max > height: y_max = height
            
            y_min,y_max,x_min,x_max = int(y_min), int(y_max), int(x_min), int(x_max)
            face_cut = img[y_min:y_max, x_min:x_max]
            faces_cut.append(face_cut)
            faces_position.append([x_min,y_min,x_max,y_max])
    
    faces_name = []
    for i, face_cut in enumerate(faces_cut):
        face_name = f'{img_name}_{i}_face_unmasked.jpg'
        faces_name.append(face_name)
        cv2.imwrite(os.path.join('tmp/detect_face', face_name), face_cut)
        
    return faces_name, faces_position


# pix2pix用にAを繋げる
def combineAA(img_name):
    # img_name : str ('kao_unmasked.jpg')
    
    img_name = os.path.join('tmp/detect_face', img_name)
    im_A = cv2.imread(img_name, 1)
    im_AA = np.concatenate([im_A, im_A], 1)
    img_name = img_name.split('/')[-1].split('_face')[0]
    cv2.imwrite(f'datasets/tmp/test/{img_name}.jpg', im_AA)


# 256から元のheightへリサイズする（pix2pix後）
def resizeB(img_name, img_name_before, path_1, path_2):
    # img_name : str ('kao_masked.jpg')
    # img_name_before : str ('kao_unmasked.jpg')
    
    img_name_before = os.path.join('tmp/detect_face', img_name_before)
    img_name = os.path.join('results', path_1, f'test_{path_2}', 'images', img_name)
    
    img_before = Image.open(img_name_before)
    height = img_before.size[1]
    width = img_before.size[0]
    
    img = Image.open(img_name)
    img_resize = img.resize((width, height))
    img_name_before = img_name_before.split('/')[-1].split('_face')[0] + '_face_masked.jpg'
    img_resize.save(os.path.join('tmp/resize', img_name_before))


# 元画像へマスク付与顔画像を戻す
def return_face(img_name, faces_name, faces_position):
    # img_name : str ('image.jpg')
    # face_name : str ('kao_masked.jpg')
    
    img = cv2.imread(os.path.join('test_imgs', img_name), 1)
    for face_name, face_position in zip(faces_name, faces_position):
        face_name = face_name.split('_face')[0] + '_face_masked.jpg'
        
        face = cv2.imread(os.path.join('tmp/resize', face_name), 1)
        height = face.shape[0]
        height = int(round(0.4*height))
        face = face[height:, :]
        x_min,y_min,x_max,y_max = face_position
        ex_y = y_max - y_min
        ex_y = int(round(0.4*ex_y))
        img[y_min+ex_y:y_max, x_min:x_max] = face
    
    img_name = img_name.split('.')[0] + '_masked.jpg'
    cv2.imwrite(os.path.join('tmp/result', img_name), img)