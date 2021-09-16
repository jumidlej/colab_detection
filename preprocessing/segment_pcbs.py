import cv2 as cv
import numpy as np
import math
import sys
import imutils
from os import listdir
from colab_detection.preprocessing.filter import filter_screws

def closing(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Detecção de bordas
    gray = cv.Canny(gray,100,150)
    # Fechamento
    Kernel = cv.getStructuringElement(cv.MORPH_RECT,(9,9))
    close = cv.morphologyEx(gray, cv.MORPH_CLOSE, Kernel)

    return close

def fill_holes(mask, size):
    mask = cv.bitwise_not(mask)
    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        if cv.contourArea(contour) < size:
            cv.fillPoly(mask, pts =[contour], color=(0,0,0))

    mask = cv.bitwise_not(mask)

    return mask

'''
Does: find the two pcbs and crop them in to two different images
Arguments: image
'''
def segment_pcbs_template(image):
    # resize
    image = imutils.resize(image, width=1920)

    # height, width = image.shape[:2]
    # center = (width/2, height/2)
    # rotate_matrix = cv.getRotationMatrix2D(center=center, angle=45+180, scale=1)
    # image = cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height), borderMode=cv.BORDER_CONSTANT)
    # image1 = imutils.resize(image, width=1000)
    
    image = imutils.rotate_bound(image, 180+45)
    
    # cv.imshow("image1", image1)
    # cv.waitKey(0)

    # grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)
    # cv.waitKey(0)

    # grayscale e clahe novamente
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)

    # detecção de bordas
    edges = cv.Canny(cl,100,150)
    # cv.imshow("edges2", edges)
    # cv.waitKey(0)

    # fechamento
    Kernel = cv.getStructuringElement(cv.MORPH_RECT,(21,21))
    close = cv.morphologyEx(edges, cv.MORPH_CLOSE, Kernel)
    # cv.imshow("close", close)
    # cv.waitKey(0)

    # preencher contornos pequenos
    mask = fill_holes(close, 1000)
    # holes = imutils.resize(mask, width=1000)
    # cv.imshow("holes", holes)
    # cv.waitKey(0)

    # ler template
    template = cv.imread('preprocessing/template-fhd.jpeg',0)

    # resize pra uma proporção boa pra a distância da camera
    template = imutils.resize(template, width=400)
    w, h = template.shape[::-1]

    # specify a threshold
    threshold = 0.2

    # template matching
    res = cv.matchTemplate(mask,template,cv.TM_CCOEFF_NORMED)
    
    if np.all(res < threshold):
        return None, None

    # pegar melhor resultado e recortar
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    x1 = top_left[0]

    # pintar o que já foi pego de preto
    cv.rectangle(mask,top_left, bottom_right, 0, -1)
    # image = imutils.resize(image, width=1920)
    # cv.imshow("template 1", mask)
    # cv.waitKey(0)
    
    add_w = 20
    add_h = 120

    pcb1 = image[top_left[1]-add_w:top_left[1]+w+add_w, top_left[0]-add_h:top_left[0]+h+add_h+20]
    # gray = cv.cvtColor(pcb1, cv.COLOR_BGR2GRAY)
    # gray = clahe.apply(gray)
    # gray = cv.Canny(gray,100,150)

    # # encontrar maior linha da imagem (esperamos que seja sempre a horizontal da placa)
    # lines = cv.HoughLines(gray,1,np.pi/180,150)
    # try:
    #     if lines == None:
    #         ang = 0
    # except ValueError: 
    #     rho, theta = lines[0][0][0], lines[0][0][1]
    #     ang = np.degrees(theta)
    #     pcb1 = imutils.rotate_bound(pcb1, -ang)

    # template matching
    res = cv.matchTemplate(mask,template,cv.TM_CCOEFF_NORMED)
    
    if np.all(res < threshold):
        return None, None

    # pegar melhor resultado e recortar
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    x2 = top_left[0]

    cv.rectangle(mask,top_left, bottom_right, 0, -1)
    # final = imutils.resize(mask, width=800)
    # cv.imshow("template", final)
    # cv.waitKey(0)
    
    pcb2 = image[top_left[1]-add_w:top_left[1]+w+add_w, top_left[0]-add_h:top_left[0]+h+add_h+20]

    # right or left
    if x1 < x2:
        return pcb1, pcb2
    return pcb2, pcb1

'''
Does: find the two pcbs and crop them in to two different images
Arguments: image
'''
def segment_pcbs_screws(image, screw_cascade):
    # resize
    if (image.shape[1] != 1920):
        image = imutils.resize(image, width=1920)

    # grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)  #<-- Precisso para o cascade
    
    #screws = screw_cascade.detectMultiScale(gray)
    screws = screw_cascade.detectMultiScale(gray, minNeighbors = 5)
    cx = 0.0
    cy = 0.0
    
    try: 
        if (screws.shape[0] != 4):
            pcbs = None
            return pcbs, pcbs
    except AttributeError:
        pcbs = None
        return pcbs, pcbs
    
    # Procurand o centro de rotação
    for(x, y, w, h) in screws:
        # cv.circle(image, (int(x+w//2), int(y+h//2)), 10, (255,0,255), -1)
        cx = cx + x + w//2
        cy = cy + y + h//2

    center = (int(cx//4), int(cy//4))
    cv.circle(image, center, 10, (0,255,0), -1)
    
    # Ordenando os screws para dimensionar a imagem
    scrD = {}
    for(x, y, w, h) in screws:
        if (x < center[0]):
            if (y < center[1]):
                scrD["lt"] = (x, y)
            else:
                scrD["lb"] = (x, y)
        else: 
            if (y < center[1]):
                scrD["rt"] = (x, y)
            else:
                scrD["rb"] = (x, y)

    # translação para o centro da imagem
    image_h, image_w = image.shape[0], image.shape[1]
    cix = image_w//2
    ciy = image_h//2
    tx = cix - center[0]
    ty = ciy - center[1]
    M = np.float32([ [1,0,tx], [0,1,ty] ])
    image = cv.warpAffine(image, M, (image_w, image_h))
    
    final = imutils.resize(image, width=400)
    cv.imshow("translated", final)
    cv.waitKey(0)

    try:
        ang = np.degrees(np.arctan2(scrD["lb"][1] - scrD["rt"][1], scrD["rt"][0] - scrD["lb"][0]))
    except:
        print("Algo deu errado com os parafusos!\n")
        pcbs = None
        return pcbs, pcbs
    ang -= 2.61 # correção do ângulo dos parafusos
        
    rows,cols = image.shape[0], image.shape[1]
    # distância em pixels entre os screws
    c1 = np.sqrt((scrD["lb"][0] - scrD["rb"][0])**2 + (scrD["lb"][1] - scrD["rb"][1])**2)
    c2 = np.sqrt((scrD["lt"][0] - scrD["rt"][0])**2 + (scrD["lt"][1] - scrD["rt"][1])**2)
    c = (c1 + c2)/2
    ang *= -1
    ang += 180
    
    M = cv.getRotationMatrix2D((cix, ciy),ang,1)
    image = cv.warpAffine(image,M, (cols,rows))

    final = imutils.resize(image, width=400)
    cv.imshow("rotated", final)
    cv.waitKey(0)
    
    #pxm = c/182  #pixels por milímetro
    pxm = c/163  #pixels por milímetro (distancia entre os parafusos)
    pcbx = 140 * pxm # largura da pcb em pixels (140 mm)
    pcby = 120 * pxm # altura da pcb em pixels
    extra = 25 * pxm # folga horizaontal
    
    lb = rows//2
    lt = int(lb - pcby)
    if lt < 0: 
        lt = 0
    lr = cols//2
    ll = int(lr - pcbx)
    lr = int(lr + extra)
    if ll < 0: 
        ll = 0
    
    #left = image[lt:lb, ll:lr,:].copy()
    left = image[lt:lb, ll:lr,:]

    rt = rows//2
    rb = int(rt + pcby)
    if rb > rows: 
        rb = rows
    rl = cols//2
    rr = int(rl + pcbx)
    rl = int(rl - extra)
    if rr > cols: 
        rr = cols

    #right = image[rt:rb, rl:rr, :].copy()
    right = image[rt:rb, rl:rr, :]
    return left, right

'''
Does: find the two pcbs and crop them in to two different images
Arguments: image
'''
def segment_pcbs(image, screw_cascade):
    # resize
    if (image.shape[1] != 1920):
        image = imutils.resize(image, width=1920)

    # grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)  #<-- Precisso para o cascade
    
    #screws = screw_cascade.detectMultiScale(gray)
    screws = screw_cascade.detectMultiScale(gray, minNeighbors = 3)
    cx = 0.0
    cy = 0.0

    # aqui a gente tem um erro quando não encontra nenhum screw, aparentemente não retorna um numpy array
    # ERRO: AttributeError: 'tuple' object has no attribute 'shape'
    try: 
        if (screws.shape[0] < 4):
            print("Screws < 4")
            pcbs = None
            return pcbs, pcbs
        elif screws.shape[0] >= 4:
            print("Screws >= 4")
            screws = filter_screws(gray, screws)
            if screws == None:
                pcbs = None
                return pcbs, pcbs
    except AttributeError:
        pcbs = None
        return pcbs, pcbs
    
    # Procurand o centro de rotação
    for(x, y) in screws:
        # cv.circle(image, (int(x), int(y)), 10, (0,255,0), -1)
        cx = cx + x
        cy = cy + y
    center = (int(cx//4), int(cy//4))

    # cv.circle(image, center, 10, (0,0,255), -1)
    
    # Ordenando os screws para dimensionar a imagem
    scrD = {}
    for(x, y) in screws:
        if (x < center[0]):
            if (y < center[1]):
                scrD["lt"] = (x, y)
            else:
                scrD["lb"] = (x, y)
        else: 
            if (y < center[1]):
                scrD["rt"] = (x, y)
            else:
                scrD["rb"] = (x, y)

    # print(str(screws))

    # translação para o centro da imagem
    image_h, image_w = image.shape[0], image.shape[1]
    cix = image_w//2
    ciy = image_h//2
    tx = cix - center[0]
    ty = ciy - center[1]
    M = np.float32([ [1,0,tx], [0,1,ty] ])
    image = cv.warpAffine(image, M, (image_w, image_h))

    # final = imutils.resize(image, width=800)
    # cv.imshow("translated", final)
    # cv.waitKey(0)

    try:
        ang = np.degrees(np.arctan2(scrD["lb"][1] - scrD["rt"][1], scrD["rt"][0] - scrD["lb"][0]))
    except:
        print("Algo deu errado com os parafusos!\n")
        pcbs = None
        return pcbs, pcbs  
        
    ang -= 2.61 # correção do ângulo dos parafusos
        
    rows,cols = image.shape[0], image.shape[1]
    # distância em pixels entre os screws
    c1 = np.sqrt((scrD["lb"][0] - scrD["rb"][0])**2 + (scrD["lb"][1] - scrD["rb"][1])**2)
    c2 = np.sqrt((scrD["lt"][0] - scrD["rt"][0])**2 + (scrD["lt"][1] - scrD["rt"][1])**2)
    c = (c1 + c2)/2
    ang *= -1
    ang += 180
    ciy += 10
    
    M = cv.getRotationMatrix2D((cix, ciy),ang,1)
    image = cv.warpAffine(image,M, (cols,rows))

    # image = np.pad(image, pad_with=1)
    # print(image.shape)
    # rows,cols = image.shape[0], image.shape[1]

    # final = imutils.resize(image, width=800)
    # cv.imshow("rotated", final)
    # cv.waitKey(0)
    
    #pxm = c/182  #pixels por milímetro
    pxm = c/163  #pixels por milímetro (distancia entre os parafusos)
    pcbx = 140 * pxm # largura da pcb em pixels (140 mm)
    pcby = 140 * pxm # altura da pcb em pixels
    extra = 25 * pxm # folga horizaontal
    
    lb = rows//2+20
    lt = int(lb - pcby)
    if lt < 0: 
        lt = 0
    # print(lt)
    lr = cols//2
    ll = int(lr - pcbx)
    lr = int(lr + extra)
    if ll < 0: 
        ll = 0
    
    #left = image[lt:lb, ll:lr,:].copy()
    left = image[lt:lb, ll:lr,:]

    rt = rows//2+20
    rb = int(rt + pcby)
    # print(rb)
    if rb > rows: 
        rb = rows
    rl = cols//2
    rr = int(rl + pcbx)
    rl = int(rl - extra)
    if rr > cols: 
        rr = cols

    #right = image[rt:rb, rl:rr, :].copy()
    right = image[rt:rb, rl:rr, :]
    return left, right