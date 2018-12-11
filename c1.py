import pygame
import pygame.camera

#import random

import time

def takephoto():
 r=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

 pygame.camera.init()
 cam = pygame.camera.Camera("/dev/video0",(640,480))
 cam.start()
 img = cam.get_image()
 tm = time.strftime('%H:%M:%S')
 #filename=random.choice(r)+"filename"+random.choice(r)
 filename=tm+".jpg"
 print(filename)
 pygame.image.save(img,filename)
 cam.stop()
 pygame.camera.quit()
 return
