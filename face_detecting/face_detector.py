import cv2

# create the cascade object in order to search for face in image
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# color image
img = cv2.imread("haha.jpg")

# change color to gray to increase detecting accuracy
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face stores coordinates of face in image, in this case [[157  84 379 379]]
# scaleFactor means smaller value, higher accuracy, the scale will be decreased by 5 %
# minNeighbors means how many neighbors to search around the window.
face = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=10)

# draw rectangle on face
for x, y, w, h in face:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # (0,255,0) is green, 3 is width of rectangle

# print(type(face))  # <class 'numpy.ndarray'>
print(face)

resized_img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))

cv2.imshow("detecting face", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("resized_haha.jpg", resized_img)

