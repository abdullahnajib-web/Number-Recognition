import numpy as np
import cv2
from keras.models import load_model
import serial


ser = serial.Serial('COM11', 115200)

if not ser.isOpen():
    ser.open()
print('com11 is open', ser.isOpen())

model = load_model('mnist.h5')

webcam = cv2.VideoCapture()
webcam.open(1,cv2.CAP_DSHOW)

def firstEl(elem):
    return elem[0]


num =''
while True:
    status , image = webcam.read()
        
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    if len(contours) < 5:
        listDig = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # make a rectangle box around each curve
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = th[y:y + h, x:x + w]

            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit, (18, 18))

            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

            digit = padded_digit.reshape(1, 28, 28, 1)
            digit = digit / 255.0

            pred = model.predict([digit])[0]
            final_pred = np.argmax(pred)

            #data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'
            data = str(final_pred)
            t = (x,data)
            listDig.append(t)
            
            cv2.putText(image, data, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
            
        listDig.sort(key=firstEl)
        ret = ''.join([lis[1] for lis in listDig])
        if num != ret:
            #print(ret)
            num = ret
            ser.write(ret[::-1].encode())
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
webcam.release()
