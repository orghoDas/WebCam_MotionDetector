import cv2, time, pandas 
from datetime import datetime


# FIRST FRAME
first_frame = None


status_list = [None, None]
times = []

df = pandas.DataFrame(columns=["Start", "End"])


# VIDEO CAPTURE
video = cv2.VideoCapture(0)


while True:
    # READING THE VIDEO 
    check, frame = video.read()

    status = 0

    # GRAYSCALE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)


    if first_frame is None:
        first_frame = gray
        continue
    

        # DELTA FRAME WITH ABSOLUTE DIFFERENCE
    delta_frame = cv2.absdiff(first_frame, gray)

    # THRESHOLDING THE DELTA FRAME 
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # RETRIEVING EXTERNAL OBJECTS
    (_,cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # CONTOUR LOOP
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)


    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())


    # FRAME WINDOWS 
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    # WAIT KEY 
    key = cv2.waitKey(1)

    # BREAK KEY 
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())

        break

    
    
print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start" : times[i], "End" : times[i+1]}, ignore_index = True)


df.to_csv('Times.csv')

# CALLING FUNCTONS
video.release()
cv2.destroyAllWindows()