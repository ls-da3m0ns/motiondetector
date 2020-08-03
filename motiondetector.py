import cv2,time
from datetime import datetime
import pandas


first_frame=None
video=cv2.VideoCapture(0)

status_list=[None,None]
times=[None,None]

df=pandas.DataFrame(columns=["Start","End"])

while True:
    check,frame=video.read()

    status=0

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_delta=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_delta=cv2.dilate(thresh_delta,None,iterations=2)
    _,cnts,__=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for countours in cnts:
        if cv2.contourArea(countours) < 20000:
            continue

        status=1

        (x,y,w,h)=cv2.boundingRect(countours)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


    status_list.append(status)

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-2]==0 and status_list[-1]==1:
        times.append(datetime.now())


    cv2.imshow("gray",gray)
    cv2.imshow("delta_frame",delta_frame)
    cv2.imshow("thresh_delta",thresh_delta)
    cv2.imshow("Color",frame)

    key=cv2.waitKey(1)
    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("times.csv")
print(status_list)
print(times)

video.release()
cv2.destroyAllWindows()
