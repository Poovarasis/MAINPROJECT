from flask import Flask, render_template, flash, request, session
from flask import render_template, redirect, url_for, request
#from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
#from werkzeug.utils import secure_filename

import mysql.connector
import sys, fsdk, math, ctypes, time

import time
#import yagmail
import sys
app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/AdminLogin")
def AdminLogin():
    return render_template('AdminLogin.html')

@app.route("/DriverLogin")
def DriverLogin():
    return render_template('DriverLogin.html')

@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1drowsydb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb")
    data = cur.fetchall()
    return render_template('AdminHome.html', data=data)

@app.route("/NewOwner")
def NewOwner():
    return render_template('NewOwner.html')

@app.route("/OwnerInfo")
def OwnerInfo():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1drowsydb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM ownertb")
    data = cur.fetchall()
    return render_template('OwnerInfo.html', data=data)
@app.route("/NewDriver")
def NewDriver():
    #import LiveRecognition  as liv

    #liv.att()
    #dname = session['name']
    #print(dname)
    #del sys.modules["LiveRecognition"]

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1drowsydb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM ownertb")
    data = cur.fetchall()
    #return render_template('AdminHome.html', data=data)

    return render_template('NewDriver.html', company=data )
@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    error = None
    if request.method == 'POST':
       if request.form['uname'] == 'admin' or request.form['password'] == 'admin':
           conn = mysql.connector.connect(user='root', password='', host='localhost', database='1drowsydb')
           cur = conn.cursor()
           cur.execute("SELECT * FROM regtb")
           data = cur.fetchall()
           return render_template('AdminHome.html', data=data)

       else:
        return render_template('index.html', error=error)


@app.route("/newdriver", methods=['GET', 'POST'])
def newdriver():
     if request.method == 'POST':
          uname =  request.form['uname']
          company = request.form['company']
          dno = request.form['dno']
          ano = request.form['ano']
          exp = request.form['exp']
          password = request.form['password']

          conn = mysql.connector.connect(user='root', password='', host='localhost', database='1drowsydb')
          cursor = conn.cursor()
          cursor.execute("SELECT  *  FROM ownertb where  CompanyName='" + company + "'")
          data = cursor.fetchone()

          if data:
              Mobile = data[3]
              Email = data[4]
              Address = data[5]
         else:
              return 'Incorrect username / password !'
          conn = mysql.connector.connect(user='root', password='', host='localhost', database='1drowsydb')
          cursor = conn.cursor()
          cursor.execute("insert into regtb values('','"+company+"','"+Mobile+"','"+Email+"','"+Address+"','"+dno+"','"+ano+"','"+exp +"','"+uname+"','"+password+"')")
          conn.commit()
          conn.close()
     return render_template("DriverLogin.html")
@app.route("/newowner", methods=['GET', 'POST'])
def newowner():
     if request.method == 'POST':
          oname = request.form['oname']
          cname = request.form['cname']
          mobile = request.form['mobile']
          email = request.form['email']
          address = request.form['address']

          conn = mysql.connector.connect(user='root', password='', host='localhost', database='1drowsydb')
          cursor = conn.cursor()
          cursor.execute("insert into ownertb values('','"+oname+"','"+cname+"','"+mobile+"','"+email+"','"+address+"')")
          conn.commit()
          conn.close()

     conn = mysql.connector.connect(user='root', password='', host='localhost', database='1drowsydb')
     cur = conn.cursor()
     cur.execute("SELECT * FROM ownertb")
     data = cur.fetchall()
     return render_template('OwnerInfo.html', data=data)

@app.route("/driverlogin", methods=['GET', 'POST'])
def userlogin():
    error = None
    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['password']
        session['dname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1drowsydb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where UserName='" + username + "' and Password='" + password + "'")
        data = cursor.fetchone()
        if data is None:
            return render_template('index.html')
            return 'Username or Password is wrong'
        else:
            session['mob'] = data[2]
            session['email'] = data[3]
            #driveralert()
            driver()
        return render_template('DriverLogin.html')
def driveralert():
    from scipy.spatial import distance
    from imutils import face_utils
    import imutils
    import dlib
    import cv2

    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    thresh = 0.25

    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("static/models/shape_predictor_68_face_landmarks.dat")  # Dat file is the crux of the code

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)
    flag = 0
    frame_check = 20
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag += 1
                print(flag)
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    import winsound
                    filename = 'alert.wav'
                    winsound.PlaySound(filename, winsound.SND_FILENAME)
                    #play("alert.wav")
                    #cv2.imwrite("alert.jpg", frame)
                    #sendmail()
                    #sendmsg(session['mob'],session['dname'] +  " Driver Has Sleep")
            else:
                flag = 0
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release()
def driver():
    import cv2 as cv
    import mediapipe as mp
    import time
    import utils, math
    import numpy as np
    # variables
    frame_counter = 0
    CEF_COUNTER = 0
    TOTAL_BLINKS = 0
    # constants
    CLOSED_EYES_FRAME = 3
    FONTS = cv.FONT_HERSHEY_COMPLEX
    flag = 0
    frame_check = 20

    # face bounder indices
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
                 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # lips indices for Landmarks
    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40,
            39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
    LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
    # Left eyes indices
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

    # right eyes indices
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

    map_face_mesh = mp.solutions.face_mesh
    # camera object
    camera = cv.VideoCapture(0)

    # landmark detection function
    def landmarksDetection(img, results, draw=False):
        img_height, img_width = img.shape[:2]
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                      results.multi_face_landmarks[0].landmark]
        if draw:
            [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

        # returning the list of tuples for each landmarks
        return mesh_coord

    # Euclaidean distance
    def euclaideanDistance(point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance

    # Blinking Ratio
    def blinkRatio(img, landmarks, right_indices, left_indices):
        # Right eyes
        # horizontal line
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]
        # vertical line
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]
        # draw lines on right eyes
        # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
        # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

        # LEFT_EYE
        # horizontal line
        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]
        # vertical line
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]
        rhDistance = euclaideanDistance(rh_right, rh_left)
        rvDistance = euclaideanDistance(rv_top, rv_bottom)
        lvDistance = euclaideanDistance(lv_top, lv_bottom)
        lhDistance = euclaideanDistance(lh_right, lh_left)
        reRatio = rhDistance / rvDistance
        leRatio = lhDistance / lvDistance
        ratio = (reRatio + leRatio) / 2
        return ratio
    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # starting time here
        start_time = time.time()
        # starting Video loop here.

        while True:
            frame_counter += 1  # frame counter
            ret, frame = camera.read()  # getting frame from camera
            if not ret:
                break  # no more frames break
            #  resizing frame
            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                # utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

                if ratio > 4.5:
                    CEF_COUNTER += 1
                    # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                    utils.colorBackgroundText(frame, f'Drowsiness', FONTS, 1.7, (int(frame_height / 2), 100), 2,
                                              utils.YELLOW, pad_x=6, pad_y=6, )
                    flag +=1
                    #print(flag)
                    if(flag==30):
                        flag=0
                        print('alert')
                        import winsound
                        filename = 'alert.wav'
                        winsound.PlaySound(filename, winsound.SND_FILENAME)
                        cv.imwrite("alert.jpg", frame)
                        #sendmail()
                        print(session['mob'])
                        #sendmsg(session['mob'],session['dname'] +  " Driver Has Sleep")
                else:
                    flag = 0
                    if CEF_COUNTER > CLOSED_EYES_FRAME:
                        TOTAL_BLINKS += 1
                        CEF_COUNTER = 0
                # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                # utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)

                cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN,
                             1, cv.LINE_AA)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN,
                             1, cv.LINE_AA)
            # calculating  frame per seconds FPS
            end_time = time.time() - start_time
            fps = frame_counter / end_time

            frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9,
                                             textThickness=2)
            # writing image for thumbnail drawing shape
            # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
            cv.imshow('frame', frame)
            key = cv.waitKey(2)
            if key == ord('q') or key == ord('Q'):
                break
        cv.destroyAllWindows()
        camera.release()
def sendmail():
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    fromaddr = "testsam360@gmail.com"
    toaddr = session['email']
    # instance of MIMEMultipart
    msg = MIMEMultipart()
    # storing the senders email address
    msg['From'] = fromaddr
    # storing the receivers email address
    msg['To'] = toaddr
    # storing the subject
    msg['Subject'] = "Alert"
    # string to store the body of the mail
    body = "Drowsy Driver Detection"
    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))
    # open the file to be sent
    filename = "alert.jpg"
    attachment = open("alert.jpg", "rb")
    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')
    # To change the payload into encoded form
    p.set_payload((attachment).read())
    # encode into base64
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    # attach the instance 'p' to instance 'msg'
    msg.attach(p)
    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)
    # start TLS for security
    s.starttls()
    # Authentication
    s.login(fromaddr, "mailtest4")
    # Converts the Multipart msg into a string
    text = msg.as_string()
    # sending the mail
    s.sendmail(fromaddr, toaddr, text)
    # terminating the session
    s.quit()
def mail_send():
    mail = 'testsam360@gmail.com';
    password = 'rddwmbynfcbgpywf';
    # list of email_id to send the mail
    li = ["sundarsamcore@gmail.com"]
    body = "Emergency Alert... Help Required...!"
    filename = "test.py"
    yag = yagmail.SMTP(mail, password)
    for dest in li:
        yag.send(
            to=dest,
            subject="Emergency Alert...!",
            contents=body,
            attachments=filename,
        )
    print("Mail sent to all...!")
    time.sleep(1)
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
        			
