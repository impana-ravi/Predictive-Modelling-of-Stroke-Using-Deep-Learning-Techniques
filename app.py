from flask import Flask, render_template, request, redirect, url_for, flash, session
import mysql.connector as mq
from mysql.connector import Error
from markupsafe import Markup
import os
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
model=load_model("best_model.h5")
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

import smtplib
from email.mime.text import MIMEText

def send_email(result_class, probability, receiver):
    sender_email = "smonisha913@gmail.com"
    sender_password = "tmuxwhqojcgbdflx"
    receiver_email = receiver

    # -------- Different Plans --------
    if result_class == "ISCHEMIC":
        diet = """
- High fiber foods (oats, brown rice)
- Fruits and vegetables
- Low-fat dairy
- Avoid oily food
"""
        exercise = """
- Walking 30 mins daily
- Light stretching
- Breathing exercises
"""

    elif result_class == "HAEMORRHAGIC":
        diet = """
- Low salt diet
- Fresh fruits & vegetables
- Whole grains
- Avoid alcohol & caffeine
"""
        exercise = """
- Light walking only
- Physiotherapy
- Avoid heavy workouts
"""

    # -------- Email Content --------
    body = f"""
Prediction Result: {result_class}
Confidence: {probability}

DIET PLAN:
{diet}

EXERCISE PLAN:
{exercise}
"""

    msg = MIMEText(body)
    msg['Subject'] = "Health Alert - Stroke Prediction"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print("Email Error:", e)

def dbconnection():
    con = mq.connect(host='localhost', database='mydb',user='root',password='root')
    return con

@app.route('/')
def home():
    return render_template('index.html', title='Login')


@app.route('/userloginpage')
def userloginpage():
    return render_template('userlogin.html', title='Login')

@app.route('/userregisterpage')
def userregisterpage():
    return render_template('userreg.html', title='reg')


@app.route('/uploadimagepage')
def uploadimagepage():
    return render_template('uploadimage.html', title='upload image')


       
@app.route('/userregister', methods=['GET', 'POST'])
def userregister():
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        address = request.form['address']
        password = request.form['password']
        
        con = dbconnection()
        cursor = con.cursor()
        cursor.execute("select * from user where email='{}' or phone='{}'".format(email,phone))
        res = cursor.fetchall()
        if res==[]:
                cursor.execute("insert into user(name,phone,email,address,password)values('{}','{}','{}','{}','{}')".format(name,phone,email,address,password))
                con.commit()
                con.close()
                message = Markup("<h3> User Registration Success!</h3>")
                flash(message)
                return redirect(url_for('userloginpage'))
        else:
           message = Markup("<h3>Failed! Email Id or phone number already Exist</h3>")
           flash(message)
           return redirect(url_for('userregisterpage'))
       

       

            
@app.route('/userlogin', methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        con = dbconnection()
        cursor = con.cursor()
        cursor.execute("select * from user where email='{}' and password='{}'".format(email,password))
        res = cursor.fetchall()
        if res==[]:
                message = Markup("<h3>Failed! Invalid Email or Password</h3>")
                flash(message)
                return redirect(url_for('userloginpage'))
        else:
            session["email"]=email
            return render_template('uploadimage.html')
        

def preprocess(image_path):
    test_image = load_img(image_path, target_size = (224,224))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255
    return test_image



@app.route('/uploadimage', methods=['GET','POST'])
def uploadimage():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
            # Save the uploaded file to a specific folder
            filename = 'static/uploads/' + uploaded_file.filename
            onlyfilename=uploaded_file.filename
            print(filename)
            uploaded_file.save(filename)
            dic={0:'HAEMORRHAGIC',1:'ISCHEMIC',2:'NORMAL'
                 }
            test_image= preprocess(filename)
            result = model.predict(test_image)
            probabilities = result[0]
            result = np.argmax(result)
            predicted_class_probability = probabilities[result] * 100
            predicted_class_probability= f"{predicted_class_probability:.2f}%"
            print(np.argmax(result))
            
            detec=dic[result]
            if detec !="NORMAL":
                 
                send_email(detec,predicted_class_probability,session["email"])
            print(detec)
            return render_template('uploadimage.html',result=detec+" "+predicted_class_probability, cl=detec)
    return redirect(url_for('uploadimagepage'))

if __name__ == '__main__':
    app.run(debug=True)
