from flask import Flask, request, jsonify
import os
import pymysql
import pandas as pd
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
import warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'si211000',
    'database': 'graduation_project'
}


app = Flask(__name__)


@app.route('/userinfo', methods=['GET'])
def get_userinfo():
    db_connection = pymysql.connect(**db_config)
    cursor = db_connection.cursor(pymysql.cursors.DictCursor)
    user_email = request.args.get('email', '')
    query = "SELECT * FROM user_information WHERE email = %s"
    cursor.execute(query, (user_email,))
    result = cursor.fetchone()
    if result:
        return jsonify(result)
    else:
        return jsonify({'message': 'User not found'}), 404
    cursor.close()
    db_connection.close()
    return result


@app.route('/usertable', methods=['GET'])
def get_alluser():
    query = "SELECT * FROM user_information"
    db_connection = pymysql.connect(**db_config)
    cursor = db_connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    db_connection.close()
    if result:
        return jsonify(result)
    else:
        return jsonify({'message': 'No User'}), 404


@app.route('/spamtable', methods=['GET'])
def get_spamtable():
    db_connection = pymysql.connect(**db_config)
    cursor = db_connection.cursor(pymysql.cursors.DictCursor)
    query = "SELECT * FROM spam_caught"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    db_connection.close()
    if result:
        return jsonify(result)
    else:
        return jsonify({'message': 'No spam caught'}), 404


@app.route('/spamcheck', methods=['GET'])
def process_string():
    db_connection = pymysql.connect(**db_config)
    cursor = db_connection.cursor(pymysql.cursors.DictCursor)
    input_string = request.args.get('input_string', '')
    input_string = input_string.replace(',', '')
    phonenum = request.args.get('phonenum', '')
    email = request.args.get('email', '')
    temp = []
    temp.append(input_string)
    df = pd.DataFrame(temp, columns=["text"])
    df.to_csv("./test_data.csv", index=True, index_label="id")
    text = []
    with open('./test_data.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for i in reader:
            if i[1] == "text":
                continue
            text.append(i[1])

    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            return text
    test_dataset = TextDataset(text)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = torch.load("./model.pt", map_location=torch.device('cpu'))
    model.eval()

    with torch.no_grad():
        for x in test_data_loader:
            x = model.embed_texts(x[0])
            x = torch.tensor(x)
            result = model(x)
            result = float(result)

    yes = round(result*100, 2)
    no = round(100-yes, 2)

    yes += 30
    no -= 30
    yes = round(yes, 2)
    no = round(no, 2)
    ############################# new_word_method ################################
    normal_neo = pd.read_csv('normal_neo.csv')
    normal_neo = list(normal_neo)
    normal_detect = ''
    normal_cnt = 1
    spam_neo = pd.read_csv('spam_neo.csv')
    spam_neo = list(spam_neo)
    spam_detect = ''
    spam_cnt = 1
    input_word = input_string.split(' ')
    for word in input_word:
        if word in normal_neo:
            normal_cnt += 1
            normal_detect += word
            normal_detect += ', '
        if word in spam_neo:
            spam_cnt += 1
            spam_detect += word
            spam_detect += ', '

    yes = yes*spam_cnt
    no = no*normal_cnt
    Y = round(yes*100/(yes+no), 2)
    N = round(no*100/(yes+no), 2)
    decision = 'spam'
    if (Y <= 45):
        decision = 'normal'
    elif (Y <= 60):
        decision = 'suspected'
    elif (Y <= 75):
        decision = 'expected'
    else:
        decision = 'confirmed'
    # 스팸이라면 그 회원이 잡은 스팸건수 업데이트
    if (decision == 'suspected' or decision == 'expected' or decision == 'confirmed'):
        query = "UPDATE user_information SET spamcaught=spamcaught+1 WHERE email = %s"
        cursor.execute(query, (email,))
        db_connection.commit()

    # 번호별 스팸 건수 테이블 업데이트
    if decision == 'normal':
        normal = 1
    else:
        normal = 0

    if decision == 'suspected':
        suspected = 1
    else:
        suspected = 0
    if decision == 'expected':
        expected = 1
    else:
        expected = 0
    if decision == 'confirmed':
        confirmed = 1
    else:
        confirmed = 0

    query = ''' 
        INSERT INTO spam_caught (phonenum, suspected, expected, confirmed, normal)
        VALUES (%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            suspected = CASE WHEN %s = 'suspected' THEN suspected + 1 ELSE suspected END,
            expected = CASE WHEN %s = 'expected' THEN expected +1 ELSE expected END,
            confirmed = CASE WHEN %s = 'confirmed' THEN confirmed +1 ELSE confirmed END,
            normal = CASE WHEN %s = 'normal' THEN normal + 1 ELSE normal END;
    '''
    cursor.execute(query, (phonenum, suspected, expected, confirmed,
                   normal, decision, decision, decision, decision))
    db_connection.commit()
    ################################### print result ##################################
    result = {'spam': yes, 'notspam': no, 'neo_spam': Y,
              'neo_notspam': N, 'decision': decision}
    cursor.close()
    db_connection.close()
    return result


@app.route('/register', methods=['POST'])
def register():
    db_connection = pymysql.connect(**db_config)
    cursor = db_connection.cursor(pymysql.cursors.DictCursor)
    data = request.get_json()
    email2 = data['email']
    password2 = data['password']
    name2 = data['name']
    phonenum2 = data['phonenum']

    cursor.execute(
        "SELECT * FROM user_information WHERE email = %s", (email2,))
    existing_member = cursor.fetchone()
    if existing_member:
        return ({'message': 'Email address is already Registered'}), 400

    cursor.execute("INSERT INTO user_information (email, password, name, phonenum) VALUES (%s, %s, %s, %s)",
                   (email2, password2, name2, phonenum2))
    db_connection.commit()
    cursor.close()
    db_connection.close()
    return jsonify({'message': 'registeration successful'}), 201


@app.route('/login', methods=['POST'])
def login():
    db_connection = pymysql.connect(**db_config)
    cursor = db_connection.cursor(pymysql.cursors.DictCursor)
    data = request.get_json()
    email3 = data['email']
    password3 = data['password']
    cursor.execute(
        "SELECT * FROM user_information WHERE email = %s AND password = %s", (email3, password3))
    user = cursor.fetchone()
    cursor.close()
    db_connection.close()
    if user:
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'message': 'Login failed. Invalid credentials'}), 401


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
