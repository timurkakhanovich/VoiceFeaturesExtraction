from fourier import mfcc_alg
import librosa as ls
import numpy as np
import pymysql as sql
import sys

def get_password():
    with open(sys.path[0] + r'\password.txt') as fin:
        return fin.readline()

def get_connection():
    password = get_password()

    return sql.connect('localhost', 'root', password, 'voiceDB', charset='utf8mb4', cursorclass=sql.cursors.DictCursor)

def insertData(conn, name, coefs):
    with conn.cursor() as cursor:
        cmd = 'SELECT * FROM voice_data WHERE `name`=%s'
        cursor.execute(cmd, (name,))

        assert cursor.fetchone() is None, 'Try the other username!'

    with conn.cursor() as cursor:
        cmd = 'INSERT INTO voice_data (`name`, coefs) VALUES (%s, %s)'
        cursor.execute(cmd, (name, coefs))

        conn.commit()

def deleteData(conn, name):
    with conn.cursor() as cursor:
        cmd = 'DELETE FROM voice_data WHERE `name`=%s'
        cursor.execute(cmd, (name,))

        conn.commit()

def selectCoefs(conn, name):
    with conn.cursor() as cursor:
        cmd = 'SELECT coefs FROM voice_data WHERE `name`=%s'
        cursor.execute(cmd, (name,))

        result = cursor.fetchone()

        assert result is not None, 'Data belonging {} hasn\'t been found!'.format(name)

        return np.array(eval(result['coefs']))

def selectAllCoefs(conn):
    with conn.cursor() as cursor:
        cmd = 'SELECT * FROM voice_data'
        cursor.execute(cmd)

        result = cursor.fetchall()

        return result

def callInsert(name, coefs):
    connection = get_connection()
    insertData(connection, name, coefs)
    connection.close()
    
def callDelete(name):
    connection = get_connection()
    deleteData(connection, name)
    connection.close()

def callSelect(name):
    connection = get_connection()
    coefs = selectCoefs(connection, name)
    connection.close()

    return coefs

def callSelectAll():
    connection = get_connection()
    result = selectAllCoefs(connection)
    connection.close()

    return result

def main():
    speaker1, _ = ls.load(sys.path[0] + r'\Data\test11.wav', sr=16000)
    speaker2, _ = ls.load(sys.path[0] + r'\Data\test21.wav', sr=16000)

    #callInsert('Temur low', str(mfcc_alg(speaker1)))
    #callInsert('Temur high', str(mfcc_alg(speaker2)))

    #print(callSelect('Temur'))
    print(callSelectAll()[3])
    #print(selectCoefs(connection, 'Another'))    

if __name__ == "__main__":
    main()
