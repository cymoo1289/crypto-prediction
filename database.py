import sqlite3
#import streamlit_authenticator as stauth 



class SqliteDB():
    def __init__(self, db_path = 'crypto_app.db'):
        self.db_path = db_path

    def create_creds_table(self):
        sql = """CREATE TABLE IF NOT EXISTS creds(
        yaml_path VARCHAR NULL
        )"""
        self.execute_sql(sql,())

    def insert_creds_table(self, yaml_path):
        params = (yaml_path,)
        print(yaml_path)
        sql="""INSERT INTO creds VALUES (?)"""
        self.execute_sql(sql,params)

    def get_creds(self):
        sql = """SELECT * FROM creds"""
        user = self.execute_sql(sql,())
        return user 

    def create_feedback_table(self):
        sql = """CREATE TABLE IF NOT EXISTS feedback(
        feedback_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        feedback_text CHAR NOT NULL
        )"""
        self.execute_sql(sql,())

    def insert_feedback(self,feedback_text):
        rows = self.get_feedback()
        sql="""INSERT INTO feedback VALUES (?,?)"""
        params = (len(rows) , feedback_text)
        self.execute_sql(sql,params)

    def get_feedback(self):
        sql = """SELECT * FROM feedback"""
        feedback = self.execute_sql(sql,())
        return feedback 

    def execute_sql(self,sql,params):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if len(params) == 0:
            cursor.execute(sql)
        else:
            cursor.execute(sql,params)
        conn.commit()
        return cursor.fetchall()
if __name__ == "__main__":
    db = SqliteDB()
    db.create_creds_table()
    db.insert_creds_table('config.yaml')
    creds = db.get_creds()
    print(creds)
    db.create_feedback_table()
    db.insert_feedback("hi")
    feedbacks = db.get_feedback()
    print(feedbacks)

