import mysql.connector
import sqlalchemy
import pandas as pd

# print("mysql-connector-python version:", mysql.connector.__version__)
# print("SQLAlchemy version:", sqlalchemy.__version__)
# print("pandas version:", pd.__version__)
# print("All libraries installed and imported successfully!")

mydb = mysql.connector.connect(
  host="localhost",
  port = "3306",
  user="root",
  password="admin"
)

mycursor = mydb.cursor()

mycursor.execute("select * from finance.accounting_dataset LIMIT 10;")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)