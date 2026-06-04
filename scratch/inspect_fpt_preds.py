import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)
cursor = conn.cursor()
cursor.execute("""
    SELECT date, prediction, probability_up, predict_date 
    FROM model_predictions 
    WHERE ticker = 'FPT' 
    ORDER BY date DESC LIMIT 15
""")
rows = cursor.fetchall()
print("DATE | PREDICT | PROB_UP | TARGET_DATE")
for r in rows:
    print(r)
cursor.close()
conn.close()
