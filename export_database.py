import os
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"
BACKUP_DIR = "database_backup"

def export_database():
    print("=" * 60)
    print("  EXPORT DU LIEU DATABASE & SCHEMA")
    print("=" * 60)

    # Tao thu muc backup neu chua ton tai
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"-> Da tao thu muc luu tru: {BACKUP_DIR}/")

    # 1. Connect to MySQL to export Schema (DDL)
    print("\n[1/3] Dang ket noi MySQL de xuat cau truc bang (Schema)...")
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cursor = conn.cursor()
    except Exception as e:
        print(f"X Connection that bai: {e}")
        print("Vui long kiem tra MySQL server da khoi dong va dung thong tin cau hinh chua.")
        return

    # Lay danh sach cac bang
    cursor.execute("SHOW TABLES;")
    tables = [row[0] for row in cursor.fetchall()]
    
    schema_file = os.path.join(BACKUP_DIR, "schema.sql")
    with open(schema_file, "w", encoding="utf-8") as f:
        f.write(f"-- SQL Schema Dump\n-- Database: {DB_NAME}\n-- Generated on: {pd.Timestamp.now()}\n\n")
        f.write("CREATE DATABASE IF NOT EXISTS `kltn_stock_db` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;\n")
        f.write("USE `kltn_stock_db`;\n\n")
        
        for table in tables:
            cursor.execute(f"SHOW CREATE TABLE {table};")
            create_stmt = cursor.fetchone()[1]
            f.write(f"-- Table structure for table `{table}`\n")
            f.write(f"DROP TABLE IF EXISTS `{table}`;\n")
            f.write(create_stmt + ";\n\n")
            
    print(f"-> Da xuat cau truc bang thanh cong ra file: {schema_file}")
    cursor.close()
    conn.close()

    # 2. Export du lieu cua tung bang ra file CSV
    print("\n[2/3] Dang xuat du lieu tu cac bang ra file CSV...")
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
    
    for table in tables:
        try:
            print(f"  - Dang doc du lieu bang `{table}`...")
            # Doc toan bo bang bang pandas
            df = pd.read_sql_table(table, con=engine)
            
            csv_path = os.path.join(BACKUP_DIR, f"{table}.csv")
            # Luu ra file CSV (nam trong thu muc database_backup va se duoc Git ignore)
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"    -> Da luu {len(df):,} dong vao: {csv_path}")
        except Exception as e:
            print(f"    X Loi khi xuat bang {table}: {e}")

    print("\n[3/3] Hoan thanh xuat du lieu!")
    print(f"Luu y: Cac file .csv trong '{BACKUP_DIR}/' se tu dong bi bo qua (ignored) boi Git de tranh lam phinh repo.")
    print("Chi file 'schema.sql' duoc chuan bi de push len Git phuc vu viec tai tao cau truc database.")
    print("=" * 60)

if __name__ == "__main__":
    export_database()
