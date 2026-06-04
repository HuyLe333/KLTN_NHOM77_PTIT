import os
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804" # Thay doi mat khau MySQL cua may ban cho phu hop
DB_NAME = "kltn_stock_db"
BACKUP_DIR = "database_backup"

def import_database():
    print("=" * 60)
    print("  KHOI PHUC / NHAP DU LIEU (IMPORT) DATABASE")
    print("=" * 60)

    # 1. Kiem tra thu muc backup va file schema
    schema_file = os.path.join(BACKUP_DIR, "schema.sql")
    if not os.path.exists(schema_file):
        print(f"❌ Khong tim thay file schema tai: {schema_file}")
        print("Vui long coppy thu muc 'database_backup/' vao thu muc du an truoc.")
        return

    # 2. Ket noi MySQL de tao database va khoi tao bang (Schema)
    print("\n[1/3] Dang ket noi MySQL va chay file schema.sql de khoi tao cau truc...")
    try:
        # Ket noi khong database truoc de tao database neu chua co
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS
        )
        cursor = conn.cursor()
        
        # Doc va thuc thi schema.sql
        with open(schema_file, "r", encoding="utf-8") as f:
            sql_script = f.read()
        
        # Tach script thanh cac cau lenh don le bang dau cham phay ';'
        sql_commands = sql_script.split(";")
        for command in sql_commands:
            command = command.strip()
            if command and not command.startswith("--"):
                cursor.execute(command)
        
        conn.commit()
        print("  -> Khoi tao database va cau truc cac bang thanh cong!")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ That bai khi tao Schema: {e}")
        print("Vui long kiem tra MySQL server da khoi dong va dung thong tin mat khau chua.")
        return

    # 3. Import du lieu tu cac file CSV
    print("\n[2/3] Dang doc cac file CSV va import du lieu vao cac bang...")
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
    
    # Tim tat ca cac file CSV trong thu muc backup
    csv_files = [f for f in os.listdir(BACKUP_DIR) if f.endswith(".csv")]
    
    if not csv_files:
        print("⚠️ Canh bao: Khong tim thay file du lieu .csv nao de import.")
    else:
        for file in csv_files:
            table_name = file.replace(".csv", "")
            csv_path = os.path.join(BACKUP_DIR, file)
            
            try:
                print(f"  - Dang doc va import file {file} vao bang `{table_name}`...")
                df = pd.read_csv(csv_path, encoding="utf-8")
                
                # Ingest data dung sqlalchemy
                # Xoa du lieu cu neu co va thay the bang du lieu backup
                df.to_sql(
                    name=table_name,
                    con=engine,
                    if_exists="replace", # 'replace' se xoa bang cu va tao lai dung schema cua DF,
                                         # hoac 'append' neu chi muon ghi de dong.
                                         # Vi schema da duoc tao o buoc 1, ta can lam sach bang va chen du lieu vao.
                                         # Neu schema phuc tap co PRIMARY KEY thi phai thuc hien DELETE truoc roi append.
                    index=False,
                    chunksize=5000
                )
                print(f"    -> Thanh cong: Da import {len(df):,} dong.")
            except Exception as e:
                # Neu gap loi do 'replace' lam mat PRIMARY KEY, ta dung phuong phap xoa va chen
                try:
                    # Clear data
                    conn = mysql.connector.connect(
                        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME
                    )
                    cursor = conn.cursor()
                    cursor.execute(f"TRUNCATE TABLE {table_name};")
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    # Append lai
                    df.to_sql(name=table_name, con=engine, if_exists="append", index=False, chunksize=5000)
                    print(f"    -> Thanh cong (append): Da import {len(df):,} dong.")
                except Exception as ex:
                    print(f"    ❌ Loi khi import bang {table_name}: {ex}")

    print("\n[3/3] Hoan thanh qua trinh khoi phuc / import du lieu database!")
    print("=" * 60)

if __name__ == "__main__":
    import_database()
