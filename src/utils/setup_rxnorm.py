import os
import sys
import sqlite3
import zipfile

# Thêm project root vào sys.path để hỗ trợ import src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.paths import DB_PATH, RXNORM_TXT_PATH

ZIP_PATH = r"D:\AI Race Viettel\data\kb\RxNorm_full_07062026.zip"

def setup_rxnorm():
    if not os.path.exists(ZIP_PATH):
        print(f"[ERROR] RxNorm zip file not found at: {ZIP_PATH}")
        return False

    print("=== STARTING RXNORM DATABASE SETUP ===")
    print(f"Reading zip file: {ZIP_PATH}...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Tạo các bảng mới trong SQLite
    print("Creating RxNorm tables...")
    cursor.execute("DROP TABLE IF EXISTS rxnorm;")
    cursor.execute("""
    CREATE TABLE rxnorm (
        rxcui TEXT,
        name TEXT,
        tty TEXT
    );
    """)

    cursor.execute("DROP TABLE IF EXISTS rxnorm_mapping;")
    cursor.execute("""
    CREATE TABLE rxnorm_mapping (
        old_cui TEXT,
        new_cui TEXT,
        PRIMARY KEY (old_cui, new_cui)
    );
    """)
    conn.commit()

    # 2. Đọc và chèn dữ liệu RXNCONSO.RRF (từ vựng thuốc)
    print("Parsing rrf/RXNCONSO.RRF from zip...")
    conso_records = []
    unique_conso = set()
    conso_count = 0
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        with zip_ref.open("rrf/RXNCONSO.RRF") as f:
            for line in f:
                line_str = line.decode('utf-8')
                parts = line_str.split('|')
                if len(parts) > 16:
                    sab = parts[11]
                    suppress = parts[16]
                    
                    # Áp dụng bộ lọc tối ưu: SAB='RXNORM' và không bị ẩn hoàn toàn (SUPPRESS != 'Y')
                    if sab == "RXNORM" and suppress != "Y":
                        rxcui = parts[0].strip()
                        name = parts[14].strip()
                        tty = parts[12].strip()
                        
                        record_key = (rxcui, name, tty)
                        if record_key not in unique_conso:
                            unique_conso.add(record_key)
                            conso_records.append(record_key)
                            
                        if len(conso_records) >= 20000:
                            cursor.executemany("INSERT INTO rxnorm VALUES (?, ?, ?);", conso_records)
                            conn.commit()
                            conso_count += len(conso_records)
                            conso_records = []
                            print(f"  Inserted {conso_count} records into rxnorm...")

            if conso_records:
                cursor.executemany("INSERT INTO rxnorm VALUES (?, ?, ?);", conso_records)
                conn.commit()
                conso_count += len(conso_records)
                print(f"  Inserted total {conso_count} records into rxnorm.")

    # Tạo index cho bảng rxnorm
    print("Creating indexes on rxnorm table...")
    cursor.execute("CREATE INDEX idx_rxnorm_rxcui ON rxnorm (rxcui);")
    cursor.execute("CREATE INDEX idx_rxnorm_name ON rxnorm (name);")
    conn.commit()

    # 3. Đọc và chèn dữ liệu RXNATOMARCHIVE.RRF (ánh xạ lịch sử mã đầy đủ)
    print("Parsing rrf/RXNATOMARCHIVE.RRF from zip...")
    mapping_records = []
    unique_mapping = set()
    mapping_count = 0

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        if "rrf/RXNATOMARCHIVE.RRF" in zip_ref.namelist():
            with zip_ref.open("rrf/RXNATOMARCHIVE.RRF") as f:
                for line in f:
                    line_str = line.decode('utf-8')
                    parts = line_str.split('|')
                    if len(parts) > 13:
                        old_cui = parts[0].strip()   # RXCUI cũ (đã archive)
                        new_cui = parts[6].strip()   # RXCUI hiện hành nó trỏ tới
                        sab = parts[13].strip()
                        
                        # Lọc theo nguồn RXNORM và chỉ lấy các ánh xạ thay đổi mã thực sự
                        if sab == "RXNORM" and old_cui and new_cui and old_cui != new_cui:
                            record_key = (old_cui, new_cui)
                            if record_key not in unique_mapping:
                                unique_mapping.add(record_key)
                                mapping_records.append(record_key)
                                
                            if len(mapping_records) >= 10000:
                                cursor.executemany("INSERT OR IGNORE INTO rxnorm_mapping VALUES (?, ?);", mapping_records)
                                conn.commit()
                                mapping_count += len(mapping_records)
                                mapping_records = []
                                print(f"  Inserted {mapping_count} records into rxnorm_mapping...")

                if mapping_records:
                    cursor.executemany("INSERT OR IGNORE INTO rxnorm_mapping VALUES (?, ?);", mapping_records)
                    conn.commit()
                    mapping_count += len(mapping_records)
                    print(f"  Inserted total {mapping_count} records into rxnorm_mapping.")
        else:
            print("  [WARNING] rrf/RXNATOMARCHIVE.RRF not found in zip.")

    conn.close()

    # 4. Sinh file context rxnorm_context.txt
    print(f"Generating flat context file: {RXNORM_TXT_PATH}...")
    try:
        # Nhóm các tên theo rxcui để ghi ra file context gọn gàng
        cui_names = {}
        for rxcui, name, _ in unique_conso:
            if rxcui not in cui_names:
                cui_names[rxcui] = []
            cui_names[rxcui].append(name)

        with open(RXNORM_TXT_PATH, "w", encoding="utf-8") as f:
            f.write("=== DANH MỤC MÃ THUỐC RXNORM ===\n\n")
            for rxcui in sorted(cui_names.keys()):
                names_str = " | ".join(cui_names[rxcui])
                f.write(f"[{rxcui}] {names_str}\n")
        print("Flat context file generated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to generate context file: {e}")

    print("=== RXNORM SETUP COMPLETED SUCCESSFULLY ===")
    return True

if __name__ == "__main__":
    setup_rxnorm()
