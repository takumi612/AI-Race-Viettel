import os
import sys
import sqlite3
import re

# Thêm project root vào sys.path để hỗ trợ import src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.paths import DB_PATH

class ClinicalValidator:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.load_rules()

    def load_rules(self):
        """Tải toàn bộ quy tắc từ SQLite vào bộ nhớ RAM để tối ưu hóa tốc độ."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 1. Luật giới tính
        cursor.execute("SELECT code, allowed_sex FROM icd10_rules_sex;")
        self.sex_rules = {row[0]: row[1] for row in cursor.fetchall()}

        # 2. Luật độ tuổi
        cursor.execute("SELECT code, min_days, max_days, description FROM icd10_rules_age;")
        self.age_rules = {row[0]: (row[1], row[2], row[3]) for row in cursor.fetchall()}

        # 3. Luật mã kép
        cursor.execute("SELECT dagger_code, asterisk_code FROM icd10_rules_dual;")
        self.dual_rules = {}
        for dagger, asterisk in cursor.fetchall():
            if dagger not in self.dual_rules:
                self.dual_rules[dagger] = set()
            self.dual_rules[dagger].add(asterisk)

        # 4. Luật không được làm bệnh chính
        cursor.execute("SELECT code FROM icd10_rules_not_primary;")
        self.not_primary_rules = {row[0] for row in cursor.fetchall()}

        conn.close()
        print(f"ClinicalValidator initialized:")
        print(f"  - Loaded {len(self.sex_rules)} sex rules.")
        print(f"  - Loaded {len(self.age_rules)} age rules.")
        print(f"  - Loaded {len(self.dual_rules)} dual code mapping groups.")
        print(f"  - Loaded {len(self.not_primary_rules)} non-primary codes.")

    def validate_sex(self, code, patient_sex):
        """
        Kiểm tra xem mã bệnh có vi phạm giới tính của bệnh nhân hay không.
        patient_sex: 'M' (Nam), 'F' (Nữ) hoặc None (nếu không xác định).
        Trả về True nếu hợp lệ, False nếu vi phạm.
        """
        if not patient_sex or code not in self.sex_rules:
            return True
        return self.sex_rules[code] == patient_sex

    def validate_age(self, code, patient_age_days):
        """
        Kiểm tra xem mã bệnh có vi phạm tuổi của bệnh nhân (tính bằng ngày) hay không.
        patient_age_days: Số ngày tuổi của bệnh nhân (hoặc None nếu không xác định).
        """
        if patient_age_days is None or code not in self.age_rules:
            return True
        min_days, max_days, _ = self.age_rules[code]
        return min_days <= patient_age_days <= max_days

    def check_and_fix_candidates(self, entity, patient_info=None):
        """
        Kiểm tra danh sách candidates của một thực thể (dự đoán) và lọc bỏ các mã vi phạm.
        patient_info: dict, ví dụ: {'sex': 'M', 'age_days': 7200} (20 tuổi)
        """
        if not entity.get("candidates") or entity.get("type") not in ("CHẨN_DOÁN", "THUỐC"):
            return entity

        sex = patient_info.get("sex") if patient_info else None
        age_days = patient_info.get("age_days") if patient_info else None

        valid_candidates = []
        for code in entity["candidates"]:
            code_clean = str(code).strip().upper()
            
            # Kiểm tra giới tính và độ tuổi
            if not self.validate_sex(code_clean, sex):
                print(f"  [RULE VIOLATION] Candidate {code} rejected for patient sex {sex}")
                continue
            if not self.validate_age(code_clean, age_days):
                print(f"  [RULE VIOLATION] Candidate {code} rejected for patient age {age_days} days")
                continue
                
            valid_candidates.append(code)

        entity["candidates"] = valid_candidates
        return entity

    def check_dual_codes(self, entities):
        """
        Kiểm tra tính toàn vẹn của mã kép (Dagger/Asterisk).
        Nếu phát hiện mã Dagger trong danh sách chẩn đoán, kiểm tra xem
        đã có mã Asterisk tương ứng chưa. Nếu chưa có, đưa ra cảnh báo hoặc đề xuất bổ sung.
        """
        all_codes = set()
        for ent in entities:
            if ent.get("type") == "CHẨN_DOÁN" and ent.get("candidates"):
                for c in ent["candidates"]:
                    all_codes.add(str(c).strip().upper())

        for ent in entities:
            if ent.get("type") == "CHẨN_DOÁN" and ent.get("candidates"):
                for c in ent["candidates"]:
                    c_upper = str(c).strip().upper()
                    if c_upper in self.dual_rules:
                        required_asterisks = self.dual_rules[c_upper]
                        # Kiểm tra xem có mã asterisk nào có mặt trong CSDL chẩn đoán chưa
                        if not required_asterisks.intersection(all_codes):
                            print(f"  [CLINICAL WARNING] Dagger code {c_upper} found, but missing required Asterisk code: {required_asterisks}")
                            
        return entities

if __name__ == "__main__":
    # Test thử Validator
    validator = ClinicalValidator()
    
    # Test case 1: Uốn ván sơ sinh (A33) chỉ dùng cho trẻ < 1 tuổi (365 ngày)
    print("A33 check for 10yo:", validator.validate_age("A33", 3650)) # Kỳ vọng: False
    print("A33 check for 50 days baby:", validator.validate_age("A33", 50)) # Kỳ vọng: True
    
    # Test case 2: Uốn ván sản khoa (A34) chỉ dùng cho Nữ
    print("A34 check for Male:", validator.validate_sex("A34", "M")) # Kỳ vọng: False
    print("A34 check for Female:", validator.validate_sex("A34", "F")) # Kỳ vọng: True
