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

        # 5. Luật ánh xạ RxNorm lịch sử (nếu có bảng rxnorm_mapping)
        self.rxnorm_mapping = {}
        try:
            cursor.execute("SELECT old_cui, new_cui FROM rxnorm_mapping;")
            for old_cui, new_cui in cursor.fetchall():
                if new_cui not in self.rxnorm_mapping:
                    self.rxnorm_mapping[new_cui] = []
                self.rxnorm_mapping[new_cui].append(old_cui)
        except Exception as e:
            # Nếu chưa chạy setup_rxnorm.py thì bảng này chưa có, bỏ qua không lỗi
            pass

        conn.close()
        print(f"ClinicalValidator initialized:")
        print(f"  - Loaded {len(self.sex_rules)} sex rules.")
        print(f"  - Loaded {len(self.age_rules)} age rules.")
        print(f"  - Loaded {len(self.dual_rules)} dual code mapping groups.")
        print(f"  - Loaded {len(self.not_primary_rules)} non-primary codes.")
        print(f"  - Loaded {len(self.rxnorm_mapping)} RxNorm historical CUI mappings.")

    def get_historical_cuis(self, rxcui):
        """
        Trả về danh sách các mã RxCUI lịch sử (mã cũ) tương ứng với mã hiện tại.
        """
        return self.rxnorm_mapping.get(str(rxcui).strip(), [])

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

    def get_rxnorm_name(self, rxcui):
        """Truy vấn tên của RxCUI từ SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM rxnorm WHERE rxcui = ? LIMIT 1;", (str(rxcui).strip(),))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else ""

    def get_ingredients(self, rxnorm_name):
        """Truy vấn mã IN và PIN từ tên thuốc."""
        if not rxnorm_name:
            return []
        ingredients = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tìm mã IN
            cursor.execute("""
                SELECT rxcui FROM rxnorm 
                WHERE tty = 'IN' AND ? LIKE name || '%'
                ORDER BY length(name) DESC LIMIT 1;
            """, (rxnorm_name.lower(),))
            row_in = cursor.fetchone()
            if row_in:
                ingredients.append(row_in[0])
                
            # Tìm mã PIN
            cursor.execute("""
                SELECT rxcui FROM rxnorm 
                WHERE tty = 'PIN' AND ? LIKE name || '%'
                ORDER BY length(name) DESC LIMIT 1;
            """, (rxnorm_name.lower(),))
            row_pin = cursor.fetchone()
            if row_pin:
                ingredients.append(row_pin[0])
                
            conn.close()
        except Exception:
            pass
        return ingredients

    def validate_dose_form(self, drug_text, rxcui_name):
        """
        Kiểm tra xem tên dạng bào chế trong rxcui_name (tiếng Anh) có mâu thuẫn 
        với từ chỉ dạng dùng/dạng bào chế tiếng Việt trong drug_text hay không.
        Trả về True nếu không có mâu thuẫn (hợp lệ), False nếu mâu thuẫn rõ ràng.
        """
        if not drug_text or not rxcui_name:
            return True
            
        drug_text_lower = drug_text.lower()
        rxcui_name_lower = rxcui_name.lower()
        
        # 1. Định nghĩa các nhóm từ khóa dạng dùng tiếng Việt và dạng bào chế tiếng Anh tương ứng
        # Nhóm Oral (uống)
        oral_keywords_vi = ["uống", "viên", "gói", "capsule", "tablet", "nén", "nang"]
        oral_keywords_en = ["oral", "tablet", "capsule", "packet", "pill", "lozenge"]
        
        # Nhóm Injectable (tiêm)
        inj_keywords_vi = ["tiêm", "truyền", "ống", "tĩnh mạch", "bắp", "lọ tiêm", "dịch truyền"]
        inj_keywords_en = ["injectable", "injection", "infusion", "intravenous"]
        
        # Nhóm Topical (bôi/ngoài da)
        topical_keywords_vi = ["bôi", "thoa", "kem", "mỡ", "gel", "ngoài da", "xịt da"]
        topical_keywords_en = ["topical", "cream", "ointment", "gel", "spray", "patch"]

        # Nhóm Inhalant (hít/xịt xông)
        inhalant_keywords_vi = ["hít", "xịt mũi", "xịt họng", "xông", "khí dung"]
        inhalant_keywords_en = ["inhalant", "inhalation", "nasal spray", "aerosol"]

        # 2. Phát hiện dạng dùng trong văn bản tiếng Việt
        has_oral_vi = any(kw in drug_text_lower for kw in oral_keywords_vi)
        has_inj_vi = any(kw in drug_text_lower for kw in inj_keywords_vi)
        has_topical_vi = any(kw in drug_text_lower for kw in topical_keywords_vi)
        has_inhalant_vi = any(kw in drug_text_lower for kw in inhalant_keywords_vi)
        
        # 3. Phát hiện dạng bào chế của RxCUI tiếng Anh
        has_oral_en = any(kw in rxcui_name_lower for kw in oral_keywords_en)
        has_inj_en = any(kw in rxcui_name_lower for kw in inj_keywords_en)
        has_topical_en = any(kw in rxcui_name_lower for kw in topical_keywords_en)
        has_inhalant_en = any(kw in rxcui_name_lower for kw in inhalant_keywords_en)
        
        # 4. Kiểm duyệt mâu thuẫn chéo (Cross-contradiction checking)
        # Bác sĩ kê dạng uống (viên, uống) nhưng mã thuốc lại là dạng tiêm hoặc bôi ngoài da
        if has_oral_vi and not has_oral_en:
            if has_inj_en or has_topical_en or has_inhalant_en:
                return False
                
        # Bác sĩ kê dạng tiêm (ống, tiêm) nhưng mã thuốc lại là dạng uống hoặc bôi ngoài da
        if has_inj_vi and not has_inj_en:
            if has_oral_en or has_topical_en:
                return False
                
        # Bác sĩ kê dạng bôi ngoài da nhưng mã thuốc lại là dạng uống hoặc tiêm
        if has_topical_vi and not has_topical_en:
            if has_oral_en or has_inj_en:
                return False

        # Bác sĩ kê dạng hít/xịt nhưng mã thuốc lại là dạng uống hoặc tiêm
        if has_inhalant_vi and not has_inhalant_en:
            if has_oral_en or has_inj_en:
                return False
                
        return True

    def check_and_fix_candidates(self, entity, patient_info=None):
        """
        Kiểm tra danh sách candidates của một thực thể (dự đoán) và lọc bỏ các mã vi phạm.
        patient_info: dict, ví dụ: {'sex': 'M', 'age_days': 7200} (20 tuổi)
        """
        if not entity.get("candidates") or entity.get("type") not in ("CHẨN_ĐOÁN", "THUỐC"):
            return entity

        sex = patient_info.get("sex") if patient_info else None
        age_days = patient_info.get("age_days") if patient_info else None

        valid_candidates = []
        for code in entity["candidates"]:
            code_clean = str(code).strip().upper()
            
            # Nếu là chẩn đoán: Kiểm tra giới tính và độ tuổi
            if entity["type"] == "CHẨN_ĐOÁN":
                if not self.validate_sex(code_clean, sex):
                    print(f"  [RULE VIOLATION] Candidate {code} rejected for patient sex {sex}")
                    continue
                if not self.validate_age(code_clean, age_days):
                    print(f"  [RULE VIOLATION] Candidate {code} rejected for patient age {age_days} days")
                    continue
            
            # Nếu là thuốc: Kiểm tra dạng bào chế lâm sàng
            elif entity["type"] == "THUỐC":
                rxcui_name = self.get_rxnorm_name(code_clean)
                if rxcui_name:
                    if not self.validate_dose_form(entity.get("text", ""), rxcui_name):
                        print(f"  [CLINICAL VIOLATION] RxNorm Candidate {code} rejected due to dose form contradiction")
                        continue
                
            valid_candidates.append(code)

        entity["candidates"] = valid_candidates
        return entity

    def check_dual_codes(self, entities):
        """
        Kiểm tra tính toàn vẹn của mã kép (Dagger/Asterisk).
        Nếu phát hiện mã Dagger trong danh sách chẩn đoán, kiểm tra xem
        đã có mã Asterisk tương ứng chưa. Nếu chưa có, tự động bổ sung mã Asterisk
        đầu tiên tìm thấy vào danh sách dưới dạng một thực thể chẩn đoán phụ trùng tọa độ.
        """
        all_codes = set()
        for ent in entities:
            if ent.get("type") == "CHẨN_ĐOÁN" and ent.get("candidates"):
                for c in ent["candidates"]:
                    all_codes.add(str(c).strip().upper())

        additional_entities = []
        for ent in entities:
            if ent.get("type") == "CHẨN_ĐOÁN" and ent.get("candidates"):
                for c in ent["candidates"]:
                    c_upper = str(c).strip().upper()
                    if c_upper in self.dual_rules:
                        required_asterisks = self.dual_rules[c_upper]
                        # Kiểm tra xem có mã asterisk nào có mặt trong danh sách chẩn đoán chưa
                        if not required_asterisks.intersection(all_codes):
                            # Lấy mã Asterisk đầu tiên để bổ sung
                            best_asterisk = sorted(list(required_asterisks))[0]
                            asterisk_name = ""
                            try:
                                conn = sqlite3.connect(self.db_path)
                                cursor = conn.cursor()
                                cursor.execute("SELECT name_vi FROM icd10 WHERE code = ? LIMIT 1;", (best_asterisk,))
                                row = cursor.fetchone()
                                conn.close()
                                asterisk_name = row[0] if row else f"Biểu hiện lâm sàng của {ent.get('text')}"
                            except Exception:
                                asterisk_name = f"Biểu hiện lâm sàng của {ent.get('text')}"
                                
                            print(f"  [CLINICAL UPDATE] Automatically adding required Asterisk code {best_asterisk} for Dagger {c_upper}")
                            
                            # Tạo thực thể mới trùng tọa độ với thực thể Dagger gốc
                            new_ent = {
                                "text": asterisk_name,
                                "type": "CHẨN_ĐOÁN",
                                "position": ent.get("position"),
                                "assertions": ent.get("assertions", []),
                                "candidates": [best_asterisk]
                            }
                            additional_entities.append(new_ent)
                            all_codes.add(best_asterisk)
                            
        return entities + additional_entities

if __name__ == "__main__":
    # Test thử Validator
    validator = ClinicalValidator()
    
    # Test case 1: Uốn ván sơ sinh (A33) chỉ dùng cho trẻ < 1 tuổi (365 ngày)
    print("A33 check for 10yo:", validator.validate_age("A33", 3650)) # Kỳ vọng: False
    print("A33 check for 50 days baby:", validator.validate_age("A33", 50)) # Kỳ vọng: True
    
    # Test case 2: Uốn ván sản khoa (A34) chỉ dùng cho Nữ
    print("A34 check for Male:", validator.validate_sex("A34", "M")) # Kỳ vọng: False
    print("A34 check for Female:", validator.validate_sex("A34", "F")) # Kỳ vọng: True
