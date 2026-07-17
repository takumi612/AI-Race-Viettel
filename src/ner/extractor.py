import re
import sqlite3
import os
from src.utils.paths import DB_PATH

# Danh sách triệu chứng lâm sàng toàn diện được xây dựng từ chuẩn y tế lâm sàng
SYMPTOMS = [
    "khó thở", "ho", "ho đờm", "ho ra máu", "ho đờm xanh", "ho khan",
    "đờm", "tiếng rít", "khò khè", "tức ngực", "thở khò khè",
    "đánh trống ngực", "đau ngực", "hồi hộp", "tím tái",
    "buồn nôn", "nôn", "nôn ói", "nôn mửa", "nôn ra máu",
    "đau bụng", "chướng bụng", "ợ hơi", "ợ chua", "ợ nóng",
    "tiêu chảy", "táo bón", "đi ngoài ra máu", "phân đen",
    "đau thượng vị", "đau hạ vị",
    "đau đầu", "chóng mặt", "choáng váng", "ngất", "ngất xỉu",
    "co giật", "tê bì", "yếu liệt", "liệt", "mất ý thức",
    "run", "mất trí nhớ", "lú lẫn", "mất phương hướng",
    "đau lưng", "đau khớp", "đau cơ", "sưng khớp",
    "đau chân", "đau vai", "đau cổ", "cứng khớp",
    "đau hông", "đau đầu gối",
    "sốt", "mệt mỏi", "sụt cân", "tăng cân", "ớn lạnh",
    "đổ mồ hôi", "vã mồ hôi", "mất ngủ", "chán ăn",
    "suy nhược", "kiệt sức",
    "tiểu khó", "tiểu buốt", "tiểu máu", "tiểu ra máu",
    "tiểu đêm", "tiểu gắt",
    "phát ban", "ngứa", "mẩn đỏ", "sưng", "phù",
    "phù mắt cá chân", "phù chân",
    "mờ mắt", "nhìn đôi", "ù tai", "giảm thính lực",
    "khó nuốt", "đau họng", "sổ mũi", "nghẹt mũi",
    "hoa mắt", "xỉu", "khó chịu",
    "giảm dung nạp gắng sức", "mất cảm giác",
    "đau khi ấn xoang",
    "cảm giác thắt chặt ngực vùng trước tim",
    "cảm giác thắt chặt ngực",
    "đau vùng hạ sườn phải", "đau bụng vùng hạ sườn phải",
    "đau bụng vùng thượng vị", "đau vùng hạ vị",
    "đau bụng hạ sườn phải",
    "hội chứng não gan",
    "ý thức suy giảm",
    "đau bụng quặn",
    "tiêu chảy ra máu",
    "đau lưng âm ỉ",
    "giọng khàn", "loét", "chảy dịch", "đau quanh vết mổ",
    "tiểu tiện không tự chủ", "bí tiểu", "cơn co tử cung",
    "ra huyết âm đạo", "chảy máu", "cục máu đông",
    "tiểu ra máu dai dẳng", "đau khi ấn", "đau khi nắm tay",
    "sưng tím", "mất màu", "yếu", "đau bẹn",
    "đau hạ vị", "khó khăn",
    "chảy dịch liên tục", "đau quanh vết mổ",
    "liệt hai chân", "sa âm đạo", "bàng quang căng",
    "đau vùng xoang", "ho nhẹ",
    "đau đầu dữ dội", "đau hạ vị",
    "tăng cân trở lại",
    "vết thương thấu bụng",
    "rối loạn dáng đi",
    "trung tiện",
    "ảo giác", "ảo giác thính giác",
    "rối loạn giấc ngủ", "lo âu", "cảm giác tuyệt vọng",
    "tư tưởng tự sát", "chán ăn",
    "suy giáp trí nhớ", "mất phương hướng",
    "nước tiểu đỏ", "phù nề",
    "gãy xương sườn"
]

# Danh sách các tên xét nghiệm phổ biến
TEST_NAMES = [
    "chụp x-quang ngực", "chụp x-quang",
    "chụp ct", "chụp cắt lớp vi tính", "chụp cắt lớp vi tính sọ não",
    "chụp ct ngực không thuốc cản quang", "chụp ct ngực",
    "chụp cộng hưởng từ", "mri", "chụp cộng hưởng từ mật tụy",
    "siêu âm", "siêu âm tim", "siêu âm tim qua thành ngực",
    "siêu âm gan mật", "siêu âm hôm nay",
    "xạ hình tưới máu cơ tim", "xạ hình tưới máu cơ tim mibi",
    "xét nghiệm máu", "xét nghiệm chức năng gan",
    "bảng công thức sinh hóa máu cơ bản", "bảng công thức máu",
    "bảng chức năng gan", "bản phân tích nước tiểu",
    "phân tích nước tiểu",
    "nội soi", "nội soi đại tràng",
    "nội soi thực quản - dạ dày - tá tràng",
    "nội soi mật tụy ngược dòng",
    "sinh thiết",
    "điện tâm đồ", "ecg",
    "monitor holter",
    "điện não đồ", "eeg",
    "troponin", "cea",
    "ast", "alt", "phosphatase kiềm", "bilirubin toàn phần",
    "bạch cầu", "wbc",
    "hemoglobin", "hematocrit",
    "creatinine", "bun",
    "glucose", "hba1c",
    "lưu lượng đỉnh thở ra",
]

CLINICAL_MEDICATIONS = [
    "metoprolol", "atenolol", "propranolol", "carvedilol", "bisoprolol",
    "lisinopril", "enalapril", "ramipril", "losartan", "valsartan", "irbesartan",
    "captopril", "telmisartan", "amlodipine", "nifedipine", "diltiazem", "verapamil",
    "furosemide", "hydrochlorothiazide", "spironolactone", "bumetanide", "chlorthalidone",
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lovastatin",
    "aspirin", "clopidogrel", "warfarin", "heparin", "enoxaparin", "rivaroxaban",
    "apixaban", "dabigatran", "ticagrelor", "prasugrel", "omeprazole", "pantoprazole",
    "esomeprazole", "lansoprazole", "ranitidine", "famotidine", "sucralfate",
    "amoxicillin", "azithromycin", "doxycycline", "ciprofloxacin", "levofloxacin",
    "metronidazole", "ceftriaxone", "vancomycin", "meropenem", "piperacillin",
    "trimethoprim", "clindamycin", "gentamicin", "ampicillin", "cephalexin",
    "cefazolin", "cefuroxime", "cefepime", "imipenem", "linezolid", "erythromycin",
    "acetaminophen", "ibuprofen", "naproxen", "morphine", "fentanyl", "tramadol",
    "ketorolac", "celecoxib", "gabapentin", "pregabalin", "oxycodone", "hydrocodone",
    "codeine", "metformin", "insulin", "glipizide", "glyburide", "pioglitazone",
    "sitagliptin", "empagliflozin", "dapagliflozin", "prednisone", "prednisolone",
    "dexamethasone", "hydrocortisone", "methylprednisolone", "budesonide",
    "albuterol", "ipratropium", "tiotropium", "montelukast", "fluticasone",
    "salmeterol", "theophylline", "digoxin", "amiodarone", "nitroglycerin",
    "isosorbide", "dopamine", "dobutamine", "milrinone", "sertraline", "fluoxetine",
    "citalopram", "escitalopram", "paroxetine", "venlafaxine", "duloxetine",
    "bupropion", "mirtazapine", "trazodone", "amitriptyline", "quetiapine",
    "olanzapine", "risperidone", "haloperidol", "lorazepam", "diazepam",
    "alprazolam", "clonazepam", "lithium", "valproic acid", "carbamazepine",
    "lamotrigine", "topiramate", "levothyroxine", "allopurinol", "colchicine",
    "hydroxychloroquine", "sildenafil", "tamsulosin", "propofol", "levophed",
    "norepinephrine", "epinephrine", "phentolamine", "combivent", "magnesium",
    "chlorpheniramine", "capsaicin", "lactulose", "solumedrol", "natriclorid",
    "vincardipin", "nicardipine", "lipitor", "viacoram", "octreotide", "flagyl",
    "vicodin", "suboxone", "clonidine", "klonopin", "nsaid", "nsaids", "bipap",
    "cpap", "nitroglycerin dưới lưỡi"
]

CLINICAL_DIAGNOSES = [
    'tăng huyết áp', 'nhồi máu cơ tim', 'suy tim', 'rung nhĩ', 'rung nhĩ kịch phát', 
    'bệnh tim mạch do xơ vữa động mạch', 'bệnh mạch vành', 'ngoại tâm thu nhĩ', 
    'ngoại tâm thu thất', 'ngoại tâm thu', 'hẹp động mạch cảnh', 'phình động mạch chủ', 
    'phình động mạch chủ nhỏ', 'hạ huyết áp', 'hạ huyết áp, không đặc hiệu', 
    'tràn dịch màng tim', 'tràn dịch màng phổi', 'hở van ba lá', 'hở van hai lá', 
    'tim to', 'xẹp phổi', 'viêm phổi', 'hen suyễn', 'hen phế quản', 
    'bệnh phổi tắc nghẽn mạn tính', 'bệnh phổi tắc nghẽn mạn tính, không xác định', 
    'xơ gan', 'xơ gan do rượu', 'viêm dạ dày', 'viêm gan', 'viêm gan do men', 
    'sỏi ống mật chủ', 'bệnh trào ngược dạ dày thực quản', 'bệnh trào ngược dạ dày- thực quản', 
    'bệnh trào ngược dạ dày- thực quản không có viêm thực quản', 'loét dạ dày tá tràng', 
    'viêm tụy cấp', 'bệnh thận mạn', 'bệnh thận mạn, không đặc hiệu', 'suy thận', 
    'đái tháo đường', 'đái tháo đường loại 2', 'đái tháo đường type ii', 'cường giáp', 
    'suy giáp', 'gãy cổ xương đùi', 'gãy xương đùi', 'gãy xương hông', 
    'gãy cổ xương đùi di lệch', 'u trực tràng', 'u ác trực tràng', 'khối u trực tràng', 
    'u tuyến', 'ung thư phổi không tế bào nhỏ', 'ung thư vú di căn', 
    'ung thư biểu mô tuyến phổi không tế bào nhỏ di căn', 'viêm tuyến mồ hôi', 
    'tăng sản tuyến tiền liệt', 'xuất huyết nội sọ', 'xuất huyết nội sọ không do chấn thương, không đặc hiệu', 
    'tổn thương dây thanh quản', 'bệnh rễ thần kinh', 'hẹp ống sống', 'rối loạn lipid máu', 
    'tăng lipid máu', 'tăng lipid máu, không đặc hiệu', 'béo phì', 'thiếu máu', 
    'ngừng thở khi ngủ', 'trầm cảm', 'vết thương hở', 'sa âm đạo', 'tiểu tiện không tự chủ', 
    'bàn chân vẹo bẩm sinh', 'tách thành động mạch chủ', 'rò động - tĩnh mạch', 
    'khối máu tụ dưới màng cứng', 'bệnh mạch máu', 'bệnh mạch máu ngoại biên', 
    'bệnh động mạch vành', 'tâm thần phân liệt', 'rối loạn lưỡng cực', 'rối loạn lo âu', 
    'rối loạn cảm xúc', 'sa van hai lá', 'thoát vị hoành', 'thực quản barrett', 
    'huyết khối tĩnh mạch sâu', 'viêm dạ dày ruột do virus', 'sỏi bàng quang', 
    'diverticulosis', 'trĩ nội', 'loét', 'phì đại vú', 'tràn dịch màng ngoài tim', 
    'men gan tăng', 'tăng men gan', 'rò ống tuỵ'
]

class TrieMatcher:
    def __init__(self):
        self.root = {}
        
    def insert(self, word, type_name):
        word = word.lower().strip()
        if len(word) < 3: # Bỏ qua từ khóa quá ngắn để tránh nhận diện sai
            return
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        if None not in node:
            node[None] = set()
        node[None].add(type_name)
        
    def search_in_text(self, text):
        text_lower = text.lower()
        matches = []
        n = len(text)
        for i in range(n):
            node = self.root
            j = i
            last_match = None
            last_match_end = -1
            while j < n and text_lower[j] in node:
                node = node[text_lower[j]]
                if None in node:
                    last_match = node[None]
                    last_match_end = j + 1
                j += 1
            if last_match is not None:
                # Kiểm tra ranh giới từ (Word Boundary)
                is_start_boundary = (i == 0 or not text_lower[i-1].isalnum())
                is_end_boundary = (last_match_end == n or not text_lower[last_match_end].isalnum())
                if is_start_boundary and is_end_boundary:
                    for type_name in last_match:
                        matches.append((i, last_match_end, type_name))
        return matches

class BaselineExtractor:
    def __init__(self):
        print("Initializing BaselineExtractor and loading dictionaries from SQLite...")
        self.matcher = TrieMatcher()
        
        # Bộ lọc tránh nhận diện nhầm triệu chứng thành chẩn đoán và chất thông thường thành thuốc
        symptom_set = {s.lower().strip() for s in SYMPTOMS}
        blacklist_drugs = {
            "caffeine", "water", "alcohol", "oxygen", "glucose", "sodium chloride",
            "carbon dioxide", "nitrogen", "helium", "air", "diet", "food",
            "caffeine anhydrous", "caffeine citrate"
        }
        blacklist_diagnoses = {
            "sử dụng rượu", "uống rượu", "hút thuốc", "hút thuốc lá", "sử dụng thuốc lá",
            "ăn uống", "hoạt động thể lực", "lối sống", "tiền sử", "tiền sử gia đình",
            "tiền sử bản thân", "dị ứng", "dị ứng thuốc", "rượu", "bia", "lạm dụng rượu",
            "lạm dụng chất", "nghiện rượu", "nghiện thuốc lá"
        }
        blacklist_common = {
            "nam", "nữ", "mẹ", "con", "ngày", "tháng", "năm", "tiền", "sau", "trên", "dưới", "đau",
            "có", "không", "trong", "cho", "và", "là", "được", "đại", "thể", "bình", "thường", 
            "hạn", "kỳ", "nhân", "viên", "đầu", "tay", "chân", "miệng", "răng", "tai", "mắt", "mũi",
            "họng", "cổ", "ngực", "bụng", "lưng", "vai", "gối", "hông", "khớp", "da", "tim", "gan",
            "thận", "phổi", "mật", "ruột", "dạ dày", "thực quản", "não", "máu", "nước tiểu",
            "u", "kết quả", "chụp", "siêu âm", "nội soi", "sinh thiết", "điện tâm đồ", "xét nghiệm",
            "thuốc", "bệnh", "hội chứng", "viêm", "loét", "suy", "tăng", "hạ", "rối loạn", "uống",
            "tiêm", "truyền", "bôi", "đặt", "ngạt", "ho", "sốt", "phù", "liệt", "yếu", "mệt", "mỏi",
            "nôn", "tiêu chảy", "táo bón", "ngất", "run", "ngứa", "sưng", "rụng", "tóc", "móng",
            "tin", "vết thương", "dị ứng", "béo phì", "thiếu máu", "trầm cảm", "lo âu", "sa",
            "rò", "trĩ", "men", "nước", "muối", "oxy", "cồn", "rượu", "bia", "khói", "bụi",
            "thở", "hoạt động", "ăn", "ngủ", "nghỉ", "lối sống", "gia đình", "bố", "anh", "chị",
            "em", "cháu", "ông", "bà", "thân", "người", "bản thân", "tiền sử", "quá khứ", "tương lai",
            "hiện tại", "bệnh viện", "phòng khám", "bác sĩ", "y tá", "điều dưỡng", "bệnh nhân",
            "khó khăn", "chảy dịch", "chảy máu", "ho nhẹ", "lo âu", "chán ăn", "phù nề"
        }

        def is_blacklisted(text):
            text_clean = text.lower().strip()
            if len(text_clean) < 4:
                return True
            if text_clean in blacklist_common:
                return True
            words = text_clean.split()
            if all(w in blacklist_common for w in words):
                return True
            return False

        # Kết nối CSDL và tự động tải từ điển
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # 1. Tải danh mục chẩn đoán ICD-10
            cursor.execute("SELECT code, name_vi, name_en FROM icd10;")
            for code, name_vi, name_en in cursor.fetchall():
                # Bỏ qua các mã thuộc nhóm triệu chứng R-codes hoặc trùng với triệu chứng lâm sàng
                if code and code.upper().startswith('R'):
                    continue
                if name_vi:
                    name_vi_clean = name_vi.lower().strip()
                    if name_vi_clean in symptom_set or name_vi_clean in blacklist_diagnoses or is_blacklisted(name_vi_clean):
                        continue
                if name_en:
                    name_en_clean = name_en.lower().strip()
                    if name_en_clean in symptom_set or name_en_clean in blacklist_diagnoses or is_blacklisted(name_en_clean):
                        continue
                    
                if name_vi:
                    self.matcher.insert(name_vi, "CHẨN_ĐOÁN")
                if name_en:
                    self.matcher.insert(name_en, "CHẨN_ĐOÁN")
                    
            # 2. Tải danh mục thuốc RxNorm (BN, IN, PIN, SBD, SCD, SCDC)
            cursor.execute("SELECT name FROM rxnorm WHERE tty IN ('BN', 'IN', 'PIN', 'SBD', 'SCD', 'SCDC');")
            for (name,) in cursor.fetchall():
                if name:
                    name_clean = name.lower().strip()
                    if name_clean not in blacklist_drugs and not is_blacklisted(name_clean):
                        self.matcher.insert(name, "THUỐC")
                    
            conn.close()
            print("  - Successfully loaded ICD-10 and RxNorm dictionaries.")
        except Exception as e:
            print(f"  - [WARNING] Failed to load dictionaries from SQLite database: {e}")
            
        # 3. Tải danh mục thuốc lâm sàng bổ sung
        for med in CLINICAL_MEDICATIONS:
            self.matcher.insert(med, "THUỐC")

        # 4. Tải danh mục chẩn đoán lâm sàng bổ sung
        for diag in CLINICAL_DIAGNOSES:
            if diag.lower().strip() not in blacklist_diagnoses and not is_blacklisted(diag):
                self.matcher.insert(diag, "CHẨN_ĐOÁN")
            
        # 4. Tải danh mục triệu chứng
        for symp in SYMPTOMS:
            self.matcher.insert(symp, "TRIỆU_CHỨNG")
            
        # 5. Tải danh mục xét nghiệm
        for test in TEST_NAMES:
            self.matcher.insert(test, "TÊN_XÉT_NGHIỆM")
            
        print("BaselineExtractor initialization completed.")

    def extract_entities(self, text):
        entities = []
        if not text:
            return entities
            
        text_lower = text.lower()
        
        # So khớp bằng Trie
        raw_matches = self.matcher.search_in_text(text)
        
        # Sắp xếp các cụm khớp để xử lý chồng lấn (ưu tiên cụm dài nhất, sau đó đến vị trí xuất hiện trước)
        raw_matches = sorted(raw_matches, key=lambda x: (x[0], -(x[1] - x[0])))
        
        used_positions = []
        def is_overlap(start, end):
            for s, e in used_positions:
                if not (end <= s or start >= e):
                    return True
            return False

        # Hàm mở rộng ranh giới thuốc để lấy liều lượng
        def get_dose_expansion(start, end):
            tail_text = text[end:end+40]
            dosage_match = re.match(r'^\s*(?:\d+(?:[\.,]\d+)?\s*(?:mg/ml|mg|ml|g|mcg|ui|iu|viên|ống|lọ|gói|bơm)?(?:\s+(?:po|bid|qid|tid|ac|pc|daily|uống|tiêm|bôi|đặt|truyền|ngày\s+\d+\s+lần|sáng|chiều|tối))*)+', tail_text, flags=re.IGNORECASE)
            if dosage_match:
                return end + dosage_match.end()
            return end

        # Bước 1: Trích xuất THUỐC và CHẨN_ĐOÁN trước (độ ưu tiên cao hơn)
        for start, end, type_name in raw_matches:
            if type_name == "THUỐC":
                extended_end = get_dose_expansion(start, end)
                if not is_overlap(start, extended_end):
                    used_positions.append((start, extended_end))
                    entities.append({
                        'text': text[start:extended_end].strip(),
                        'type': 'THUỐC',
                        'position': [start, extended_end]
                    })
            elif type_name == "CHẨN_ĐOÁN":
                if not is_overlap(start, end):
                    used_positions.append((start, end))
                    entities.append({
                        'text': text[start:end],
                        'type': 'CHẨN_ĐOÁN',
                        'position': [start, end]
                    })

        # Bước 2: Trích xuất TRIỆU_CHỨNG và TÊN_XÉT_NGHIỆM
        for start, end, type_name in raw_matches:
            if type_name in ("TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM"):
                if not is_overlap(start, end):
                    used_positions.append((start, end))
                    entities.append({
                        'text': text[start:end],
                        'type': type_name,
                        'position': [start, end]
                    })

        # Bước 3: Trích xuất KẾT_QUẢ_XÉT_NGHIỆM bằng biểu thức chính quy (Regex)
        result_patterns = [
            r'(?:troponin|cea|ast|alt|wbc|hemoglobin|hematocrit|creatinine|bun|glucose|hba1c|bilirubin toàn phần|bilirubin trực tiếp|phosphatase kiềm|amylase|crp|neu)\s*(?:\([^)]*\))?\s*(?:là|:|\s)\s*(\d+[\.,]?\d*)',
        ]
        for pattern in result_patterns:
            for match in re.finditer(pattern, text_lower):
                val_text = match.group(1)
                val_start = match.start(1)
                val_end = match.end(1)
                if not is_overlap(val_start, val_end):
                    used_positions.append((val_start, val_end))
                    entities.append({
                        'text': val_text,
                        'type': 'KẾT_QUẢ_XÉT_NGHIỆM',
                        'position': [val_start, val_end]
                    })

        # Sắp xếp lại theo vị trí bắt đầu
        entities.sort(key=lambda x: x['position'][0])
        return entities

if __name__ == "__main__":
    extractor = BaselineExtractor()
    sample_text = (
        "Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ hơi, "
        "bệnh nhân có tiền sử sử dụng Chlorpheniramine 0.4 MG/ML, Capsaicin 0.38 MG/ML, đã tiến hành "
        "tổng phân tích tế bào máu bằng máy lazer (tbm): WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung tính):76,4; "
        "LYPH% (Tỷ lệ bạch cầu lympho):12,8; bệnh trào ngược dạ dày - thực quản"
    )
    
    extracted = extractor.extract_entities(sample_text)
    for ent in extracted:
        print(f"[{ent['type']}] {ent['text']} at {ent['position']}")
