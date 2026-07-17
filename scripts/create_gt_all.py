"""
Script tạo Ground Truth (GT) cho 100 file input Part1.
Trích xuất thực thể y tế: TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM
Tính position chính xác, phát hiện assertions, tra cứu ICD-10/RxNorm.
"""
import json
import re
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ============================================================
# BỘ TỪ ĐIỂN Y KHOA
# ============================================================

# --- THUỐC (tên thuốc phổ biến) với RxNorm CUI ---
MEDICATIONS = {
    # Beta-blockers
    "metoprolol": ["6918"], "atenolol": ["1202"], "propranolol": ["8787"],
    "carvedilol": ["20352"], "bisoprolol": ["19484"],
    # ACE inhibitors / ARBs
    "lisinopril": ["29046"], "enalapril": ["3827"], "ramipril": ["35296"],
    "losartan": ["52175"], "valsartan": ["69749"], "irbesartan": ["83818"],
    "captopril": ["1998"], "telmisartan": ["73494"],
    # Calcium channel blockers
    "amlodipine": ["17767"], "nifedipine": ["7417"], "diltiazem": ["3443"],
    "verapamil": ["11170"],
    # Diuretics
    "furosemide": ["4603"], "hydrochlorothiazide": ["5487"],
    "spironolactone": ["9997"], "bumetanide": ["1808"],
    "chlorthalidone": ["2409"],
    # Statins
    "atorvastatin": ["83367"], "rosuvastatin": ["301542"],
    "simvastatin": ["36567"], "pravastatin": ["42463"],
    "lovastatin": ["6472"],
    # Anticoagulants / Antiplatelets
    "aspirin": ["1191"], "clopidogrel": ["32968"], "warfarin": ["11289"],
    "heparin": ["5224"], "enoxaparin": ["67108"], "rivaroxaban": ["1114195"],
    "apixaban": ["1364430"], "dabigatran": ["1037042"],
    "ticagrelor": ["1116632"], "prasugrel": ["613391"],
    # PPIs / GI
    "omeprazole": ["7646"], "pantoprazole": ["40790"],
    "esomeprazole": ["283742"], "lansoprazole": ["17128"],
    "ranitidine": ["9143"], "famotidine": ["4278"],
    "sucralfate": ["10156"],
    # Antibiotics
    "amoxicillin": ["723"], "azithromycin": ["18631"],
    "doxycycline": ["3640"], "ciprofloxacin": ["2551"],
    "levofloxacin": ["82122"], "metronidazole": ["6922"],
    "ceftriaxone": ["2193"], "vancomycin": ["11124"],
    "meropenem": ["29561"], "piperacillin": ["8339"],
    "trimethoprim": ["10829"], "clindamycin": ["2582"],
    "gentamicin": ["4750"], "ampicillin": ["733"],
    "cephalexin": ["2231"], "cefazolin": ["2180"],
    "cefuroxime": ["2194"], "cefepime": ["25037"],
    "imipenem": ["5690"], "linezolid": ["190376"],
    "erythromycin": ["3952"],
    # Pain / Anti-inflammatory
    "acetaminophen": ["161"], "ibuprofen": ["5640"],
    "naproxen": ["7258"], "morphine": ["7052"],
    "fentanyl": ["4337"], "tramadol": ["10689"],
    "ketorolac": ["6809"], "celecoxib": ["140587"],
    "gabapentin": ["25480"], "pregabalin": ["187832"],
    "oxycodone": ["7804"], "hydrocodone": ["5489"],
    "codeine": ["2670"],
    # Diabetes
    "metformin": ["6809"], "insulin": ["5856"],
    "glipizide": ["4815"], "glyburide": ["4821"],
    "pioglitazone": ["33738"], "sitagliptin": ["593411"],
    "empagliflozin": ["1545653"], "dapagliflozin": ["1488564"],
    # Corticosteroids
    "prednisone": ["8640"], "prednisolone": ["8638"],
    "dexamethasone": ["3264"], "hydrocortisone": ["5492"],
    "methylprednisolone": ["6902"], "budesonide": ["19831"],
    # Respiratory
    "albuterol": ["435"], "ipratropium": ["7213"],
    "tiotropium": ["274783"], "montelukast": ["88249"],
    "fluticasone": ["41126"], "salmeterol": ["36117"],
    "theophylline": ["10438"],
    # Cardiac
    "digoxin": ["3407"], "amiodarone": ["703"],
    "nitroglycerin": ["7417"], "isosorbide": ["6058"],
    "dopamine": ["3616"], "dobutamine": ["3616"],
    "milrinone": ["30125"],
    # Psych
    "sertraline": ["36437"], "fluoxetine": ["4493"],
    "citalopram": ["2556"], "escitalopram": ["321988"],
    "paroxetine": ["32937"], "venlafaxine": ["39786"],
    "duloxetine": ["72625"], "bupropion": ["42347"],
    "mirtazapine": ["15996"], "trazodone": ["10737"],
    "amitriptyline": ["704"], "quetiapine": ["51272"],
    "olanzapine": ["61381"], "risperidone": ["35636"],
    "haloperidol": ["5093"], "lorazepam": ["6470"],
    "diazepam": ["3322"], "alprazolam": ["596"],
    "clonazepam": ["2598"], "lithium": ["6448"],
    "valproic acid": ["11118"], "carbamazepine": ["2002"],
    "lamotrigine": ["28439"], "topiramate": ["38404"],
    # Others
    "levothyroxine": ["10582"], "allopurinol": ["519"],
    "colchicine": ["2683"], "hydroxychloroquine": ["5521"],
    "sildenafil": ["136411"], "tamsulosin": ["77492"],
    "propofol": ["8712"], "levophed": ["7512"],
    "norepinephrine": ["7512"], "epinephrine": ["3992"],
    "phentolamine": ["8163"],
    "combivent": ["344919"], "magnesium": ["6625"],
    "chlorpheniramine": ["2264"], "capsaicin": ["1873"],
    "lactulose": ["6316"],
    # Thuốc Việt Nam phổ biến
    "solumedrol": ["6902"], "natriclorid": [],
    "vincardipin": [], "nicardipine": ["51091"],
    "lipitor": ["83367"],
    "viacoram": [],
    # Bổ sung thêm
    "octreotide": ["7649"], "flagyl": ["4550"],
    "vicodin": ["857002"], "suboxone": ["351264"],
    "clonidine": ["2599"], "klonopin": ["2598"],
    "nsaid": [], "nsaids": [],
    "bipap": [], "cpap": [],
    "nitroglycerin dưới lưỡi": ["7417"],
}

# --- TRIỆU CHỨNG phổ biến ---
SYMPTOMS = [
    # Hô hấp
    "khó thở", "ho", "ho đờm", "ho ra máu", "ho đờm xanh", "ho khan",
    "đờm", "tiếng rít", "khò khè", "tức ngực", "thở khò khè",
    # Tim mạch
    "đánh trống ngực", "đau ngực", "hồi hộp", "tím tái",
    # Tiêu hóa
    "buồn nôn", "nôn", "nôn ói", "nôn mửa", "nôn ra máu",
    "đau bụng", "chướng bụng", "ợ hơi", "ợ chua", "ợ nóng",
    "tiêu chảy", "táo bón", "đi ngoài ra máu", "phân đen",
    "đau thượng vị", "đau hạ vị",
    # Thần kinh
    "đau đầu", "chóng mặt", "choáng váng", "ngất", "ngất xỉu",
    "co giật", "tê bì", "yếu liệt", "liệt", "mất ý thức",
    "run", "mất trí nhớ", "lú lẫn", "mất phương hướng",
    # Cơ xương khớp
    "đau lưng", "đau khớp", "đau cơ", "sưng khớp",
    "đau chân", "đau vai", "đau cổ", "cứng khớp",
    "đau hông", "đau đầu gối",
    # Toàn thân
    "sốt", "mệt mỏi", "sụt cân", "tăng cân", "ớn lạnh",
    "đổ mồ hôi", "vã mồ hôi", "mất ngủ", "chán ăn",
    "suy nhược", "kiệt sức",
    # Tiết niệu
    "tiểu khó", "tiểu buốt", "tiểu máu", "tiểu ra máu",
    "tiểu đêm", "tiểu gắt",
    # Da
    "phát ban", "ngứa", "mẩn đỏ", "sưng", "phù",
    "phù mắt cá chân", "phù chân",
    # Mắt / Tai
    "mờ mắt", "nhìn đôi", "ù tai", "giảm thính lực",
    # Khác
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
    # Bổ sung từ phân tích file thiếu
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
    "suy giảm trí nhớ", "mất phương hướng",
    "nước tiểu đỏ", "phù nề",
    "gãy xương sườn",
]

# --- CHẨN ĐOÁN phổ biến với ICD-10 ---
DIAGNOSES = {
    # Tim mạch
    "tăng huyết áp": ["I10"],
    "nhồi máu cơ tim": ["I21.9"],
    "suy tim": ["I50.9"],
    "rung nhĩ": ["I48.9"],
    "rung nhĩ kịch phát": ["I48.0"],
    "bệnh tim mạch do xơ vữa động mạch": ["I25.1"],
    "bệnh mạch vành": ["I25.1"],
    "ngoại tâm thu nhĩ": ["I49.1"],
    "ngoại tâm thu thất": ["I49.3"],
    "ngoại tâm thu": ["I49.4"],
    "hẹp động mạch cảnh": ["I65.2"],
    "phình động mạch chủ": ["I71.9"],
    "phình động mạch chủ nhỏ": ["I71.9"],
    "hạ huyết áp": ["I95.9"],
    "hạ huyết áp, không đặc hiệu": ["I95.9"],
    "tràn dịch màng tim": ["I31.3"],
    "tràn dịch màng phổi": ["J91.8"],
    "hở van ba lá": ["I07.1"],
    "hở van hai lá": ["I05.1"],
    "tim to": ["I51.7"],
    "xẹp phổi": ["J98.1"],
    # Hô hấp
    "viêm phổi": ["J18.9"],
    "hen suyễn": ["J45.9"],
    "hen phế quản": ["J45.9"],
    "bệnh phổi tắc nghẽn mạn tính": ["J44.9"],
    "bệnh phổi tắc nghẽn mạn tính, không xác định": ["J44.9"],
    # Tiêu hóa
    "xơ gan": ["K74.6"],
    "xơ gan do rượu": ["K70.3"],
    "viêm dạ dày": ["K29.7"],
    "viêm gan": ["K75.9"],
    "viêm gan do men": ["K75.9"],
    "sỏi ống mật chủ": ["K80.5"],
    "bệnh trào ngược dạ dày thực quản": ["K21.0"],
    "bệnh trào ngược dạ dày- thực quản": ["K21.0"],
    "bệnh trào ngược dạ dày- thực quản không có viêm thực quản": ["K21.0"],
    "loét dạ dày tá tràng": ["K27.9"],
    "viêm tụy cấp": ["K85.9"],
    # Thận
    "bệnh thận mạn": ["N18.9"],
    "bệnh thận mạn, không đặc hiệu": ["N18.9"],
    "suy thận": ["N19"],
    # Nội tiết
    "đái tháo đường": ["E11.9"],
    "đái tháo đường loại 2": ["E11.9"],
    "đái tháo đường type ii": ["E11.9"],
    "cường giáp": ["E05.9"],
    "suy giáp": ["E03.9"],
    # Chấn thương
    "gãy cổ xương đùi": ["S72.0"],
    "gãy xương đùi": ["S72.9"],
    "gãy xương hông": ["S72.0"],
    "gãy cổ xương đùi di lệch": ["S72.0"],
    # Ung thư
    "u trực tràng": ["C20"],
    "u ác trực tràng": ["C20"],
    "khối u trực tràng": ["C20"],
    "u tuyến": ["D12.8"],
    "ung thư phổi không tế bào nhỏ": ["C34.9"],
    "ung thư vú di căn": ["C50.9"],
    "ung thư biểu mô tuyến phổi không tế bào nhỏ di căn": ["C34.9"],
    # Thần kinh
    "viêm tuyến mồ hôi": ["L73.2"],
    "tăng sản tuyến tiền liệt": ["N40.0"],
    "xuất huyết nội sọ": ["I62.9"],
    "xuất huyết nội sọ không do chấn thương, không đặc hiệu": ["I62.9"],
    "tổn thương dây thanh quản": ["J38.3"],
    "bệnh rễ thần kinh": ["M54.1"],
    "hẹp ống sống": ["M48.0"],
    # Lipid
    "rối loạn lipid máu": ["E78.5"],
    "tăng lipid máu": ["E78.5"],
    "tăng lipid máu, không đặc hiệu": ["E78.5"],
    # Khác
    "béo phì": ["E66.9"],
    "thiếu máu": ["D64.9"],
    "ngừng thở khi ngủ": ["G47.3"],
    "trầm cảm": ["F32.9"],
    "vết thương hở": ["T14.1"],
    "sa âm đạo": ["N81.9"],
    "tiểu tiện không tự chủ": ["N39.3"],
    "bàn chân vẹo bẩm sinh": ["Q66.0"],
    "tách thành động mạch chủ": ["I71.0"],
    "rò động - tĩnh mạch": ["I77.0"],
    "khối máu tụ dưới màng cứng": ["I62.0"],
    "bệnh mạch máu": ["I73.9"],
    "bệnh mạch máu ngoại biên": ["I73.9"],
    "bệnh động mạch vành": ["I25.1"],
    "tâm thần phân liệt": ["F20.9"],
    "rối loạn lưỡng cực": ["F31.9"],
    "rối loạn lo âu": ["F41.9"],
    "rối loạn cảm xúc": ["F39"],
    "sa van hai lá": ["I34.1"],
    "thoát vị hoành": ["K44.9"],
    "thực quản barrett": ["K22.7"],
    "huyết khối tĩnh mạch sâu": ["I82.9"],
    "viêm dạ dày ruột do virus": ["A08.4"],
    "sỏi bàng quang": ["N21.0"],
    "diverticulosis": ["K57.3"],
    "trĩ nội": ["K64.0"],
    "loét": ["L98.4"],
    "phì đại vú": ["N62"],
    "tràn dịch màng ngoài tim": ["I31.3"],
    "men gan tăng": ["R74.0"],
    "tăng men gan": ["R74.0"],
    "rò ống tuỵ": ["K86.8"],
}

# --- TÊN XÉT NGHIỆM ---
TEST_NAMES = [
    # Hình ảnh
    "chụp x-quang ngực", "chụp x-quang",
    "chụp ct", "chụp cắt lớp vi tính", "chụp cắt lớp vi tính sọ não",
    "chụp ct ngực không thuốc cản quang", "chụp ct ngực",
    "chụp cộng hưởng từ", "mri", "chụp cộng hưởng từ mật tụy",
    "siêu âm", "siêu âm tim", "siêu âm tim qua thành ngực",
    "siêu âm gan mật", "siêu âm hôm nay",
    "xạ hình tưới máu cơ tim", "xạ hình tưới máu cơ tim mibi",
    # Xét nghiệm máu
    "xét nghiệm máu", "xét nghiệm chức năng gan",
    "bảng công thức sinh hóa máu cơ bản", "bảng công thức máu",
    "bảng chức năng gan", "bản phân tích nước tiểu",
    "phân tích nước tiểu",
    # Thủ thuật chẩn đoán
    "nội soi", "nội soi đại tràng",
    "nội soi thực quản - dạ dày - tá tràng",
    "nội soi mật tụy ngược dòng",
    "sinh thiết",
    "điện tâm đồ", "ecg",
    "monitor holter",
    "điện não đồ", "eeg",
    # Chỉ số xét nghiệm
    "troponin", "cea",
    "ast", "alt", "phosphatase kiềm", "bilirubin toàn phần",
    "bạch cầu", "wbc",
    "hemoglobin", "hematocrit",
    "creatinine", "bun",
    "glucose", "hba1c",
    "lưu lượng đỉnh thở ra",
]

# --- Từ khóa NEGATION ---
NEGATION_WORDS = [
    "không", "không có", "chưa", "phủ nhận", "loại trừ",
    "chưa phát hiện", "không ghi nhận", "không còn",
    "chưa có", "chưa thấy", "không thấy",
]

# --- Từ khóa HISTORICAL ---
HISTORICAL_WORDS = [
    "tiền sử", "trước khi nhập viện", "đã dùng", "đã từng",
    "trước đây", "trong quá khứ", "đã ngừng", "đã sử dụng",
    "từ nhiều năm", "mạn tính", "mãn tính",
]

# --- Từ khóa FAMILY ---
FAMILY_WORDS = [
    "gia đình", "người nhà", "bố", "mẹ", "anh", "chị", "em",
    "con", "cháu", "ông", "bà", "người thân",
]

# ============================================================
# ICD-10 DICTIONARY
# ============================================================
def load_icd10():
    path = r'D:\AI Race Viettel\docs\icd10_dictionary.json'
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def lookup_icd10(diagnosis_text, icd10_dict):
    """Tra cứu ICD-10 code cho một chẩn đoán."""
    text_lower = diagnosis_text.lower().strip()
    # Kiểm tra trong DIAGNOSES dict trước
    if text_lower in DIAGNOSES:
        return DIAGNOSES[text_lower]
    # Tìm trong ICD-10 dictionary
    candidates = []
    for code, info in icd10_dict.items():
        vi = info.get('vi', '').lower()
        en = info.get('en', '').lower()
        if text_lower in vi or vi in text_lower:
            candidates.append(code)
        if len(candidates) >= 5:
            break
    return candidates

# ============================================================
import sys
sys.path.append(r'D:\AI Race Viettel')
from src.assertion.rule_based import AssertionAnalyzer
analyzer = AssertionAnalyzer()

def detect_assertions(content, start_pos, end_pos, entity_type):
    """Phát hiện assertions dựa trên ngữ cảnh xung quanh entity."""
    if entity_type not in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
        return []

    # Sử dụng AssertionAnalyzer của dự án (chứa logic scope termination rất tốt)
    assertions = analyzer.analyze(content, start_pos, end_pos)

    # Bổ sung thêm rule đặc thù cho section "thuốc trước khi nhập viện"
    if entity_type == "THUỐC":
        section_start = max(0, start_pos - 500)
        section_context = content[section_start:start_pos].lower()
        if "thuốc trước khi nhập viện" in section_context:
            section_marker = section_context.rfind("thuốc trước khi nhập viện")
            after_marker = section_context[section_marker:]
            if not any(h in after_marker for h in ["tiền sử bệnh hiện tại", "đánh giá tại bệnh viện", "các sự kiện"]):
                if "isHistorical" not in assertions:
                    assertions.append("isHistorical")

    return assertions

# ============================================================
# ENTITY EXTRACTION
# ============================================================
def find_all_occurrences(content, text):
    """Tìm tất cả vị trí xuất hiện của text trong content."""
    import re
    positions = []
    # Dùng regex với word boundary \b để không bắt nhầm chuỗi con (ví dụ 'ho' trong 'khoa')
    pattern = r'\b' + re.escape(text) + r'\b'
    for match in re.finditer(pattern, content, re.IGNORECASE):
        start = match.start()
        end = match.end()
        original_text = content[start:end]
        positions.append((start, end, original_text))
    return positions

def extract_entities_from_file(content, icd10_dict):
    """Trích xuất tất cả entities từ nội dung file."""
    entities = []
    used_positions = set()  # Tránh trùng lặp

    # 1. Trích xuất THUỐC
    for med_name, rxnorm_codes in MEDICATIONS.items():
        occurrences = find_all_occurrences(content, med_name)
        for start, end, original_text in occurrences:
            # Kiểm tra word boundary (tránh match trong từ dài hơn)
            # Cho phép match trong từ compound (ví dụ "atenololtrong")
            pos_key = (start, end)
            if pos_key in used_positions:
                continue

            # Mở rộng để bắt dosage nếu có
            extended_text = original_text
            extended_end = end
            # Tìm dosage pattern sau tên thuốc
            after = content[end:end + 50]
            dose_match = re.match(r'[\s]*(\d+[\.,]?\d*\s*(mg|mg/ml|mg/ngày|g|ml|mcg|ug|MG|MG/ML)\s*(po|iv|sc|im|tid|bid|qd|q\d+h)?(\s*(x\s*\d+|/\s*ngày))?)', after, re.IGNORECASE)
            if dose_match:
                extended_text = content[start:end + dose_match.end()]
                extended_end = end + dose_match.end()
                # Trim trailing whitespace
                extended_text = extended_text.rstrip()
                extended_end = start + len(extended_text)

            pos_key = (start, extended_end)
            if pos_key in used_positions:
                continue
            used_positions.add(pos_key)

            assertions = detect_assertions(content, start, extended_end, "THUỐC")
            entity = {
                "text": extended_text,
                "type": "THUỐC",
                "position": [start, extended_end],
                "assertions": assertions,
                "candidates": rxnorm_codes
            }
            entities.append(entity)

    # 2. Trích xuất CHẨN ĐOÁN (tìm dài trước, ngắn sau để tránh overlap)
    sorted_diagnoses = sorted(DIAGNOSES.keys(), key=len, reverse=True)
    for diag_name in sorted_diagnoses:
        icd_codes = DIAGNOSES[diag_name]
        occurrences = find_all_occurrences(content, diag_name)
        for start, end, original_text in occurrences:
            # Kiểm tra overlap với entity đã có
            overlap = False
            for pos_key in used_positions:
                if start < pos_key[1] and end > pos_key[0]:
                    overlap = True
                    break
            if overlap:
                continue

            pos_key = (start, end)
            used_positions.add(pos_key)

            assertions = detect_assertions(content, start, end, "CHẨN_ĐOÁN")
            # Tra cứu ICD-10
            candidates = icd_codes if icd_codes else lookup_icd10(original_text, icd10_dict)

            entity = {
                "text": original_text,
                "type": "CHẨN_ĐOÁN",
                "position": [start, end],
                "assertions": assertions,
                "candidates": candidates
            }
            entities.append(entity)

    # 3. Trích xuất TRIỆU CHỨNG (tìm dài trước)
    sorted_symptoms = sorted(SYMPTOMS, key=len, reverse=True)
    for symp in sorted_symptoms:
        occurrences = find_all_occurrences(content, symp)
        for start, end, original_text in occurrences:
            # Kiểm tra overlap
            overlap = False
            for pos_key in used_positions:
                if start < pos_key[1] and end > pos_key[0]:
                    overlap = True
                    break
            if overlap:
                continue

            pos_key = (start, end)
            used_positions.add(pos_key)

            assertions = detect_assertions(content, start, end, "TRIỆU_CHỨNG")
            entity = {
                "text": original_text,
                "type": "TRIỆU_CHỨNG",
                "position": [start, end],
                "assertions": assertions
            }
            entities.append(entity)

    # 4. Trích xuất TÊN XÉT NGHIỆM (tìm dài trước)
    sorted_tests = sorted(TEST_NAMES, key=len, reverse=True)
    for test in sorted_tests:
        occurrences = find_all_occurrences(content, test)
        for start, end, original_text in occurrences:
            overlap = False
            for pos_key in used_positions:
                if start < pos_key[1] and end > pos_key[0]:
                    overlap = True
                    break
            if overlap:
                continue

            pos_key = (start, end)
            used_positions.add(pos_key)

            entity = {
                "text": original_text,
                "type": "TÊN_XÉT_NGHIỆM",
                "position": [start, end],
                "assertions": []
            }
            entities.append(entity)

    # 5. Trích xuất KẾT_QUẢ_XÉT_NGHIỆM bằng regex
    # Pattern: tên_xét_nghiệm + dấu phân cách + giá trị số
    result_patterns = [
        # Pattern: "tên: giá_trị" hoặc "tên là giá_trị"
        r'(?:troponin|cea|ast|alt|wbc|hemoglobin|hematocrit|creatinine|bun|glucose|hba1c|bilirubin toàn phần|bilirubin trực tiếp|phosphatase kiềm|amylase|crp|neu)\s*(?:\([^)]*\))?\s*(?:là|:|\s)\s*(\d+[\.,]?\d*)',
    ]
    for pattern in result_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            value_text = match.group(1)
            value_start = match.start(1)
            value_end = match.end(1)
            
            pos_key = (value_start, value_end)
            if pos_key in used_positions:
                continue
            
            # Kiểm tra overlap
            overlap = False
            for pk in used_positions:
                if value_start < pk[1] and value_end > pk[0]:
                    overlap = True
                    break
            if overlap:
                continue

            used_positions.add(pos_key)
            entity = {
                "text": value_text,
                "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                "position": [value_start, value_end],
                "assertions": []
            }
            entities.append(entity)

    # Sắp xếp theo position
    entities.sort(key=lambda e: (e["position"][0], -e["position"][1]))
    return entities

# ============================================================
# VERIFY POSITIONS
# ============================================================
def verify_entities(content, entities):
    """Kiểm tra tất cả positions có chính xác không."""
    errors = 0
    for e in entities:
        s, end = e["position"]
        extracted = content[s:end]
        if extracted.lower() != e["text"].lower():
            print(f"  LỖI: [{s}:{end}] '{e['text']}' != '{extracted}'", file=sys.stderr)
            errors += 1
    return errors

# ============================================================
# MAIN
# ============================================================
def process_file(file_id, input_dir, output_dir, icd10_dict):
    """Xử lý 1 file input và tạo GT."""
    input_path = f"{input_dir}\\{file_id}.txt"
    output_path = f"{output_dir}\\{file_id}.json"

    if not os.path.exists(input_path):
        print(f"  Không tìm thấy: {input_path}", file=sys.stderr)
        return None

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entities = extract_entities_from_file(content, icd10_dict)
    errors = verify_entities(content, entities)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=4)

    return len(entities), errors

def main():
    from pathlib import Path
    input_dir = Path(r'D:\AI Race Viettel\data\dev\input')
    output_dir = Path(r'D:\AI Race Viettel\data\dev\gt')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Đang tải ICD-10 dictionary...", file=sys.stderr)
    icd10_dict = load_icd10()
    print(f"Đã tải {len(icd10_dict)} mã ICD-10", file=sys.stderr)

    total_entities = 0
    total_errors = 0
    total_files = 0

    for file_id in range(1, 101):
        result = process_file(file_id, str(input_dir), str(output_dir), icd10_dict)
        if result:
            n_entities, n_errors = result
            total_entities += n_entities
            total_errors += n_errors
            total_files += 1
            print(f"File {file_id:3d}: {n_entities:3d} entities, {n_errors} lỗi")

    print(f"\n{'='*50}")
    print(f"TỔNG KẾT: {total_files} files, {total_entities} entities, {total_errors} lỗi position")

if __name__ == "__main__":
    main()
