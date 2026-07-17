import re
import json
import os
import unicodedata

class TextNormalizer:
    def __init__(self, abbrev_path=None):
        if abbrev_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            abbrev_path = os.path.join(current_dir, "abbreviations.json")
            
        self.abbreviations = {}
        if os.path.exists(abbrev_path):
            with open(abbrev_path, 'r', encoding='utf-8') as f:
                self.abbreviations = json.load(f)
                
    def normalize_unicode(self, text):
        """Chuẩn hóa Unicode tiếng Việt về dựng sẵn (NFC)."""
        if not text:
            return ""
        return unicodedata.normalize("NFC", text)

    def expand_clinical_shorthand(self, text):
        """Mở rộng ký hiệu lâm sàng."""
        if not text:
            return ""
        text = text.replace("(+)", "dương tính")
        text = text.replace("(-)", "âm tính")
        return text

    def clean_text(self, text):
        """Xóa khoảng trắng thừa và ký tự đặc biệt ở đầu/cuối."""
        if not text:
            return ""
        text = self.normalize_unicode(text)
        text = self.expand_clinical_shorthand(text)
        # Bỏ khoảng trắng thừa
        text = text.strip()
        # Bỏ dấu câu ở cuối thường gặp
        text = re.sub(r'[,;.]$', '', text)
        return text

    def expand_abbreviation(self, text):
        """Mở rộng từ viết tắt ở mức độ từ (token-level) trong câu."""
        if not text:
            return ""
            
        text_upper = text.upper().strip()
        if text_upper in self.abbreviations:
            return self.abbreviations[text_upper]
            
        words = text.split()
        expanded_words = []
        for word in words:
            clean_word = re.sub(r'^[^\w\d_]+|[^\w\d_]+$', '', word).upper()
            if clean_word in self.abbreviations:
                expanded_val = self.abbreviations[clean_word]
                prefix = re.match(r'^[^\w\d_]+', word)
                suffix = re.search(r'[^\w\d_]+$', word)
                pref_str = prefix.group(0) if prefix else ""
                suff_str = suffix.group(0) if suffix else ""
                expanded_words.append(pref_str + expanded_val + suff_str)
            else:
                expanded_words.append(word)
                
        return " ".join(expanded_words)

    def remove_dosage(self, drug_name):
        """
        Bóc tách liều lượng, cách dùng và các ký hiệu viết tắt Latin khỏi tên thuốc.
        Bảo toàn hàm lượng thuốc (mg, ml, g, ui...) để phục vụ đối sánh RxNorm.
        """
        if not drug_name:
            return ""
            
        cleaned_drug = drug_name.lower()
        
        # 1. Trích xuất hàm lượng trước (ví dụ: 5mg, 0.4 MG/ML)
        dosage_pattern = r'\b\d+(?:[\.,]\d+)?\s*(?:mg/ml|mg|ml|g|mcg|ui|iu)\b'
        dosages = re.findall(dosage_pattern, cleaned_drug, flags=re.IGNORECASE)
        
        # 2. Loại bỏ hàm lượng khỏi tên thuốc tạm thời để lọc các thành phần rác khác
        cleaned_drug = re.sub(dosage_pattern, '', cleaned_drug, flags=re.IGNORECASE)
        
        # 3. Chuẩn hóa các từ viết tắt Latin có dấu chấm (vd: p.o. -> po, a.c. -> ac, q.6.h. -> q6h)
        cleaned_drug = re.sub(r'\b([a-z])\.([a-z])\.(?:([a-z])\.)?(?![a-z])', lambda m: m.group(0).replace('.', ''), cleaned_drug)
        cleaned_drug = re.sub(r'\bq\.(\d+)\.h\.\b', r'q\1h', cleaned_drug)
        
        # 4. Các pattern phổ biến chỉ tần suất/đơn vị/đường dùng
        patterns = [
            r'\b(?:po|bid|qid|tid|ac|pc|hs|prn|stat|daily|q\d+h|iv|im|bds|gtt|q\d+d|qod|qam|qpm|qhs|qds|tds|ud)\b', 
            r'\b(?:sr|la|xr|er|xl|mr|cr|zok)\b',                    
            r'\b\d+\s*(?:viên|ống|lọ|chai|gói|bơm)\b',               
            r'\b(?:uống|tiêm|bôi|đặt|truyền)\b',                     
            r'\bngày\s+\d+\s+lần\b',                                 
            r'\bsáng|chiều|tối\b',                                   
            r'\b(?:ii|iii)\b'                                        
        ]
        
        for pattern in patterns:
            cleaned_drug = re.sub(pattern, '', cleaned_drug, flags=re.IGNORECASE)
            
        cleaned_drug = re.sub(r'[/,;\(\)]', ' ', cleaned_drug)
        cleaned_drug = re.sub(r'\s+', ' ', cleaned_drug)
        cleaned_drug = self.clean_text(cleaned_drug)
        
        # 5. Ghép trả lại hàm lượng đã chuẩn hóa vào cuối chuỗi tên thuốc
        if dosages:
            cleaned_dosages = []
            for ds in dosages:
                ds_clean = ds.strip().upper()
                ds_clean = re.sub(r'(\d+)([A-Z])', r'\1 \2', ds_clean)
                cleaned_dosages.append(ds_clean)
            cleaned_drug = f"{cleaned_drug} {' '.join(cleaned_dosages)}"
            
        return cleaned_drug

if __name__ == "__main__":
    normalizer = TextNormalizer()
    
    # Test Abbreviation
    assert normalizer.expand_abbreviation("NMCT") == "nhồi máu cơ tim"
    assert normalizer.expand_abbreviation("SA") == "siêu âm"
    assert normalizer.expand_abbreviation("HCTZ") == "hydrochlorothiazide"
    # Test token-level abbreviation
    assert normalizer.expand_abbreviation("asa 325mg po x1") == "aspirin 325mg po x1"
    print("Abbreviation Expansion Test: Passed")
    
    # Test Clinical Shorthand & Unicode
    assert normalizer.clean_text("HBsAg (+)") == "HBsAg dương tính"
    assert normalizer.clean_text("HIV (-)") == "HIV âm tính"
    print("Shorthand & Unicode Test: Passed")
    
    # Test Dosage Preservation with slow release modifiers & Latin with dots
    test_drugs = [
        ("Amlodipine 5mg p.o. daily", "amlodipine 5 MG"),
        ("Chlorpheniramine 0.4 MG/ML", "chlorpheniramine 0.4 MG/ML"),
        ("Metformin 500mg ngày 2 lần", "metformin 500 MG"),
        ("Paracetamol 1 viên sủi", "paracetamol sủi"),
        ("Omeprazole 20mg a.c.", "omeprazole 20 MG"),
        ("Metformin XR 500mg", "metformin 500 MG"),
        ("Metoprolol succinate ZOK 50mg", "metoprolol succinate 50 MG"),
        ("Augmentin 1g i.v. stat", "augmentin 1 G")
    ]
    
    for raw, expected in test_drugs:
        result = normalizer.remove_dosage(raw)
        assert result == expected, f"Failed: [{raw}] -> expected [{expected}], got [{result}]"
        print(f"[{raw}] -> [{result}]")
    print("Dosage Preservation Test: Passed")
