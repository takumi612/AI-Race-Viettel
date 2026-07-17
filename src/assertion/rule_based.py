import re

class AssertionAnalyzer:
    def __init__(self):
        # Các trigger words cho từng loại assertion
        self.negation_triggers = [r'\bkhông\b', r'\bchưa\b', r'\bchưa phát hiện\b', r'\bchưa thấy\b', r'\bphủ nhận\b', r'\bbình thường\b', r'\bâm tính\b']
        self.historical_triggers = [r'\btiền sử\b', r'\btrước đây\b', r'\bcũ\b', r'\bnăm ngoái\b', r'\bđã từng\b']
        self.family_triggers = [r'\bbố\b', r'\bmẹ\b', r'\banh\b', r'\bchị\b', r'\bem\b', r'\bgia đình\b', r'\bngười nhà\b']
        
        # Các từ khóa ngắt phạm vi (scope termination)
        self.termination_words = [r'\bnhưng\b', r'\btuy nhiên\b', r'\bmặc dù\b']
        self.termination_chars = ['.', ';']

    def analyze(self, full_text, start_idx, end_idx):
        """
        Phân tích văn bản để tìm assertions cho entity.
        Trả về danh sách các nhãn (ví dụ: ['isNegated', 'isHistorical']).
        """
        assertions = set()
        text_lower = full_text.lower()
        
        # 1. Lấy prefix window 60 ký tự về trước
        window_start = max(0, start_idx - 60)
        prefix_window = text_lower[window_start:start_idx]
        
        # Ngắt câu bằng các dấu câu mạnh (dấu chấm, dấu chấm phẩy, hoặc dấu xuống dòng)
        for char in ['.', ';', '\n']:
            last_idx = prefix_window.rfind(char)
            if last_idx != -1:
                prefix_window = prefix_window[last_idx + 1:]
                
        # 2. Định nghĩa termination words riêng cho từng loại assertion để tránh lan truyền nhầm
        neg_terminations = [r'\bnhưng\b', r'\btuy nhiên\b', r'\bmặc dù\b', r'\bcó\b', r'\bđã\b', r'\bbị\b', r'\bphát hiện\b', r'\bchẩn đoán\b', r'\btiền sử\b']
        hist_terminations = [r'\bnhưng\b', r'\btuy nhiên\b', r'\bmặc dù\b', r'\bhiện tại\b', r'\bnay\b', r'\bvừa\b']
        fam_terminations = [r'\bnhưng\b', r'\btuy nhiên\b', r'\bmặc dù\b', r'\bbản thân\b', r'\bbệnh nhân\b', r'\bbn\b']
        
        # Kiểm tra isNegated
        neg_window = prefix_window
        # Loại trừ các cụm y khoa có chứa chữ "không" nhưng mang nghĩa "unspecified" hoặc thủ thuật
        exclude_patterns = [
            r'\bkhông đặc hiệu\b', r'\bkhông xác định\b',
            r'\bkhông thuốc cản quang\b', r'\bkhông cản quang\b',
            r'\bkhông tiêm cản quang\b', r'\bkhông tiêm thuốc cản quang\b',
            r'\bkhông tiêm thuốc\b', r'\bkhông rõ nguyên nhân\b',
            r'\bkhông rõ lí do\b', r'\bkhông rõ lý do\b',
            r'\bkhông chuẩn bị\b'
        ]
        for exclude in exclude_patterns:
            neg_window = re.sub(exclude, '', neg_window)
            
        for word in neg_terminations:
            matches = list(re.finditer(word, neg_window))
            if matches:
                neg_window = neg_window[matches[-1].end():]
        for trigger in self.negation_triggers:
            if re.search(trigger, neg_window):
                assertions.add("isNegated")
                break
                
        # Kiểm tra isHistorical
        hist_window = prefix_window
        for word in hist_terminations:
            matches = list(re.finditer(word, hist_window))
            if matches:
                hist_window = hist_window[matches[-1].end():]
        for trigger in self.historical_triggers:
            if re.search(trigger, hist_window):
                assertions.add("isHistorical")
                break
                
        # Kiểm tra isFamily
        fam_window = prefix_window
        for word in fam_terminations:
            matches = list(re.finditer(word, fam_window))
            if matches:
                fam_window = fam_window[matches[-1].end():]
        for trigger in self.family_triggers:
            if re.search(trigger, fam_window):
                assertions.add("isFamily")
                break
                
        return list(assertions)

if __name__ == "__main__":
    analyzer = AssertionAnalyzer()
    text1 = "Bệnh nhân nam 70 tuổi, không ho, có tiền sử tăng huyết áp."
    
    # Test entity "ho" (vị trí: 29-31)
    res1 = analyzer.analyze(text1, 29, 31)
    assert "isNegated" in res1, f"Failed res1: {res1}"
    print(f"Test 1 (ho): {res1}")
    
    # Test entity "tăng huyết áp" (vị trí: 46-59)
    res2 = analyzer.analyze(text1, 46, 59)
    assert "isHistorical" in res2, f"Failed res2: {res2}"
    print(f"Test 2 (tăng huyết áp): {res2}")
    
    text2 = "Gia đình có người bị ĐTĐ, nhưng bệnh nhân bình thường."
    # Test entity "ĐTĐ" (vị trí: 21-24)
    res3 = analyzer.analyze(text2, 21, 24)
    assert "isFamily" in res3, f"Failed res3: {res3}"
    print(f"Test 3 (ĐTĐ): {res3}")
    
    print("Assertion Rule-based Test: Passed")
