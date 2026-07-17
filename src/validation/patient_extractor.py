import re

class PatientExtractor:
    def __init__(self):
        # Từ khóa gây nhiễu cho biết thông tin tuổi/giới tính thuộc về người khác
        self.noise_keywords = [
            r'\bbố\b', r'\bmẹ\b', r'\bvợ\b', r'\bchồng\b', r'\bcon\b', 
            r'\bngười nhà\b', r'\bbác sĩ\b', r'\bđiều dưỡng\b', r'\bnvyt\b'
        ]
        
    def extract(self, text):
        """
        Trích xuất giới tính và tuổi (quy đổi ra ngày) của bệnh nhân từ 150 ký tự đầu.
        Trả về: dict {'sex': 'M'/'F'/None, 'age_days': int/None}
        """
        if not text:
            return {'sex': None, 'age_days': None}
            
        # Chỉ quét trong 150 ký tự đầu tiên
        header = text[:150].lower()
        
        # 1. Cơ chế fail-safe: kiểm tra từ khóa nhiễu liên quan người nhà/nhân viên y tế
        for noise in self.noise_keywords:
            if re.search(noise, header):
                # Phát hiện nhiễu -> Trả về None an toàn
                return {'sex': None, 'age_days': None}
                
        # 2. Trích xuất giới tính
        sex = None
        # Kiểm tra "nữ" trước vì "nữ" có thể chứa cụm từ khác hoặc ưu tiên chính xác
        if re.search(r'\bnữ\b', header):
            sex = 'F'
        elif re.search(r'\bnam\b', header):
            sex = 'M'
            
        # 3. Trích xuất tuổi và quy đổi về ngày
        age_days = None
        
        # Thử tìm các pattern kết hợp trước (ví dụ: 1 tháng 15 ngày, 2 tuổi 3 tháng)
        combined_match = re.search(r'\b(\d+)\s*(?:tháng)\s*(?:và\s*)?(\d+)\s*(?:ngày)\b', header)
        combined_yr_mon = re.search(r'\b(\d+)\s*(?:tuổi|t)\s*(?:và\s*)?(\d+)\s*(?:tháng)\b', header)
        
        if combined_match:
            age_days = int(combined_match.group(1)) * 30 + int(combined_match.group(2))
        elif combined_yr_mon:
            age_days = int(combined_yr_mon.group(1)) * 365 + int(combined_yr_mon.group(2)) * 30
        else:
            # Thử tìm theo tuổi (năm) trước để tránh bắt nhầm "ngày thứ X của bệnh"
            year_match = re.search(r'\b(\d+)\s*(?:tuổi|t)\b', header)
            month_match = re.search(r'\b(\d+)\s*(?:tháng tuổi|tháng)\b', header)
            day_match = re.search(r'\b(\d+)\s*(?:ngày tuổi|ngày)\b', header)
            
            if year_match:
                val = int(year_match.group(1))
                match_str = year_match.group(0)
                start_pos = year_match.start()
                prefix = header[max(0, start_pos - 5):start_pos]
                # Tránh bắt nhầm viết tắt l/t (lần/phút)
                if "/" in prefix and match_str.strip().endswith("t"):
                    clear_year = re.search(r'\b(\d+)\s*tuổi\b', header)
                    if clear_year:
                        age_days = int(clear_year.group(1)) * 365
                else:
                    age_days = val * 365
            elif month_match:
                age_days = int(month_match.group(1)) * 30
            elif day_match:
                # Kiểm tra tránh nhầm "ngày thứ X", "ngày bệnh", "ngày thứ" của bệnh
                start_pos = day_match.start()
                prefix = header[max(0, start_pos - 15):start_pos]
                if "ngày thứ" in prefix or "ngày bệnh" in prefix or "bị bệnh" in prefix or "vào viện ngày" in prefix:
                    age_days = None
                else:
                    age_days = int(day_match.group(1))
            
        return {
            'sex': sex,
            'age_days': age_days
        }

if __name__ == "__main__":
    extractor = PatientExtractor()
    
    # Test case 1: Bình thường
    txt1 = "Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh..."
    print("Test 1:", extractor.extract(txt1)) # Expect: {'sex': 'M', 'age_days': 25550}
    assert extractor.extract(txt1) == {'sex': 'M', 'age_days': 25550}
    
    # Test case 2: Trẻ sơ sinh
    txt2 = "Bệnh nhi nữ 50 ngày tuổi vào viện vì sốt..."
    print("Test 2:", extractor.extract(txt2)) # Expect: {'sex': 'F', 'age_days': 50}
    assert extractor.extract(txt2) == {'sex': 'F', 'age_days': 50}
    
    # Test case 3: Có từ gây nhiễu (mẹ đưa đi khám)
    txt3 = "Bệnh nhi nam 3 tuổi, mẹ bệnh nhân khai thấy cháu ho nhiều..."
    print("Test 3 (noise):", extractor.extract(txt3)) # Expect: {'sex': None, 'age_days': None}
    assert extractor.extract(txt3) == {'sex': None, 'age_days': None}
    
    print("PatientExtractor Test: Passed")
