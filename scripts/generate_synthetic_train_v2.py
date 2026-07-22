from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


DISEASE = "CHẨN_ĐOÁN"
SYMPTOM = "TRIỆU_CHỨNG"
DRUG = "THUỐC"
LAB_NAME = "TÊN_XÉT_NGHIỆM"
LAB_RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"
TYPES = {DISEASE, SYMPTOM, DRUG, LAB_NAME, LAB_RESULT}


GENRES = {
    "tiep_nhan": ("BÁO CÁO TIẾP NHẬN BỆNH NHÂN", "Lý do tiếp nhận"),
    "noi_tru": ("BỆNH ÁN ĐIỀU TRỊ NỘI TRÚ", "Diễn biến khi nhập khoa"),
    "ngoai_tru": ("PHIẾU KHÁM BỆNH NGOẠI TRÚ", "Lý do đến khám"),
    "cap_cuu": ("PHIẾU CẤP CỨU", "Tình trạng khi vào khoa Cấp cứu"),
    "dien_bien": ("NHẬT KÝ DIỄN BIẾN BỆNH", "Thay đổi trong ca theo dõi"),
    "ban_giao": ("PHIẾU BÀN GIAO CA TRỰC", "Nội dung cần bàn giao"),
    "hoi_chan": ("BIÊN BẢN HỘI CHẨN", "Vấn đề được đưa ra hội chẩn"),
    "xet_nghiem": ("PHIẾU KẾT QUẢ CẬN LÂM SÀNG", "Thông tin lâm sàng kèm mẫu"),
    "don_thuoc": ("ĐƠN THUỐC VÀ KẾ HOẠCH ĐIỀU TRỊ", "Căn cứ lập kế hoạch"),
    "ra_vien": ("TÓM TẮT RA VIỆN", "Tóm tắt lý do nhập viện"),
    "benh_man": ("HỒ SƠ THEO DÕI BỆNH MẠN", "Mục tiêu của lần tái khám"),
    "thai_nhi": ("HỒ SƠ THEO DÕI THAI SẢN/NHI KHOA", "Lý do theo dõi chuyên khoa"),
}


GENRE_ORDERS: dict[str, tuple[str, ...]] = {
    "tiep_nhan": ("reason", "history", "medication", "negative", "lab", "assessment", "plan"),
    "noi_tru": ("history", "reason", "negative", "lab", "assessment", "medication", "plan"),
    "ngoai_tru": ("reason", "history", "lab", "assessment", "plan", "medication", "negative"),
    "cap_cuu": ("reason", "negative", "lab", "assessment", "plan", "history", "medication"),
    "dien_bien": ("assessment", "reason", "lab", "medication", "negative", "history", "plan"),
    "ban_giao": ("assessment", "medication", "reason", "lab", "negative", "plan", "history"),
    "hoi_chan": ("history", "lab", "reason", "assessment", "medication", "plan", "negative"),
    "xet_nghiem": ("reason", "lab", "history", "assessment", "negative", "medication", "plan"),
    "don_thuoc": ("assessment", "history", "medication", "plan", "reason", "negative", "lab"),
    "ra_vien": ("history", "reason", "assessment", "lab", "medication", "negative", "plan"),
    "benh_man": ("history", "medication", "lab", "reason", "assessment", "plan", "negative"),
    "thai_nhi": ("reason", "history", "lab", "assessment", "negative", "plan", "medication"),
}


PROFILES: dict[str, dict[str, Any]] = {
    "coronary": {
        "age": (45, 84), "sex": ("nam", "nữ"), "dx": "I25.1", "history": "I10",
        "drugs": ("1191", "83367"),
        "symptoms": ("đau ngực khi gắng sức", "khó thở khi đi nhanh", "cảm giác nặng ngực"),
        "negatives": ("ngất", "ho ra máu", "sốt cao"),
        "labs": (("điện tâm đồ 12 chuyển đạo", "nhịp xoang, chưa thấy biến đổi ST-T cấp"),
                 ("troponin tim", "troponin trong giới hạn tham chiếu")),
    },
    "heart_failure": {
        "age": (50, 88), "sex": ("nam", "nữ"), "dx": "I50", "history": "I10",
        "drugs": ("4603", "52175"),
        "symptoms": ("khó thở khi nằm", "phù hai chân", "mệt khi gắng sức"),
        "negatives": ("đau ngực dữ dội", "sốt", "ho ra máu"),
        "labs": (("siêu âm tim", "phân suất tống máu thất trái 42%"),
                 ("định lượng BNP", "BNP tăng so với giới hạn tham chiếu")),
    },
    "hypertension": {
        "age": (35, 85), "sex": ("nam", "nữ"), "dx": "I10", "history": "E11.9",
        "drugs": ("17767", "52175"),
        "symptoms": ("đau đầu vùng chẩm", "chóng mặt thoáng qua", "mệt khi làm việc"),
        "negatives": ("yếu liệt khu trú", "đau ngực", "khó thở khi nghỉ"),
        "labs": (("đo huyết áp tại phòng khám", "huyết áp 158/92 mmHg"),
                 ("creatinine huyết thanh", "creatinine 92 µmol/L")),
    },
    "pneumonia": {
        "age": (18, 82), "sex": ("nam", "nữ"), "dx": "J18.9", "history": "I10",
        "drugs": ("723", "2193"),
        "symptoms": ("ho khạc đờm vàng", "sốt kèm rét run", "khó thở tăng dần"),
        "negatives": ("ho ra máu", "đau ngực kiểu mạch vành", "phù chân"),
        "labs": (("chụp X-quang ngực", "đám mờ phế nang thùy dưới phổi phải"),
                 ("công thức máu", "bạch cầu 15,2 G/L, ưu thế bạch cầu trung tính")),
    },
    "asthma": {
        "age": (18, 70), "sex": ("nam", "nữ"), "dx": "J45.9", "history": "J30.4",
        "drugs": ("435", "8640"),
        "symptoms": ("khò khè từng cơn", "khó thở về đêm", "ho khan tái diễn"),
        "negatives": ("sốt", "đau ngực liên tục", "ho ra máu"),
        "labs": (("đo chức năng hô hấp", "FEV1 cải thiện sau nghiệm pháp giãn phế quản"),
                 ("đo SpO2", "SpO2 96% khí phòng")),
    },
    "copd": {
        "age": (45, 86), "sex": ("nam", "nữ"), "dx": "J44.9", "history": "I10",
        "drugs": ("435", "8640"),
        "symptoms": ("ho khạc đờm kéo dài", "khó thở khi gắng sức", "khò khè"),
        "negatives": ("sốt cao", "ho ra máu", "đau ngực cấp"),
        "labs": (("đo chức năng hô hấp", "rối loạn thông khí tắc nghẽn không hồi phục hoàn toàn"),
                 ("khí máu động mạch", "chưa ghi nhận toan hô hấp")),
    },
    "diabetes": {
        "age": (35, 82), "sex": ("nam", "nữ"), "dx": "E11.9", "history": "I10",
        "drugs": ("6809", "83367"),
        "symptoms": ("khát nước nhiều", "tiểu nhiều về đêm", "mệt sau bữa ăn"),
        "negatives": ("hạ đường huyết có triệu chứng", "đau ngực", "khó thở"),
        "labs": (("đường huyết lúc đói", "glucose 9,8 mmol/L"),
                 ("HbA1c", "HbA1c 8,1%")),
    },
    "stroke": {
        "age": (50, 88), "sex": ("nam", "nữ"), "dx": "I63.9", "history": "I10",
        "drugs": ("1191", "32968"),
        "symptoms": ("yếu nửa người phải", "nói khó khởi phát đột ngột", "méo miệng"),
        "negatives": ("co giật", "sốt", "chấn thương đầu"),
        "labs": (("chụp cắt lớp vi tính sọ não", "không thấy xuất huyết nội sọ"),
                 ("siêu âm Doppler động mạch cảnh", "có mảng xơ vữa, chưa gây hẹp khít")),
    },
    "gerd": {
        "age": (20, 78), "sex": ("nam", "nữ"), "dx": "K21.9", "history": "K29.7",
        "drugs": ("7646",),
        "symptoms": ("nóng rát sau xương ức", "ợ chua sau ăn", "đau thượng vị âm ỉ"),
        "negatives": ("nôn ra máu", "đi ngoài phân đen", "sụt cân nhanh"),
        "labs": (("nội soi dạ dày", "niêm mạc thực quản viêm nhẹ, không thấy ổ loét chảy máu"),
                 ("xét nghiệm Helicobacter pylori", "kết quả âm tính")),
    },
    "ckd": {
        "age": (40, 86), "sex": ("nam", "nữ"), "dx": "N18.9", "history": "I10",
        "drugs": ("4603", "17767"),
        "symptoms": ("phù mắt cá chân", "tiểu ít", "mệt kéo dài"),
        "negatives": ("đau quặn thận", "sốt", "tiểu máu đại thể"),
        "labs": (("creatinine huyết thanh", "creatinine 186 µmol/L"),
                 ("ước tính mức lọc cầu thận", "eGFR 38 mL/phút/1,73 m2")),
    },
    "renal_stone": {
        "age": (20, 75), "sex": ("nam", "nữ"), "dx": "N20.0", "history": "I10",
        "drugs": ("161",),
        "symptoms": ("đau quặn hông lưng", "tiểu buốt", "tiểu máu"),
        "negatives": ("bí tiểu hoàn toàn", "sốt cao", "phù toàn thân"),
        "labs": (("siêu âm hệ tiết niệu", "sỏi thận phải 6 mm, chưa gây ứ nước nặng"),
                 ("tổng phân tích nước tiểu", "hồng cầu niệu, nitrit âm tính")),
    },
    "urticaria": {
        "age": (16, 72), "sex": ("nam", "nữ"), "dx": "L50.9", "history": "J30.4",
        "drugs": ("20610",),
        "symptoms": ("mày đay rải rác", "ngứa da", "ban đỏ xuất hiện từng đợt"),
        "negatives": ("khó thở", "phù môi lưỡi", "sốt"),
        "labs": (("công thức máu", "bạch cầu ái toan trong giới hạn tham chiếu"),
                 ("khám da liễu", "sẩn phù ranh giới rõ, mất màu khi ấn")),
    },
    "lung_cancer": {
        "age": (40, 85), "sex": ("nam", "nữ"), "dx": "C34.9", "history": "J44.9",
        "drugs": ("161",),
        "symptoms": ("ho kéo dài", "sụt cân không chủ ý", "mệt tăng dần"),
        "negatives": ("sốt cao", "phù chân", "nôn ra máu"),
        "labs": (("chụp cắt lớp vi tính ngực", "khối bất thường vùng phổi phải cần đánh giá mô bệnh học"),
                 ("sinh thiết tổn thương phổi", "mẫu bệnh phẩm đã được gửi giải phẫu bệnh")),
    },
    "head_injury": {
        "age": (18, 75), "sex": ("nam", "nữ"), "dx": "S06.9", "history": "I10",
        "drugs": ("161",),
        "symptoms": ("đau đầu sau tai nạn", "chóng mặt", "buồn nôn nhẹ"),
        "negatives": ("co giật", "yếu liệt khu trú", "chảy dịch tai"),
        "labs": (("chụp cắt lớp vi tính sọ não", "không thấy tổn thương nội sọ cấp"),
                 ("đánh giá thang điểm Glasgow", "Glasgow 15 điểm")),
    },
    "pregnancy_diabetes": {
        "age": (20, 42), "sex": ("nữ",), "dx": "O24.9", "history": "E11.9",
        "drugs": (),
        "symptoms": ("khát nước nhiều trong thai kỳ", "tiểu nhiều", "mệt sau ăn"),
        "negatives": ("ra huyết âm đạo", "đau bụng từng cơn", "giảm thai máy"),
        "labs": (("nghiệm pháp dung nạp glucose", "đường huyết sau uống glucose cao hơn ngưỡng thai kỳ"),
                 ("siêu âm thai", "thai sống trong buồng tử cung, tăng trưởng phù hợp tuổi thai")),
    },
    "pediatric_asthma": {
        "age": (6, 15), "sex": ("nam", "nữ"), "dx": "J45.9", "history": "J30.4",
        "drugs": ("435",),
        "symptoms": ("thở khò khè", "ho về đêm", "khó thở khi vận động"),
        "negatives": ("sốt cao", "tím môi", "bỏ ăn hoàn toàn"),
        "labs": (("đo SpO2", "SpO2 97% khí phòng"),
                 ("đo lưu lượng đỉnh", "lưu lượng đỉnh cải thiện sau thuốc giãn phế quản")),
    },
}


SCENARIOS: dict[str, tuple[str, ...]] = {
    "cardiology": ("coronary", "heart_failure", "hypertension"),
    "respiratory": ("pneumonia", "asthma", "copd"),
    "diabetes": ("diabetes",),
    "stroke": ("stroke",),
    "gastrointestinal": ("gerd",),
    "renal": ("ckd", "renal_stone"),
    "obstetric": ("pregnancy_diabetes",),
    "pediatric": ("pediatric_asthma",),
    "dermatology": ("urticaria",),
    "oncology": ("lung_cancer",),
    "trauma": ("head_injury",),
    "longtail": tuple(PROFILES),
}


NEUTRAL_SENTENCES = (
    "Thông tin hành chính được đối chiếu với vòng nhận dạng trước khi cập nhật hồ sơ điện tử.",
    "Nhân viên phụ trách ghi nhận thời điểm tiếp xúc và nguồn cung cấp thông tin trong từng mục.",
    "Người bệnh được giải thích quy trình, quyền đặt câu hỏi và cách liên hệ khi cần hỗ trợ.",
    "Các ghi chép của ca trước được so sánh với lời kể hiện tại để phát hiện điểm chưa thống nhất.",
    "Kế hoạch chăm sóc được bàn giao bằng văn bản, kèm mốc đánh giá lại và người chịu trách nhiệm.",
    "Hồ sơ giấy và dữ liệu điện tử được kiểm tra chéo trước khi kết thúc lượt làm việc.",
    "Mọi thay đổi trong quá trình theo dõi phải ghi rõ thời gian, người ghi và lý do điều chỉnh.",
    "Người bệnh và người hỗ trợ đã được hướng dẫn mang theo tài liệu liên quan trong lần hẹn kế tiếp.",
    "Nhóm chăm sóc thống nhất sử dụng một đầu mối liên lạc để tránh truyền đạt thiếu nhất quán.",
    "Khả năng tự chăm sóc tại nhà được đánh giá cùng điều kiện sinh hoạt và mức hỗ trợ sẵn có.",
    "Nội dung tư vấn được diễn đạt bằng ngôn ngữ dễ hiểu và được người nhận nhắc lại để xác nhận.",
    "Phiếu theo dõi tiếp tục được cập nhật theo từng ca, không sao chép nhận định chưa được kiểm chứng.",
    "Các mốc hẹn được ghi trên giấy ra về và đồng thời lưu trong hệ thống quản lý lịch.",
    "Nhóm chuyên môn sẽ xem lại hồ sơ khi có dữ liệu mới hoặc khi diễn biến không theo dự kiến.",
    "Việc bàn giao bao gồm tình trạng chung, mục tiêu ngắn hạn và những nội dung còn chờ xác minh.",
    "Người ghi hồ sơ xác nhận đã kiểm tra tính đầy đủ trước khi ký điện tử và chuyển bước xử lý.",
    "Thông tin do thân nhân cung cấp được đánh dấu nguồn, tránh trình bày như kết luận của nhân viên y tế.",
    "Khi trao đổi qua điện thoại, nhân viên đọc lại nội dung quan trọng để người nhận xác nhận.",
    "Các biểu mẫu không áp dụng cho trường hợp này được để trống có chủ đích thay vì điền dữ liệu suy đoán.",
    "Nhân viên trực sau có trách nhiệm rà soát mục chưa hoàn tất ngay khi bắt đầu ca làm việc.",
    "Tài liệu đính kèm được đặt tên theo mã hồ sơ và ngày thực hiện để thuận tiện truy vết.",
    "Quyết định tiếp theo chỉ được đưa ra sau khi thông tin bắt buộc đã được đối chiếu đầy đủ.",
    "Người bệnh được nhắc giữ lại bản tóm tắt và xuất trình khi đến cơ sở y tế khác.",
    "Các trao đổi giữa nhiều chuyên khoa được tổng hợp thành một bản thống nhất trong hồ sơ chính.",
    "Thời điểm đánh giá lại được lựa chọn dựa trên mức ổn định và khả năng quay lại của người bệnh.",
    "Dữ liệu nhận từ hệ thống khác được ghi rõ ngày truy xuất và không thay thế bản gốc.",
    "Người phụ trách kiểm tra rằng kế hoạch sau cùng phù hợp với điều kiện thực hiện tại nơi cư trú.",
    "Nếu thông tin còn thiếu, hồ sơ nêu rõ nội dung cần bổ sung thay vì tự động suy diễn.",
)


def _canonical_icd(code: str) -> str:
    return re.sub(r"[†*]", "", str(code)).strip()


@lru_cache(maxsize=4)
def _load_ontology(kb_path: Path, artifact_root: Path) -> dict[str, Any]:
    with gzip.open(artifact_root / "icd10" / "icd10_dictionary.jsonl.gz", "rt", encoding="utf-8") as stream:
        icd_artifact_ids = {str(json.loads(line)["candidate_id"]) for line in stream if line.strip()}
    with gzip.open(artifact_root / "rxnorm" / "rxnorm_dictionary.jsonl.gz", "rt", encoding="utf-8") as stream:
        rx_artifact_ids = {str(json.loads(line)["candidate_id"]) for line in stream if line.strip()}

    connection = sqlite3.connect(kb_path)
    icd: dict[str, str] = {}
    for raw_code, name_vi, name_en in connection.execute("select code, name_vi, name_en from icd10"):
        code = _canonical_icd(str(raw_code))
        name = str(name_vi or name_en or "").strip()
        if code in icd_artifact_ids and name and code not in icd:
            icd[code] = name

    rx_rows: dict[str, list[tuple[str, str]]] = {}
    for rxcui, name, tty in connection.execute("select rxcui, name, tty from rxnorm"):
        candidate_id = str(rxcui)
        if candidate_id in rx_artifact_ids and name:
            rx_rows.setdefault(candidate_id, []).append((str(name).strip(), str(tty or "")))
    preferred_tty = {"IN": 0, "PIN": 1, "BN": 2, "SCD": 3, "SBD": 4}
    rx = {
        candidate_id: sorted(rows, key=lambda row: (preferred_tty.get(row[1], 9), len(row[0]), row[0]))[0][0]
        for candidate_id, rows in rx_rows.items()
    }

    sex_rules = {
        _canonical_icd(str(code)): str(allowed)
        for code, allowed in connection.execute("select code, allowed_sex from icd10_rules_sex")
    }
    age_rules = {
        _canonical_icd(str(code)): (int(min_days), int(max_days))
        for code, min_days, max_days in connection.execute(
            "select code, min_days, max_days from icd10_rules_age"
        )
    }
    excluded_codes = {
        _canonical_icd(str(code))
        for (code,) in connection.execute("select code from icd10_rules_not_primary")
    }
    for dagger, asterisk in connection.execute("select dagger_code, asterisk_code from icd10_rules_dual"):
        excluded_codes.update({_canonical_icd(str(dagger)), _canonical_icd(str(asterisk))})
    connection.close()

    suspicious_drug_terms = (
        "veterinary", "animal", "extract", "venom", "insecticide", "pollen",
        "allergen", "feather", "cockroach", "mouse", "rat ", "dog ", "cat ",
    )
    longtail_rx = sorted(
        candidate_id
        for candidate_id, name in rx.items()
        if 4 <= len(name) <= 70
        and not any(term in name.casefold() for term in suspicious_drug_terms)
        and candidate_id not in {drug for profile in PROFILES.values() for drug in profile["drugs"]}
    )
    common_codes = {str(profile["dx"]) for profile in PROFILES.values()}
    longtail_icd = sorted(
        code
        for code, name in icd.items()
        if code not in common_codes
        and code not in excluded_codes
        and not code.startswith(("O", "P", "Q", "R", "V", "W", "X", "Y", "Z"))
        and 8 <= len(name) <= 140
    )
    return {
        "icd": icd,
        "rx": rx,
        "sex_rules": sex_rules,
        "age_rules": age_rules,
        "longtail_icd": longtail_icd,
        "longtail_rx": longtail_rx,
    }


def _add(
    text_parts: list[str],
    entities: list[dict[str, Any]],
    text: str,
    kind: str,
    assertions: Iterable[str] = (),
    candidates: Iterable[str] = (),
) -> None:
    start = sum(len(part) for part in text_parts)
    text_parts.append(text)
    item: dict[str, Any] = {"text": text, "type": kind, "position": [start, start + len(text)]}
    if kind in {DISEASE, DRUG, SYMPTOM}:
        item["assertions"] = list(assertions)
    if kind in {DISEASE, DRUG}:
        item["candidates"] = list(candidates)
    entities.append(item)


def _block(render) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    entities: list[dict[str, Any]] = []
    render(parts, entities)
    return "".join(parts), entities


def _eligible_longtail_codes(ontology: dict[str, Any], age: int, sex: str) -> list[str]:
    sex_code = "F" if sex == "nữ" else "M"
    age_days = age * 365
    eligible: list[str] = []
    for code in ontology["longtail_icd"]:
        allowed_sex = ontology["sex_rules"].get(code)
        if allowed_sex and allowed_sex != sex_code:
            continue
        age_range = ontology["age_rules"].get(code)
        if age_range and not (age_range[0] <= age_days <= age_range[1]):
            continue
        eligible.append(code)
    return eligible


def build_document(
    case_id: int,
    scenario_name: str,
    kb_path: Path,
    seed: int | None = None,
    genre: str | None = None,
    longtail_rank: int | None = None,
    artifact_root: Path | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    rng = random.Random((seed or 20260722) + case_id * 1009)
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    artifact_root = artifact_root or Path(__file__).resolve().parents[1] / "v2" / "artifacts"
    ontology = _load_ontology(Path(kb_path).resolve(), artifact_root.resolve())
    genre = genre or list(GENRES)[case_id % len(GENRES)]
    if genre not in GENRES:
        raise ValueError(f"Unknown genre: {genre}")

    profile_keys = list(SCENARIOS[scenario_name])
    if genre == "thai_nhi":
        profile_keys = ["pregnancy_diabetes", "pediatric_asthma"]
    profile_key = rng.choice(profile_keys)
    profile = PROFILES[profile_key]
    age = rng.randint(*profile["age"])
    sex = rng.choice(profile["sex"])
    current_code = str(profile["dx"])
    history_code = str(profile["history"])
    positive_symptoms = rng.sample(
        list(profile["symptoms"]),
        k=rng.choice((2, 2, 3)),
    )
    negative_count = rng.choice((0, 1, 1, 2))
    negative_symptoms = rng.sample(list(profile["negatives"]), k=negative_count)
    lab_pairs = list(profile["labs"])
    rng.shuffle(lab_pairs)
    drug_ids = list(profile["drugs"])
    rng.shuffle(drug_ids)
    drug_ids = drug_ids[: rng.randint(1, len(drug_ids))] if drug_ids else []

    longtail_code: str | None = None
    longtail_drug: str | None = None
    if scenario_name == "longtail":
        eligible_codes = _eligible_longtail_codes(ontology, age, sex)
        if eligible_codes:
            rank = longtail_rank if longtail_rank is not None else rng.randrange(len(eligible_codes))
            longtail_code = eligible_codes[(rank * 37 + case_id) % len(eligible_codes)]
        if ontology["longtail_rx"]:
            rank = longtail_rank if longtail_rank is not None else rng.randrange(len(ontology["longtail_rx"]))
            longtail_drug = ontology["longtail_rx"][(rank * 53 + case_id) % len(ontology["longtail_rx"])]

    title, reason_label = GENRES[genre]
    header = (
        f"{title}\n"
        f"Mã hồ sơ HS-{case_id:04d}. Bệnh nhân {sex} {age} tuổi. "
        f"Tài liệu được lập tại lượt ghi nhận {1 + case_id % 4}, ca {1 + case_id % 3}.\n"
    )

    blocks: dict[str, tuple[str, list[dict[str, Any]]]] = {}

    def render_reason(parts, entities):
        parts.append(f"{reason_label}: người bệnh trình bày ")
        for index, symptom in enumerate(positive_symptoms):
            if index:
                parts.append(rng.choice((" kèm ", "; đồng thời có ", " và sau đó xuất hiện ")))
            _add(parts, entities, symptom, SYMPTOM)
        parts.append(rng.choice((" trong những ngày gần đây.\n", " trước thời điểm đánh giá.\n", " với mức độ dao động.\n")))
    blocks["reason"] = _block(render_reason)

    def render_history(parts, entities):
        parts.append(rng.choice(("Bệnh sử trước đây ghi nhận ", "Trong danh sách vấn đề cũ có ", "Tiền sử cá nhân có ")))
        _add(parts, entities, ontology["icd"][history_code].casefold(), DISEASE, ("isHistorical",), (history_code,))
        parts.append("; thông tin này được giữ riêng với đánh giá của lần hiện tại. ")
        if rng.random() < 0.25:
            parts.append("Mẹ của người bệnh từng được chẩn đoán ")
            _add(parts, entities, ontology["icd"][current_code].casefold(), DISEASE, ("isFamily",), (current_code,))
            parts.append(" theo lời kể của gia đình. ")
        parts.append("Nguồn khai thác đã được ghi rõ trong hồ sơ.\n")
    blocks["history"] = _block(render_history)

    def render_negative(parts, entities):
        if not negative_symptoms:
            parts.append("Khai thác có định hướng chưa ghi nhận thêm biểu hiện cảnh báo liên quan.\n")
            return
        parts.append(rng.choice(("Hiện người bệnh phủ nhận ", "Khi hỏi trực tiếp, chưa ghi nhận ", "Đánh giá hiện tại không có ")))
        for index, symptom in enumerate(negative_symptoms):
            if index:
                parts.append(" và không có ")
            _add(parts, entities, symptom, SYMPTOM, ("isNegated",))
        parts.append(". Phạm vi phủ định chỉ áp dụng cho câu này.\n")
    blocks["negative"] = _block(render_negative)

    def render_lab(parts, entities):
        parts.append(rng.choice(("Cận lâm sàng đã thực hiện gồm ", "Dữ liệu hỗ trợ hiện có là ", "Kết quả được đối chiếu từ ")))
        for index, (lab_name, lab_result) in enumerate(lab_pairs):
            if index:
                parts.append("; ngoài ra, ")
            _add(parts, entities, lab_name, LAB_NAME)
            parts.append(rng.choice((", ghi nhận ", " với kết luận ", "; kết quả mô tả ")))
            _add(parts, entities, lab_result, LAB_RESULT)
        parts.append(". Thời điểm lấy mẫu hoặc thực hiện được lưu cùng bản gốc.\n")
    blocks["lab"] = _block(render_lab)

    def render_assessment(parts, entities):
        parts.append(rng.choice(("Đánh giá hiện tại phù hợp với ", "Chẩn đoán làm việc của lượt này là ", "Sau khi tổng hợp dữ liệu, ghi nhận ")))
        _add(parts, entities, ontology["icd"][current_code].casefold(), DISEASE, (), (current_code,))
        parts.append(". Nhận định này không thay đổi trạng thái của các vấn đề chỉ xuất hiện trong tiền sử.\n")
        if longtail_code:
            parts.append("Đối chiếu mã hóa từ hồ sơ cũ còn có ")
            _add(
                parts,
                entities,
                ontology["icd"][longtail_code].casefold(),
                DISEASE,
                ("isHistorical",),
                (longtail_code,),
            )
            parts.append("; lượt này không có đủ tài liệu để xác định vấn đề đó còn hoạt động.\n")
    blocks["assessment"] = _block(render_assessment)

    def render_medication(parts, entities):
        if drug_ids:
            parts.append(rng.choice(("Kế hoạch dùng thuốc của lượt này gồm ", "Điều trị được ghi nhận với ", "Danh sách thuốc đang áp dụng có ")))
            for index, candidate_id in enumerate(drug_ids):
                if index:
                    parts.append(" và ")
                _add(parts, entities, ontology["rx"][candidate_id], DRUG, (), (candidate_id,))
            parts.append("; liều và đường dùng phải theo y lệnh đã ký.\n")
        else:
            parts.append("Lượt ghi nhận này chưa bổ sung một thuốc điều trị đặc hiệu vào phần kế hoạch.\n")
        if longtail_drug:
            parts.append("Danh sách thuốc mang theo khi tiếp nhận có ")
            _add(parts, entities, ontology["rx"][longtail_drug], DRUG, (), (longtail_drug,))
            parts.append("; thông tin hiện có không dùng để suy diễn chỉ định hoặc liên hệ thuốc này với chẩn đoán hiện tại.\n")
    blocks["medication"] = _block(render_medication)

    def render_plan(parts, entities):
        parts.append(
            rng.choice(
                (
                    "Kế hoạch tiếp theo ưu tiên đánh giá lại theo mốc đã thống nhất và cập nhật khi có dữ liệu mới.\n",
                    "Nhóm chăm sóc thống nhất theo dõi diễn biến, rà soát kết quả còn chờ và bàn giao rõ trách nhiệm.\n",
                    "Hướng xử lý được ghi theo từng mục, kèm thời điểm xem lại và tiêu chí liên hệ sớm hơn.\n",
                )
            )
        )
    blocks["plan"] = _block(render_plan)

    text_parts = [header]
    entities: list[dict[str, Any]] = []
    for block_name in GENRE_ORDERS[genre]:
        block_text, block_entities = blocks[block_name]
        offset = sum(len(part) for part in text_parts)
        text_parts.append(block_text)
        for entity in block_entities:
            shifted = dict(entity)
            shifted["position"] = [entity["position"][0] + offset, entity["position"][1] + offset]
            entities.append(shifted)

    target_words = rng.randint(330, 470)
    neutral = list(NEUTRAL_SENTENCES)
    rng.shuffle(neutral)
    selected: list[str] = []
    while len("".join(text_parts).split()) + sum(len(sentence.split()) for sentence in selected) < target_words:
        if neutral:
            selected.append(neutral.pop())
        else:
            selected.append(
                f"Ghi chú bổ sung của hồ sơ {case_id:04d} được rà soát ở vòng {len(selected) + 1} và chưa làm thay đổi kết luận đã nêu."
            )
    text_parts.append("Theo dõi và phối hợp: " + " ".join(selected) + "\n")

    full_text = "".join(text_parts)
    entities.sort(key=lambda entity: (entity["position"][0], entity["position"][1], entity["type"]))
    for entity in entities:
        start, end = entity["position"]
        if full_text[start:end] != entity["text"]:
            raise ValueError(f"Offset mismatch in case {case_id}: {entity}")
    return full_text, entities


def generate_dataset(
    kb_path: Path,
    output_root: Path,
    count: int = 2000,
    seed: int = 20260722,
    start_id: int = 1,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "input").mkdir(exist_ok=True)
    (output_root / "gt").mkdir(exist_ok=True)
    (output_root / "reports").mkdir(exist_ok=True)
    scenario_names = [name for name in SCENARIOS if name != "longtail"]
    genres = list(GENRES)
    genre_counts: dict[str, int] = {name: 0 for name in genres}
    manifest: list[dict[str, Any]] = []
    longtail_rank = 0
    for case_id in range(start_id, start_id + count):
        relative_id = case_id - start_id + 1
        scenario = "longtail" if relative_id % 5 == 0 else scenario_names[(relative_id - 1) % len(scenario_names)]
        genre = genres[(relative_id - 1) % len(genres)]
        genre_counts[genre] += 1
        rank = longtail_rank if scenario == "longtail" else None
        text, entities = build_document(
            case_id,
            scenario,
            kb_path,
            seed=seed,
            genre=genre,
            longtail_rank=rank,
            artifact_root=artifact_root,
        )
        if scenario == "longtail":
            longtail_rank += 1
        (output_root / "input" / f"{case_id}.txt").write_text(text, encoding="utf-8")
        (output_root / "gt" / f"{case_id}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "document_id": str(case_id),
                "source_bucket": "synthetic",
                "genre": genre,
                "scenario": scenario,
                "template_group": f"{genre}:{scenario}:{case_id % 7}",
                "long_tail": scenario == "longtail",
                "train_eligible": True,
                "linking_train_eligible": True,
                "train_exclusion_reason": None,
                "primary_candidates": sorted(
                    {
                        candidate
                        for entity in entities
                        if entity["type"] in {DISEASE, DRUG}
                        for candidate in entity.get("candidates", [])
                    }
                ),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    (output_root / "reports" / "genre_manifest.json").write_text(
        json.dumps(genre_counts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "reports" / "dataset_manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in manifest),
        encoding="utf-8",
    )
    return {
        "count": count,
        "longtail_count": longtail_rank,
        "scenarios": len(SCENARIOS),
        "genres": genre_counts,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        generate_dataset(
            root / "data" / "kb" / "metadata.db",
            root / "data_v2" / "Training_data" / "synthetic_train_v2",
            count=2000,
            start_id=201,
        )
    )
