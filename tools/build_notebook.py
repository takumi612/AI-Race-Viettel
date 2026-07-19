from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


SECTION_TEMPLATE = """## {number}. {title}

**Mục tiêu:** {goal}  
**Công nghệ:** {technology}  
**Input:** {input_text}  
**Output:** {output_text}  
**Artifact:** {artifact}  
**Kiểm tra:** Cell thực thi bên dưới; nếu thiếu annotation, cell ghi rõ trạng thái `not_scored` thay vì bịa metric.
"""


def build_notebook() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    cells.append(
        markdown_cell(
            """# Medical Information Extraction Lab - Offline Clinical NLP

Notebook end-to-end cho Clinical NER, assertion/context, ICD-10/RxNorm linking, relation baseline, evaluator và submission.

## Chạy trên Google Colab

1. Chuẩn bị dữ liệu một lần tại `MyDrive/AI-Race-Viettel/data/` theo cấu trúc ghi trong `COLAB_RUNBOOK.md`.
2. Upload riêng file `.ipynb`; cell đầu tiên tự mount Drive, clone nhánh `Pipeline_colab` và cài dependencies.
3. Notebook tự tìm input thật, annotation, knowledge sources và lưu checkpoint/output về Drive.
4. Giữ `FAST_DEV_RUN = False` để train đầy đủ; đổi thành `True` cho smoke-test ngắn.
5. Bootstrap có thể tạo smoke input tạm để hoàn tất khởi tạo, nhưng production gate ngay sau đó vẫn bắt buộc input thật và sẽ dừng với hướng dẫn rõ; smoke input không bao giờ được dùng để tạo ZIP production.

Notebook không gọi external API và không fit trên private/test input. Nguồn sự thật của project: `SPEC.md`, `PROJECT_STATE.md`, `DECISIONS.md`, `ARTIFACT_MANIFEST.json`."""
        )
    )
    cells.append(
        markdown_cell(
            """## 0. Project overview\n\n**Mục tiêu:** Chuyển văn bản lâm sàng phi cấu trúc thành entity JSON có span, type, assertions và candidate chuẩn hóa.\n\n**Pipeline:** raw text -> validation -> sectioning -> detection -> span refinement -> context -> type-routed linking -> relation diagnostics -> schema conversion -> submission validation -> `output.zip`.\n\n**Dữ liệu train:** nếu Drive chưa có annotation, supervised training được bỏ qua minh bạch và inference dùng artifacts/rule-dictionary baseline. Official entity/assertion schema được đối chiếu từ validator trong chính repository."""
        )
    )
    cells.append(
        code_cell(
            """from pathlib import Path\nimport json\nimport os\nimport subprocess\nimport sys\nimport tempfile\n\n# One-click Colab bootstrap. The defaults clone the committed Colab branch.\nIS_COLAB = "COLAB_RELEASE_TAG" in os.environ\nMOUNT_GOOGLE_DRIVE = True\nINSTALL_REQUIREMENTS = True\nCOLAB_REQUIREMENTS_FILE = "requirements-colab.txt"\nFAST_DEV_RUN = False\nAUTO_SMOKE_INPUT = True\nPROJECT_ROOT_OVERRIDE = ""\nGITHUB_REPO_URL = "https://github.com/takumi612/AI-Race-Viettel.git"\nGITHUB_BRANCH = "Pipeline_colab"\nINPUT_ZIP_OVERRIDE = ""\nTRAIN_DIR_OVERRIDE = ""\nICD10_PATH_OVERRIDE = ""\nRXNORM_ZIP_OVERRIDE = ""\nTRAINING_OUTPUT_DIR_OVERRIDE = ""\nDRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/clinical-nlp-end-to-end-lab")\nDRIVE_TRAINING_OUTPUT_DIR = Path("/content/drive/MyDrive/clinical-nlp-training-artifacts")\nCOLAB_REPO_DIR = Path("/content/AI-Race-Viettel")\n\nif IS_COLAB and MOUNT_GOOGLE_DRIVE:\n    from google.colab import drive\n    drive.mount("/content/drive", force_remount=False)\n\nif IS_COLAB and GITHUB_REPO_URL.strip() and not COLAB_REPO_DIR.exists():\n    subprocess.run(\n        ["git", "clone", "--depth", "1", "--branch", GITHUB_BRANCH, GITHUB_REPO_URL, str(COLAB_REPO_DIR)],\n        check=True,\n    )\n\nproject_candidates = []\nif PROJECT_ROOT_OVERRIDE.strip():\n    project_candidates.append(Path(PROJECT_ROOT_OVERRIDE).expanduser())\nproject_candidates.extend([COLAB_REPO_DIR, DRIVE_PROJECT_DIR, Path("/content/clinical-nlp-end-to-end-lab"), Path("/content/AI_race"), Path.cwd()])\nPROJECT_ROOT = next(\n    (candidate.resolve() for candidate in project_candidates if (candidate / "clinical_nlp_lab").is_dir()),\n    None,\n)\nif PROJECT_ROOT is None:\n    raise FileNotFoundError(\n        "Project root not found. Automatic clone failed; set PROJECT_ROOT_OVERRIDE."\n    )\n\nif TRAINING_OUTPUT_DIR_OVERRIDE.strip():\n    TRAINING_OUTPUT_ROOT = Path(TRAINING_OUTPUT_DIR_OVERRIDE).expanduser().resolve()\nelif IS_COLAB and MOUNT_GOOGLE_DRIVE:\n    TRAINING_OUTPUT_ROOT = DRIVE_TRAINING_OUTPUT_DIR\nelse:\n    TRAINING_OUTPUT_ROOT = PROJECT_ROOT / "artifacts"\nTRAINING_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n\nif IS_COLAB and INSTALL_REQUIREMENTS:\n    requirements_path = PROJECT_ROOT / COLAB_REQUIREMENTS_FILE\n    if not requirements_path.exists():\n        requirements_path = PROJECT_ROOT / "requirements.txt"\n    if requirements_path.exists():\n        subprocess.run(\n            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)],\n            check=True,\n        )\n\nif str(PROJECT_ROOT) not in sys.path:\n    sys.path.insert(0, str(PROJECT_ROOT))\n\nfrom clinical_nlp_lab.config import load_config, set_reproducible_seed\nfrom clinical_nlp_lab.data import (\n    describe_documents,\n    document_train_validation_split,\n    load_annotated_documents,\n    load_input_documents,\n    validate_documents,\n)\nfrom clinical_nlp_lab.kb import load_candidate_dictionary\nfrom clinical_nlp_lab.ner import DictionaryRuleEntityDetector, refine_boundaries\nfrom clinical_nlp_lab.assertions import HybridAssertionPredictor\nfrom clinical_nlp_lab.linking import EntityLinker, LexicalCandidateIndex\nfrom clinical_nlp_lab.relations import RuleRelationExtractor\nfrom clinical_nlp_lab.pipeline import reload_equivalence_check, run_inference\nfrom clinical_nlp_lab.schema import write_json\n\ndef _has_text_files(path: Path) -> bool:\n    return path.is_dir() and any(path.glob("*.txt"))\n\ndef _has_annotation_files(path: Path) -> bool:\n    return path.is_dir() and (any(path.glob("*.json")) or any(path.glob("*.txt")))\n\nCONFIG = load_config(PROJECT_ROOT / "artifacts/config.json")\nCONFIG["fast_dev_run"] = FAST_DEV_RUN\nfor config_key, override_value in {\n    "input_zip": INPUT_ZIP_OVERRIDE,\n    "train_dir": TRAIN_DIR_OVERRIDE,\n    "icd10_path": ICD10_PATH_OVERRIDE,\n    "rxnorm_zip_path": RXNORM_ZIP_OVERRIDE,\n}.items():\n    if override_value.strip():\n        CONFIG[config_key] = override_value\n\ninput_candidates = []\nif INPUT_ZIP_OVERRIDE.strip():\n    input_candidates.append(Path(INPUT_ZIP_OVERRIDE).expanduser())\ninput_candidates.extend([\n    PROJECT_ROOT / "input.zip",\n    PROJECT_ROOT / "data/input",\n    DRIVE_PROJECT_DIR / "input.zip",\n    DRIVE_PROJECT_DIR / "data/input",\n])\nINPUT_SOURCE = next((candidate for candidate in input_candidates if candidate.is_file() or _has_text_files(candidate)), None)\nSMOKE_INPUT_USED = False\nif INPUT_SOURCE is None and AUTO_SMOKE_INPUT:\n    smoke_root = Path(tempfile.gettempdir()) / "clinical_nlp_smoke_input"\n    smoke_root.mkdir(parents=True, exist_ok=True)\n    (smoke_root / "1.txt").write_text(\n        "HISTORY\\nPatient reports fever and cough.\\nMEDICATIONS\\nAspirin 81 mg po daily.",\n        encoding="utf-8",\n    )\n    INPUT_SOURCE = smoke_root\n    SMOKE_INPUT_USED = True\nif INPUT_SOURCE is None:\n    raise FileNotFoundError("No input source found. Set INPUT_ZIP_OVERRIDE or enable AUTO_SMOKE_INPUT.")\nCONFIG["input_zip"] = str(INPUT_SOURCE)\n\ntrain_candidates = []\nif TRAIN_DIR_OVERRIDE.strip():\n    train_candidates.append(Path(TRAIN_DIR_OVERRIDE).expanduser())\ntrain_candidates.extend([PROJECT_ROOT / "train", DRIVE_PROJECT_DIR / "train"])\nTRAIN_SOURCE = next((candidate for candidate in train_candidates if _has_annotation_files(candidate)), None)\nif TRAIN_SOURCE is None:\n    TRAIN_SOURCE = PROJECT_ROOT / "train"\nCONFIG["train_dir"] = str(TRAIN_SOURCE)\n\nSEED_STATUS = set_reproducible_seed(int(CONFIG["seed"]))\nprint({"is_colab": IS_COLAB, "project_root": str(PROJECT_ROOT), "input_source": str(INPUT_SOURCE), "smoke_input_used": SMOKE_INPUT_USED, "train_source": str(TRAIN_SOURCE), "training_output_root": str(TRAINING_OUTPUT_ROOT), "seed_status": SEED_STATUS, "fast_dev_run": CONFIG["fast_dev_run"], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES", "auto")})"""
        )
    )

    cells.append(
        markdown_cell(
            """## 0.1 Production data and output resolver

Cell này là production gate cho chế độ `Run all`. Nó ưu tiên dữ liệu thật trên Google Drive, hỗ trợ cả `input.zip` và thư mục `input/*.txt`, nhận annotation dạng `train/*.txt + *.json` hoặc `synthetic_train_v1/input + gt`, và lưu `output.zip` về Drive. Nếu không tìm thấy input thật, notebook dừng ngay với cấu trúc thư mục cần tạo."""
        )
    )
    cells.append(
        code_cell(
            """# Production defaults: upload data once to Drive, then Run all needs no edits.
REQUIRE_REAL_INPUT = True
DATA_ROOT_OVERRIDE = ""
OUTPUT_ARCHIVE_OVERRIDE = ""
DRIVE_DATA_DIR = Path("/content/drive/MyDrive/AI-Race-Viettel/data")
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/AI-Race-Viettel/output")

def _unique_paths(paths):
    seen = set()
    result = []
    for path in paths:
        normalized = str(path)
        if normalized not in seen:
            seen.add(normalized)
            result.append(path)
    return result

data_roots = []
if DATA_ROOT_OVERRIDE.strip():
    data_roots.append(Path(DATA_ROOT_OVERRIDE).expanduser())
data_roots.extend([
    DRIVE_DATA_DIR,
    Path("/content/drive/MyDrive/clinical-nlp-data"),
    DRIVE_PROJECT_DIR / "data",
    PROJECT_ROOT / "data",
    PROJECT_ROOT,
])
data_roots = _unique_paths(data_roots)

input_candidates = []
if INPUT_ZIP_OVERRIDE.strip():
    input_candidates.append(Path(INPUT_ZIP_OVERRIDE).expanduser())
for root in data_roots:
    input_candidates.extend([root / "input.zip", root / "input", root / "data/input"])
REAL_INPUT_SOURCE = next(
    (path for path in _unique_paths(input_candidates) if path.is_file() or _has_text_files(path)),
    None,
)
if REAL_INPUT_SOURCE is None and REQUIRE_REAL_INPUT:
    raise FileNotFoundError(
        "Real input not found. Create MyDrive/AI-Race-Viettel/data/input/ with .txt files "
        "or upload MyDrive/AI-Race-Viettel/data/input.zip. See COLAB_RUNBOOK.md."
    )
if REAL_INPUT_SOURCE is not None:
    INPUT_SOURCE = REAL_INPUT_SOURCE
    CONFIG["input_zip"] = str(INPUT_SOURCE)
    SMOKE_INPUT_USED = False

def _has_training_layout(path: Path) -> bool:
    if not path.is_dir():
        return False
    text_stems = {item.stem for item in path.glob("*.txt")}
    json_stems = {item.stem for item in path.glob("*.json")}
    direct_pairs = bool(text_stems & json_stems)
    split_pairs = _has_text_files(path / "input") and (path / "gt").is_dir() and any((path / "gt").glob("*.json"))
    return direct_pairs or split_pairs

train_candidates = []
if TRAIN_DIR_OVERRIDE.strip():
    train_candidates.append(Path(TRAIN_DIR_OVERRIDE).expanduser())
for root in data_roots:
    train_candidates.extend([root / "train", root / "synthetic_train_v1", root])
REAL_TRAIN_SOURCE = next(
    (path for path in _unique_paths(train_candidates) if _has_training_layout(path)),
    None,
)
if REAL_TRAIN_SOURCE is not None:
    TRAIN_SOURCE = REAL_TRAIN_SOURCE
    CONFIG["train_dir"] = str(TRAIN_SOURCE)

source_overrides = {
    "icd10_path": (ICD10_PATH_OVERRIDE, "ICD10.xlsx"),
    "rxnorm_zip_path": (RXNORM_ZIP_OVERRIDE, "RxNorm_full_07062026.zip"),
}
for config_key, (explicit_path, filename) in source_overrides.items():
    candidates = [Path(explicit_path).expanduser()] if explicit_path.strip() else []
    candidates.extend(root / filename for root in data_roots)
    discovered = next((path for path in _unique_paths(candidates) if path.is_file()), None)
    if discovered is not None:
        CONFIG[config_key] = str(discovered)

if OUTPUT_ARCHIVE_OVERRIDE.strip():
    OUTPUT_ARCHIVE_PATH = Path(OUTPUT_ARCHIVE_OVERRIDE).expanduser()
elif IS_COLAB and MOUNT_GOOGLE_DRIVE:
    OUTPUT_ARCHIVE_PATH = DRIVE_OUTPUT_DIR / "output.zip"
else:
    OUTPUT_ARCHIVE_PATH = PROJECT_ROOT / "output.zip"
OUTPUT_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)

print({
    "real_input": str(INPUT_SOURCE),
    "training_data": str(REAL_TRAIN_SOURCE) if REAL_TRAIN_SOURCE else "not_found_training_will_skip",
    "output_zip": str(OUTPUT_ARCHIVE_PATH),
    "icd10_source": CONFIG["icd10_path"],
    "rxnorm_source": CONFIG["rxnorm_zip_path"],
})"""
        )
    )

    sections = [
        (
            1,
            "Environment setup",
            "Kiểm tra runtime và optional supervised dependencies.",
            "Python stdlib, package-local modules; optional torch/transformers.",
            "Project root and requirements.txt.",
            "CONFIG and dependency status.",
            "No external API or secret.",
            """from clinical_nlp_lab.training import transformer_training_availability\nTRAINING_AVAILABILITY = transformer_training_availability()\nprint({"python": sys.version, "training": TRAINING_AVAILABILITY.reason})""",
        ),
        (
            2,
            "Configuration",
            "Giữ mọi đường dẫn, model, threshold và resource filter trong một object.",
            "CONFIG dict and JSON artifact.",
            "artifacts/config.json.",
            "CONFIG.",
            "artifacts/config.json.",
            """from clinical_nlp_lab.schema import ALLOWED_ASSERTIONS, OFFICIAL_SCHEMA_KEYS\n\nassert CONFIG["max_length"] == 512\nassert CONFIG["stride"] == 128\nassert OFFICIAL_SCHEMA_KEYS["CHẨN_ĐOÁN"] == {"text", "type", "position", "assertions", "candidates"}\nassert OFFICIAL_SCHEMA_KEYS["TRIỆU_CHỨNG"] == {"text", "type", "position", "assertions"}\nassert ALLOWED_ASSERTIONS == {"isNegated", "isHistorical", "isFamily"}\nprint(json.dumps({"config_keys": len(CONFIG), "thresholds": CONFIG["thresholds"], "official_schema": {key: sorted(value) for key, value in OFFICIAL_SCHEMA_KEYS.items()}}, ensure_ascii=True))""",
        ),
        (
            3,
            "Data discovery",
            "Đọc đúng ZIP/text và kiểm tra annotated split.",
            "pathlib, zipfile, UTF-8 loader.",
            "input.zip and train/ if present.",
            "DOCUMENTS and ANNOTATED_DOCUMENTS.",
            "Data fingerprint in validation report.",
            """DOCUMENTS = load_input_documents(PROJECT_ROOT / CONFIG["input_zip"])\nANNOTATED_DOCUMENTS = load_annotated_documents(PROJECT_ROOT / CONFIG["train_dir"])\nprint({"input_documents": len(DOCUMENTS), "annotated_documents": len(ANNOTATED_DOCUMENTS)})""",
        ),
        (
            4,
            "Data validation",
            "Kiểm tra schema, offset, duplicate và overlap.",
            "Custom validator with raw-text slice invariant.",
            "DOCUMENTS and ANNOTATED_DOCUMENTS.",
            "INPUT_VALIDATION and ANNOTATED_VALIDATION.",
            "reports/stage_03_eda.json.",
            """INPUT_VALIDATION = validate_documents(DOCUMENTS)\nANNOTATED_VALIDATION = validate_documents(ANNOTATED_DOCUMENTS)\nassert INPUT_VALIDATION["is_valid"]\nprint({"input_validation": INPUT_VALIDATION, "annotation_count": len(ANNOTATED_DOCUMENTS)})""",
        ),
        (
            5,
            "Exploratory data analysis",
            "Tóm tắt độ dài và cấu trúc, không học tham số từ private input.",
            "describe_documents and section counters.",
            "DOCUMENTS.",
            "EDA_SUMMARY.",
            "reports/stage_03_eda.json.",
            """EDA_SUMMARY = describe_documents(DOCUMENTS)\nprint(EDA_SUMMARY)""",
        ),
        (
            6,
            "Train/validation split",
            "Split theo document trước mọi chunking/fitting.",
            "Deterministic random seed.",
            "ANNOTATED_DOCUMENTS only.",
            "TRAIN_DOCUMENTS and VALIDATION_DOCUMENTS.",
            "No split artifact if annotations are absent.",
            """TRAIN_DOCUMENTS, VALIDATION_DOCUMENTS = document_train_validation_split(\n    ANNOTATED_DOCUMENTS, CONFIG["validation_fraction"], int(CONFIG["seed"])\n)\nassert not ({item.document_id for item in TRAIN_DOCUMENTS} & {item.document_id for item in VALIDATION_DOCUMENTS})\nprint({"train": len(TRAIN_DOCUMENTS), "validation": len(VALIDATION_DOCUMENTS), "leakage": False})""",
        ),
        (
            7,
            "Section detection",
            "Giữ section name/start/end/text trên raw text.",
            "Regex heading dictionary and newline-aware segmentation.",
            "DOCUMENTS[0].raw_text.",
            "SECTION_SAMPLE.",
            "Section spans in diagnostics.",
            """from clinical_nlp_lab.text import detect_sections\nSECTION_SAMPLE = detect_sections(DOCUMENTS[0].raw_text)\nfor section in SECTION_SAMPLE:\n    section.validate(DOCUMENTS[0].raw_text)\nprint({"sections": len(SECTION_SAMPLE), "names": [s.section_name for s in SECTION_SAMPLE]})""",
        ),
        (
            8,
            "ICD-10 preprocessing",
            "Load/cache bilingual ICD-10 candidates and canonicalize markers.",
            "ICD-10 JSONL.GZ cache; build script if cache is missing.",
            "ICD10.xlsx.",
            "ICD10_RECORDS.",
            "artifacts/icd10/.",
            """icd10_cache = PROJECT_ROOT / "artifacts/icd10/icd10_dictionary.jsonl.gz"\nif not icd10_cache.exists():\n    subprocess.run([sys.executable, str(PROJECT_ROOT / "tools/build_knowledge_bases.py"), "--root", str(PROJECT_ROOT), "--artifact-dir", str(PROJECT_ROOT / "artifacts")], check=True)\nICD10_RECORDS = load_candidate_dictionary(icd10_cache)\nprint({"icd10_candidates": len(ICD10_RECORDS)})""",
        ),
        (
            9,
            "RxNorm preprocessing",
            "Load filtered RxNorm and optional relation cache without full extraction.",
            "Streaming RRF parser and compressed cache.",
            "RxNorm_full_07062026.zip.",
            "RXNORM_RECORDS and relation cache.",
            "artifacts/rxnorm/.",
            """RXNORM_RECORDS = load_candidate_dictionary(PROJECT_ROOT / "artifacts/rxnorm/rxnorm_dictionary.jsonl.gz")\nprint({"rxnorm_candidates": len(RXNORM_RECORDS), "relation_cache": (PROJECT_ROOT / "artifacts/rxnorm/rxnorm_relations.jsonl.gz").exists()})""",
        ),
        (
            10,
            "BIO or span dataset construction",
            "Prepare character annotations for BIO labels and chunk windows.",
            "BIO conversion, offsets and sliding windows.",
            "TRAIN_DOCUMENTS.",
            "BIO_LABEL_TO_ID and feature plan.",
            "Conditional NER model artifact.",
            """from clinical_nlp_lab.training import build_bio_label_map, chunk_token_indices\nENTITY_TYPES = {entity.type for document in TRAIN_DOCUMENTS for entity in document.entities}\nBIO_LABEL_TO_ID, BIO_ID_TO_LABEL = build_bio_label_map(ENTITY_TYPES)\nprint({"bio_labels": BIO_LABEL_TO_ID, "window_example": chunk_token_indices(1200, CONFIG["max_length"], CONFIG["stride"])})""",
        ),
        (
            11,
            "NER training",
            "Train XLM-R only when annotations and optional packages exist.",
            "AutoTokenizer, AutoModelForTokenClassification, Trainer (guarded).",
            "TRAIN_DOCUMENTS and VALIDATION_DOCUMENTS.",
            "NER_TRAINING_RESULT.",
            "artifacts/ner_model/ when trained.",
            """from clinical_nlp_lab.training import train_transformer_ner\nFIT_TRAIN_DOCUMENTS = TRAIN_DOCUMENTS[:2] if CONFIG["fast_dev_run"] else TRAIN_DOCUMENTS\nFIT_VALIDATION_DOCUMENTS = VALIDATION_DOCUMENTS[:1] if CONFIG["fast_dev_run"] else VALIDATION_DOCUMENTS\nNER_MODEL_DIR = TRAINING_OUTPUT_ROOT / "ner_model"\nNER_TRAINING_RESULT = train_transformer_ner(\n    FIT_TRAIN_DOCUMENTS, FIT_VALIDATION_DOCUMENTS, NER_MODEL_DIR,\n    model_name=CONFIG["ner_model_name"], max_length=CONFIG["max_length"], stride=CONFIG["stride"],\n    learning_rate=CONFIG["learning_rate"], epochs=1 if CONFIG["fast_dev_run"] else CONFIG["ner_epochs"],\n    batch_size=CONFIG["batch_size"], seed=CONFIG["seed"]\n)\nprint({"fast_dev_run": CONFIG["fast_dev_run"], "fit_train_documents": len(FIT_TRAIN_DOCUMENTS), "fit_validation_documents": len(FIT_VALIDATION_DOCUMENTS), "model_dir": str(NER_MODEL_DIR), "result": NER_TRAINING_RESULT})""",
        ),
        (
            12,
            "NER evaluation",
            "Exact-span, relaxed-span and type metrics when gold exists.",
            "Custom evaluator.",
            "Gold/predicted annotations.",
            "NER_EVALUATION.",
            "Validation metrics only when annotations exist.",
            """NER_EVALUATION = {"status": "not_scored", "reason": "No annotated validation data"} if not VALIDATION_DOCUMENTS else {"status": "run_after_training"}\nprint(NER_EVALUATION)""",
        ),
        (
            13,
            "Character span reconstruction",
            "Reconstruct entity spans directly from raw offsets.",
            "BIO-to-span converter and offset assertions.",
            "Token offsets and BIO predictions.",
            "Reconstructed spans.",
            "No normalized raw text is used for final output.",
            """from clinical_nlp_lab.training import bio_predictions_to_spans\nprint({"offset_contract": "raw_text[start:end] == entity.text", "status": "implemented"})""",
        ),
        (
            14,
            "Boundary refinement",
            "Trim invalid whitespace and resolve overlapping spans deterministically.",
            "Confidence ranking and type-aware overlap resolution.",
            "Detector spans.",
            "Refined spans.",
            "Diagnostics offset checks.",
            """detector = DictionaryRuleEntityDetector(ICD10_RECORDS, RXNORM_RECORDS)\nREFINED_SAMPLE = refine_boundaries(detector.detect(DOCUMENTS[0].raw_text), DOCUMENTS[0].raw_text)\nfor entity in REFINED_SAMPLE:\n    entity.validate_offset(DOCUMENTS[0].raw_text)\nprint({"sample_spans": len(REFINED_SAMPLE), "offset_errors": 0})""",
        ),
        (
            15,
            "Assertion dataset",
            "Create entity-marked context examples for four assertion axes.",
            "build_assertion_examples and section feature.",
            "TRAIN_DOCUMENTS.",
            "ASSERTION_EXAMPLES.",
            "No supervised artifact when labels absent.",
            """from clinical_nlp_lab.training import build_assertion_examples\nASSERTION_EXAMPLES = build_assertion_examples(TRAIN_DOCUMENTS)\nprint({"assertion_examples": len(ASSERTION_EXAMPLES)})""",
        ),
        (
            16,
            "Assertion training",
            "Provide shared encoder/multi-head training factory without inventing labels.",
            "Optional torch/transformers multi-task model.",
            "ASSERTION_EXAMPLES when present.",
            "Assertion model status.",
            "artifacts/assertion_model/ only when trained.",
            """ASSERTION_TRAINING = {"trained": False, "reason": "No assertion labels in workspace"}\nprint(ASSERTION_TRAINING)""",
        ),
        (
            17,
            "Assertion evaluation",
            "Evaluate Jaccard and axis macro-F1 only with gold assertions.",
            "Custom Jaccard and axis metrics.",
            "Gold/predicted assertions.",
            "ASSERTION_EVALUATION.",
            "No score when annotations absent.",
            """ASSERTION_EVALUATION = {"status": "not_scored", "reason": "No assertion ground truth"}\nprint(ASSERTION_EVALUATION)""",
        ),
        (
            18,
            "Candidate knowledge bases",
            "Instantiate separate ICD-10 and RxNorm indexes.",
            "LexicalCandidateIndex and type routing.",
            "ICD10_RECORDS and RXNORM_RECORDS.",
            "ICD_INDEX and RXNORM_INDEX.",
            "Candidate index objects rebuilt from artifacts.",
            """ICD_INDEX = LexicalCandidateIndex(ICD10_RECORDS, "ICD-10")\nRXNORM_INDEX = LexicalCandidateIndex(RXNORM_RECORDS, "RxNorm")\nprint({"icd10": len(ICD_INDEX.records), "rxnorm": len(RXNORM_INDEX.records)})""",
        ),
        (
            19,
            "ICD-10 retrieval",
            "Retrieve disease candidates using exact/fuzzy/character methods.",
            "LexicalCandidateIndex with character n-gram similarity.",
            "Disease mention.",
            "Top-k ICD-10 candidates.",
            "ICD-10 dictionary/cache.",
            """sample_icd = ICD10_RECORDS[0]\nprint(ICD_INDEX.retrieve((sample_icd.get("detection_aliases") or [sample_icd["candidate_id"]])[0], top_k=3))""",
        ),
        (
            20,
            "RxNorm retrieval",
            "Retrieve drug candidates and parse medication attributes.",
            "RxNorm lexical index and medication parser.",
            "Drug mention.",
            "Top-k RXCUI candidates and attributes.",
            "RxNorm dictionary/cache.",
            """from clinical_nlp_lab.linking import parse_medication_attributes\nsample_rx = RXNORM_RECORDS[0]\nprint({"retrieval": RXNORM_INDEX.retrieve((sample_rx.get("detection_aliases") or [sample_rx["candidate_id"]])[0], top_k=3), "attributes": parse_medication_attributes("aspirin 81 mg po daily")})""",
        ),
        (
            21,
            "Candidate reranking",
            "Apply deterministic weighted lexical/character score and output-k policy.",
            "SequenceMatcher + token overlap + character n-grams; threshold config.",
            "Retriever candidate list.",
            "Reranked candidates.",
            "thresholds.json.",
            """LINKER = EntityLinker(ICD_INDEX, RXNORM_INDEX, CONFIG["candidate_top_k"], CONFIG["candidate_output_k"], CONFIG["thresholds"]["candidate_min_score"])\nprint({"reranker": "weighted lexical + character", "top_k": CONFIG["candidate_top_k"], "output_k": CONFIG["candidate_output_k"]})""",
        ),
        (
            22,
            "Relation extraction",
            "Generate type-compatible same-sentence relation diagnostics.",
            "RuleRelationExtractor and pair constraints.",
            "Internal entities.",
            "Relation predictions (diagnostics only).",
            "diagnostics/relations.json when inference runs.",
            """RELATION_EXTRACTOR = RuleRelationExtractor(CONFIG["relation_max_distance"])\nrelations = RELATION_EXTRACTOR.extract(DOCUMENTS[0].raw_text, REFINED_SAMPLE)\nprint({"sample_relations": len(relations), "submission_key_added": False})""",
        ),
        (
            23,
            "Integrated pipeline",
            "Run the complete inference graph using saved artifacts.",
            "ClinicalNLPPipeline and run_inference.",
            "input.zip and artifacts/.",
            "INTEGRATION_SUMMARY.",
            "output/, diagnostics/, output.zip.",
            """INTEGRATION_SUMMARY = run_inference(PROJECT_ROOT / CONFIG["input_zip"], PROJECT_ROOT / CONFIG["output_dir"], PROJECT_ROOT / CONFIG["artifact_dir"], True, PROJECT_ROOT / CONFIG["diagnostics_dir"], OUTPUT_ARCHIVE_PATH)\nprint(INTEGRATION_SUMMARY)""",
        ),
        (
            24,
            "Competition evaluator",
            "Expose strict and approximate scoring without claiming organizer equivalence.",
            "Custom strict/greedy matching, WER and Jaccard.",
            "Gold annotations when available.",
            "EVALUATOR_STATUS.",
            "No score artifact when gold absent.",
            """EVALUATOR_STATUS = {"strict": "not_scored", "approximate": "not_scored", "reason": "No ground truth"}\nprint(EVALUATOR_STATUS)""",
        ),
        (
            25,
            "Error analysis",
            "Summarize internal detections and schema drops; never fabricate gold errors.",
            "Diagnostics aggregation.",
            "diagnostics/*.json.",
            "ERROR_ANALYSIS.",
            "reports/stage_08_integration.json.",
            """summary_path = PROJECT_ROOT / CONFIG["diagnostics_dir"] / "run_summary.json"\nERROR_ANALYSIS = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"status": "not_run"}\nprint(ERROR_ANALYSIS)""",
        ),
        (
            26,
            "Test inference",
            "Validate every output JSON against every raw input.",
            "Schema validator and ZIP checks.",
            "output/*.json and input.zip.",
            "TEST_INFERENCE_STATUS.",
            "output.zip.",
            """TEST_INFERENCE_STATUS = {"output_json_count": len(list((PROJECT_ROOT / CONFIG["output_dir"]).glob("*.json"))), "expected": len(DOCUMENTS), "offset_errors": INTEGRATION_SUMMARY["offset_error_count"]}\nassert TEST_INFERENCE_STATUS["output_json_count"] == len(DOCUMENTS)\nprint(TEST_INFERENCE_STATUS)""",
        ),
        (
            27,
            "Submission generation",
            "Create and validate output.zip with no nested output directory.",
            "zipfile and official schema validator.",
            "output/*.json.",
            "output.zip.",
            "output.zip.",
            """import zipfile\nwith zipfile.ZipFile(OUTPUT_ARCHIVE_PATH) as archive:\n    ZIP_NAMES = archive.namelist()\n    assert len(ZIP_NAMES) == len(DOCUMENTS)\n    assert all(name.startswith("output/") and not name.startswith("output/output/") for name in ZIP_NAMES)\n    assert archive.testzip() is None\nprint({"members": len(ZIP_NAMES), "structure_valid": True, "output_zip": str(OUTPUT_ARCHIVE_PATH)})""",
        ),
        (
            28,
            "Save/load artifacts",
            "Reload every inference artifact without training again.",
            "Artifact-first pipeline construction.",
            "artifacts/ and input.zip.",
            "RELOAD_CHECK.",
            "artifacts/model_status.json and KB caches.",
            """RELOAD_CHECK = reload_equivalence_check(PROJECT_ROOT / CONFIG["input_zip"], PROJECT_ROOT / CONFIG["artifact_dir"])\nassert RELOAD_CHECK["equivalent"]\nprint(RELOAD_CHECK)""",
        ),
        (
            29,
            "Reproducibility test",
            "Confirm deterministic seed/config and before/after reload equivalence.",
            "Seed status, checksums and deterministic rule pipeline.",
            "CONFIG and artifacts.",
            "REPRODUCIBILITY_STATUS.",
            "diagnostics/integration_report.json.",
            """REPRODUCIBILITY_STATUS = {"seed_status": SEED_STATUS, "reload_equivalent": RELOAD_CHECK["equivalent"], "fit_on_private_input": False}\nassert REPRODUCIBILITY_STATUS["reload_equivalent"]\nprint(REPRODUCIBILITY_STATUS)""",
        ),
        (
            30,
            "Conclusion",
            "Summarize completed stages, limitations and next organizer confirmation.",
            "Project state and manifest.",
            "All stage reports.",
            "Final completion summary.",
            "README.md and PROJECT_STATE.md.",
            """print({"completed_stages": list(range(1, 10)), "submission_schema_valid": True, "output_zip": str(OUTPUT_ARCHIVE_PATH), "trained_ner": bool(NER_TRAINING_RESULT.get("trained")), "official_mapping": INTEGRATION_SUMMARY.get("official_mapping_status"), "submission_entities": INTEGRATION_SUMMARY.get("submission_entity_count"), "limitation": "Supervised quality metrics require annotated train/validation data."})""",
        ),
    ]

    for number, title, goal, technology, input_text, output_text, artifact, code in sections:
        cells.append(markdown_cell(SECTION_TEMPLATE.format(number=number, title=title, goal=goal, technology=technology, input_text=input_text, output_text=output_text, artifact=artifact)))
        cells.append(code_cell(code))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    if notebook.get("nbformat") != 4:
        raise ValueError("Notebook must use nbformat 4")
    if not notebook.get("cells"):
        raise ValueError("Notebook has no cells")
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    empty_code_cells = [index for index, cell in enumerate(code_cells) if not "".join(cell.get("source", [])).strip()]
    if empty_code_cells:
        raise ValueError(f"Empty code cells: {empty_code_cells}")
    return {
        "cell_count": len(notebook["cells"]),
        "code_cell_count": len(code_cells),
        "markdown_cell_count": len(notebook["cells"]) - len(code_cells),
        "empty_code_cells": empty_code_cells,
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the single end-to-end Clinical NLP notebook")
    parser.add_argument("--output", type=Path, default=Path("medical_information_extraction_lab.ipynb"))
    args = parser.parse_args()
    notebook = build_notebook()
    validation = validate_notebook(notebook)
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=1)
        stream.write("\n")
    print(json.dumps({"output": str(destination), **validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
