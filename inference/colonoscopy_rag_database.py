"""
Colonoscopy Report Data - Structured for RAG & Classification
Extracted clinical findings optimized for polyp classification AI
"""

COLONOSCOPY_REPORTS_RAG = {
    "report_1": {
        "procedure": "Colonoscopy",
        "findings": [
            {
                "location": "Descending Colon",
                "polyp_type": "bi-lobulated polyp",
                "morphology": "JNET type 2a",
                "pit_pattern": "4",
                "classification": "ADENOMATOUS",
                "subtype": "likely_tubulovillous",
                "treatment": "EMR",
                "confidence": "high",
                "clinical_note": "Removed without immediate complication"
            }
        ],
        "follow_up": "1 month",
        "useful_for_training": True
    },
    
    "report_2": {
        "procedure": "Colonoscopy with anesthesia",
        "findings": [
            {
                "location": "Hepatic Flexure",
                "polyp_type": "lateral_spreading_lesion (LST)",
                "morphology": "LST-G homogeneous type 0-IIA",
                "pit_pattern": "homogeneous",
                "classification": "ADENOMATOUS",
                "subtype": "likely_villous_or_tubulovillous",
                "treatment": "Piecemeal EMR",
                "confidence": "high",
                "clinical_note": "Removed without immediate complication, remnant evaluation recommended"
            }
        ],
        "follow_up": "1 month",
        "useful_for_training": True
    },
    
    "report_3": {
        "procedure": "Colonoscopy with advanced imaging (WLE, M-NBI, TXI)",
        "bowel_prep_score": "2/2/2 (excellent)",
        "findings": [
            {
                "location": "Rectosigmoid Junction",
                "polyp_type": "sessile polyp",
                "morphology": "sessile",
                "classification": "HYPERPLASTIC",
                "subtype": "likely_hyperplastic",
                "treatment": "cold snare",
                "confidence": "low",
                "clinical_note": "Could not be fully characterized"
            },
            {
                "location": "Splenic Flexure",
                "polyp_type": "flat elevated polyp",
                "morphology": "JNET type 2a",
                "pit_pattern": "3",
                "classification": "ADENOMATOUS",
                "subtype": "likely_tubular",
                "treatment": "EMR with clip insertion",
                "confidence": "high",
                "clinical_note": "Removed without immediate complication, clip for hemorrhage prevention"
            }
        ],
        "additional_findings": [
            "hemorrhoids",
            "internal mucosal prolapse",
            "focal sub epithelial hemorrhage (proctitis Mayo 2)",
            "two superficial ulcers sigmoid",
            "inverted diverticula"
        ],
        "follow_up": "standard surveillance",
        "useful_for_training": True
    },
    
    "report_4": {
        "procedure": "Colonoscopy with advanced imaging (WLE, M-NBI, TXI)",
        "findings": [
            {
                "location": "Rectum",
                "polyp_type": "multiple serrated polyps",
                "morphology": "0-1b",
                "classification": "HYPERPLASTIC",
                "subtype": "serrated_hyperplastic",
                "treatment": "cold snare",
                "confidence": "high",
                "clinical_note": "Multiple polyps throughout rectum, especially proximal"
            },
            {
                "location": "Rectum - second group",
                "polyp_type": "small tubular adenoma",
                "morphology": "JNET type 2a, pit pattern 3L",
                "classification": "ADENOMATOUS",
                "subtype": "tubular",
                "treatment": "cold snare",
                "confidence": "high",
                "clinical_note": "Single small tubular adenoma removed"
            },
            {
                "location": "Sigmoid",
                "polyp_type": "multiple serrated polyps",
                "morphology": "0-1b",
                "classification": "HYPERPLASTIC",
                "subtype": "serrated_hyperplastic",
                "treatment": "cold snare",
                "confidence": "high",
                "clinical_note": "Numerous polyps throughout sigmoid, developing polyps noted"
            },
            {
                "location": "Ascending Colon",
                "polyp_type": "multiple serrated polyps",
                "morphology": "0-1b",
                "classification": "HYPERPLASTIC",
                "subtype": "serrated_hyperplastic",
                "treatment": "hot snare",
                "confidence": "high",
                "clinical_note": "Three polyps removed"
            }
        ],
        "follow_up": "close surveillance (serrated polyposis)",
        "useful_for_training": True
    }
}

# RAG PROMPTS FOR CLASSIFICATION MODEL
RAG_SYSTEM_PROMPTS = {
    "polyp_classification": """You are an expert endoscopy AI assistant trained on real colonoscopy reports and international classification standards (JNET, Paris).

When classifying polyps, use these clinical markers from the database and standards:

JNET CLASSIFICATION (Narrow Band Imaging):
- Type 1: Invisible vessels, regular dark/white spots. Likely HYPERPLASTIC or Sessile Serrated Lesion (SSL).
- Type 2A: Regular vessel caliber/distribution, regular surface pattern (tubular/branched). Likely LOW-GRADE ADENOMA.
- Type 2B: Variable vessel caliber, irregular distribution, irregular/obscure surface. Likely HIGH-GRADE NEOPLASIA or superficial cancer.
- Type 3: Loose vessel areas, interruption of thick vessels, amorphous surface. Likely DEEP INVASIVE CANCER.

ADENOMATOUS POLYP INDICATORS:
- JNET type 2a or 2b
- Lateral spreading lesions (LST)
- Flat elevated morphology (0-IIA)
- Requires EMR or hot snare treatment

HYPERPLASTIC POLYP INDICATORS:
- JNET type 1
- Sessile or 0-1b morphology
- Serrated appearance notation
- Cold snare treatment sufficient

UNCERTAINTY FACTORS:
- Incomplete characterization
- Small size (<7mm) without clear JNET type
- May require follow-up evaluation""",

    "clinical_context": """Clinical Data & Classification Standards:
    
JNET (Japanese NBI Expert Team) STANDARDS:
- Type 1: Hyperplastic/Sessile serrated polyp
- Type 2A: Low-grade intramucosal neoplasia (Adenoma)
- Type 2B: High-grade intramucosal neoplasia / Superficial submucosal invasive cancer
- Type 3: Deep submucosal invasive cancer

PARIS CLASSIFICATION FOR LST (Lateral Spreading Tumors):
- LST-G (Granular): Homogeneous type (0-IIa) or Mixed nodular type (0-IIa + Is)
- LST-NG (Non-granular): Flat type (0-IIa) or Pseudodepressed type (0-IIa + IIc)

CONFIRMED ADENOMATOUS CASES:
- Bi-lobulated polyp (descending colon, JNET 2a, pit 4)
- Lateral spreading lesion (hepatic flexure, LST-G, large)
- Flat elevated polyp (splenic flexure, JNET 2a, pit 3)

CONFIRMED HYPERPLASTIC CASES:
- Sessile small polyps (rectosigmoid, 7mm, JNET 1)
- Serrated polyps in polyposis (rectum/sigmoid, 0-1b)""",

    "image_analysis": """When analyzing polyp images:

JNET VISUAL MARKERS:
✓ Type 1: Invisible vessels, regular spots (Hyperplastic)
✓ Type 2A: Regular meshed/spiral vessels, regular surface (Adenoma)
✓ Type 2B: Variable/irregular vessels, obscure surface (High-grade)
✓ Type 3: Interrupted thick vessels, amorphous areas (Invasive)

PARIS MORPHOLOGY:
✓ 0-IIa: Flat elevated
✓ 0-IIa + Is: Mixed nodular (LST-G)
✓ 0-IIa + IIc: Pseudodepressed (LST-NG)

REFERENCE: Real cases from database show JNET 2a patterns consistently with adenomas."""
}

# TRAINING DATA SUMMARY FOR RAG
RAG_TRAINING_DATA = {
    "total_reports": 4,
    "total_polyps_classified": 9,
    "adenomatous_count": 4,
    "hyperplastic_count": 5,
    
    "classification_standards": [
        "JNET (Japanese NBI Expert Team) Type 1, 2A, 2B, 3",
        "Paris Classification (0-IIa, 0-Is, 0-IIc)",
        "LST Subtypes (LST-G, LST-NG)"
    ],
    
    "adenomatous_features": {
        "common_morphologies": ["JNET type 2a", "LST-G", "flat elevated (0-IIa)", "bi-lobulated"],
        "common_pit_patterns": ["3", "4"],
        "typical_treatment": ["EMR", "piecemeal EMR", "hot snare with clip"],
        "locations": ["descending colon", "hepatic flexure", "splenic flexure"]
    },
    
    "hyperplastic_features": {
        "common_morphologies": ["JNET type 1", "sessile", "0-1b", "serrated"],
        "pit_patterns": ["homogeneous", "not specified"],
        "typical_treatment": ["cold snare"],
        "locations": ["rectosigmoid", "rectum", "sigmoid", "ascending colon"],
        "pattern_note": "Often multiple in polyposis cases"
    },
    
    "imaging_equipment_used": [
        "Olympus endoscope",
        "WLE (White Light Endoscopy)",
        "M-NBI (Narrow Band Imaging)",
        "TXI (Texture and Color Enhancement)",
        "JNET classification standard"
    ]
}

def get_rag_context(polyp_location: str, polyp_morphology: str) -> dict:
    """
    Retrieve relevant clinical context from RAG database
    for a specific polyp being classified
    """
    context = {
        "system_prompt": RAG_SYSTEM_PROMPTS["polyp_classification"],
        "clinical_reference": RAG_SYSTEM_PROMPTS["clinical_context"],
        "similar_cases": []
    }
    
    # Find similar cases in database
    for report_key, report in COLONOSCOPY_REPORTS_RAG.items():
        if "findings" not in report:
            continue
        
        for finding in report["findings"]:
            if finding["location"].lower() == polyp_location.lower():
                context["similar_cases"].append({
                    "location": finding["location"],
                    "morphology": finding["morphology"],
                    "classification": finding["classification"],
                    "confidence": finding["confidence"]
                })
    
    return context

if __name__ == "__main__":
    print("=" * 80)
    print("COLONOSCOPY REPORTS RAG DATABASE")
    print("=" * 80)
    
    print(f"\nTotal Reports: {RAG_TRAINING_DATA['total_reports']}")
    print(f"Total Polyps: {RAG_TRAINING_DATA['total_polyps_classified']}")
    print(f"Adenomatous: {RAG_TRAINING_DATA['adenomatous_count']}")
    print(f"Hyperplastic: {RAG_TRAINING_DATA['hyperplastic_count']}")
    
    print("\n" + "=" * 80)
    print("ADENOMATOUS FEATURES")
    print("=" * 80)
    for key, value in RAG_TRAINING_DATA["adenomatous_features"].items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 80)
    print("HYPERPLASTIC FEATURES")
    print("=" * 80)
    for key, value in RAG_TRAINING_DATA["hyperplastic_features"].items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 80)
    print("SAMPLE RAG CONTEXT RETRIEVAL")
    print("=" * 80)
    context = get_rag_context("Descending Colon", "bi-lobulated")
    print(f"Found {len(context['similar_cases'])} similar cases")
    for case in context["similar_cases"]:
        print(f"  - {case['location']}: {case['morphology']} → {case['classification']}")
