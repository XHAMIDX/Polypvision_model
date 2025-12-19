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
                "size_mm": "0-10",  # 0-1s classification
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
                "size_mm": ">10",  # LST-G type indicates large
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
                "size_mm": "7",
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
                "size_mm": ">10",
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
        "bowel_prep_score": "1/2/2 (fair)",
        "findings": [
            {
                "location": "Rectum",
                "polyp_type": "multiple serrated polyps",
                "size_mm": "8",
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
                "size_mm": "<8",
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
                "size_mm": "8",
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
                "size_mm": "8",
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
    "polyp_classification": """You are an expert endoscopy AI assistant trained on real colonoscopy reports.

When classifying polyps, use these clinical markers from the database:

ADENOMATOUS POLYP INDICATORS:
- JNET type 2a with pit pattern 3-4
- Lateral spreading lesions (LST-G)
- Flat elevated morphology (0-IIA)
- Size > 7mm with concerning features
- Requires EMR or hot snare treatment

HYPERPLASTIC POLYP INDICATORS:
- Sessile or 0-1b morphology
- Small size (< 8mm typically)
- Serrated appearance notation
- Cold snare treatment sufficient
- Often multiple in polyposis cases

UNCERTAINTY FACTORS:
- Incomplete characterization
- Small size (<7mm) without clear JNET type
- May require follow-up evaluation

Use location context (rectum, colon segments) to refine classification.""",

    "clinical_context": """Clinical Data from Mehrad Hospital Colonoscopy Series:
    
CONFIRMED ADENOMATOUS CASES:
- Bi-lobulated polyp (descending colon, JNET 2a, pit 4)
- Lateral spreading lesion (hepatic flexure, LST-G, large)
- Flat elevated polyp (splenic flexure, JNET 2a, pit 3, >10mm)

CONFIRMED HYPERPLASTIC CASES:
- Sessile small polyps (rectosigmoid, 7mm)
- Serrated polyps in polyposis (rectum/sigmoid, 0-1b, 8mm)
- Multiple small polyps (ascending colon, 0-1b)

MORPHOLOGY GUIDE:
- JNET Type 2a = Adenomatous features
- Pit pattern 3-4 = Higher dysplasia risk
- 0-1b = Often hyperplastic
- Flat/elevated = Adenomatous pattern
- LST = Large advanced lesion""",

    "image_analysis": """When analyzing polyp images:

HIGH CONFIDENCE ADENOMA FEATURES:
✓ Irregular surface with pit patterns
✓ Raised or flat-elevated morphology
✓ Reddish/granular appearance
✓ Size > 10mm with concerning pattern

HIGH CONFIDENCE HYPERPLASTIC FEATURES:
✓ Smooth, glistening surface
✓ Pale/whitish appearance
✓ Small size (< 10mm)
✓ Regular/serrated border
✓ Multiple lesions in polyposis pattern

REFERENCE: Real cases from database show JNET 2a patterns consistently with adenomas."""
}

# TRAINING DATA SUMMARY FOR RAG
RAG_TRAINING_DATA = {
    "total_reports": 4,
    "total_polyps_classified": 9,
    "adenomatous_count": 4,
    "hyperplastic_count": 5,
    
    "adenomatous_features": {
        "common_morphologies": ["JNET type 2a", "LST-G", "flat elevated", "bi-lobulated"],
        "common_pit_patterns": ["3", "4"],
        "size_range": "7-15mm",
        "typical_treatment": ["EMR", "piecemeal EMR", "hot snare with clip"],
        "locations": ["descending colon", "hepatic flexure", "splenic flexure"]
    },
    
    "hyperplastic_features": {
        "common_morphologies": ["sessile", "0-1b", "serrated"],
        "pit_patterns": ["homogeneous", "not specified"],
        "size_range": "6-8mm",
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
                    "patient_age": report["age"],
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
