"""
Generate synthetic doctor data for the demo.

Creates realistic-looking doctor records with names, specialties, locations,
ratings, fees, reviews, and bios — enriching real OSM hospital/clinic coordinates
or generating fully synthetic records for demo purposes.
"""

import json
import random
import math
from pathlib import Path

# ── Configuration ──
OUTPUT_DIR = Path(__file__).parent.parent / "app" / "data" / "seed"
DOCTORS_PER_CITY = 250

# Target cities with approximate center coordinates
CITIES = {
    "Bangalore": {"lat": 12.9716, "lng": 77.5946, "country": "India", "currency": "INR"},
    "San Francisco": {"lat": 37.7749, "lng": -122.4194, "country": "USA", "currency": "USD"},
}

SPECIALTIES = [
    "Cardiologist", "Dermatologist", "Endocrinologist", "Gastroenterologist",
    "Neurologist", "Orthopedist", "Pulmonologist", "ENT Specialist",
    "Ophthalmologist", "Psychiatrist", "Urologist", "Rheumatologist",
    "Oncologist", "Gynecologist", "Nephrologist", "Allergist/Immunologist",
    "General Practitioner", "Pediatrician",
]

# Weighted distribution — GPs and common specialties appear more often
SPECIALTY_WEIGHTS = [
    8, 6, 4, 7, 5, 8, 4, 5, 5, 6, 4, 3, 3, 6, 3, 3, 15, 8,
]

SUB_SPECIALIZATIONS = {
    "Cardiologist": ["Interventional Cardiology", "Electrophysiology", "Heart Failure", "Preventive Cardiology"],
    "Dermatologist": ["Cosmetic Dermatology", "Pediatric Dermatology", "Dermatopathology", "Mohs Surgery"],
    "Neurologist": ["Stroke Neurology", "Epilepsy", "Movement Disorders", "Neuro-oncology"],
    "Orthopedist": ["Sports Medicine", "Spine Surgery", "Joint Replacement", "Hand Surgery", "Pediatric Orthopedics"],
    "Gastroenterologist": ["Hepatology", "IBD Specialist", "Endoscopy", "Motility Disorders"],
    "Oncologist": ["Breast Cancer", "Lung Cancer", "Hematologic Oncology", "Pediatric Oncology"],
    "Psychiatrist": ["Child & Adolescent", "Addiction Psychiatry", "Geriatric Psychiatry", "Forensic Psychiatry"],
    "Gynecologist": ["Maternal-Fetal Medicine", "Reproductive Endocrinology", "Gynecologic Oncology"],
}

# Name pools
FIRST_NAMES_MALE_IN = ["Rajesh", "Arun", "Vikram", "Suresh", "Amit", "Sanjay", "Pradeep", "Kiran", "Deepak", "Manoj", "Ravi", "Ashok", "Naveen", "Venkat", "Ramesh", "Prakash", "Harish", "Girish", "Mohan", "Vijay"]
FIRST_NAMES_FEMALE_IN = ["Priya", "Anita", "Kavita", "Sunita", "Lakshmi", "Meena", "Anjali", "Divya", "Nandini", "Pooja", "Swati", "Rekha", "Shalini", "Neha", "Asha", "Vandana", "Smita", "Shobha", "Geeta", "Usha"]
LAST_NAMES_IN = ["Sharma", "Patel", "Reddy", "Kumar", "Singh", "Rao", "Iyer", "Nair", "Gupta", "Joshi", "Desai", "Murthy", "Shetty", "Hegde", "Kulkarni", "Bhat", "Menon", "Pillai", "Choudhury", "Agarwal"]

FIRST_NAMES_MALE_US = ["James", "Robert", "Michael", "David", "John", "William", "Richard", "Daniel", "Christopher", "Matthew", "Andrew", "Brian", "Kevin", "Steven", "Edward", "Mark", "Thomas", "Charles", "Joseph", "Paul"]
FIRST_NAMES_FEMALE_US = ["Sarah", "Jennifer", "Emily", "Jessica", "Amanda", "Rachel", "Megan", "Lauren", "Stephanie", "Nicole", "Elizabeth", "Katherine", "Maria", "Lisa", "Rebecca", "Michelle", "Samantha", "Ashley", "Christina", "Laura"]
LAST_NAMES_US = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris"]

LANGUAGES_IN = ["English", "Hindi", "Kannada", "Telugu", "Tamil", "Malayalam"]
LANGUAGES_US = ["English", "Spanish", "Mandarin", "Hindi", "Korean", "Vietnamese"]

INSURANCE_IN = ["Star Health", "HDFC Ergo", "ICICI Lombard", "Bajaj Allianz", "Max Bupa", "Aditya Birla", "None"]
INSURANCE_US = ["Aetna", "Blue Cross Blue Shield", "Cigna", "United Healthcare", "Kaiser", "Medicare", "Medicaid", "None"]

HOSPITALS_IN = ["Apollo Hospital", "Fortis Hospital", "Manipal Hospital", "Columbia Asia", "Narayana Health", "Sakra World Hospital", "BGS Gleneagles", "Aster CMI", "Vikram Hospital", "MS Ramaiah", "Private Clinic"]
HOSPITALS_US = ["UCSF Medical Center", "Stanford Health", "Kaiser SF", "Sutter Health CPMC", "Zuckerberg SF General", "Chinese Hospital", "St. Mary's Medical", "VA Medical Center", "Private Practice", "One Medical"]

EDUCATION_IN = ["MBBS, MD - {spec} (AIIMS Delhi)", "MBBS, MD - {spec} (NIMHANS Bangalore)", "MBBS, MS - {spec} (KMC Manipal)", "MBBS, DNB - {spec} (Bangalore Medical College)", "MBBS, MD - {spec} (St. John's Medical College)"]
EDUCATION_US = ["MD - {spec} (Stanford School of Medicine)", "MD - {spec} (UCSF School of Medicine)", "MD - {spec} (Johns Hopkins)", "DO - {spec} (Harvard Medical School)", "MD - {spec} (UCLA David Geffen)"]

REVIEW_TEMPLATES = [
    ("Excellent doctor. Very thorough and takes time to explain everything clearly.", 5),
    ("Dr. {name} is extremely knowledgeable. Helped me understand my condition fully.", 5),
    ("Great experience. The doctor was patient and answered all my questions.", 5),
    ("Very professional. Diagnosed my issue quickly and the treatment worked perfectly.", 5),
    ("One of the best {specialty}s I've visited. Highly recommend.", 5),
    ("Good doctor overall. Wait time was a bit long but the consultation was worth it.", 4),
    ("Competent doctor with good bedside manner. Would visit again.", 4),
    ("Dr. {name} was helpful. Prescribed the right medication and I felt better within a week.", 4),
    ("Decent experience. The clinic is well-maintained and staff is friendly.", 4),
    ("Good doctor but slightly rushed during the consultation.", 3),
    ("Average experience. The doctor was okay but didn't explain much.", 3),
    ("Long wait time and the consultation felt too short for the fee charged.", 3),
    ("The doctor was knowledgeable but could improve on communication.", 3),
    ("Not the best experience. Had to follow up multiple times for test results.", 2),
    ("Dr. {name} seemed distracted during my visit. Expected better for the consultation fee.", 2),
]


def random_point_within_radius(center_lat: float, center_lng: float, radius_km: float = 15) -> tuple[float, float]:
    """Generate a random point within radius_km of center coordinates."""
    r = radius_km / 111.32  # Approximate degrees
    u = random.random()
    v = random.random()
    w = r * math.sqrt(u)
    t = 2 * math.pi * v
    x = w * math.cos(t)
    y = w * math.sin(t)
    return round(center_lat + y, 6), round(center_lng + x, 6)


def generate_doctors_for_city(city_name: str, city_info: dict, start_id: int) -> tuple[list[dict], list[dict]]:
    """Generate synthetic doctor records and reviews for a city."""
    doctors = []
    reviews = []
    review_id = start_id * 100

    is_india = city_info["country"] == "India"
    first_names_m = FIRST_NAMES_MALE_IN if is_india else FIRST_NAMES_MALE_US
    first_names_f = FIRST_NAMES_FEMALE_IN if is_india else FIRST_NAMES_FEMALE_US
    last_names = LAST_NAMES_IN if is_india else LAST_NAMES_US
    languages = LANGUAGES_IN if is_india else LANGUAGES_US
    insurances = INSURANCE_IN if is_india else INSURANCE_US
    hospitals = HOSPITALS_IN if is_india else HOSPITALS_US
    education_templates = EDUCATION_IN if is_india else EDUCATION_US

    for i in range(DOCTORS_PER_CITY):
        doc_id = start_id + i
        specialty = random.choices(SPECIALTIES, weights=SPECIALTY_WEIGHTS, k=1)[0]

        # Gender (60/40 split for demo diversity)
        gender = random.choice(["male"] * 6 + ["female"] * 4)
        first_name = random.choice(first_names_m if gender == "male" else first_names_f)
        last_name = random.choice(last_names)
        name = f"Dr. {first_name} {last_name}"

        lat, lng = random_point_within_radius(city_info["lat"], city_info["lng"])
        years_exp = random.randint(3, 35)

        # Fee based on specialty and country
        if is_india:
            base_fee = random.randint(300, 2500)
            if specialty in ["Oncologist", "Neurologist", "Cardiologist"]:
                base_fee = random.randint(800, 3000)
        else:
            base_fee = random.randint(100, 400)
            if specialty in ["Oncologist", "Neurologist", "Cardiologist"]:
                base_fee = random.randint(200, 600)

        rating = round(random.uniform(3.0, 5.0), 1)
        review_count = random.randint(10, 500)

        sub_spec = None
        if specialty in SUB_SPECIALIZATIONS and random.random() > 0.4:
            sub_spec = random.choice(SUB_SPECIALIZATIONS[specialty])

        n_languages = random.randint(1, 3)
        doc_languages = random.sample(languages, min(n_languages, len(languages)))
        if "English" not in doc_languages:
            doc_languages[0] = "English"

        n_insurance = random.randint(1, 4)
        doc_insurance = random.sample(insurances, min(n_insurance, len(insurances)))

        education = random.choice(education_templates).format(spec=specialty)
        hospital = random.choice(hospitals)

        certs = [f"Board Certified - {specialty}"]
        if sub_spec:
            certs.append(f"Fellowship - {sub_spec}")

        doctor = {
            "id": doc_id,
            "name": name,
            "gender": gender,
            "specialty": specialty,
            "sub_specialization": sub_spec,
            "years_experience": years_exp,
            "board_certifications": certs,
            "education": education,
            "lat": lat,
            "lng": lng,
            "address": f"{random.randint(1, 999)} {random.choice(['Main St', 'MG Road', 'Health Ave', 'Medical Lane', 'Hospital Rd', 'Clinic Street', 'Park Road', 'Lake View'])}",
            "city": city_name,
            "rating": rating,
            "review_count": review_count,
            "consultation_fee": base_fee,
            "insurance_accepted": doc_insurance,
            "languages": doc_languages,
            "telehealth_available": random.random() > 0.5,
            "phone": f"+{'91' if is_india else '1'}-{''.join([str(random.randint(0,9)) for _ in range(10)])}",
            "website": None,
            "hospital_name": hospital,
            "source": "synthetic",
        }
        doctors.append(doctor)

        # Generate 3-8 reviews per doctor
        n_reviews = random.randint(3, 8)
        chosen_reviews = random.sample(REVIEW_TEMPLATES, min(n_reviews, len(REVIEW_TEMPLATES)))
        for tmpl_text, tmpl_rating in chosen_reviews:
            review_id += 1
            text = tmpl_text.format(name=name, specialty=specialty)
            # Adjust rating slightly
            adj_rating = max(1, min(5, tmpl_rating + random.choice([-1, 0, 0, 0, 1])))

            reviews.append({
                "id": str(review_id),
                "doctor_id": str(doc_id),
                "doctor_name": name,
                "rating": adj_rating,
                "text": text,
            })

    return doctors, reviews


def main():
    """Generate all synthetic data and save to JSON files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_doctors = []
    all_reviews = []
    start_id = 1

    for city_name, city_info in CITIES.items():
        print(f"Generating {DOCTORS_PER_CITY} doctors for {city_name}...")
        doctors, reviews = generate_doctors_for_city(city_name, city_info, start_id)
        all_doctors.extend(doctors)
        all_reviews.extend(reviews)
        start_id += DOCTORS_PER_CITY

    # Save
    doctors_path = OUTPUT_DIR / "doctors.json"
    reviews_path = OUTPUT_DIR / "reviews.json"

    with open(doctors_path, "w", encoding="utf-8") as f:
        json.dump(all_doctors, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_doctors)} doctors to {doctors_path}")

    with open(reviews_path, "w", encoding="utf-8") as f:
        json.dump(all_reviews, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_reviews)} reviews to {reviews_path}")


if __name__ == "__main__":
    main()
