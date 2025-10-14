import time
import random
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from datetime import datetime
from selenium import webdriver
from time import sleep
import os
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import unicodedata
import re
import asyncio
import httpx
import nest_asyncio
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
from openai import OpenAI
import instructor
from enum import Enum
from pydantic import BaseModel, Field, field_validator, constr, ValidationError, model_validator
from typing import List, Optional, Any

# Définir les paramètres de recherche
url_base = f"https://www.hosco.com/fr/emplois?locations=ChIJMVd4MymgVA0R99lHx5Y__Ws&sort=posted"


def restart_driver():
    """Quit current driver, wait 61–70 seconds, then return a new driver instance."""
    global driver
    try:
        driver.quit()
    except:
        pass
    wait_time = random.uniform(61, 70)
    print(f"[INFO] Restarting driver after waiting {wait_time:.2f} seconds...")
    time.sleep(wait_time)
    # Reinitialize the driver (adjust options as you had them before)
    driver = webdriver.Chrome()  
    print("[INFO] New driver started.")
    return driver

def extraire_offres(start_page=1, max_pages=4, limit=40):
    offres_totales = []

    try:
        # ⚡ create driver once per batch
        options = uc.ChromeOptions()
        options.add_argument('--headless=new')
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-setuid-sandbox")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)')

        driver = uc.Chrome(options=options)

        for page in range(start_page, start_page + max_pages):
            if len(offres_totales) >= limit:
                break

            url = f"{url_base}&page={page}"
            print(f"Scraping page {page} from URL: {url}")
            try:
                driver.get(url)
            except Exception as e:
                print(f"⚠️ Driver crashed at page {page}, skipping. Error: {e}")
                break

            sleep(random.uniform(3, 4))

            offres = driver.find_elements(By.CSS_SELECTOR, '[data-cy="job-tile"] h2 a')
            if not offres:
                print("Aucune offre trouvée sur cette page.")
                break

            for offre in offres:
                if len(offres_totales) >= limit:
                    break
                try:
                    url_offre = offre.get_attribute("href")
                except Exception:
                    url_offre = "N/A"

                offres_totales.append({
                    'url': url_offre,
                    'salaire_1': None
                })

            sleep(random.uniform(1, 2))

    finally:
        try:
            driver.quit()
        except:
            pass
        sleep(random.uniform(1, 2))

    return offres_totales



# 🚀 wrapper pour boucler par batchs de 4 pages
def extraire_toutes_offres(total_pages=20, batch_size=4, limit=200):
    all_results = []
    for start in range(1, total_pages + 1, batch_size):
        print(f"\n=== Batch {start} → {start+batch_size-1} ===")
        batch_results = extraire_offres(start_page=start, max_pages=batch_size, limit=limit)
        all_results.extend(batch_results)

        if len(all_results) >= limit:
            break
        sleep(random.uniform(15, 20))
    return all_results


resultats_part1 = extraire_toutes_offres(total_pages=40, batch_size=4, limit=400)
resultats_part1 = pd.DataFrame(resultats_part1)
job_urls = resultats_part1.url.tolist()

# Setup undetected Chrome driver
options = uc.ChromeOptions()
options.add_argument('--headless=new')
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-setuid-sandbox")
options.add_argument("--disable-extensions")
options.add_argument("--disable-infobars")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')


# Launch the driver
driver = uc.Chrome(options=options)


def restart_driver():
    """Quit current driver, wait 61–70 seconds, then return a new driver instance."""
    global driver
    try:
        driver.quit()
    except:
        pass
    wait_time = random.uniform(61, 70)
    print(f"[INFO] Restarting driver after waiting {wait_time:.2f} seconds...")
    time.sleep(wait_time)
    # Reinitialize the driver (adjust options as you had them before)
    driver = webdriver.Chrome(options=options)  
    print("[INFO] New driver started.")
    return driver
    
# Function to extract text safely
def get_text(selector, multiple=False):
    try:
        if multiple:
            return [elem.text.strip() for elem in driver.find_elements(By.CSS_SELECTOR, selector)]
        return driver.find_element(By.CSS_SELECTOR, selector).text.strip()
    except NoSuchElementException:
        return "" if not multiple else []

# Initialize list to store job data
job_data = []
start_index = 0  # if you want to resume manually later

for i, job_url in enumerate(job_urls[start_index:], start=start_index):
    try:
        driver.get(job_url)
        time.sleep(random.uniform(4, 5))  # Random delay for human-like behavior

        try:
            title = get_text("h2.sc-95116d32-0.sc-95116d32-1")
        except NoSuchElementException:
            print(f"[ERROR] Title not found for {job_url}. Restarting driver...")
            driver = restart_driver()
            # retry current URL once with the new driver
            try:
                driver.get(job_url)
                time.sleep(random.uniform(4, 5))
                title = get_text("h2.sc-95116d32-0.sc-95116d32-1")
            except NoSuchElementException:
                print(f"[ERROR] Title still not found after restart. Skipping {job_url}.")
                continue  # skip this job and move to next

        entreprise = get_text("a.sc-7e2951f8-3.kJCWHp > h2.sc-7e2951f8-4.gBRCjz")
        details = driver.find_elements(By.CSS_SELECTOR, "p.sc-95116d32-2.dEzmLv")
        location = details[0].text.strip()
        Temps_partiel_plein = details[1].text.strip()

        parts = [p.strip() for p in Temps_partiel_plein.split(",")]
        if "Indéfini" in parts and len(parts) > 1:
            Temps_partiel_plein = parts[0]

        date_debut_raw = details[2].text.strip()
        date_value = date_debut_raw.split(":")[-1].strip()
        date_debut = f"Début: {date_value}"

        # --- Description ---
        if entreprise != "EXTRACADABRA":
            try:
                section = driver.find_element(By.XPATH, "//section[h2[contains(text(), 'À propos du poste')]]")
                paragraphs = section.find_elements(By.TAG_NAME, "p")
                list_items = section.find_elements(By.TAG_NAME, "li")
                description_parts = [p.text.strip() for p in paragraphs if p.text.strip()]
                list_parts = [li.text.strip() for li in list_items if li.text.strip()]
                description = "\n".join(description_parts + list_parts)
            except NoSuchElementException:
                description = ""
        else:
            try:
                description = get_text("div.sc-be9911fe-1.gvJWJp > div:nth-of-type(1)")
            except NoSuchElementException:
                description = ""

        # --- About you ---
        if entreprise != "EXTRACADABRA":
            try:
                section = driver.find_element(By.XPATH, "//section[h2[contains(text(), 'À propos de vous')]]")
                paragraphs = section.find_elements(By.TAG_NAME, "p")
                list_items = section.find_elements(By.TAG_NAME, "li")
                about_you_parts = [p.text.strip() for p in paragraphs if p.text.strip()]
                about_you_list = [li.text.strip() for li in list_items if li.text.strip()]
                about_you_description = "\n".join(about_you_parts + about_you_list)
            except NoSuchElementException:
                about_you_description = ""
        else:
            about_you_description = ""

        # --- Salaire ---
        if entreprise == "EXTRACADABRA":
            try:
                section = driver.find_element(By.XPATH, "//section[h2[contains(text(), 'À propos du poste')]]")
                paragraphs = section.find_elements(By.TAG_NAME, "p")
                list_items = section.find_elements(By.TAG_NAME, "li")
                description_parts = [p.text.strip() for p in paragraphs if p.text.strip()]
                list_parts = [li.text.strip() for li in list_items if li.text.strip()]
                salaire = "\n".join(description_parts + list_parts)
            except NoSuchElementException:
                salaire = None
        else:
            salaire = None

        type_contrat = "EXTRA" if entreprise == "EXTRACADABRA" else "CDD/CDI"

        if about_you_description:
            description += "\n\n" + about_you_description

        date_scraping = datetime.datetime.now().strftime("%Y-%m-%d")
        tags = date_debut

        job_data.append({
            "titre": title,
            "localisation": location,
            "entreprise": entreprise,
            "salaire_2": salaire,
            "description": description,
            "date_scraping": date_scraping,
            "Temps_partiel_plein": Temps_partiel_plein,
            "Tags": tags,
            "type_contrat": type_contrat,
        })

        print(f"[INFO] Scraped {i+1}/{len(job_urls)}: {title}")

    except Exception as e:
        print(f"[ERROR] Unexpected error at index {i}: {e}")
        driver = restart_driver()
        continue  # Continue from next job_url

driver.quit()
print("[DONE] Scraping completed.")

# Convert list to Pandas DataFrame
resultats_part2 = pd.DataFrame(job_data)

#Concat axis 0 resultats_part1 and resultats_part2
df_jobs = pd.concat([resultats_part1, resultats_part2], axis=1)

def fusionner_salaire(row):
    if pd.notna(row['salaire_2']):
        return row['salaire_2']
    elif pd.notna(row['salaire_1']):
        return row['salaire_1']
    else:
        return None

df_jobs["salaire"] = df_jobs.apply(fusionner_salaire, axis=1)
df_jobs = df_jobs.drop(columns=["salaire_1", "salaire_2"])

scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

credentials_info = json.loads(os.environ.get("GOOGLE_CREDENTIALS"))
credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
client = gspread.authorize(credentials)

# Open the Google Sheet
spreadsheet = client.open('hoscoScrapper')  # Use your sheet's name
worksheet = spreadsheet.sheet1

# Read existing data from Google Sheets into a DataFrame
existing_data = pd.DataFrame(worksheet.get_all_records())

# Convert scraped results into a DataFrame
new_data = df_jobs

# Apply nest_asyncio to fix event loop issue in Jupyter
nest_asyncio.apply()

# Data Gouv API URL
API_URL = "https://api-adresse.data.gouv.fr/search"

# Function to call API asynchronously with retries
async def get_geodata(client, address, retries=3):
    params = {"q": address, "limit": 1}

    for attempt in range(retries):
        try:
            response = await client.get(API_URL, params=params, timeout=5)

            if response.status_code == 503:  # Server overloaded
                print(f"503 Error - Retrying {address} (Attempt {attempt+1})...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue

            response.raise_for_status()  # Raise error if response is bad
            data = response.json()

            if data["features"]:
                props = data["features"][0]["properties"]
                geo = data["features"][0]["geometry"]["coordinates"]

                ville = props.get("city", "")
                code_postal = props.get("postcode", "")
                longitude = geo[0] if geo else None
                latitude = geo[1] if geo else None
                contexte = props.get("context", "")

                # Extract region name (after second comma)
                region = contexte.split(", ")[-1] if contexte.count(",") >= 2 else ""

                return ville, code_postal, longitude, latitude, region
        
        except Exception as e:
            print(f"Error fetching data for {address} (Attempt {attempt+1}): {e}")
        
        await asyncio.sleep(2 ** attempt)  # Exponential backoff for retries

    return None, None, None, None, None  # Return empty values if all retries fail

# Async function to process all addresses with rate limiting
async def process_addresses(address_list, delay_between_requests=0.017):  # 1/60 = ~0.017s
    results = []
    async with httpx.AsyncClient() as client:
        for i, address in enumerate(address_list):
            result = await get_geodata(client, address)
            results.append(result)
            
            print(f"Processed {i + 1} / {len(address_list)}")

            # Respect 60 requests per second limit
            await asyncio.sleep(delay_between_requests)  

    return results

# Run API calls asynchronously
addresses = new_data["localisation"].tolist()
geodata_results = asyncio.run(process_addresses(addresses))

# Assign the results to the DataFrame
new_data[["Ville", "Code Postal", "Longitude", "Latitude", "Region"]] = pd.DataFrame(geodata_results)

# Add "France Travail" column
new_data["Source"] = "hosco"

print(f"Post geo new data Check length {len(new_data)}")
print(f"Post geo Check existing length {len(existing_data)}")

new_data = new_data[~new_data["url"].isin(existing_data["url"])]

# -------- DEBUT CHATGPT DATA ENRICHMENT --------------------------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_ai = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))

class Loge(str, Enum):
    LOGE = "Logé"
    NON_LOGE = "Non Logé"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"

class TypeContrat(str, Enum):
    CDD = "CDD"
    CDI = "CDI"
    STAGE = "Stage"
    APPRENTISSAGE = "Apprentissage"
    INTERIM = "Interim"
    EXTRA = "Extra"
    SAISONNIER = "Saisonnier"
    ALTERNANCE = "Alternance"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CoupleAccepte(str, Enum):
    ACCEPTE = "Couple accepté"
    NON_ACCEPTE = "Couple non accepté"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieEtablissement(str, Enum):
    GASTRONOMIQUE = "Gastronomique"
    BRASSERIE = "Brasserie"
    BAR = "Bar"
    RAPIDE = "Restauration rapide"
    COLLECTIVE = "Restauration collective"
    RESTAURANT = "Restaurant"
    HOTEL_LUXE = "Hôtel luxe"
    HOTEL = "Hôtel"
    CAMPING = "Camping"
    CAFE = "Café/Salon de thé"
    BOULANGERIE = "Boulangerie/Patisserie"
    ETOILE = "Etoile Michelin"
    PALACE = "Palace"
    TRAITEUR = "Traiteur/Événementiel/Banquet"
    SPA = "Spa"
    LABORATOIRE = "Laboratoire"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieJob1(str, Enum):
    RESTAURATION = "Restauration"
    HOTELLERIE = "Hôtellerie"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieJob2(str, Enum):
    SALLE = "Salle & Service"
    DIRECTION = "Direction & Management"
    SUPPORT = "Support & Back-office"
    CUISINE = "Cuisine"
    SPA = "Spa & Bien-être"
    ETAGES = "Étages & Housekeeping"
    BAR = "Bar & Sommellerie"
    RECEPTION = "Réception & Hébergement"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieJob3(str, Enum):
    CHEF_EXECUTIF = "Chef exécutif"
    CHEF_CUISINE = "Chef de cuisine"
    SOUS_CHEF = "Sous-chef"
    CHEF_PARTIE = "Chef de partie"
    COMMIS_CUISINE = "Commis de cuisine"
    PATISSIER = "Pâtissier"
    BOULANGER = "Boulanger"
    PIZZAIOLO = "Pizzaiolo"
    TRAITEUR = "Traiteur"
    MANAGER = "Manager / Responsable"
    EMPLOYE = "Employé polyvalent"
    PLONGEUR = "Plongeur"
    STEWARD = "Steward"
    DIRECTEUR = "Directeur"
    RESPONSABLE_SALLE = "Responsable de salle"
    MAITRE_HOTEL = "Maître d’hôtel"
    CHEF_RANG = "Chef de rang"
    COMMIS_SALLE = "Commis de salle / Runner"
    SERVEUR = "Serveur"
    SOMMELIER = "Sommelier"
    BARMAN = "Barman"
    BARISTA = "Barista"
    RECEPTIONNISTE = "Réceptionniste / Hôte d’accueil"
    CONCIERGE = "Concierge"
    BAGAGISTE = "Bagagiste / Voiturier"
    VALET = "Valet / Femme de chambre"
    MARKETING = "Marketing / Communication"
    AGENT_RESERVATIONS = "Agent de réservations"
    REVENUE_MANAGER = "Revenue manager"
    GOUVERNANT = "Gouvernant(e)"
    SPA_PRATICIEN = "Spa praticien(ne) / Ésthéticien(ne)"
    COACH = "Coach sportif"
    MAITRE_NAGEUR = "Maître-nageur"
    ANIMATION = "Animation / Événementiel"
    COMMERCIAL = "Commercial"
    RH = "RH / Paie"
    COMPTABILITE = "Comptabilité / Contrôle de gestion"
    TECHNICIEN = "Technicien / Maintenance"
    IT = "IT / Data"
    HACCP = "HACCP manager"
    CUISINIER = "Cuisinier"
    LIMONADIER = "Limonadier"
    ALLOTISSEUR = "Allotisseur"
    APPROVISIONNEUR = "Approvisionneur / Économe"
    AGENT_SECURITE = "Agent de sécurité"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class Urgent(str, Enum):
    URGENT = "Urgent"
    NON_URGENT = "Non Urgent"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"

class Environnement(str, Enum):
    CENTRE_VILLE = "Centre ville"
    BORD_MER = "Bord de mer"
    MONTAGNE = "Montagne"
    BANLIEUE = "Banlieue"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class ChaineIndependant(str, Enum):
    CHAINE = "Chaine"
    INDEPENDANT = "Indépendant"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class TempsTravail(str, Enum):
    PLEIN_TEMPS = "Plein temps"
    TEMPS_PARTIEL = "Temps partiel"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class HorairesTravail(str, Enum):
    JOUR = "Jour"
    NUIT = "Nuit"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class Experience(str, Enum):
    DEBUTANT = "Débutant"
    CONFIRME = "Confirmé"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class DureeModel(BaseModel):
    value: str


class HeuresParSemaineModel(BaseModel):
    heures: Optional[int] = None

    # v2 field validator
    @field_validator("heures", mode="before")
    def parse_heures(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            match = re.search(r"\d+", v)
            if match:
                return int(match.group())
        return None

class DateDebutModel(BaseModel):
    value: str

class SalaireModel(BaseModel):
    value: str

# --- Base model that ties everything together ---
class JobClassification(BaseModel):
    IA_Logé: Loge
    IA_Type_de_contrat: TypeContrat
    IA_Salaire: SalaireModel
    IA_Couple_accepté: CoupleAccepte
    IA_Catégorie_établissement: CategorieEtablissement
    IA_Catégorie_Job_1: CategorieJob1
    IA_Catégorie_Job_2: CategorieJob2
    IA_Catégorie_Job_3: CategorieJob3
    IA_Urgent: Urgent
    IA_Date_de_début: DateDebutModel
    IA_Durée: DureeModel
    IA_Type_environnement: Environnement
    IA_Chaine_Indépendant: ChaineIndependant
    IA_Temps_de_travail: TempsTravail
    IA_Horaires_de_travail: HorairesTravail
    IA_Heures_par_semaine: HeuresParSemaineModel
    IA_Éxpérience: Experience

SYSTEM_PROMPT = """You are a classifier for job listings in the hospitality industry in France. You are an expert and absolutely have to respect the 
instructions. Each category can ONLY take one the value that are specified for it.
The success of my business depends on you so double check!!
    "IA_Logé": when accomodation or help with accomodation is provided "Logé" else "Non logé",
        "IA_Type_de_contrat": it MUST BE one of ["CDD", "CDI", "Stage", "Apprentissage", "Interim", "Extra", "Saisonnier", "Alternance"],
        "IA_Salaire": the highest salary offered in format "X€/heure" or "X€/mois" or "X€/an", or "" if not specified,
        "IA_Couple_accepté": "Couple accepté" or "",
    	"IA_Catégorie_établissement": it MUST BE one of the following and CANNOT be empty ["Gastronomique","Brasserie","Bar","Restauration rapide","Restauration collective","Restaurant","Hôtel luxe","Hôtel","Camping","Café/Salon de thé”,”Boulangerie/Patisserie”,”Etoile Michelin","Palace”, “Traiteur/Événementiel/Banquet”,“Spa”, “Laboratoire”],
    	"IA_Catégorie_Job_1":  it MUST BE one of the following and it cannot be empty [“Restauration”, “Hôtellerie”],
    	“IA_Catégorie_Job_2”:  it MUST BE one of and the most relevant, it cannot be empty [“Salle & Service”, “Direction & Management”, “Support & Back-office”, “Cuisine”, “Spa & Bien-être”, “Étages & Housekeeping”, “Bar & Sommellerie”, “Réception & Hébergement”],
        “IA_Catégorie_Job_3”: it has to be one of the following and the most relevant, it cannot be empty ["Chef exécutif","Chef de cuisine","Sous-chef","Chef de partie","Commis de cuisine","Pâtissier","Boulanger","Pizzaiolo","Traiteur","Manager / Responsable","Employé polyvalent","Plongeur","Steward","Directeur","Responsable de salle","Maître d’hôtel","Chef de rang","Commis de salle / Runner","Serveur","Sommelier","Barman","Barista","Réceptionniste / Hôte d’accueil","Concierge","Bagagiste / Voiturier","Valet / Femme de chambre","Marketing / Communication","Agent de réservations","Revenue manager","Gouvernant(e)","Spa praticien(ne) / Ésthéticien(ne)","Coach sportif","Maître-nageur","Animation / Événementiel","Commercial","RH / Paie","Comptabilité / Contrôle de gestion","Technicien / Maintenance","IT / Data","HACCP manager","Cuisinier","Limonadier","Allotisseur","Approvisionneur / Économe","Agent de sécurité"],
    	"IA_Urgent": "Urgent" or "", it takes "Urgent" only when the starting date is within 2 weeks of the date_scraping or when it is explicitly mentioned in the description
        "IA_Date_de_début": starting date in format YYYY-MM-DD if present, else "",
        "IA_Durée": contract duration like "N days", "N weeks", "N months", or "Indéfini",
        "IA_Type_environnement”: one of ["Centre ville","Bord de mer","Montagne","Banlieue"],
    	“IA_Chaine_Indépendant”: when the company posting the job listing is part of a group or bigger company "Chaine", else ""
        "IA_Temps_de_travail": "Plein temps" or "Temps partiel",
        "IA_Horaires_de_travail": "Jour" or "Nuit",
        "IA_Heures_par_semaine": return a number not a string ! the number of hours worked per week if available, when the contract is less than a week just put how many hours it , else “”,
    	“IA_Éxpérience” one the following [“Débutant”, “Confirmé”]

    Strictly output without explanations."""


def classify_job_listing(ticket_text: str) -> JobClassification:
    response = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            max_retries=3,
            response_model=JobClassification,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ticket_text}
            ],
            temperature=0
        )
    return response

# Convert each row into a single string with "col":"value" format
new_data["row_as_string"] = new_data.apply(
    lambda row: ", ".join([f'"{col}":"{row[col]}"' for col in new_data.columns]),
    axis=1
)

# Apply your classify_job_listing function to each row
result = new_data["row_as_string"].apply(classify_job_listing)

# If you want, convert the results (list of dicts) into a DataFrame
classified_df = pd.DataFrame(result.tolist())

base_model_columns = list(JobClassification.model_fields.keys())

def get_value(cell, column_name=None):
    if isinstance(cell, tuple) and len(cell) == 2:
        val = cell[1]

        # Special case for IA_Heures_par_semaine
        if column_name == "IA_Heures_par_semaine" and hasattr(val, "heures"):
            return val.heures  # directly the int

        # Other enums / objects
        if hasattr(val, "value"):
            return val.value
        return str(val)
    elif hasattr(cell, "value"):
        return cell.value
    return str(cell)

classified_df = pd.DataFrame([
    [get_value(cell, col) for cell, col in zip(row, base_model_columns)]
    for row in classified_df.values
], columns=base_model_columns)

new_data = new_data.drop(columns=["row_as_string"])

# -------- FIN CHATGPT DATA ENRICHMENT ----------------------------------------------------------------------------------------------
# Merge with original sample
new_data = pd.concat([new_data.reset_index(drop=True), classified_df], axis=1)

# Convert list to Pandas DataFrame
resultats_part2 = pd.DataFrame(job_data)

#Concat axis 0 resultats_part1 and resultats_part2
df_jobs = pd.concat([resultats_part1, resultats_part2], axis=1)

def fusionner_salaire(row):
    if pd.notna(row['salaire_2']):
        return row['salaire_2']
    elif pd.notna(row['salaire_1']):
        return row['salaire_1']
    else:
        return None

df_jobs["salaire"] = df_jobs.apply(fusionner_salaire, axis=1)
df_jobs = df_jobs.drop(columns=["salaire_1", "salaire_2"])

# Google Sheets API setup
scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

credentials_info = json.loads(os.environ.get("GOOGLE_CREDENTIALS"))
credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
client = gspread.authorize(credentials)

# Open the Google Sheet
spreadsheet = client.open('hoscoScrapper')  # Use your sheet's name
worksheet = spreadsheet.sheet1


# Read existing data from Google Sheets into a DataFrame
existing_data = pd.DataFrame(worksheet.get_all_records())

# Convert scraped results into a DataFrame
new_data = df_jobs

# Apply nest_asyncio to fix event loop issue in Jupyter
nest_asyncio.apply()

# Data Gouv API URL
API_URL = "https://api-adresse.data.gouv.fr/search"

# Function to call API asynchronously with retries
async def get_geodata(client, address, retries=3):
    params = {"q": address, "limit": 1}

    for attempt in range(retries):
        try:
            response = await client.get(API_URL, params=params, timeout=5)

            if response.status_code == 503:  # Server overloaded
                print(f"503 Error - Retrying {address} (Attempt {attempt+1})...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue

            response.raise_for_status()  # Raise error if response is bad
            data = response.json()

            if data["features"]:
                props = data["features"][0]["properties"]
                geo = data["features"][0]["geometry"]["coordinates"]

                ville = props.get("city", "")
                code_postal = props.get("postcode", "")
                longitude = geo[0] if geo else None
                latitude = geo[1] if geo else None
                contexte = props.get("context", "")

                # Extract region name (after second comma)
                region = contexte.split(", ")[-1] if contexte.count(",") >= 2 else ""

                return ville, code_postal, longitude, latitude, region
        
        except Exception as e:
            print(f"Error fetching data for {address} (Attempt {attempt+1}): {e}")
        
        await asyncio.sleep(2 ** attempt)  # Exponential backoff for retries

    return None, None, None, None, None  # Return empty values if all retries fail

# Async function to process all addresses with rate limiting
async def process_addresses(address_list, delay_between_requests=0.017):  # 1/60 = ~0.017s
    results = []
    async with httpx.AsyncClient() as client:
        for i, address in enumerate(address_list):
            result = await get_geodata(client, address)
            results.append(result)
            
            print(f"Processed {i + 1} / {len(address_list)}")

            # Respect 60 requests per second limit
            await asyncio.sleep(delay_between_requests)  

    return results

# Run API calls asynchronously
addresses = new_data["localisation"].tolist()
geodata_results = asyncio.run(process_addresses(addresses))

# Assign the results to the DataFrame
new_data[["Ville", "Code Postal", "Longitude", "Latitude", "Region"]] = pd.DataFrame(geodata_results)

# Add "France Travail" column
new_data["Source"] = "hosco"

# -------- DEBUT CHATGPT DATA ENRICHMENT --------------------------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_ai = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))

class Loge(str, Enum):
    LOGE = "Logé"
    NON_LOGE = "Non Logé"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"

class TypeContrat(str, Enum):
    CDD = "CDD"
    CDI = "CDI"
    STAGE = "Stage"
    APPRENTISSAGE = "Apprentissage"
    INTERIM = "Interim"
    EXTRA = "Extra"
    SAISONNIER = "Saisonnier"
    ALTERNANCE = "Alternance"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CoupleAccepte(str, Enum):
    ACCEPTE = "Couple accepté"
    NON_ACCEPTE = "Couple non accepté"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieEtablissement(str, Enum):
    GASTRONOMIQUE = "Gastronomique"
    BRASSERIE = "Brasserie"
    BAR = "Bar"
    RAPIDE = "Restauration rapide"
    COLLECTIVE = "Restauration collective"
    RESTAURANT = "Restaurant"
    HOTEL_LUXE = "Hôtel luxe"
    HOTEL = "Hôtel"
    CAMPING = "Camping"
    CAFE = "Café/Salon de thé"
    BOULANGERIE = "Boulangerie/Patisserie"
    ETOILE = "Etoile Michelin"
    PALACE = "Palace"
    TRAITEUR = "Traiteur/Événementiel/Banquet"
    SPA = "Spa"
    LABORATOIRE = "Laboratoire"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieJob1(str, Enum):
    RESTAURATION = "Restauration"
    HOTELLERIE = "Hôtellerie"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieJob2(str, Enum):
    SALLE = "Salle & Service"
    DIRECTION = "Direction & Management"
    SUPPORT = "Support & Back-office"
    CUISINE = "Cuisine"
    SPA = "Spa & Bien-être"
    ETAGES = "Étages & Housekeeping"
    BAR = "Bar & Sommellerie"
    RECEPTION = "Réception & Hébergement"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieJob3(str, Enum):
    CHEF_EXECUTIF = "Chef exécutif"
    CHEF_CUISINE = "Chef de cuisine"
    SOUS_CHEF = "Sous-chef"
    CHEF_PARTIE = "Chef de partie"
    COMMIS_CUISINE = "Commis de cuisine"
    PATISSIER = "Pâtissier"
    BOULANGER = "Boulanger"
    PIZZAIOLO = "Pizzaiolo"
    TRAITEUR = "Traiteur"
    MANAGER = "Manager / Responsable"
    EMPLOYE = "Employé polyvalent"
    PLONGEUR = "Plongeur"
    STEWARD = "Steward"
    DIRECTEUR = "Directeur"
    RESPONSABLE_SALLE = "Responsable de salle"
    MAITRE_HOTEL = "Maître d’hôtel"
    CHEF_RANG = "Chef de rang"
    COMMIS_SALLE = "Commis de salle / Runner"
    SERVEUR = "Serveur"
    SOMMELIER = "Sommelier"
    BARMAN = "Barman"
    BARISTA = "Barista"
    RECEPTIONNISTE = "Réceptionniste / Hôte d’accueil"
    CONCIERGE = "Concierge"
    BAGAGISTE = "Bagagiste / Voiturier"
    VALET = "Valet / Femme de chambre"
    MARKETING = "Marketing / Communication"
    AGENT_RESERVATIONS = "Agent de réservations"
    REVENUE_MANAGER = "Revenue manager"
    GOUVERNANT = "Gouvernant(e)"
    SPA_PRATICIEN = "Spa praticien(ne) / Ésthéticien(ne)"
    COACH = "Coach sportif"
    MAITRE_NAGEUR = "Maître-nageur"
    ANIMATION = "Animation / Événementiel"
    COMMERCIAL = "Commercial"
    RH = "RH / Paie"
    COMPTABILITE = "Comptabilité / Contrôle de gestion"
    TECHNICIEN = "Technicien / Maintenance"
    IT = "IT / Data"
    HACCP = "HACCP manager"
    CUISINIER = "Cuisinier"
    LIMONADIER = "Limonadier"
    ALLOTISSEUR = "Allotisseur"
    APPROVISIONNEUR = "Approvisionneur / Économe"
    AGENT_SECURITE = "Agent de sécurité"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class Urgent(str, Enum):
    URGENT = "Urgent"
    NON_URGENT = "Non Urgent"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"

class Environnement(str, Enum):
    CENTRE_VILLE = "Centre ville"
    BORD_MER = "Bord de mer"
    MONTAGNE = "Montagne"
    BANLIEUE = "Banlieue"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class ChaineIndependant(str, Enum):
    CHAINE = "Chaine"
    INDEPENDANT = "Indépendant"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class TempsTravail(str, Enum):
    PLEIN_TEMPS = "Plein temps"
    TEMPS_PARTIEL = "Temps partiel"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class HorairesTravail(str, Enum):
    JOUR = "Jour"
    NUIT = "Nuit"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class Experience(str, Enum):
    DEBUTANT = "Débutant"
    CONFIRME = "Confirmé"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class DureeModel(BaseModel):
    value: str


class HeuresParSemaineModel(BaseModel):
    heures: Optional[int] = None

    # v2 field validator
    @field_validator("heures", mode="before")
    def parse_heures(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            match = re.search(r"\d+", v)
            if match:
                return int(match.group())
        return None

class DateDebutModel(BaseModel):
    value: str

class SalaireModel(BaseModel):
    value: str

# --- Base model that ties everything together ---
class JobClassification(BaseModel):
    IA_Logé: Loge
    IA_Type_de_contrat: TypeContrat
    IA_Salaire: SalaireModel
    IA_Couple_accepté: CoupleAccepte
    IA_Catégorie_établissement: CategorieEtablissement
    IA_Catégorie_Job_1: CategorieJob1
    IA_Catégorie_Job_2: CategorieJob2
    IA_Catégorie_Job_3: CategorieJob3
    IA_Urgent: Urgent
    IA_Date_de_début: DateDebutModel
    IA_Durée: DureeModel
    IA_Type_environnement: Environnement
    IA_Chaine_Indépendant: ChaineIndependant
    IA_Temps_de_travail: TempsTravail
    IA_Horaires_de_travail: HorairesTravail
    IA_Heures_par_semaine: HeuresParSemaineModel
    IA_Éxpérience: Experience

SYSTEM_PROMPT = """You are a classifier for job listings in the hospitality industry in France. You are an expert and absolutely have to respect the 
instructions. Each category can ONLY take one the value that are specified for it.
The success of my business depends on you so double check!!
    "IA_Logé": when accomodation or help with accomodation is provided "Logé" else "Non logé",
        "IA_Type_de_contrat": it MUST BE one of ["CDD", "CDI", "Stage", "Apprentissage", "Interim", "Extra", "Saisonnier", "Alternance"],
        "IA_Salaire": the highest salary offered in format "X€/heure" or "X€/mois" or "X€/an", or "" if not specified,
        "IA_Couple_accepté": "Couple accepté" or "",
    	"IA_Catégorie_établissement": it MUST BE one of the following and CANNOT be empty ["Gastronomique","Brasserie","Bar","Restauration rapide","Restauration collective","Restaurant","Hôtel luxe","Hôtel","Camping","Café/Salon de thé”,”Boulangerie/Patisserie”,”Etoile Michelin","Palace”, “Traiteur/Événementiel/Banquet”,“Spa”, “Laboratoire”],
    	"IA_Catégorie_Job_1":  it MUST BE one of the following and it cannot be empty [“Restauration”, “Hôtellerie”],
    	“IA_Catégorie_Job_2”:  it MUST BE one of and the most relevant, it cannot be empty [“Salle & Service”, “Direction & Management”, “Support & Back-office”, “Cuisine”, “Spa & Bien-être”, “Étages & Housekeeping”, “Bar & Sommellerie”, “Réception & Hébergement”],
        “IA_Catégorie_Job_3”: it has to be one of the following and the most relevant, it cannot be empty ["Chef exécutif","Chef de cuisine","Sous-chef","Chef de partie","Commis de cuisine","Pâtissier","Boulanger","Pizzaiolo","Traiteur","Manager / Responsable","Employé polyvalent","Plongeur","Steward","Directeur","Responsable de salle","Maître d’hôtel","Chef de rang","Commis de salle / Runner","Serveur","Sommelier","Barman","Barista","Réceptionniste / Hôte d’accueil","Concierge","Bagagiste / Voiturier","Valet / Femme de chambre","Marketing / Communication","Agent de réservations","Revenue manager","Gouvernant(e)","Spa praticien(ne) / Ésthéticien(ne)","Coach sportif","Maître-nageur","Animation / Événementiel","Commercial","RH / Paie","Comptabilité / Contrôle de gestion","Technicien / Maintenance","IT / Data","HACCP manager","Cuisinier","Limonadier","Allotisseur","Approvisionneur / Économe","Agent de sécurité"],
    	"IA_Urgent": "Urgent" or "", it takes "Urgent" only when the starting date is within 2 weeks of the date_scraping or when it is explicitly mentioned in the description
        "IA_Date_de_début": starting date in format YYYY-MM-DD if present, else "",
        "IA_Durée": contract duration like "N days", "N weeks", "N months", or "Indéfini",
        "IA_Type_environnement”: one of ["Centre ville","Bord de mer","Montagne","Banlieue"],
    	“IA_Chaine_Indépendant”: when the company posting the job listing is part of a group or bigger company "Chaine", else ""
        "IA_Temps_de_travail": "Plein temps" or "Temps partiel",
        "IA_Horaires_de_travail": "Jour" or "Nuit",
        "IA_Heures_par_semaine": return a number not a string ! the number of hours worked per week if available, when the contract is less than a week just put how many hours it , else “”,
    	“IA_Éxpérience” one the following [“Débutant”, “Confirmé”]

    Strictly output without explanations."""


def classify_job_listing(ticket_text: str) -> JobClassification:
    response = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            max_retries=3,
            response_model=JobClassification,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ticket_text}
            ],
            temperature=0
        )
    return response

# Convert each row into a single string with "col":"value" format
new_data["row_as_string"] = new_data.apply(
    lambda row: ", ".join([f'"{col}":"{row[col]}"' for col in new_data.columns]),
    axis=1
)

# Apply your classify_job_listing function to each row
result = new_data["row_as_string"].apply(classify_job_listing)

# If you want, convert the results (list of dicts) into a DataFrame
classified_df = pd.DataFrame(result.tolist())

base_model_columns = list(JobClassification.model_fields.keys())

def get_value(cell, column_name=None):
    if isinstance(cell, tuple) and len(cell) == 2:
        val = cell[1]

        # Special case for IA_Heures_par_semaine
        if column_name == "IA_Heures_par_semaine" and hasattr(val, "heures"):
            return val.heures  # directly the int

        # Other enums / objects
        if hasattr(val, "value"):
            return val.value
        return str(val)
    elif hasattr(cell, "value"):
        return cell.value
    return str(cell)

classified_df = pd.DataFrame([
    [get_value(cell, col) for cell, col in zip(row, base_model_columns)]
    for row in classified_df.values
], columns=base_model_columns)

new_data = new_data.drop(columns=["row_as_string"])

# -------- FIN CHATGPT DATA ENRICHMENT ----------------------------------------------------------------------------------------------
# Merge with original sample
new_data = pd.concat([new_data.reset_index(drop=True), classified_df], axis=1)

new_data = new_data[(new_data["titre"] != "") & (new_data["Ville"] != "") & (new_data["Ville"] != "0")]

print(f"Post geo new data Check length {len(new_data)}")
print(f"Post geo Check existing length {len(existing_data)}")

# Combine and remove duplicates
if not existing_data.empty:
    print(len(pd.concat([existing_data, new_data], ignore_index=True).drop_duplicates(subset=['url'])))
    combined_data = pd.concat([existing_data, new_data], ignore_index=True).drop_duplicates(
        subset=['url']
    )
else:
    combined_data = new_data

# -------- DEBUT DATA VALIDATION EMPTY VALUES OPENAI ----------------------------------------------------------------------------------------------

# Select columns starting with "IA_"
ia_cols = [col for col in combined_data.columns if col.startswith("IA_")]

# Replace "" with "Non spécifié" in those columns only
combined_data[ia_cols] = combined_data[ia_cols].replace("", "Non spécifié")

# -------- FIN DATA VALIDATION EMPTY VALUES OPENAI ----------------------------------------------------------------------------------------------



print(f"Post concat Check combined_data length {len(combined_data)}")

# Debug: Print the number of rows to append
rows_to_append = new_data.shape[0]
print(f"Rows to append: {rows_to_append}")

# Handle NaN, infinity values before sending to Google Sheets
# Replace NaN values with 0 or another placeholder (you can customize this)
combined_data = combined_data.fillna(0)

# Replace infinite values with 0 or another placeholder
combined_data.replace([float('inf'), float('-inf')], 0, inplace=True)

# Optional: Ensure all float types are valid (e.g., replace any invalid float with 0)
combined_data = combined_data.applymap(lambda x: 0 if isinstance(x, float) and (x == float('inf') or x == float('-inf') or x != x) else x)

# Optional: Ensuring no invalid values (like lists or dicts) in any column
def clean_value(value):
    if isinstance(value, (list, dict)):
        return str(value)  # Convert lists or dicts to string
    return value

combined_data = combined_data.applymap(clean_value)

#add column titre de annonce sans accents ni special characters
def remove_accents_and_special(text):
    # Normalize the text to separate characters from their accents.
    normalized = unicodedata.normalize('NFD', text)
    # Remove the combining diacritical marks.
    without_accents = ''.join(c for c in normalized if not unicodedata.combining(c))
    # Replace special characters (-, ') with a space.
    cleaned = re.sub(r"[-']", " ", without_accents)
    # Remove other special characters (retain letters, digits, and whitespace).
    cleaned = re.sub(r"[^A-Za-z0-9\s]", "", cleaned)
    return cleaned

# Create the new column "Titre annonce sans accent" by applying the function on "intitule".
combined_data["TitreAnnonceSansAccents"] = combined_data["titre"].apply(
    lambda x: remove_accents_and_special(x) if isinstance(x, str) else x
)

print(f"Post concat Check combined_data length {len(combined_data)}")

# Update Google Sheets with the combined data
worksheet.clear()  # Clear existing content
worksheet.update([combined_data.columns.tolist()] + combined_data.values.tolist())
