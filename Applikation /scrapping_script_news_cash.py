scraping_script.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import re
import sqlite3
from datetime import datetime, timedelta

DB_PATH = r"C:\Bachelorarbeit\Datenbank\news_cache.db"
BASE_URL = "https://www.cash.ch/news/alle"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT UNIQUE,
            date TEXT,
            pre_checked INTEGER,
            gpt_checked INTEGER DEFAULT 0,
            relevant INTEGER,
            stock TEXT,
            impact INTEGER,
            source TEXT,
            source_quality INTEGER DEFAULT 3,
            group_id TEXT,
            article_text TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_article_if_new(title, url, date):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO articles (title, url, date, source)
            VALUES (?, ?, ?, ?)
        """, (title, url, date, "cash.ch"))
        conn.commit()
        print("✅ Neuer Artikel gespeichert")
    except sqlite3.IntegrityError:
        print("🔁 Artikel bereits vorhanden – wird übersprungen")
    finally:
        conn.close()

def build_page_url(page_index: int) -> str:
    """Seite 0 = BASE_URL, danach ?page=1,2,..."""
    if page_index <= 0:
        return BASE_URL
    return f"{BASE_URL}?page={page_index}"

def scrape_cash_ch_news():
    # Ziel-Datum (Vortag) vorbereiten
    yesterday_str = (datetime.now().date() - timedelta(days=1)).strftime("%d.%m.%Y")
    print(f"⏱️  Stopp-Kriterium: erster Artikel mit Datum {yesterday_str}")

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)

    page_index = 0
    stop_all = False

    try:
        while not stop_all:
            page_url = build_page_url(page_index)
            print(f"\n📄 Lade Seite {page_index}: {page_url}")
            driver.get(page_url)

            if page_index == 0:
                try:
                    accept_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
                    )
                    accept_btn.click()
                    print("🍪 Cookie-Banner akzeptiert")
                except Exception:
                    print("🍪 Kein Cookie-Banner (oder bereits akzeptiert)")

            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, "html.parser")

            all_links = soup.find_all("a", href=True)
            news_links = [a for a in all_links if a["href"].startswith("/news/") and len(a.get_text(strip=True)) > 30]

            print(f"🔎 Gefundene News-Artikel auf Seite {page_index}: {len(news_links)}")

            if not news_links:
                print("ℹ️  Keine weiteren Artikel gefunden – beende.")
                break

            for a in news_links:
                raw_text = a.get_text(strip=True)
                url = "https://www.cash.ch" + a["href"]

                match = re.search(r"\d{2}\.\d{2}\.\d{4}", raw_text)
                if match:
                    article_date = match.group()
                    title = raw_text[:match.start()].strip()
                else:
                    article_date = "Unbekannt"
                    title = raw_text.strip()

                print(f"📰 Titel: {title}")
                print(f"🔗 Link:  {url}")
                print(f"📅 Datum: {article_date}")

                if article_date == yesterday_str:
                    print("🛑 Vortag erreicht – stoppe das Auslesen weiterer Artikel und Seiten.")
                    stop_all = True
                    break

                save_article_if_new(title, url, article_date)
                print("-" * 60)

            # Wenn Stopp-Kriterium ausgelöst wurde: Schleife verlassen
            if stop_all:
                break

            page_index += 1

    finally:
        driver.quit()
        print("\n✅ Browser geschlossen.")

if __name__ == "__main__":
    init_db()
    scrape_cash_ch_news()
