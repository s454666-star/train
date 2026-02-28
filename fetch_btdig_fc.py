import re
import time
import urllib.parse
import pymysql
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

DB_CONFIG = {
    "host": "mysql.mystar.monster",
    "user": "s454666",
    "password": "i06180318",
    "database": "star",
    "port": 3306,
    "charset": "utf8mb4"
}


def connect_db():
    return pymysql.connect(**DB_CONFIG)


def safe_str(value, max_len):
    if value is None:
        return ""
    s = str(value)
    if max_len is None:
        return s
    if len(s) <= max_len:
        return s
    return s[:max_len]


def normalize_keyword(s):
    return re.sub(r"\s+", "", (s or "")).strip().lower()


def result_exists_by_detail_url(conn, keyword, type_value, detail_url):
    sql = "SELECT 1 FROM btdig_results WHERE search_keyword = %s AND type = %s AND detail_url = %s LIMIT 1"
    with conn.cursor() as cursor:
        cursor.execute(sql, (keyword, type_value, detail_url))
        return cursor.fetchone() is not None


def insert_to_db(conn, keyword, type_value, detail_url, info):
    sql = """
    INSERT INTO btdig_results
    (search_keyword, type, detail_url, magnet, name, size, age, files)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    with conn.cursor() as cursor:
        cursor.execute(
            sql,
            (
                safe_str(keyword, 100),
                safe_str(type_value, 2),
                safe_str(detail_url, 500),
                safe_str(info.get("magnet", ""), None),
                safe_str(info.get("name", ""), None),
                safe_str(info.get("size", ""), 100),
                safe_str(info.get("age", ""), 100),
                safe_str(info.get("files_count", ""), 100),
            ),
        )
    conn.commit()


def create_driver():
    options = Options()

    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1400,900")

    options.add_argument("--headless=new")

    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 1,
        "profile.managed_default_content_settings.fonts": 2,
        "profile.managed_default_content_settings.notifications": 2,
    }
    options.add_experimental_option("prefs", prefs)

    service = Service("chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(25)
    return driver


def wait_search_done(driver, timeout=10):
    wait = WebDriverWait(driver, timeout)

    def done_condition(drv):
        html = drv.page_source or ""
        if "0 results found" in html:
            return True
        try:
            drv.find_element(By.CSS_SELECTOR, ".torrent_name a")
            return True
        except Exception:
            return False

    try:
        wait.until(lambda d: done_condition(d))
        return True
    except TimeoutException:
        return False


def wait_detail_ready(driver, timeout=10):
    wait = WebDriverWait(driver, timeout)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[href^='magnet:?']")))
        return True
    except TimeoutException:
        return False


def get_detail_links_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select(".torrent_name a"):
        href = a.get("href")
        if href and href.startswith("https://en.btdig.com/"):
            links.append(href)

    unique = []
    seen = set()
    for u in links:
        if u in seen:
            continue
        seen.add(u)
        unique.append(u)
    return unique


def find_row_value(soup, label):
    td = soup.find("td", string=label)
    if not td:
        return ""
    value_td = td.find_next_sibling("td")
    return value_td.get_text(strip=True) if value_td else ""


def extract_files_list_text(soup):
    th = None
    for x in soup.select("tr th"):
        if normalize_keyword(x.get_text()) == "files":
            th = x
            break
    if not th:
        th = soup.find("th", string=lambda t: normalize_keyword(t) == "files")
    if not th:
        return ""

    tr = th.find_parent("tr")
    if not tr:
        return ""

    next_tr = tr.find_next_sibling("tr")
    if not next_tr:
        return ""

    td = next_tr.find("td")
    if not td:
        return ""

    return td.get_text("\n", strip=True) or ""


def parse_size_to_bytes(size_text):
    if not size_text:
        return None

    s = size_text.replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)", s)
    if not m:
        return None

    value = float(m.group(1))
    unit = m.group(2).strip().lower()

    unit_map = {
        "b": 1,
        "bytes": 1,
        "kb": 1000,
        "k": 1000,
        "mb": 1000 ** 2,
        "m": 1000 ** 2,
        "gb": 1000 ** 3,
        "g": 1000 ** 3,
        "tb": 1000 ** 4,
        "t": 1000 ** 4,
        "kib": 1024,
        "mib": 1024 ** 2,
        "gib": 1024 ** 3,
        "tib": 1024 ** 4,
    }

    mul = unit_map.get(unit)
    if mul is None:
        unit2 = unit if unit.endswith("b") else unit + "b"
        mul = unit_map.get(unit2)

    if mul is None:
        return None

    return int(value * mul)


def size_in_range_1g_15g(size_text):
    b = parse_size_to_bytes(size_text)
    if b is None:
        return False
    min_b = 1 * (1024 ** 3)
    max_b = 15 * (1024 ** 3)
    return min_b <= b <= max_b


def parse_detail_html(html):
    soup = BeautifulSoup(html, "html.parser")

    magnet = ""
    magnet_a = soup.select_one("a[href^='magnet:?']")
    if magnet_a:
        magnet = magnet_a.get("href", "")

    canonical = ""
    canonical_el = soup.select_one("link[rel='canonical']")
    if canonical_el:
        canonical = canonical_el.get("href", "")

    q_value = ""
    q_el = soup.select_one("input#q")
    if q_el:
        q_value = q_el.get("value", "")

    files_count = find_row_value(soup, "Files:")
    files_list_text = extract_files_list_text(soup)

    return {
        "magnet": magnet,
        "name": find_row_value(soup, "Name:"),
        "size": find_row_value(soup, "Size:"),
        "age": find_row_value(soup, "Age:"),
        "files_count": files_count,
        "files_list_text": files_list_text,
        "page_title": (soup.title.get_text(strip=True) if soup.title else ""),
        "canonical": canonical,
        "q_value": q_value,
    }


def detail_page_has_exact_keyword(info, keyword):
    k = normalize_keyword(keyword)
    candidates = [
        info.get("name", ""),
        info.get("page_title", ""),
        info.get("canonical", ""),
        info.get("q_value", ""),
        info.get("files_list_text", ""),
    ]
    for c in candidates:
        if k in normalize_keyword(c):
            return True
    return False


def files_list_has_exact_keyword(info, keyword):
    k = normalize_keyword(keyword)
    t = info.get("files_list_text", "") or ""
    if not t.strip():
        return False
    return k in normalize_keyword(t)


def process_keyword(driver, conn, keyword, type_value=2, max_detail_pages=5, max_inserts_per_keyword=5):
    quoted = f"\"{keyword}\""
    q = urllib.parse.quote_plus(quoted)
    search_url = f"https://en.btdig.com/search?order=0&q={q}"
    print(f"開始抓 {keyword} (type={type_value}) query={quoted}")

    try:
        driver.get(search_url)
    except WebDriverException as ex:
        print(f"搜尋頁載入失敗: {ex}")
        return

    wait_search_done(driver, timeout=10)
    html = driver.page_source or ""

    links = get_detail_links_from_html(html)
    if not links:
        print("無搜尋結果，跳過")
        return

    checked = 0
    inserted = 0

    for link in links:
        checked = checked + 1
        if checked > max_detail_pages:
            break

        if result_exists_by_detail_url(conn, keyword, str(type_value), link):
            print(f"已存在(同 detail_url)，跳過: {link}")
            continue

        try:
            driver.get(link)
        except WebDriverException as ex:
            print(f"明細頁載入失敗: {ex} | {link}")
            continue

        wait_detail_ready(driver, timeout=10)
        info = parse_detail_html(driver.page_source or "")

        if not detail_page_has_exact_keyword(info, keyword):
            print(f"明細頁未完全匹配 {keyword}，跳過: {link}")
            continue

        if not files_list_has_exact_keyword(info, keyword):
            print(f"Files 清單未匹配 {keyword}，跳過: {link}")
            continue

        if not size_in_range_1g_15g(info.get("size", "")):
            print(f"Size 不在 1G~15G，跳過: {info.get('size', '')} | {link}")
            continue

        try:
            insert_to_db(conn, keyword, str(type_value), link, info)
        except Exception as ex:
            print(f"寫入失敗: {ex} | {link}")
            continue

        inserted = inserted + 1
        print(f"已寫入 {keyword} (type={type_value}) files={info.get('files_count', '')} url={link}")

        if inserted >= max_inserts_per_keyword:
            print(f"{keyword} 已達本次上限 {max_inserts_per_keyword} 筆，結束該 keyword")
            return

    if inserted == 0:
        print(f"{keyword} 本次前 {max_detail_pages} 筆都沒有符合條件的明細頁，略過")


def main():
    start_no = 1181072
    count = 100000
    type_value = 2

    driver = create_driver()
    conn = connect_db()

    try:
        for n in range(start_no, start_no - count, -1):
            keyword = f"FC2-PPV-{n}"
            process_keyword(
                driver,
                conn,
                keyword,
                type_value=type_value,
                max_detail_pages=5,
                max_inserts_per_keyword=5
            )
            time.sleep(0.1)
    finally:
        driver.quit()
        conn.close()


if __name__ == "__main__":
    main()