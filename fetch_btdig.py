import time
import pymysql
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

DB_CONFIG = {
    "host": "mysql.mystar.monster",
    "user": "s454666",
    "password": "i06180318",
    "database": "star",
    "port": 3306,
    "charset": "utf8mb4"
}


def create_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    service = Service("chromedriver.exe")
    return webdriver.Chrome(service=service, options=options)


def connect_db():
    return pymysql.connect(**DB_CONFIG)


def keyword_exists(conn, keyword):
    sql = "SELECT 1 FROM btdig_results WHERE search_keyword = %s LIMIT 1"
    with conn.cursor() as cursor:
        cursor.execute(sql, (keyword,))
        return cursor.fetchone() is not None


def insert_to_db(conn, keyword, detail_url, info):
    sql = """
    INSERT INTO btdig_results
    (search_keyword, detail_url, magnet, name, size, age, files)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (
            keyword,
            detail_url,
            info["magnet"],
            info["name"],
            info["size"],
            info["age"],
            info["files"]
        ))
    conn.commit()


def size_to_gb(size_str):
    if not size_str:
        return 0

    size_str = size_str.replace("\xa0", " ").strip()
    parts = size_str.split()

    if len(parts) != 2:
        return 0

    try:
        value = float(parts[0])
    except:
        return 0

    unit = parts[1].upper()

    if unit == "GB":
        return value
    if unit == "MB":
        return value / 1024
    if unit == "KB":
        return value / (1024 * 1024)
    if unit == "TB":
        return value * 1024

    return 0


def parse_detail(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")

    def find_row(label):
        td = soup.find("td", string=label)
        if not td:
            return ""
        value_td = td.find_next_sibling("td")
        return value_td.get_text(strip=True) if value_td else ""

    magnet = ""
    magnet_a = soup.select_one("a[href^='magnet:?']")
    if magnet_a:
        magnet = magnet_a.get("href")

    return {
        "magnet": magnet,
        "name": find_row("Name:"),
        "size": find_row("Size:"),
        "age": find_row("Age:"),
        "files": find_row("Files:")
    }


def get_detail_links(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    links = []

    for a in soup.select(".torrent_name a"):
        href = a.get("href")
        if href and href.startswith("https://en.btdig.com/"):
            links.append(href)

    return links


def process_keyword(driver, conn, keyword):
    if keyword_exists(conn, keyword):
        print(f"{keyword} 已存在，跳過")
        return

    search_url = f"https://en.btdig.com/search?q=tokyo-hot+{keyword}-fhd"
    print(f"開始抓 {keyword}")

    driver.get(search_url)
    time.sleep(5)

    links = get_detail_links(driver)

    if not links:
        print("無搜尋結果")
        return

    for link in links[:5]:
        driver.get(link)
        time.sleep(3)

        info = parse_detail(driver)

        size_gb = size_to_gb(info["size"])

        if size_gb < 2:
            print("容量小於 2GB，跳過")
            continue

        if size_gb > 15:
            print("容量大於 15GB，跳過")
            continue

        insert_to_db(conn, keyword, link, info)
        print(f"已寫入 {keyword} ({size_gb:.2f} GB)")


def main():
    driver = create_driver()
    conn = connect_db()

    try:
        for i in range(798, 801):
            keyword = f"n{i:04d}"
            process_keyword(driver, conn, keyword)
            time.sleep(3)

    finally:
        driver.quit()
        conn.close()


if __name__ == "__main__":
    main()