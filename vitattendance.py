import re
from playwright.sync_api import Playwright, sync_playwright, expect
import time


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://vtop.vit.ac.in/vtop/login")
    page.get_by_role("button", name="").click()
    page.get_by_placeholder("Username").click()
    page.get_by_placeholder("Username").press("CapsLock")
    page.get_by_placeholder("Username").fill("ANKITJOSEPH0305")
    page.get_by_placeholder("Password").click()
    page.get_by_placeholder("Password").fill("")
    page.get_by_placeholder("Password").fill("AlakazamBlast#230303")
    page.wait_for_load_state("networkidle")
    time.sleep(10)
    page.get_by_role("button", name="Submit").click()
    page.wait_for_load_state("networkidle")
    page.get_by_role("link", name=" Class Attendance").click()
    page.locator("#semesterSubId").select_option("VL20232405")
    time.sleep(10)

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
