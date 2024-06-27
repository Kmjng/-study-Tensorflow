# -*- coding: utf-8 -*-
"""
1. google 페이지 이동 
2. 입력상자 가져오기 
3. 검색 입력 -> 엔터  
4. 검색 페이지 이동
"""

from selenium import webdriver # driver 
from selenium.webdriver.chrome.service import Service # Chrom 서비스
from webdriver_manager.chrome import ChromeDriverManager # 크롬드라이버 관리자 
from selenium.webdriver.common.by import By # 로케이터(locator) 제공
from selenium.webdriver.common.keys import Keys # 엔터키 사용(Keys.ENTER) 
import time

# 1. driver 객체 생성 
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))


# 2. 대상 url 이동 
driver.get('https://www.google.co.kr/') # google 페이지 이동 


# 3. 검색 입력상자 tag -> 검색어 입력   
search_box = driver.find_element(By.NAME, 'q') # name = 'q'
search_box.send_keys('딥러닝') # 검색어 입력     
search_box.send_keys(Keys.ENTER)

# 4. 검색 결과 페이지 
time.sleep(3) # 3초 대기(자원 loading)

driver.close() # 현재 창 닫기  



