# -*- coding: utf-8 -*-
"""
셀레리움 설치 
(base) conda activate tensorflow 
(tensorflow) pip install selenium
(tensorflow) pip install webdriber_manager
"""

from selenium import webdriver # 드라이버 
from selenium.webdriver.chrome.service import Service # Chrom Service
from webdriver_manager.chrome import ChromeDriverManager # 크롬브라우저 관리자  
import time # 화면 일시 정지

 
# 1. driver 객체 생성
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
# driver 객체 생성 : 크롬브라우저 창 띄움   

dir(driver)
'''
driver.get(url) # url 이동 
driver.close() # 창 닫기 
'''

# 2. 대상 url 이동 
driver.get('https://www.naver.com/') # url 이동 

# 3. 일시 중지 & driver 종료 
time.sleep(5) # 5초 일시 중지 
driver.close() # 현재 창 닫기  

