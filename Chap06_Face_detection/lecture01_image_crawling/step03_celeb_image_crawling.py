# -*- coding: utf-8 -*-
"""
셀럽 이미지 수집 
 Selenium + Driver + BeautifulSoup
"""

from selenium import webdriver # driver 
from selenium.webdriver.chrome.service import Service # Chrom 서비스
from webdriver_manager.chrome import ChromeDriverManager # 크롬드라이버 관리자
from selenium.webdriver.common.by import By # By.NAME
from selenium.webdriver.common.keys import Keys # 엔터키 사용(Keys.ENTER) 
from bs4 import BeautifulSoup # html 파싱(find, select)
from urllib.request import urlretrieve # server image 
import os # dir 경로/생성/이동
import time

def celeb_crawler(name) :    
    # 1. dirver 객체 생성  
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    
    # 1. 이미지 검색 url 
    driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")
    
    # 2. 검색 입력상자 tag -> 검색어 입력   
    search_box = driver.find_element(By.NAME, 'q')
    search_box.send_keys(name) # 검색어 입력     
    search_box.send_keys(Keys.ENTER)
    time.sleep(3) # 3초 대기(자원 loading)
    
    
    # ------------ 스크롤바 내림 ------------------------------------------------------ 
    last_height = driver.execute_script("return document.body.scrollHeight") #현재 스크롤 높이 계산
    
    while True: # 무한반복
        # 브라우저 끝까지 스크롤바 내리기
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") 
        
        time.sleep(2) # 2초 대기 - 화면 스크롤 확인
    
        # 화면 갱신된 화면의 스크롤 높이 계산
        new_height = driver.execute_script("return document.body.scrollHeight")

        # 새로 계산한 스크롤 높이와 같으면 stop
        if new_height == last_height: 
            break
        last_height = new_height # 새로 계산한 스크롤 높이로 대체 
    #-------------------------------------------------------------------------
    
    
    # 3. 이미지 div 태그 수집  
    image_url = []
    for i in range(50) : # image 개수 지정                 
        src = driver.page_source # 현재페이지 source 수집 
        html = BeautifulSoup(src, "html.parser")
               
        # 상위태그 : <div class="wIjY0d jFk0f"> 하위태그 : div[n]
        div_img = html.select_one(f'div[class="wIjY0d jFk0f"] > div:nth-of-type({i+1})') # div 1개 수집
    
         
        # 4. img 태그 수집 & image url 추출
        img_tag = div_img.select_one('img[class="YQ4gaf"]') 
        
        try :
            image_url.append(img_tag.attrs['src']) # url 추출 
            print(str(i+1) + '번째 image url 추출')
        except :
            print(str(i+1) + '번째 image url 없음')
      
    print(image_url)        
    
    # 5. 중복 image url 삭제      
    print('중복 삭제 전 :',len(image_url)) # 44      
    image_url = list(set(image_url)) # 중복 url  삭제 
    print('중복 삭제 후 :', len(image_url)) # 22
    
       
    # 6. image 저장 폴더 생성과 이동 
    pwd = r'C:\ITWILL\7_Tensorflow\workspace' # base 저장 경로 
    os.mkdir(pwd + '/' + name) # 폴더 만들기(셀럽 이) 
    os.chdir(pwd + '/' + name) # 폴더 이동
        
    # 7. image url -> image save
    for i in range(len(image_url)) :
        try : # 예외처리 : server file 없음 예외처리 
            file_name = "test"+str(i+1)+".jpg" # test1.jsp
            # server image -> file save
            urlretrieve(image_url[i], filename=file_name)#(url, filepath)
            print(str(i+1) + '번째 image 저장')
        except :
            print('해당 url에 image 없음 : ', image_url[i])        
            
    driver.close() # driver 닫기 
    
   
# 1차 테스트 함수 호출 
'''
celeb_crawler("하정우")   

'''
# 2차 테스트 : 여러명 셀럽 이미지 수집  
namelist = ["조인성", "송강호", "전지현"] # 32 29, 30

for name in namelist :
    celeb_crawler(name) # image crawling
 

    