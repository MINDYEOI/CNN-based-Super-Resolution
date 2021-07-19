# 🌈 CNN based Super Resolution

## 🌊 Tensor 객체
`CTensor.h`를 통해 `_1_main_tensor.cpp` 수행.<br><br>
![image](https://user-images.githubusercontent.com/71140885/126168115-e1c4ca37-96e9-4271-b389-14d655cba7dd.png)
<br>
## 📚 Layer 객체
`_2_main_layer.cpp` 구현.<br><br>
![image](https://user-images.githubusercontent.com/71140885/126168386-b89c4d9b-11ab-43ed-9914-82b3ebc50fb9.png)
<br>
## 🎞 Model 객체 
`_3_main_model_mean.cpp` 구현. 
  * Layer와 Tensor를 vector로 관리
  * 순차적으로 Layer를 수행하고 출력 결과를 저장함.<br><br>
![image](https://user-images.githubusercontent.com/71140885/126168617-5fadec8c-dd36-42dc-b7ce-0baad7a80f8a.png)
<br>
  * 테스트 결과로 더 blur된 영상이 생성됨(평균값 필터링을 적용함.)
    * 입력 : baby_512x512_input.bmp
    * 출력 : baby_512x512_output_mean.bmp
    
 `_4_main_model_srcnn.cpp` 구현.
 * 최초 CNN 기반 SR 모델 (SRCNN : Super-Resolution CNN)을 구현함. <br>
 ![image](https://user-images.githubusercontent.com/71140885/126173247-57c55d96-67a4-44cf-9bc2-7f746e9b0952.png)
* 테스트 결과로 더 선명한 영상이 생성(SRCNN의 효과)
    * 입력 : baby_512x512_input.bmp
    * 출력 : baby_512x512_output_srcnn.bmp
 
 ## 주관적 평가
 * `baby_512x512_input.bmp` <br>
 ![image](https://user-images.githubusercontent.com/71140885/126173668-f23f7fc3-4de1-4fd1-9466-ab1ecc64c0ec.png)

 * `baby_512x512_output_mean.bmp`<br>
 ![image](https://user-images.githubusercontent.com/71140885/126173707-3f1433d5-fe3b-4d1d-b6ac-333ebf4311cf.png)

 * `baby_512x512_output_srcnn.bmp`<br>
![image](https://user-images.githubusercontent.com/71140885/126173735-59782faa-5758-411d-b6c3-77a5810e551e.png)
