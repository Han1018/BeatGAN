# BeatGAN

### 概要
在本專題中使用的是對抗式神經網路模型，用Lakh Midi Dataset作為訓練資料集，其中包括 14,600多首不同歌曲。因此除了可以快速且大量的生成音樂外，使用者也可以通過選擇曲風來生成 自己所需要的音樂。 我們希望可以實現並優化現有的模型，使參數能達到更好的表現，並以此設計 一個平台，此平台可以達到兩種目標:   
1.用網頁呈現出透過樂器間的生成器，讓使用者選擇曲風，隨機生成各自的音軌，並整合每個音軌的參數，生成獨立的MIDI。  
2.用一個樂器作為主要的生成動機，接著以此動機將使用者所上傳的指定格式的midi檔音樂為基準聯合生成剩餘的樂器音軌，將音樂常用的Jamming的概念用在伴奏生成上，進而生成曲風類似的伴奏。

### 研究範圍
#### （一）訓練音樂生成模型

限制於資料集的特性，音樂性的部分較難掌控。主要是針對現行的模型架構、演算法進行實
作和優化改善，讓整體的客觀指標能夠有所進步，聽感上能夠更符合直覺。目標為讓DP、ISR上升 (現行演算法分別大約在50%、65%左右)、以及UPC、PR下降(現行演算法大約在45%左右)。從圖1可以看到音樂訓練的過程，隨著訓練階段逐漸越來越進步，學習到音樂的規則。
![image align='center'](https://user-images.githubusercontent.com/61962782/197689841-0c7fedaa-f548-468c-a14f-be34cd7e473e.png)

#### （二）合併樂曲自動產生伴奏
主要的輸入和輸出會是以平台的方式呈現，除了可以讓使用者選擇曲風來生成音樂，使用者也可以透過平台上傳樂曲，當接收到樂曲後，後端會將樂曲解析成npz文件，進而分析音符與樂曲進程的排列來生成相關的伴奏(e.g,和弦進程、鼓節奏、鋼琴音等)。
從圖2可以看到經過訓練後，生成的音樂架構是完整的且有邏輯性、結構。

<img src='https://user-images.githubusercontent.com/61962782/197690015-5fda1390-1974-41aa-8e6c-46b480b529f4.png' align='center' />  
            圖2 各樂器軌道音樂波形
            
### 技術與方法
#### （一）對抗式生成網路
對抗式生成網絡(GAN)有兩個核心組件：生成器 Generator 和鑑別器 Discriminator。前者將從潛在空間(latent space)中採樣一隨機向量 z 作為輸入，並生成假樣本 G(z)。 Discriminator則是將真實數據 x 或 Generator 生成的假數據作為輸入，學習區分真實或假樣本。在訓練期間，兩個網絡相互對抗、不斷調整參數，最終目的是使其判別網絡無法判斷生成網絡的輸出結果是否真實。GAN的目標函數可以表示為 :   
$$minmaxV(D,G) = Ex∼pdata(x)[logD(x)]+Ez∼pz(z)[log(1−D(G(z)))]$$

#### （二）DCGAN
在DCGAN中，除了將模型中的Generator和Discriminator換成卷積神經網路(CNN)外，DCGAN對卷積神經網路的結構亦做了一些改變。像是取消所有pooling層，G網路中使用轉置卷積(transposed convolutional layer)進行上取樣，D網路中用加入stride的卷積代替pooling等等。此外在D和G中均使用batch normalization，並去掉FC層，使網路變為全卷積網路。圖3為DCGAN中的Generator示意：
![image align='center'](https://user-images.githubusercontent.com/61962782/197691626-ee58b3b2-c4d9-4f13-a8b6-50158569329d.png)

#### （三）多序列軌道音樂生成
音樂生成的部分我們使用的是DCGAN，為了讓不同樂器間能夠維持相同的聽感伴奏並且存在彼此間的和諧，我們將一個隨機向量Z，和樂器間每個生成器中產生的隨機向量Zi同時作為輸入，即Ｇi(Z,Zi)，期望每個樂器的隨機向量可以協同不同音樂的生成，就像作曲家一樣。此外，我們僅使用一個鑑別器來共同評估5個音樂軌道，以此希望模型在各自生成獨立音軌時，亦能用一個整體的概觀使音樂更統一性。
在模型中，每個音軌獨立生成的好處可以讓我們在不同樂器中調整適合特定樂器的模型架構（例如，層數、過濾器大小等）。因此，我們可以改變一個特定軌道的生成而不會失去軌道間的相互依賴性。圖4為音樂生成模型的核心架構：  
![image](https://user-images.githubusercontent.com/61962782/197691838-901d2f64-9c0c-422a-b184-2a15bb703de5.png)

音樂生成的第二個架構是以人類做音樂Jamming時的角度出發，給定一個特定軌道的序列 y ，藉由給定的音樂軌道序列 y 的生成剩餘的軌道完成歌曲段落。這樣音樂架構類似於隨機生成，不同的是，我們需要先將指定的輸入-音樂軌道序列 y 映射到空間向量 Z 中，並將向量 Z 與剩餘樂器中的隨機向量Zi同時作為輸入生成音樂，即Ｇi(Z,Zi)，以此達到類似於AI協同或輔助人類生成音樂伴奏。圖5為 Multi-Track Conditional GAN生成模型架構：  
![image](https://user-images.githubusercontent.com/61962782/197692005-09bc8a0e-a313-4a5b-941d-62fc9fd5ebb9.png)

### 實驗優化
#### 優化模型 - Wasserstein Distance
由於音樂軌道間的資訊量大，在初期訓練期間並不穩定，梯度下降的過程緩慢。因此為了讓模型得到更精確真實和生成樣本間的距離，我們引入了Wasserstein Distance。Wasserstein Distance的數學表示式為 :
W( P,Q )=〖inf〗_(γ∼Π( P,Q ))  E_(( x ,y ) ∼ γ) |(| x-y |)|
在使用Wasserstein-Distance為計算距離公式後，相比起JS-divergence，可以看到圖6中Wasserstein Distance在初期的訓練表現得比JS-divergence更好。
![image](https://user-images.githubusercontent.com/61962782/197692196-9669fa93-275f-4d0e-b081-7d08b19c0e61.png)
            圖 Polyphonic Rate（紅色:Wass用Wasserstein-Distance 藍色:JS-divergence）
#### 參數調整
Batch Size大小將決定一次訓練的樣本數目，在訓練過程中發現太小的Batch Size會使得模型underfitting難以收斂。為了解決這樣問題，我們適當的調整了Batch Size讓模型能夠找到更多音符間的關係。從圖7可以看到在調整Batch Size後，Model的技術指標相比起調整前鼓組在4/4拍上的比例明顯增加。
![image](https://user-images.githubusercontent.com/61962782/197692347-f45b4bcb-96ca-4c54-9d17-98bf92119e63.png)
            圖  Drum in 4/4 Rate（紅色: Old Batch Size 藍色: New Batch Size）




### 成果
音樂生成的架構是以人類做音樂Jamming時的角度出發，給定一個特定軌道的序列 y ，藉由給定的音樂軌道序列 y 的生成剩餘的軌道完成歌曲段落。這樣音樂架構類似於隨機生成，不同的是，我們需要先將指定的輸入-音樂軌道序列 y 映射到空間向量 Z 中，並將向量 Z 與剩餘樂器中的隨機向量Zi同時作為輸入生成音樂，即Ｇi(Z,Zi)，以此達到類似於AI協同或輔助人類生成音樂伴奏。
下圖中可以看到BeatGAN中的核心兩個概念，上半部分為指揮家概念，在生成音樂時同時給予不同樂器相同的輸入，因爲輸入的相同，所以不同樂器間能夠生成和諧的節奏。下半部分為Jamming概念，引入了在真實世界做音樂時，不同樂手是獨立於彼此的音樂，雖然是同一首歌但是卻擁有自己的特色，樂器間相輔相成使音樂更豐富。
在BeatGAN中，我們結合了指揮家與Jamming兩種樣式，使其擁有和諧節奏的同時能夠在樂器間有自己的特色，獨立於樂器彼此，使其在聽感上更多樣性。
下圖為 Multi-Track Conditional GAN生成模型架構：(#Model細節已開源在BeatGAN的Repo上，可詳看Model中的Generator&Discriminator)


<img width="452" alt="image" src="https://user-images.githubusercontent.com/61962782/197690196-565f7ae5-1003-4bfc-99cf-a6a2b27cc126.png">


### 

## contributor
夏念愷 t107590019@ntut.org.tw
陳思齊 t107590025@ntut.org.tw
謝宗翰 t107590040@ntut.org.tw
