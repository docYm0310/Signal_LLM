from utils import *
import matplotlib.pyplot as plt
data_dir = "../data_radar_34_512/testData"
filename =  "Data_Source (1).mat"
dataset = loadmatWithRF_Sig(data_dir,filename)

# 提取 I 通道
i_channel = dataset[0][0].numpy()

# 绘图
plt.figure(figsize=(10, 4))
plt.plot(i_channel, label='I Channel')
plt.title("I Channel Waveform")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


