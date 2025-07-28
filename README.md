## 电磁大模型操作说明

### 1.准备环境

python版本：3.10.14

Anaconda-linux版本：24.5.0  

配置python库环境：

```cmd
pip install -r requirements.txt
```

### 2.准备数据

修改main.sh中的dataset_dir参数内容为数据集data_radar_34_512中训练集路径。

修改test.sh中的dataset_dir参数内容为数据集data_radar_34_512中测试集路径。

### 3.训练流程

执行训练：

```cmd
sh main.sh
```

### 4.推理流程

替换test.sh文件中model_path参数内容为save文件中训练后保存的最优模型。

执行测试：

```cmd
sh test.sh
```

