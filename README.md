# OVAE
OVAE: Out-of-Distribution Detection with Multi-Label-Enhanced Variational Autoencoders

Accept by CCF BigData 2021，[paper link](https://link.springer.com/chapter/10.1007/978-981-16-9709-8_16)

使用vae+多分类的方法进行域外样本检测（异常检测）
使用vae生成得到标签信息

多分类包含：多分类和多标签两种
主要使用prob进行检测任务
已经分离了train和test过程，train过程保存所有模型数据，以供test期间使用
test中分离正常和异常数据，正常数据只需要在train中测试一次就可以，后面重复使用其结果就好，test只使用异常数据，可以提高速度。

### 运行

运行代码

```shell
python run.py 0 train svhn cifar10 1 0 0 1 densenet121 0.0
```

主要文件见下表，后面具体介绍每个文件的功能。

| 文件名                  | 功能                            |
| ----------------------- | ------------------------------- |
| Main.py                 | 主代码，运行入口                |
| Run.py                  | 执行main.py代码，方便调参       |
| Solver.py               | Train和test等相关函数，主要功能 |
| Model.py                | 网络模型代码                    |
| Dataset.py              | 数据集处理代码                  |
| Iforest.py              | 孤立森林代码                    |
| Read_result.py          | 读取test后的结果：auc、prc      |
| Iforest_on_images文件夹 | 直接使用孤立森林在图像数据上    |



### Main.py

首先定义一些参数，用argparse.ArgumentParser封装；然后包含对应的train和test分支，将train和test过程分开，节省时间。

模型主要参数包含

| 参数              | 意义                                       |
| ----------------- | ------------------------------------------ |
| run-state         | 运行状态：train、test、test_normal         |
| dataset           | 正常数据集名称                             |
| anomaly-dataset   | 异常数据集名称                             |
| lambda-ae         | 对于ae分支的loss权重设置                   |
| lambda-ce         | Softmax分类的loss权重设置（无用）          |
| lambda-mem        | Membenship loss权重设置（无用）            |
| lambda-ce-sigmoid | Sigmoid分类的loss权重设置                  |
| unlabel-percent   | 在半监督任务下的无标签训练数据比例         |
| net-type          | 指定网络模型参数：resnet、densenet         |
| type-name         | 读结果时指定的结果类别：prob、iforest、all |



### Solver.py

包含train和test的实际实现逻辑。

- train()函数:首先将数据encode、decode、mult_class得到结果，计算loss和梯度，更新参数，同时保存训练过程loss。
- Test_normal()函数:对正常数据encode、decode、mult_class得到结果。
- test_anomaly()函数:对异常数据encode、decode、mult_class得到结果，拿到normal的结果和现在的结果一起计算auc、prc指标。



### Model.py

- 网络模型的函数，方便前面直接调用模型，包含不同数据的ae和分类模型。
- models文件夹下包含resnet和densenet模型文件。



### Iforeset.py

包含孤立森林的代码，将前面的多特征组合，放进孤立森林中计算，包括计算auc、prc等指标。iforest_test_epoch()使用ifoest计算一次的结果。



### Read_result.py

读取之前test阶段保存的结果，其中重要的type_name参数：prob、iforst、all。

显示的结果为auc和prc。

- Prob是OVAE结果，只用最大概率值区分的结果

- Iforest是OVAE+孤立森林的结果

- All就是同时显示prob和iforest