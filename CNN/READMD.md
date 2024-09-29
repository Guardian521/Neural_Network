一、整体功能
这段代码的主要功能是构建一个基于预训练的 ResNet50 模型的情绪分类模型，并从图像中提取多种特征，包括主导颜色、颜色组合、形状特征、纹理特征、对比度特征和饱和度特征，然后将这些特征和预测的情绪类别写入一个 CSV 文件中。
二、主要模块分析
模型定义：
MoodModel类继承自nn.Module，使用预训练的 ResNet50 提取图像特征，然后通过额外的全连接层进行特征转换，最后输出预测的情绪类别。
在__init__方法中，加载预训练的 ResNet50，截取除最后两层之外的部分作为特征提取器，然后添加一个线性层将特征维度从 2048 降为 256，最后再添加一个线性层输出指定数量的情绪类别。
forward方法中，首先通过特征提取器得到特征，然后对特征进行平均池化，经过两个线性层得到最终的输出。
特征提取函数：
extract_dominant_color：使用 KMeans 聚类算法找到图像的主导颜色，将其归一化到[0, 1]范围。
extract_color_combinations：同样使用 KMeans 聚类算法找到图像的三种主要颜色组合，并归一化到[0, 1]范围。
extract_shape_feature：将图像转换为灰度图，使用 Canny 边缘检测算法检测边缘，然后计算边缘像素占总像素的比例作为形状特征。
extract_texture_feature：将图像转换为灰度图后，计算灰度共生矩阵（GLCM），并使用 GLCM 的平方和作为纹理特征。
extract_contrast_feature：将图像转换为灰度图后，计算灰度值的标准差作为对比度特征。
extract_saturation_feature：将图像转换为 HSV 颜色空间，计算饱和度通道的平均值并归一化到[0, 1]范围作为饱和度特征。
主程序部分：
加载预训练的情绪分类模型，并设置为评估模式。
定义图像的预处理变换，包括调整大小、转换为张量、归一化。
遍历指定目录下的 JPG 图像文件。
对于每个图像文件，打开图像并转换为 RGB 格式，然后应用预处理变换得到图像张量。
将图像张量输入到模型中进行预测，得到预测的情绪类别。
调用各种特征提取函数提取图像的特征。
将图像文件名、预测的情绪类别和提取的特征写入 CSV 文件中。