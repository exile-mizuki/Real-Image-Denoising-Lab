# 真实自然图像三种传统滤波实验设计

## 研究问题
在不人为添加噪声的条件下，比较高斯滤波、普通中值滤波、双边滤波对真实自然图像噪声的影响。关注其噪声抑制、细节/边缘保留、参数敏感性和运行时间。目的不是训练深度学习网络。

## 直接参考的开源项目

1. **JAshehri/Image_Denoising_and_Enhancement** — https://github.com/JAshehri/Image_Denoising_and_Enhancement  
   学习点：统一实验输入，多组窗口/参数配置，使用 PSNR、SSIM 衡量质量并呈现算法对比。该仓库包含其他算法或人为加噪流程，**本工程不照搬这些部分**。
2. **Lucydln/Image_Denoising_and_Filtering_Techniques** — https://github.com/Lucydln/Image_Denoising_and_Filtering_Techniques  
   学习点：OpenCV 实现 Gaussian/Median/Bilateral，以及滤波前后可视化。**代码由本工程独立编写**。
3. **PolyU 数据集原始项目** — https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset  
   用于对照数据来源和实拍噪声实验背景。你的包可能是由其他仓库重新打包的 `train_sub/GT, LQ` 子集，应在报告中说明具体使用的是下载包的该子集。

## 参数方案（统一固定）

| 算法 | 设置1 | 设置2 | 设置3 |
| --- | --- | --- | --- |
| 高斯滤波 Gaussian | 3×3, σ=0.8 | 5×5, σ=1.1 | 7×7, σ=1.5 |
| 中值滤波 Median | 3×3 | 5×5 | 7×7 |
| 双边滤波 Bilateral | d=5, σColor=20, σSpace=3 | d=7, σColor=35, σSpace=5 | d=9, σColor=50, σSpace=7 |

每幅带噪输入都另算 **NO_FILTER** 作为基线。所有滤波均在真实图像上直接运行，不执行噪声注入。`σ` 是高斯滤波的空间标准差，`d` 是双边滤波邻域直径，`σColor` 和 `σSpace` 分别控制颜色与距离加权。算法实现在 `run_real_denoise.py`。

## 数据集分工与重要区别

| 数据集 | 已获取部分 | 参考图评价 | 不得做的表述 |
| --- | --- | --- | --- |
| NIND | 836个原始文件（用户电脑已下载） | 使用用户现有独立工程的ISO参考及配准方案 | NIND运行中不等于已完成全部结果 |
| PolyU | 下载包的 train_sub/GT 和 train_sub/LQ | 同名配对，验证尺寸及对齐后可做 PSNR/SSIM/MSE | 不要把训练子集说成该数据集的全部原始样本 |
| SIDD | 仅场景0001的一组GT与NOISY sRGB | 同编号PNG配对并评分 | 不要称“已完成SIDD全量实验” |
| RENOIR | Canon S90 Aligned 批次 | 默认只做视觉图与耗时，核实配准/掩膜后再设计官方一致的评分 | 不要把 Full.bmp 当GT或拿未验证分数作结论 |
| DND | 下载进度待定 | 官方提交评测，测试参考图不可随意本地获取 | 不要随意编造 PSNR/SSIM |

## 指标说明

- **MSE**：每个颜色通道的平方误差取均值，越小意味着与给定GT更接近。
- **PSNR**：按 8-bit RGB 的动态范围 255，由 MSE 计算；越高越接近参考图（但不一定意味着主观画质一定最好）。
- **SSIM**：结构相似性，使用 `skimage.metrics.structural_similarity(..., channel_axis=2, data_range=255)`。
- **耗时**：当前设备对一个裁剪图块一次算法调用的墙钟时间，只用于该硬件和实现；不同机器不能直接比较绝对毫秒数。
- **视觉对比**：观察纹理、文字、平坦区域彩色噪点、边缘细线、模糊与伪影；最好同时保存局部放大。

## 实验可复现性与质量控制

1. CSV 保存输入路径、配对GT路径、原生分辨率裁剪位置、9组参数和各指标。
2. 不缩放输入或GT；尺寸不同直接跳过，避免虚假高分。
3. 先确认基线（NO_FILTER）指标，再判断每种滤波是否真正改善。
4. 逐数据集单独计算平均与方差；不同数据集的图像和标注质量不同，不混成一个不加权平均。
5. 用户电脑上运行的 PolyU/SIDD/RENOIR 实验结果，需待脚本运行后才能用作报告数字；不能用本地开发时的合成测试素材代替。
6. 若有完整样本数，推荐进一步报告按场景、相机、ISO或难度分层的均值与标准差，避免只看一项分数。
