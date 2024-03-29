# DetDak
Patch adversarial attack; object detection; CIKM2020 安全AI挑战者计划第四期：通用目标检测的对抗攻击

最终排名[7/1701]

介绍：

深度神经网络已经在各种视觉识别问题上取得了最先进的性能。尽管取得了极大成功，深度神经网络很容易遭受输入上微小和不可察觉的干扰导致的误分类（这些输入也被称作对抗样本），深度模型的安全问题也在业内引起了不少担忧。

为了发现目标检测模型的脆弱性、为此领域的工作者敲响警钟。我们举办了全球首个结合黑盒白盒场景，针对多种目标检测模型的对抗攻击竞赛。比赛采用COCO数据集，其中包含20类物体。任务是通过向原始图像中添加对抗补丁（adversarial patch）的方式，使得典型的目标检测模型不能够检测到图像中的物体，绕过目标定位。为了更好的评价选手的攻击效果，我们创造了全新的得分计算准则。除了加入攻击成功率之外，我们还对添加补丁的数量和大小进行了约束。选手添加的补丁数量、修改的像素和模型识别到的包围盒越少，则代表攻击更加成功，得分则越高。为了保证比赛的难度，我们选取了4个近期的State-of-the-art检测模型作为攻击目标，包括两个白盒模型——YOLO v4和Faster RCNN和另外两个未知的黑盒模型。

方案介绍：

https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.18.5ca36163kksFzU&postId=127867

论文参考：
https://arxiv.org/abs/2010.14974

please cite：

```
@article{zhao2020object,
  title={Object Hider: Adversarial Patch Attack Against Object Detectors},
  author={Zhao, Yusheng and Yan, Huanqian and Wei, Xingxing},
  journal={arXiv preprint arXiv:2010.14974},
  year={2020}
}
```
