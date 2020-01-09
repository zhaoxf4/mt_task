# mt_task
任务1：利用前馈神经网络实现0~7数字的异或输出，比如说输入3和5，输出6。
设计思路：先训练一个二分类器，然后按位异或
机器环境：python3.6 + numpy + matplotlib
运行XOP.py，根据提示得到你的结果
ps：程序首先会画出损失函数，观察迭代10000次时损失函数是否小于0.01，小于则继续根据提示进行，否则重新运行程序

任务2：对前馈神经网络调参

Windows运行参数：<br>
```
-fnnlm -dev 0 -lrate 0.007 -wbatch 256 -minmax 0.1 -nepoch 5 -n 5 -hdepth 1 -hsize 128 -esize 100 -train $(ProjectDir)\data\wsj.train -test $(ProjectDir)\data\wsj.test -output $(ProjectDir)\work\wsj.prob -vsize 10000 -model $(ProjectDir)\work\wsj.model -autodiff
```

更改了：<br>
```
-lrate<br>
-wbatch<br>
-nepoch<br>
-n<br>
```
调参过程详见“process.txt”<br>
ppl值降到205.97

![image](https://github.com/zhaoxf4/mt_task/blob/master/images/result.png)
