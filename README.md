# mt_task
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
