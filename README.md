# mt_task
任务2：对前馈神经网络调参

Windows运行参数
-fnnlm -dev 0 -lrate 0.006 -wbatch 128 -minmax 0.1 -nepoch 3 -n 5 -hdepth 1 -hsize 128 -esize 100 -train $(ProjectDir)\data\wsj.train -test $(ProjectDir)\data\wsj.test -output $(ProjectDir)\work\wsj.prob -vsize 10000 -model $(ProjectDir)\work\wsj.model -autodiff

更改了：
-lrate
-nepoch
-n

ppl值降到216.62

![image](http://github.com/zhaoxf4/mt_task/raw/master/images/result.png)