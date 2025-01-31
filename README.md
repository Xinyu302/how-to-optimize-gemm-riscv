# How To Optimize GEMM

**行主序**的GEMM优化方案, 新增riscv的测试。

## RISC-V

### 实验设备
D1 是全志科技首款基于 RISC-V 指令集的 SoC，主核是来自阿里平头哥的 64 位的 玄铁 C906。「哪吒」开发板 是全志在线基于全志科技 D1 芯片定制的 AIoT 开发板， 参考[哪吒开发板固件烧写](https://verimake.com/d/35-risc-v-soc-ai-d1-ncnn-demo)完成哪吒开发板的配置。

交叉编译环境的搭建参考[NCNN文档](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-allwinner-d1)。使用[平头哥官方提供的qemu](https://xuantie.t-head.cn/community/download?id=4168444414324183040)对可执行文件进行模拟。编译及运行方式可参考makefile。

CPU: 全志D1-H C906 RISC-V 1GHz 

OS: D1-H Tina Open v1.01

### 测试方法
将`run.py`中的main入口后的语句更改为:

```python
    for ix, header in enumerate(header_file_list):
        compile(header)
```

将`run.py`中的main入口后的语句更改为:
```python
    main("MMult_4x4_13.h")
    main("MMult_xxxxx.h")
```
使用qemu-riscv64进行对不同版本Gemm的测试。

可在当前目录下新建bin目录，目录中即为可在C906 CPU上执行的文件。使用adb将bin目录拷贝到开发板中，加入执行权限即可运行。

### 实验结果
M=N=K=400

|文件名|优化方法|gFLOPs|峰值占比|线程数|
|--|--|--|--|--|
|MMult1.h|无任何优化|0.07|?%|1|
|MMult2.h|一次计算4个元素|0.11|?%|1|
|MMult_1x4_3.h|一次计算4个元素|0.1|?%|1|
|MMult_1x4_4.h|一次计算4个元素|0.1|?%|1|
|MMult_1x4_5.h|一次计算4个元素(将4个循环合并为1个)|0.19|?%|1|
|MMult_1x4_6.h|一次计算4个元素(将4个循环合并为1个)|0.23|?%|1|
|MMult_1x4_7.h|一次计算4个元素(我们在寄存器中累加C的元素，并对A的元素使用寄存器),用指针来寻址B中的元素|0.09|?%|1|
|MMult_1x4_8.h|在MMult_1x4_7的基础上循环展开四个（展开因子的相对任意选择）|0.09|?%|1|
|MMult_4x4_3.h|一次计算C中的4x4小块|0.11|?%|1|
|MMult_4x4_4.h|一次计算C中的4x4小块|0.11|?%|1|
|MMult_4x4_5.h|一次计算C中的4x4小块,将16个循环合并一个|0.13|?%|1|
|MMult_4x4_6.h|一次计算C中的4x4小块(我们在寄存器中累加C的元素，并对a的元素使用寄存器)|0.3|?%|1|
|MMult_4x4_7.h|在MMult_4x4_6的基础上用指针来寻址B中的元素|0.3|?%|1|
|MMult_4x4_8.h|使用更多的寄存器|0.29|?%|1|
|MMult_4x4_10.h|RISCV向量化指令集优化|0.33|?%|1|
|MMult_4x4_11.h|RISCV向量化指令集优化, 并且为了保持较小问题规模所获得的性能，我们分块矩阵C（以及相应的A和B） |0.49|?%|1
|MMult_4x4_12.h|RISCV向量化指令集优化, 对矩阵A和B进行Pack，这样就可以连续访问内存|1.03|?%|1|
|MMult_4x4_13.h|RISCV向量化指令集优化, 新的pack方法，来源于tpoisonooo,但是效果不好，原因未知|0.51|?%|1|

## x86

### cpufp

克隆高叔叔的X86浮点峰值测试工具，使用方法见README。

本机（Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz 12 Cores）测试结果如下：

```markdown
Thread(s): 1
fma fp32 perf: 97.3358 gflops.
fma fp64 perf: 48.8463 gflops.
avx fp32 perf: 48.7343 gflops.
avx fp64 perf: 24.3121 gflops.
sse fp32 perf: 25.9305 gflops.
sse fp64 perf: 12.9604 gflops.
```

```markdown
Thread(s): 4
fma fp32 perf: 356.8164 gflops.
fma fp64 perf: 178.6532 gflops.
avx fp32 perf: 178.6297 gflops.
avx fp64 perf: 89.3076 gflops.
sse fp32 perf: 93.3519 gflops.
sse fp64 perf: 46.6798 gflops.
```

### src

测试不同的优化方法对应的gflops。

峰值占比的计算方法为当前算法的gflops除以浮点测试工具中相同线程测出的最高峰值。

|文件名|优化方法|gFLOPs|峰值占比|线程数|
|--|--|--|--|--|
|MMult1.h|无任何优化|1.4gflops|1.4%|1|
|MMult2.h|一次计算4个元素|1.5gflops|1.5%|1|
|MMult_1x4_3.h|一次计算4个元素|1.4gflops|1.4%|1|
|MMult_1x4_4.h|一次计算4个元素|1.4gflops|1.4%|1|
|MMult_1x4_5.h|一次计算4个元素(将4个循环合并为1个)|1.5gflops|1.5%|1|
|MMult_1x4_6.h|一次计算4个元素(我们在寄存器中累加C的元素，并对a的元素使用寄存器)|1.6gflops|1.6%|1|
|MMult_1x4_7.h|在MMult_1x4_6的基础上用指针来寻址B中的元素|5.0gflops|6%|1|
|MMult_1x4_8.h|在MMult_1x4_7的基础上循环展开四个（展开因子的相对任意选择）|5gflops|6%|1|
|MMult_1x4_9.h|在MMult_1x4_8的基础上使用间接寻址的方法|5gflops|6%|1|
|MMult_4x4_3.h|一次计算C中的4x4小块|1.4gflops|1.4%|1|
|MMult_4x4_4.h|一次计算C中的4x4小块|1.4gflops|1.4%|1|
|MMult_4x4_5.h|一次计算C中的4x4小块,将16个循环合并一个|1.5gflops|1.5%|1|
|MMult_4x4_6.h|一次计算C中的4x4小块(我们在寄存器中累加C的元素，并对a的元素使用寄存器)|8.2gflops|8.4%|1|
|MMult_4x4_7.h|在MMult_4x4_6的基础上用指针来寻址B中的元素|8.4gflops|8.6%|1|
|MMult_4x4_8.h|使用更多的寄存器|7.7gflops|7.7%|1|
|MMult_4x4_10.h|SSE指令集优化|8.5gflops|8.7%|1|
|MMult_4x4_11.h|SSE指令集优化, 并且为了保持较小问题规模所获得的性能，我们分块矩阵C（以及相应的A和B） |8.5gflops|8.7%|1|
|MMult_4x4_13.h|SSE指令集优化, 对矩阵A和B进行Pack，这样就可以连续访问内存|33.0gflops|34.0%|1|


## armv7a

### armv7afp

```markdown
理论浮点峰值: 

fmla: 4x2(mul+add)*1.8gHz=14.4gFLOPs

实际浮点峰值测试

|架构|浮点峰值(GFlops)|
|--|--|
|Cortex-A53，armv7a|10.88|

达到硬件浮点峰值的75%
```

### src

- MMult_4x4_19和MMult_4x4_20来自tpoisonooo(白牛大佬)。
- 额外测试了conv1x1s1.h（version3）（默认实现为8x4）的4x4分块方法，gflops为4.8gflops。

|文件名|优化方法|gFLOPs|峰值占比|线程数|
|--|--|--|--|--|
|MMult1.h|无任何优化|0.24gflops|2.1%|1|
|MMult2.h|一次计算4个元素|0.24gflops|2.1%|1|
|MMult_1x4_3.h|一次计算4个元素|0.24gflops|2.1%|1|
|MMult_1x4_4.h|一次计算4个元素|0.24gflops|2.1%|1|
|MMult_1x4_5.h|一次计算4个元素(将4个循环合并为1个)|0.25gflops|2.2%|1|
|MMult_1x4_7.h|一次计算4个元素(我们在寄存器中累加C的元素，并对a的元素使用寄存器),用指针来寻址B中的元素|0.98gflops|9.0%|1|
|MMult_1x4_8.h|在MMult_1x4_7的基础上循环展开四个（展开因子的相对任意选择）|1.1gflops|10%|1|
|MMult_4x4_3.h|一次计算C中的4x4小块|0.24gflops|2.1%|1|
|MMult_4x4_4.h|一次计算C中的4x4小块|0.24gflops|2.1%|1|
|MMult_4x4_5.h|一次计算C中的4x4小块,将16个循环合并一个|0.25gflops|2.2%|1|
|MMult_4x4_6.h|一次计算C中的4x4小块(我们在寄存器中累加C的元素，并对a的元素使用寄存器)|1.75gflops|16.0%|1|
|MMult_4x4_7.h|在MMult_4x4_6的基础上用指针来寻址B中的元素|1.75gflops|16.0%|1|
|MMult_4x4_8.h|使用更多的寄存器|1.75gflops|16.0%|1|
|MMult_4x4_10.h|NEON指令集优化|2.6gflops|23.8%|1|
|MMult_4x4_11.h|NEON指令集优化, 并且为了保持较小问题规模所获得的性能，我们分块矩阵C（以及相应的A和B） |2.6gflops|23.8%|1|
|MMult_4x4_13.h|NEON指令集优化, 对矩阵A和B进行Pack，这样就可以连续访问内存|2.6gflops|23.8%|1|
|MMult_4x4_18.h|Neon Assembly，Cache优化|3.0gflops|27.5%|1|
|MMult_4x4_19.h|MMult_4x4_18基础上+更长的pld+ldd+指令重排|3.8gflops|34.59%|1|
|MMult_4x4_20.h|MMult_4x4_19基础上更换vldr + 简单调整ping pong|4.0gflops|36.7%|1|
|conv1x1s1.h（version1）|一次计算多行，neon汇编优化|3.4gflops|31.0%|1|
|conv1x1s1.h（version2）|pack，kernel提前做，neon汇编优化，8x4分块|4.9gflops|45%|1|
|conv1x1s1.h（version3）|pack，kernel提前做，输入NC4HW4，neon汇编优化，8x4分块|5.5gflops|50.5%|1|
|conv1x1s1.h（version4） idea from megengine|pack，kernel提前做，输入NC4HW4，neon汇编优化，12x4分块|5.2gflops|47.8%|1|

- 猜测，分块时块的大小需要尽量大，恰好可以塞进Cache能获得最大性能，目前测试情况来看：
```
1x1 < 1x4 < 4x4 < 8x4 ...
```

- 测试了12x4，发现并不是这样，和MegEngine的大佬交流之后知道A53的硬件利用率极限是75%，另外ldq和fmla不能双发射，所以需要将指令拆成ldx和ldr+ins，因为ldx可以和fmla双发射，ldr和ins可以双发射，就可以突破到75%，待编码验证。

- 为什么A53极限硬件利用率为75%，因为ldr和ins都不能和fmla双发射，所以每3条fmla就需要带一条ldr+ins，fmla的吞吐极限就是3/4。

## armv8a

### armv8afp

```markdown
理论浮点峰值: 

fmla: 4x2(mul+add)*1.55gHz=12.4gFLOPs （Jetson Nano）

|架构|浮点峰值(GFlops)|
|--|--|
|Cortex-A57，armv8a|11.39|

达到硬件浮点峰值的91.8%
```

### src

|文件名|优化方法|gFLOPs|峰值占比|线程数|
|--|--|--|--|--|
|MMult1.h|无任何优化|0.75gflops|6.5%|1|
|MMult2.h|一次计算4个元素|0.8gflops|7.0%|1|
|MMult_1x4_3.h|一次计算4个元素|0.8gflops|7.0%|1|
|MMult_1x4_4.h|一次计算4个元素|0.89gflops|7.8%|1|
|MMult_1x4_5.h|一次计算4个元素(将4个循环合并为1个)|1.57gflops|13.8%|1|
|MMult_1x4_7.h|一次计算4个元素(我们在寄存器中累加C的元素，并对A的元素使用寄存器),用指针来寻址B中的元素|1.92gflops|16.8%|1|
|MMult_1x4_8.h|在MMult_1x4_7的基础上循环展开四个（展开因子的相对任意选择）|1.28gflops|11.2%|1|
|MMult_4x4_3.h|一次计算C中的4x4小块|0.75gflops|6.5%|1|
|MMult_4x4_4.h|一次计算C中的4x4小块|0.80gflops|7.0%|1|
|MMult_4x4_5.h|一次计算C中的4x4小块,将16个循环合并一个|1.26gflops|11.0%|1|
|MMult_4x4_6.h|一次计算C中的4x4小块(我们在寄存器中累加C的元素，并对a的元素使用寄存器)|4.29gflops|37.6%|1|
|MMult_4x4_7.h|在MMult_4x4_6的基础上用指针来寻址B中的元素|4.12gflops|36.0%|1|
|MMult_4x4_8.h|使用更多的寄存器|4.3gflops|37.7%|1|
|MMult_4x4_10.h|NEON指令集优化|4.3gflops|37.7%|1|
|MMult_4x4_11.h|NEON指令集优化, 并且为了保持较小问题规模所获得的性能，我们分块矩阵C（以及相应的A和B） |4.3gflops|37.7%|1
|MMult_4x4_13.h|NEON指令集优化, 对矩阵A和B进行Pack，这样就可以连续访问内存|4.77gflops|41.8%|1|
|MMult_4x4_18.h|Neon Assembly，Cache优化|8.44gflops|74.1%|1|

# 相关链接
- https://github.com/tpoisonooo/how-to-optimize-gemm/tree/master/src/HowToOptimizeGemm
- https://github.com/flame/blislab