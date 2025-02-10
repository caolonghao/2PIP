### For optional BaSiC estimate method
为方便进行效果对比，光场估计方法中集成了[BaSiC](https://github.com/peng-lab/BaSiCPy)，如果用户想要使用，需要额外安装 `jax`与`basicpy`两个库。
Windows用户安装命令如下：
```
pip install "jax[cpu]==0.3.14" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install basicpy
```
Linux用户安装命令如下：
```
pip install basicpy
```
之后在 `correct` 的 `plugin.json` 中指定 `estimatorName` 为 `BaSicEstimate` 即可使用