# court-detection

### moduleのインストール

```
$ poetry config virtualenvs.in-project true
$ poetry shell
$ poetry install
```


### landmark-toolの利用

https://github.com/donydchen/landmark-tool
をpyQt5に対応 + 点の番号が出るようにカスタマイズしている。

```
$ git clone https://github.com/donydchen/landmark-tool.git
$ patch -p1 -d landmark-tool < landmark-tool.patch
```
