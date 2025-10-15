# コンピューターシミュレーション2 第3回 演習課題

**Dataset: iris.csv**

**(1) Sample code: 1-Sample_iris-DataPlot.ipynb**

(a) サンプルコードを実行し、Colabと Python の基本操作を学ぶ。
(b) アヤメのデータセット・特徴・ターゲット値を確認する。
(c) サンプルコードで描かれる図のマーカーの色・形を変更する(5種類以上)。

**(2) Sample code: S2-Sample_iris-knnClassifier.ipynb**

(a) 全ての特徴変数(sepal.length, sepal.width, petal.length, petal.width)を用いて、サンプルコード(k=3)を実行し、KNN 分類器の基本を理解する。

(b) 他のkの値に変更してサンプルコードを実行する(例: k = 1, 2, 5, 10)。以下の3点に注意し、結果を考察する
    i. KNN 分類器を作成する。
    ii. 分類結果を図にして確認する。
    iii. 混同行列(confusion matrix)を取得する。

**追加課題:**

2つの特徴変数 sepal.length, sepal.width の組み合わせで(b)を繰り返した結果と、petal.length, petal.width の組み合わせで(b)を繰り返した結果を考察する。
Note: kの値を変更する時は、同じコード(一部変更)を複数回繰り返し実行するか、for ループを使うか選ぶこと。

**(3) Sample code: S3-Sample_iris-KmeansCluster.ipynb**

(a) 全ての特徴変数(sepal.length, sepal.width, petal.length, petal.width)を用いて、サンプルコード(k=3)を実行し、K-means Clustering の基本を理解する。

(b) k=4,5でK-means clustering を作成し、図示して結果を確認する。

**追加課題:**

2つの特徴変数 sepal.length, sepal.width の組み合わせで(b)を繰り返した結果と、petal.length, petal.widthの組み合わせで(b)を繰り返した結果を考察する。
