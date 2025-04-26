# PASpredict
基于机器学习的多聚腺苷酸化信号突变预测方法



```
python model.py --train ./data/train_feature.csv --test ./data/test_feature.csv --model_out ./model/PASpredict.joblib --output_dir ./result
```





```
python test.py --model ./model/PASpredict.joblib --data ./data/test.csv
```





```
python predict.py --model ./model/PASpredict.joblib --data ./data/feature.csv
```

