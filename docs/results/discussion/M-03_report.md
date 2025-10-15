# Auto MPG Regression Comparison — Top 3

This report summarises the strongest-performing models across KNN, linear, and polynomial regressors evaluated in tickets M-01 and M-02.

## Best Models by Mean Squared Error

```
model_type                     name           features target   k  degree        parameters       mse      mae       r2  train_samples  test_samples
       knn knn_k5_horsepower_weight horsepower, weight    mpg 5.0     NaN                   15.790349 3.037468 0.690631            313            79
       knn knn_k3_horsepower_weight horsepower, weight    mpg 3.0     NaN                   16.131435 2.991983 0.683949            313            79
    linear             linear_hp_wt horsepower, weight    mpg NaN     NaN include_bias=True 17.791776 3.505654 0.651419            313            79
```

## Best Models by R²

```
model_type                     name           features target   k  degree        parameters       mse      mae       r2  train_samples  test_samples
       knn knn_k5_horsepower_weight horsepower, weight    mpg 5.0     NaN                   15.790349 3.037468 0.690631            313            79
       knn knn_k3_horsepower_weight horsepower, weight    mpg 3.0     NaN                   16.131435 2.991983 0.683949            313            79
    linear             linear_hp_wt horsepower, weight    mpg NaN     NaN include_bias=True 17.791776 3.505654 0.651419            313            79
```

## Notes

- Metrics computed on hold-out splits with consistent random seeds.
- Feature lists and parameters are provided for reproducibility.
