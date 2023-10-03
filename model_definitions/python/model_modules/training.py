# from xgboost import XGBClassifier
# from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# from nyoka import xgboost_to_pmml
from nyoka import lgb_to_pmml
from teradataml import DataFrame
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib


def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)
    train_pdf = train_df.to_pandas(all_rows=True)

    # split data into X and y
    """
    X_train = train_pdf[feature_names]
    y_train = train_pdf[target_name]
    """
    X = train_pdf[feature_names]
    y = train_pdf[target_name]

    # early_stopping用の評価データを分割
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

    print("Starting training...")

    # fit model to training data
    """
    model = Pipeline([('scaler', MinMaxScaler()),
                      ('xgb', XGBClassifier(eta=context.hyperparams["eta"],
                                            max_depth=context.hyperparams["max_depth"]))])

    model.fit(X_train, y_train)
    """

    # LightGBMモデルの実装
    # PMML形式でモデルを出力したいため、scikit-learn APIを使用する

    # 使用するパラメータ
    params = {'objective': 'multiclass',  # 最小化させるべき損失関数
            'random_state': 42,  # 乱数シード
            'boosting_type': 'gbdt',  # boosting_type
            'n_estimators': 10000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
            'learning_rate': context.hyperparams["eta"],
            'max_depth': context.hyperparams["max_depth"]
            }

    model = Pipeline([
        ('lgbmc',lgb.LGBMClassifier(**params))
    ])

    model.fit(X_train, y_train,
              lgbmc__eval_metric='logloss', # early_stoppingの評価指標(学習用の'metric'パラメータにも同じ指標が自動入力される)
              lgbmc__eval_set=[(X_valid, y_valid)],
              lgbmc__callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True), # early_stopping用コールバック関数
                                lgb.log_evaluation(1)] # コマンドライン出力用コールバック関数
                )

    print("Finished training")

    # export model artefacts
    joblib.dump(model, f"{context.artifact_output_path}/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    """
    xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name,
                    pmml_f_name=f"{context.artifact_output_path}/model.pmml")
    """
    lgb_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name,
                pmml_f_name=f"{context.artifact_output_path}/model.pmml")

    print("Saved trained model")

    # 以下はXGBoost用コード
    """
    from xgboost import plot_importance
    model["xgb"].get_booster().feature_names = feature_names
    plot_importance(model["xgb"].get_booster(), max_num_features=10)
    save_plot("feature_importance.png", context=context)

    feature_importance = model["xgb"].get_booster().get_score(importance_type="weight")
    """
    # 以下はLightGBM用コード
    booster = model["lgbmc"].booster_
    lgb.plot_importance(booster)
    save_plot("feature_importance.png", context=context)

    feature_importance = {fname:fivalue for fname, fivalue in zip(booster.feature_name(), booster.feature_importance(importance_type='split'))}

    record_training_stats(train_df,
                          features=feature_names,
                          targets=[target_name],
                          categorical=[target_name],
                          feature_importance=feature_importance,
                          context=context)