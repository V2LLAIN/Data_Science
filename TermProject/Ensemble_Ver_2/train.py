import gc
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from config import args
from dataset import train, test, sub
from TFIDF import tf_train, tf_test

if __name__ == '__main__':
    y_train = train['label'].values
    if len(test.text.values) <= 5:
        sub.to_csv(args.submission, index=False)
    else:
        clf = MultinomialNB(alpha=0.02)
        #     clf2 = MultinomialNB(alpha=0.01)
        sgd_model = SGDClassifier(max_iter=args.max_iter, tol=1e-4, loss="modified_huber")

        p6 = {'n_iter': 1500, 'verbose': -1, 'objective': 'binary', 'metric': args.p6_metric,
              'learning_rate': args.p6_learning_rate,
              'colsample_bytree': 0.726023996436955, 'colsample_bynode': 0.5803681307354022,
              'lambda_l1': 8.562963348932286, 'lambda_l2': 4.893256185259296, 'min_data_in_leaf': 115,
              'max_depth': args.p6_max_depth, 'max_bin': args.p6_max_bin}

        lgb = LGBMClassifier(**p6)

        cat = CatBoostClassifier(iterations=1000,
                                 verbose=0,
                                 l2_leaf_reg=args.l2_leaf_reg,
                                 learning_rate=args.cat_learning_rate,
                                 allow_const_label=True, loss_function=args.cat_loss)

        ensemble = VotingClassifier(estimators=[('mnb', clf),
                                                ('sgd', sgd_model),
                                                ('lgb', lgb),
                                                ('cat', cat)],
                                    weights=args.weights, voting=args.voting, n_jobs=-1)

        ensemble.fit(tf_train, y_train)
        gc.collect()
        final_preds = ensemble.predict_proba(tf_test)[:, 1]
        sub['generated'] = final_preds
        sub.to_csv('submission.csv', index=False)
