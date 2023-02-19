import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import jax.numpy as jnp

def load_iris(concat_one=True):
    cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv('data/iris.data', names=cols)
    encoder = LabelEncoder()
    encoder.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    df['class'] = encoder.transform(df['class'])
    df = shuffle(df, random_state=0)
    X = df.values[:,:-1]
    y = df.values[:,-1]
    if concat_one:
        X = jnp.hstack([X, jnp.ones((X.shape[0],1))])
    return jnp.array(X), jnp.array(y)
