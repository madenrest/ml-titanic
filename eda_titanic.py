import pandas as pd
import numpy as np

df = pd.read_csv('titanic.csv')
# датасет: titanic
# строк: 891
# столбцов: 12
# пропуски: 1. cabin; 2. age; 3. embarked

# фильтрация, группировка, фичи
df_adult = df[df['Age'] > 18]
# print(df.groupby('Sex')['Survived'].mean())
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# print(df[['Survived', 'Pclass', 'Age', 'Fare', 'FamilySize']].corr())
# выживаемость сильно зависит от класса (чем выше класс, тем больше выживаемость)
# поля класс и цена билета сильно связаны, можно сделать из них один признак

def family_cat(size):
    if size == 1: return 1
    elif 1 < size <= 3: return 2
    else: return 3

df['FamilyCat'] = df['FamilySize'].apply(family_cat)
# print(df.groupby('FamilyCat')['Survived'].mean())
# выживаемость выше среди маленьких семей (от 2 до 3 человек)

def age_cat(age):
    if 1 <= age <= 17: return 1
    elif 18 <= age <= 25: return 2
    elif 26 <= age <= 60: return 3
    else: return 4

df['AgeCat'] = df['Age'].apply(age_cat)
# print(df.groupby('AgeCat')['Survived'].mean())
# выживаемость выше среди детей и взрослых, наименьшая - среди стариков


# обработка пропусков, кодирование категорий
# заполняем пропуски в age медианой по полям sex и pclass, т.к. возраст часто зависит от пола и класса
median_age = df.groupby(['Sex', 'Pclass'])['Age'].transform('median')
df['Age'] = df['Age'].fillna(median_age)
# заполняем пропуски в embarked модой, т.к. пропусков очень мало (2)
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

# категориальный признак -> числовой
df['Sex_enc'] = df['Sex'].map({'male': 0, 'female': 1})
# женщинам присваиваем 1, тк среди них выживаемость выше, что будет удобно при использовании логистической регрессии


# пайплайн
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


features = ['Pclass', 'Sex_enc', 'Age', 'FamilySize']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# коэффициенты
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print(coefficients)

