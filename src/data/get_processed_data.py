import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def read_data():
    raw_data_path=os.path.join(os.path.pardir,'data','raw')
    train_data_path=os.path.join(raw_data_path,'train.csv')
    test_data_path=os.path.join(raw_data_path,'test.csv')
    train_df=pd.read_csv(train_data_path,index_col='PassengerId')
    test_df=pd.read_csv(test_data_path,index_col='PassengerId')
    test_df['Survived']=-888
    df=pd.concat((train_df,test_df),axis=0)
    return df


def process_data(df):
    #chaining method
    return (df
            .assign(Title=df['Name'].map(lambda x: get_title(x)))
            .pipe(fill_missing_values)
            .assign(FareBin=lambda x :pd.qcut(x['Fare'],4,labels=['very_low','low','high','very_high']))
            .assign(AgeState=lambda x :np.where(x['Age']>=18,'Adult','child'))
            .assign(FamliySize=lambda x :x['Parch']+x['SibSp']+1)
            .assign(IsMother=lambda x:np.where(((x['Sex']=='female') & (x['Parch']>0) & (x['Age']>18) & (x['Title']!='Miss')),1,0) )
            .assign(Cabin=lambda x:np.where(x['Cabin']=='T',np.NaN,x['Cabin']) )
            .assign(Deck=lambda x:x['Cabin'].map(lambda x: get_deck(x)) )
            .assign(IsMale=lambda x :np.where(x['Sex']=='male',1,0))
            .pipe(pd.get_dummies,columns=['Deck','Pclass','Title','FareBin','Embarked','AgeState'])
            .drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis=1)
            .pipe(reorder_columns)
           )
            
def get_title(name):
    first_name_title=name.split(',')[1]
    title=first_name_title.split('.')[0]
    title=title.strip().lower()
    return title
            
def fill_missing_values(df):
    df['Embarked'].fillna('C',inplace=True)
    missing_median_fare=df.loc[(df['Pclass']==3) & (df['Embarked']=='S')]['Fare'].median()
    df['Fare'].fillna(missing_median_fare,inplace=True)
    age_median=df.groupby('Title')['Age'].transform('median')
    df['Age'].fillna(age_median,inplace=True)
    return df        
            
def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')

def reorder_columns(df):
    columns=[column for column in df.columns  if column!='Survived']
    columns=['Survived']+columns
    df=df[columns]
    return df

def write_data(df):
    processed_data_path=os.path.join(os.path.pardir,'data','processed')
    write_train_path=os.path.join(processed_data_path,'train.csv')
    write_test_path=os.path.join(processed_data_path,'test.csv')
    df.loc[df['Survived']!=-888].to_csv(write_train_path)
    columns=[column for column in df.columns if column!='Survived']
    df.loc[df['Survived']==-888,columns].to_csv(write_test_path)
    
if __name__=='__main__':
    df=read_data()
    df=process_data(df)
    write_data(df)
    
            
    