import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df=pd.read_csv(r"C:\Users\HP\Downloads\data.csv")

# print(df.head())
# print(df.isnull().sum())
# print(df.duplicated().sum())

# print(df.dtypes)


x = df[['Groceries','Transport','Eating_Out','Entertainment','Miscellaneous','Desired_Savings','Disposable_Income']]

sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# print(df.columns)

# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# ss=[]
# for i in range(2,21):
#     km=KMeans(n_clusters=i)
#     km.fit(df[num_cols])
#     ss.append(silhouette_score(df[num_cols],km.labels_))

# no_c=[j for j in range(2,21)]

# plt.plot(no_c,ss)
# plt.grid(axis='x')
# plt.xticks(no_c)
# plt.show()


wcss=[]

for i in range(2,21):
    km=KMeans(n_clusters=i,init='k-means++')
    km.fit(x_scaled)
    wcss.append(km.inertia_)

plt.plot([i for i in range(2,21)],wcss,marker='o')
plt.xticks([i for i in range(2,21)])
plt.grid(axis='x')
# plt.show()






kmn=KMeans(n_clusters=4,init='k-means++')
kmn.fit(x_scaled)



centroids=sc.inverse_transform(kmn.cluster_centers_) #array
cluster_profile=pd.DataFrame(centroids,columns=x.columns)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
# print(cluster_profile)



def recommend_budget(user_data,kmn,sc,cluster_profile):
    try:
        column_mapping = {
            'Groceries': 'Groceries',
            'Transport': 'Transport', 
            'Eating Out': 'Eating_Out',
            'Entertainment': 'Entertainment',
            'Miscellaneous': 'Miscellaneous',
            'Desired Savings': 'Desired_Savings',
            'Disposable Income': 'Disposable_Income'
        }
        
        user_data_mapped = user_data.rename(columns=column_mapping)
        
        feature = user_data_mapped[['Groceries','Transport','Eating_Out','Entertainment','Miscellaneous','Desired_Savings','Disposable_Income']]
        scaled_x=sc.transform(feature)

        cluster_id=kmn.predict(scaled_x)[0]
        cluster_pro=cluster_profile.loc[cluster_id]

        user_saving=user_data_mapped['Disposable_Income'].values[0]-user_data_mapped['Desired_Savings'].values[0]

        if user_saving >=0:
            return{
                "Message": f"Your saving goal is achievable. You are saving ₹{user_saving:.2f} monthly.",
                "Saving": user_saving,
                "Suggestions": []
            }
        
        shortfall = abs(user_saving)
        suggestions = []

        for col in ['Groceries','Transport','Eating_Out','Entertainment','Miscellaneous']:
            user_val = user_data_mapped[col].values[0]
            cluster_val = cluster_pro[col]
            if user_val > cluster_val:
                suggestions.append(f"Reduce {col}: currently ₹{user_val:.2f}, suggested ₹{cluster_val:.2f}")

        return{
            "Message": f"Your savings are not achievable. You must reduce your expenses by ₹{shortfall:.2f} to reach your saving goal.",
            "Suggestions": suggestions,
        }

    except Exception as e:
        raise Exception(f"Error in recommend_budget: {str(e)}")


# user_data = pd.DataFrame([{
#     "Eating_Out": 3000,
#     "Entertainment": 2000,
#     "Miscellaneous": 1500,
#     "Groceries": 9000,
#     "Transport": 4000,
#     "Desired_Savings": 10000,
#     "Disposable_Income": 6000
# }])


# print(recommend_budget(user_data,kmn,sc,cluster_profile))


import pickle
with open(r'C:\Users\HP\OneDrive\Documents\Etc\Project(Data Science)\Predictor\Budget Analysis\expenses_model.pkl','wb') as f:
    pickle.dump({'kmn':kmn,'sc':sc,'cluster_profile':cluster_profile},f)

