
# coding: utf-8

# # Notebook implementation showing GitHub Repository analysis for below use cases:#
# 1.  Recommend top-3 languages to the user based on the current usage of programming languages in the repositories which will show the      future popularity. This future popularity is decided on the weight of each language considering the bytes of code written in it.
# 2. Recommend repositories to users based on the stars they have given until now.
# 3. We further do the Social Analysis of top-20 recommended repos and pick three users to show case our analysis and results.

# # Acquiring Data From BigQuery #

# Using Google's BigQuery we queried the data required from the tables that we have used for the analysis. Two major tables are languages and sample_repos.

# In[ ]:


import os
import pandas
import numpy as np

from google.cloud import bigquery
client = bigquery.Client()

#Querying languages table
QUERY = """
        SELECT repo_name, language
        FROM `bigquery-public-data.github_repos.languages`
        LIMIT 300
        """

query_job = client.query(QUERY)

#Get results into the dataframe
languageData = query_job.to_dataframe()

#filtering repos which include only a single language
iterator = query_job.result(timeout=30)
rows = list(iterator)
rows = list(filter(lambda row: len(row.language)>1,rows))

#Printing first ten repositories
for i in range(10):
    print('Repository '+str(i+1))
    for j in rows[i].language:
        print(j[u'name']+': '+str(j[u'bytes'])+' bytes')
    print('')
print('...')
print(str(len(rows))+' repositories')


# This table has only the repo_name and corresponding language that it holds. If see here, we have repo_name with user and repository details.
# We feature engineered this attribute as further for matrix factorization we need specific user_id for users and specific repo_id for repositories.

# In[ ]:


languageData.head()


# # Listing the languages we got from above query after removing the repositories that have only one language#

# In[ ]:


#create dictionary of language names to matrix columns
names = {}
for i in range(len(rows)):
    for j in rows[i].language:
        if j[u'name'] in names:
            names[j[u'name']]+=1
        else:
            names[j[u'name']]=1

#filter out languages that only occur once
names = [n for n in names if names[n]>1]
# for i in range(10):
#     print(names[i])
# print('...')

#print some languages
name_to_index = {}
for j,i in enumerate(names):
    name_to_index[i] = j
print(str(len(names))+" languages")


# # Repository-Language Matrix #

# We thought of creating a matrix where rows represent the repository and column represent a language to know the relationship between the repository and language. This resulted into a sparse matrix where many repositories did had or did not had that specific language.
# 
# This was the desicion making point where we decided to go ahead with matrix factorization due to the sparsity of the data of what we aim in first use case.
# 
# We created a matrix as mentioned above. Also as mentioned, the value in the matrix is the log of number of bytes in the repository in that particular language to consider the actual weight of the code written in that language for that repository.
# 
# Log also gives us the feature minimization (feature scaling) for the random size of the bytes information we have.

# In[ ]:


from math import log

#create matrix
global mat
mat = np.zeros((len(rows),len(names)))
for i,row in enumerate(rows):
    #total = sum([log(lang[u'bytes']+1) for lang in row[1]])
    for lang in row.language:
        if lang[u'name'] in name_to_index and lang[u'bytes'] > 0:
            mat[i][name_to_index[lang[u'name']]] = log(lang[u'bytes'])
            #mat[i][name_to_index[lang[u'name']]] = log(lang[u'bytes']+1)/total
mat = mat[~np.all(mat==0,axis=1)]


# # PCA #

# Using PCA we can define roughly the number of features we want to identify the low rank matrix factorization. The graph below shows the amount of unexplained variance plotted against the number of components used. The "elbow" of the graph (at around n=12) is typically used.

# In[ ]:


from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

#compute PCA
n_components = min(50,len(names))
pca = PCA(n_components=n_components)
transformed = pca.fit_transform(mat) 

#display result
evr = [1-sum(pca.explained_variance_ratio_[:i+1]) for i in range(len(pca.explained_variance_ratio_))]
plt.plot(range(1,n_components+1),evr)


# # Loss Function and Gradient #

# We define some useful functions. 
# 
# init_mask: Create a mask matrix that indicates where Y has meaningful values. 
# 
# loss: Sum-of-squares loss, with a regularization term to prevent overfitting. The matrices theta and X are multiplied to give a "best guess", which is then compared with the target matrix Y, but only in locations where a rating has been given.
# 
# gradient: Derivative of loss with respect to theta and X, with a regularization term. 
# 
# These functions will be useful in performing gradient descent.

# In[ ]:


filter_size = min(100,len(mat[0]))
mat = mat[:,range(filter_size)] if len(mat[0])>filter_size else mat #for speed


#This function gives us the sign function implementation for where the target Y is achieved
def init_mask(Y):
    f = np.vectorize(lambda x: 1 if x>0 else 0)
    return f(Y),len(Y),len(Y[0])


#This function implements the regularization parameter to minimize the overfitting
def loss(args,Y,mask,n_repos,n_langs,n_features,reg_param):
    theta = np.reshape(args[:n_repos*n_features],(n_repos,n_features))
    X = np.reshape(args[n_repos*n_features:],(n_langs,n_features))
    g = np.vectorize(lambda x: x*x)
    return 0.5*np.sum(np.multiply(g(np.subtract(np.matmul(theta,np.transpose(X)),Y)),mask))+reg_param/2*np.sum(g(args))


#This function implements gradient calculation in vectorized way
def gradient(args,Y,mask,n_repos,n_langs,n_features,reg_param):
    theta = np.reshape(args[:n_repos*n_features],(n_repos,n_features))
    X = np.reshape(args[n_repos*n_features:],(n_langs,n_features))
    X_grad = np.matmul(np.transpose(np.multiply(np.subtract(np.matmul(theta,np.transpose(X)),Y),mask)),theta)+reg_param*X
    theta_grad = np.matmul(np.multiply(np.subtract(np.matmul(theta,np.transpose(X)),Y),mask),X)+reg_param*theta
    return np.concatenate((np.reshape(theta_grad,-1),np.reshape(X_grad,-1)))


# # Training our theta parameters using scipy's optimize library to implement gradient descent to match our target matrix achieved in mask function #

# Gradient descent is performed using loss and gradient as defined above. This will iteratively improve matrices theta and X, so that their product more closely matches the target masked matrix Y.  

# In[ ]:


import scipy.optimize as op

def train(Y,mask,n_repos,n_langs,n_features=10,reg_param=0.000001):
    #reshape into 1D format preferred by fmin_cg
    theta = np.random.rand(n_repos,n_features)
    X = np.random.rand(n_langs,n_features)
    args = np.concatenate((np.reshape(theta,-1),np.reshape(X,-1)))

    #use fmin_cg to perform gradient descent
    args = op.fmin_cg(lambda x: loss(x,Y,mask,n_repos,n_langs,n_features,reg_param),args,lambda x: gradient(x,Y,mask,n_repos,n_langs,n_features,reg_param))

    #reshape into a usable format
    theta = np.reshape(args[:n_repos*n_features],(n_repos,n_features))
    X = np.reshape(args[n_repos*n_features:],(n_langs,n_features))
    
    return theta,X


# # Giving language recommendations  based on the popularity of languages among the repositories #

# Now, we create a function for recommendations. Unfortunately Kaggle's front end doesn't allow for user input, so we will test some inputs manually.

# In[ ]:


def recommend(string,Y):
    #process input
    print('Training...')
    langs = string.split(' ')
    lc_names = {str(name).lower(): name_to_index[name] for name in name_to_index}

    #create extra row to append to Y matrix
    test = np.zeros((1,len(names)))
    known = set()
    for lang in langs:
        if lang.lower() in lc_names:
            test[0][lc_names[lang.lower()]] = 1
            known.add(lc_names[lang.lower()])

    #training
    Y = np.concatenate((Y,test[:,range(filter_size)]),0)
    mask,n_repos,n_langs = init_mask(Y)
    theta,X = train(Y,mask,n_repos,n_langs)
    Y = Y[:-1]
    
    #plot features
    for i in range(np.shape(X)[1]):
        col = sorted([(X[j,i],j) for j in range(n_langs)],reverse=True)
        #print('')
        #for k in range(10):
            #print(names[col[k][1]])

    #find top predictions
    predictions = np.matmul(theta,np.transpose(X))[-1].tolist()
    predictions = sorted([(abs(j),i) for i,j in enumerate(predictions)],reverse=True)

    #print predictions
    predictedLang = []
    i = 0
    for val,name in predictions:
        if name not in known:
#             print(str(i+1)+': '+names[name]+' - '+str(val))
            predictedLang.append(names[name])
            i+=1
        if i>=3:
            break
    return predictedLang


# The recommender system adds the extra input row to target matrix Y. Then, training is performed on your language preferences simultaneously as those of all the repositories in the sample. Finally, the trained matrices theta and X are multiplied, and the last row corresponds to the predicted ratings based on your preferences. The highest values are the languages recommended to you. 
# 
# # Manually testing for 'Java' #

# In[ ]:


languagesList = recommend('Java',mat)


# In[ ]:


languagesList


# ## Below function is used to filter the repositories to based on the recommended languages

# In[ ]:


def filterDataFrame(df, languagesList):
    reposList = []
    index = 0
    for index, row in languageData.iterrows():
       if len(row['language']) != 0:
            for itemOfLanguages in range(len(row['language'])):
                if row['language'][itemOfLanguages]['name'] in languagesList:
                    reposList.append(row['repo_name'])
                    break
    return reposList


# ## As see below we see we get the repository list to which the above recommended langauges belong

# In[ ]:


repoList = filterDataFrame(languageData, languagesList)


# ## Here in the below function we recommend the top 3 repositories which include our recommended languages

# In[ ]:


templanguageMatchingDict = {
    "reponame": "",
    "matchcount": 0
}

finalMatchingList = []
for i in range(len(repoList)):
    
    for index, row in languageData.iterrows():
        if row['repo_name'] == repoList[i]:
            templanguagelist = [li['name'] for li in row['language']]
            
            length = len(set(templanguagelist) & set(languagesList))
            templanguageMatchingDict = {
                "reponame": row['repo_name'],
                "matchcount": length
            }
            finalMatchingList.append(templanguageMatchingDict)
            break


# In[ ]:


newlist = sorted(finalMatchingList, key=lambda k: k['matchcount'], reverse=True)


# ## Recommending top 3 repositories  for the top 3 languages recommended as per first use case

# In[ ]:


newlist[:3]


# # From here on our second use case of recommending users with top 20 repositories based on Github stars begins

# ## Querying the sample_repos table to get the sample repositories and stars given to it
# 
# We could not load the current star details as we could only make 60 GET requests to GitHub APIs in an hour so had to reside with sample_repos data

# In[1]:


#random sample of 300 repositories
import os
import pandas
import numpy as np

from google.cloud import bigquery
client = bigquery.Client()
QUERY1 = """
        SELECT repo_name, watch_count
        FROM `bigquery-public-data.github_repos.sample_repos`
        LIMIT 1000
        """

query_job1 = client.query(QUERY1)

#filter out repositories with only one language
iterator = query_job1.result(timeout=30)
rows = list(iterator)


# ## Preprocessing the data to split the repository names to get users, repos, ids and corresponding stars

# In[2]:


user_repo_watch_count_dict = {
    "user": [],
    "repo_name": [],
    "stars": [],
    "user_repo_name":[]
}



for i in range(len(rows)):
    repo_name, stars = rows[i]
    temp = []
    temp = rows[i].repo_name.split('/')
    user_repo_watch_count_dict["user"].append(temp[0])
    user_repo_watch_count_dict['user_repo_name'].append(repo_name)
    user_repo_watch_count_dict["repo_name"].append(temp[1])
    
    user_repo_watch_count_dict["stars"].append(stars)
    


# In[3]:


user_repo_watch_count_df = pandas.DataFrame.from_dict(user_repo_watch_count_dict)


# ## Showing the new features we got after splitting the data

# In[4]:


user_repo_watch_count_df.head()


# In[5]:


df_user_repo_mat_fac_stars = pandas.DataFrame()


users = set(user_repo_watch_count_dict["user"])
userl = list(users)
distinct_repos = set(user_repo_watch_count_dict["repo_name"])
repol = list(distinct_repos)


# ## Here below we give unique id values to users

# In[6]:


userList = []

for i in range(len(users)):
    user = {
        "userid": i,
        "user": userl[i]
    }
    userList.append(user)
df_user_id = pandas.DataFrame(userList, columns=['userid','user'])


# In[7]:


df_user_id.head()


# ## Here below we give unique id values to repo_names

# In[8]:


repoListNew = []

for i in range(len(distinct_repos)):
    repo = {
        "repoid": i,
        "repo_name": repol[i]
    }
    repoListNew.append(repo)
df_repo_id = pandas.DataFrame(repoListNew, columns=['repoid','repo_name'])


# In[9]:


df_repo_id_copy = df_repo_id.copy(deep=True)


# In[10]:


df_repo_id.head()


# ## Here we merge the dataframe carrying the user id and user name with user repository watch_counts (stars) dataframe into one dataframe

# In[11]:


df_final_user_repo_star_1 = pandas.merge( df_user_id, user_repo_watch_count_df, how='inner', on=['user'])


# In[12]:


df_final_user_repo_star_repo_id_merge = pandas.merge( df_repo_id, user_repo_watch_count_df, how='inner', on=['repo_name'])


# In[13]:


df_final_user_repo_star_repo_id_merge.head()


# ## Below we try to see the stars for repository to check if there are any outliers that we need to take out

# In[14]:


df_final_user_repo_star_repo_id_merge.plot.scatter(x='repoid', y = 'stars')


# ## Above graph shows one repository with stars greater than 80,000 and hence we can consider this as an outlier and remove it from dataset.
# 
# ## Below we removed it and get more general plot after removing it.

# In[15]:


df_final_user_repo_star_repo_id_merge.drop(df_final_user_repo_star_repo_id_merge.loc[df_final_user_repo_star_repo_id_merge['stars']== df_final_user_repo_star_repo_id_merge['stars'].max() ].index, inplace=True)


# ## Plot after removing the outlier

# In[16]:


df_final_user_repo_star_repo_id_merge.plot.scatter(x='repoid', y = 'stars')


# ## We also see that now the dataframe has more even spreadout if you check the 25%, 50% and 75% statistics

# In[17]:


df_final_user_repo_star_repo_id_merge.describe()


# ## Here below we copy the current dataframe to get a new dataframe which we need for searching later for recommendations

# In[109]:


df_final_user_repo_star_repo_id_merge_copy = df_final_user_repo_star_repo_id_merge.copy(deep=True)


# In[19]:


df_final_user_repo_star_repo_id_merge_copy.head()


# ## We just checked if we have got the correct unique repositories as they have to be unique

# In[20]:


df_final_user_repo_star_repo_id_merge_copy['user_repo_name'].unique()


# In[21]:


df_final_user_repo_star_repo_id_merge_copy.head()


# In[22]:


df_final_user_repo_star_1.drop(df_final_user_repo_star_1.loc[df_final_user_repo_star_1['stars'] == df_final_user_repo_star_1['stars'].max() ].index, inplace=True)
#df_final_user_repo_star_1.drop(df_final_user_repo_star_1.loc[df_final_user_repo_star_1['stars'] > 6000 ].index, inplace=True)


# ## Now that we have repoid and correct repositories, we merge it with user ids to get ids concatenated to the new dataframe

# In[23]:


df_final_user_repo_star_2 = pandas.merge(df_repo_id, df_final_user_repo_star_1, how='inner', on=['repo_name'])


# In[24]:


df_final_user_repo_star_2.head()


# In[25]:


df_final_user_repo_star_2.describe()


# In[26]:


print(df_final_user_repo_star_2['stars'].max())


# In[28]:


df_final_user_repo_star_2.drop(df_final_user_repo_star_2.loc[df_final_user_repo_star_2['stars'] == df_final_user_repo_star_2['stars'].max() ].index, inplace=True)


# In[29]:


df_final_user_repo_star_2.describe()


# ## Here in this approach we used SVD standard library which expects input in the format of 'User', 'Item' and 'Rating', so do the next processing steps to create the dataframe into required labels and parameters

# In[30]:


df_final_user_repo_star_v1 = pandas.DataFrame(df_final_user_repo_star_2, columns=['userid','repoid', 'stars'])


# In[31]:


df_final_user_repo_star_v2 = df_final_user_repo_star_v1.rename(index = str, columns={'userid':'user', 'repoid':'item', 'stars':'rating'})


# In[32]:


df_final_user_repo_star_2['stars'].hist( bins = 100)


# ## Plot for 'repoid' vs 'stars'

# In[33]:


df_final_user_repo_star_2.plot.scatter(x='repoid', y = 'stars')


# ## Plot for 'userid' vs 'stars'

# In[34]:


df_final_user_repo_star_2.plot.scatter(x='userid', y = 'stars')


# In[35]:


df_final_user_repo_star_v2.describe()


# ## As seen above still the 'stars' are not in normalized range, so we have used 'divide by mean' strategy to normalize the data for 'stars'

# In[36]:


df_final_user_repo_star_v3 = df_final_user_repo_star_v2.copy(deep=True)
# rating_max = df_final_user_repo_star_v2.rating.max()
# rating_min = df_final_user_repo_star_v2.rating.min()
rating_mean = df_final_user_repo_star_v2.rating.mean()
df_final_user_repo_star_v3['rating'] = df_final_user_repo_star_v3.apply(lambda row: row['rating']/rating_mean, axis=1)


# In[37]:


df_final_user_repo_star_v3.describe()


# In[38]:


df_final_user_repo_star_v3.head()


# ## Training multiple models :
# 1. SVD
# 2. KNNBasic
# 3. CoClustering
# 4. SlopeOne

# ## SVD Implementation without train, test split

# In[83]:


listOfRMSE = []
models = []


# In[84]:


from surprise import Reader, Dataset, SVD, SVDpp, evaluate, accuracy
from surprise.model_selection import train_test_split
reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,6))
df_temp1 = df_final_user_repo_star_v3.copy(deep=True);
data = Dataset.load_from_df(df_temp1, reader)
# Test that surprise is working by running SVD on the dataset

# We'll use the famous SVD algorithm.
algo = SVD(n_factors= 100, n_epochs= 20, biased=True, init_std_dev=0.1, lr_all=0.005)

# Train the algorithm on the trainset, and predict ratings for the testset
trainset = data.build_full_trainset()

algo.fit(trainset)

testset = trainset.build_anti_testset()
svd_predictions = algo.test(testset)

rmse_svd = accuracy.rmse(svd_predictions)
print(rmse_svd)
listOfRMSE.append(rmse_svd)
models.append('SVD')


# ## SVD Implementation with KFold Cross Validators

# In[85]:


from surprise.model_selection import KFold
# define a cross-validation iterator
n_splits = 3
kf = KFold(n_splits=3)

algo = SVD()
tempRMSE = 0 
for trainset, testset in kf.split(data):

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
    tempRMSE += accuracy.rmse(predictions, verbose=False)

listOfRMSE.append(tempRMSE/n_splits)
models.append('SVD with KFold')


# ## KNNBasic Implementation using 'pearson' similarity

# In[86]:


# We'll use the famous SVD algorithm.
from surprise import KNNBasic

df_knnbasic = df_final_user_repo_star_v3.copy(deep=True);
dataknnbasic = Dataset.load_from_df(df_knnbasic, reader)

sim_options = {
    'name': 'pearson'
}

algo1 = KNNBasic(sim_options = sim_options, k=40,min_k=1)

# Train the algorithm on the trainset, and predict ratings for the testset
trainset1 = dataknnbasic.build_full_trainset()

algo1.fit(trainset1)

testset1 = trainset1.build_anti_testset()
KNNBasic_pearson_predictions = algo1.test(testset1)

accuracy.rmse(KNNBasic_pearson_predictions)
listOfRMSE.append(accuracy.rmse(KNNBasic_pearson_predictions))
models.append('KNN with Pearson')


# ## KNNBasic Implementation using 'cosine' similarity

# In[87]:


sim_options = {
    'name': 'cosine'
}

algo1 = KNNBasic(sim_options = sim_options, k=40,min_k=1)

# Train the algorithm on the trainset, and predict ratings for the testset
trainset1 = dataknnbasic.build_full_trainset()

algo1.fit(trainset1)

testset1 = trainset1.build_anti_testset()
KNNBasic_cosine_predictions = algo1.test(testset1)

cosine_rmse = accuracy.rmse(KNNBasic_cosine_predictions)
listOfRMSE.append(cosine_rmse)
models.append('KNN with cosine')


# ## CoClustering Implementation

# In[88]:


# We'll use the famous SVD algorithm.
from surprise import CoClustering

df_CoClustering = df_final_user_repo_star_v3.copy(deep=True);
dataCoClustering = Dataset.load_from_df(df_CoClustering, reader)


coClustering = CoClustering(n_cltr_u=3, n_cltr_i=3, n_epochs=20)

# Train the algorithm on the trainset, and predict ratings for the testset
trainsetcoClustering  = dataCoClustering.build_full_trainset()

coClustering.fit(trainsetcoClustering)

testcoClustering = trainsetcoClustering.build_anti_testset()
predictionscoClustering = coClustering.test(testcoClustering)

accuracy.rmse(predictionscoClustering)


listOfRMSE.append(accuracy.rmse(predictionscoClustering))
models.append('CoClustering')


# ## SlopeOne Implementation

# In[89]:


from surprise import SlopeOne
slopeOne = SlopeOne()

# Train the algorithm on the trainset, and predict ratings for the testset
trainsetslopeOne  = dataCoClustering.build_full_trainset()

slopeOne.fit(trainsetslopeOne)

testslopeOne = trainsetslopeOne.build_anti_testset()
predictionsslopeOne = slopeOne.test(testslopeOne)

accuracy.rmse(predictionsslopeOne)

listOfRMSE.append(accuracy.rmse(predictionsslopeOne))
models.append('SlopeOne')


# In[91]:


models


# In[105]:


import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = (20,3)
# x-coordinates of left sides of bars 
Algorithms = [1, 2, 3, 4, 5, 6] 

# heights of bars 
RMSE = listOfRMSE

# labels for bars 
label = models

# plotting a bar chart 
plt.bar(Algorithms, RMSE, tick_label = label, 
		width = 0.8) 

# naming the x-axis 
plt.xlabel('x - axis') 
# naming the y-axis 
plt.ylabel('y - axis') 
# plot title 
plt.title('My bar chart!') 

# function to show the plot 
plt.show()


# ## Getting top 20 recommendations (We are getting 20 because we want to do further Social Analysis of this data and see if we can do community detections)

# In[94]:


from collections import defaultdict
def get_top_n(predictions, n=10):
    
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        ## Multiplying by mean to predicted rating i.e est as we had first divided by mean for normalizing
        top_n[uid].append((iid, est*rating_mean)) 

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# ## For each user we give out top 20 recommended repositories thus our second use case is implemented firstly with SVD which gave pretty diverse recommendations compared to KNN

# In[107]:


top_n = get_top_n(KNNBasic_pearson_predictions, n=20)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])


# In[108]:


top_n = get_top_n(svd_predictions, n=20)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])


# ## Helper function program to create a list of users with repository and correspoding stars for further graph plotting and social analysis

# In[102]:


count = 0
user_graph_repo_stars_list = []
for uid, user_ratings in top_n.items():
    count += 1
    if count > 3:
        break
    graph_list1 = []
    for repoid, stars_computed in user_ratings:
        #print(repoid)
        graph_dict = {
            "repoid": repoid,
            "stars":stars_computed
        }
        graph_list1.append(graph_dict)
        temp = {
            "user": uid,
            "list": graph_list1
        }
    user_graph_repo_stars_list.append(temp)
    #print(uid, [iid for (iid, _) in user_ratings])
    
    


# ## Setting index on dataframe to search with 'repoid'

# In[110]:


df_final_user_repo_star_repo_id_merge_copy1 = df_final_user_repo_star_repo_id_merge_copy.copy(deep=True)
df_final_user_repo_star_repo_id_merge_copy1.set_index('repoid', inplace=True)


# ## Below we create final list with which we do the social analysis of user, repository and language.
# 
# ## we have implemented this as an extended functionality for study purpose so only first three users with corresponding top 20 recommended repositories and predicted stars is used

# In[111]:


graph_final_list_user_reponame = []
for i in user_graph_repo_stars_list:
    user = i['user']
    #print(df_user_id.loc[user]['user'])
    repos = i['list']
    
    temp_list = []
    for j in repos:
        id = j['repoid']
        stars = j['stars']
        temp = df_final_user_repo_star_repo_id_merge_copy1.loc[id]['user_repo_name']
        if type(temp) == str:
            repo_dict = {
                "reponame": temp,
                "stars": stars
            }
            temp_list.append(repo_dict)
    
    user_dict = {
        "user": user,
        "list": temp_list
    }    
    graph_final_list_user_reponame.append(user_dict)


# In[112]:


graph_final_list_user_reponame


# In[113]:


user_list_for_graph = []
repo_list_for_graph = []
stars_list_for_graph = []
for row in graph_final_list_user_reponame:
    index = len(row['list'])
    for i in range(index):
        user_list_for_graph.append(row['user'])
        repo_list_for_graph.append(row['list'][i]['reponame'])
        stars_list_for_graph.append(row['list'][i]['stars'])


# In[114]:


repository_url = []
for val in repo_list_for_graph:
    temp = "https://github.com/" + val
    repository_url.append(temp)
    
repository_url


# In[115]:


repo_stars_dataframe = pandas.DataFrame({'User':user_list_for_graph, 'URL':repository_url, 'Repos':repo_list_for_graph, 'Stars':stars_list_for_graph})


# In[116]:


repo_stars_dataframe.to_csv("repo_stars_dataframe.csv", sep='\t')


# # Use case 3: Network Analysis

# ## Import necessary libraries

# In[3]:


import pandas as pd
import igraph


# ## Read the input file and drop the unnecessary columns

# In[11]:


repo_stars_dataframe = pd.read_csv('repo_stars_dataframe.csv', sep='\t')
repo_stars_dataframe.drop(repo_stars_dataframe.columns[0], axis = 1, inplace=True)
repo_stars_dataframe.head()


# ## Create the list of nodes which has highest watch count

# In[5]:


indexList = [0]
for i in range(len(repo_stars_dataframe)-1):
    if repo_stars_dataframe['User'][i] != repo_stars_dataframe['User'][i+1]:
        indexList.append(i+1)


# ## Add the nodes to graph

# In[6]:


g = igraph.Graph()
graph_user = repo_stars_dataframe['User']
graph_repos_url = repo_stars_dataframe['URL']
graph_repos = repo_stars_dataframe['Repos']
graph_watchers = repo_stars_dataframe['Stars']

index = 0
for row in range(len(repo_stars_dataframe)):
    g.add_vertex(name=graph_repos_url[index],
            label=graph_repos[index],
            watchers=int(graph_watchers[0]))
    index += 1


# ## Make repo of rows having highest watch count for every user

# In[7]:


repo1 = repo_stars_dataframe.copy(deep=True)
dropIndexList = []
for index in range(len(repo_stars_dataframe)):
    if index not in indexList:
        dropIndexList.append(index)

repo1.drop(dropIndexList, inplace=True)
repo1


# ## Connect the nodes on the basis of watch count

# In[8]:


edges_dataframe = pd.merge(repo1, repo_stars_dataframe, how='inner', on=['User'])
repo1 = edges_dataframe['URL_x']
repo2 = edges_dataframe['URL_y']
weight = edges_dataframe['Stars_x'] * edges_dataframe['Stars_y']

index = 0
for row in range(len(edges_dataframe)):
    g.add_edge(repo1[index], repo2[index],
            weight=float(weight[index]))
    index += 1


# ## Export the graph file to visualize using Gephi

# In[9]:


g.write('repos.gml')
g.summary()


# ## Exported file repos.gml is processed in Gephi to generate the graph

# In[10]:


get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
figure(figsize = (50,20))
img=mpimg.imread('graph.png')
imgplot = plt.imshow(img)
plt.show()


# ## We faced a major drawback with kaggle here, we were using igraph to plot the graph but we got the error of libraries not being found to load
# 
# ## We tried a lot but at last we resided to do it on local machine

# ## Conclusion:
# 1.   We implemented matrix factorization using various approach for recommendations, we used classical gradient descent based                approach for our first use case and standard implementations for our second use case
# 2.  We studied in depth how matrix factorization will work on a real world scenario, as well we learned how important data preprocessing and how it can affect our model's behavior.
# 3.  Further we also got some hands-on on social network analysis with minimal use case applied.
# 4.  We learned how much cross validation is necessary for model selection. 
# 5.  We got lowest RMSE score with simple SVD over other models, which we selected as our final model.
# 

# ## References:
# 1. Surprise library documenation:
# https://surprise.readthedocs.io/en/stable/getting_started.html
# 2. Nick Becker's post on Matrix Factorization:
# https://beckernick.github.io/matrix-factorization-recommender/
# 3. Got first use case help from Lawrence Pang's analysis:
# https://www.kaggle.com/lpang36/recommend-languages-with-collaborative-filtering
# 4. Got third use case help from Corey Ford's analysis:
# https://github.com/coyotebush/github-network-analysis
